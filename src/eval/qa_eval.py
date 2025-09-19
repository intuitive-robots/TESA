import math
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.eval.qa_loader import (all_img_ids, filter_collection,
                                num_questions)
from src.eval.qa_statementGen import get_question_to_statements_function
from src.models.component_handler import qa_collection, similarity_from_config
from src.util.device import DEVICE

from transformers import GPT2Config, GPT2Model, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, GPT2Tokenizer

from einops import einops

DATA_BASE = os.getenv("DATA_DIR")


def eval_qa_type(qa_dicts, config, img_id_to_y, s_gen):
    r"""@return number of correct answers."""
    correct = 0
    f = 0
    part_len_data = 0
    sim = similarity_from_config(config)
    question_tqdm = tqdm(qa_dicts, desc=f"eval questions")
    for qa_dict in question_tqdm:
        part_len_data += 1
        my_embedding = img_id_to_y[qa_dict["imageId"]]
        my_embedding = my_embedding.to(DEVICE)
        question = qa_dict["question"]
        try:
            statements_on_device_vec = s_gen(
                question)  # try generating question
        except Exception:
            f += 1
            # print(f"(f) {question}")
            continue  # fail silent :)
        sims = sim(statements_on_device_vec, my_embedding)
        if sims.argmax().item() == qa_dict["answer_index"]:
            correct += 1
            # # qualitativ:
            """
            print(question)0
            vals, inds = sims.sort()
            for i in [-1,-2,-3,2,1,0]:
                print(f"{statements[inds[i]]} : {vals[i]}")
            """
        question_tqdm.set_description(
            f"eval questions - prel. accuracy: {(correct/part_len_data*100):.2f}%"
        )
    return correct, f


def eval_qa(config, graphDataset):
    # if config["dataset"] == "psg":
    #     print("skipping qa evaluation because dataset==PSG")
    #     return {}
    with torch.no_grad():
        question_collection = qa_collection(config)

        # make embeddings
        file_name_to_y = {}
        file_name_to_full_filename = {}
        for d in tqdm(graphDataset):
            parsed_filename = d["file_name"].split("/")[-1].lstrip("0")
            if parsed_filename in file_name_to_y.keys():
                fullname = file_name_to_full_filename[parsed_filename]
                duplname = d["file_name"]
                if fullname != duplname:  # ID clash
                    print(
                        f"WARN! Duplicate graph {parsed_filename} found in dataset. Ignoring it.\n Full name: {fullname}\n Dupl name: {duplname}"
                    )
                else:
                    pass  # duplicate graph annotation for same graph
            else:
                file_name_to_y[parsed_filename] = d.y
                file_name_to_full_filename[parsed_filename] = d["file_name"]

        img_id_to_y = {
            id: file_name_to_y.get(f"{id}.jpg")
            for id in tqdm(
                all_img_ids(question_collection), desc="qa: select needed embeddings"
            )
        }
        for k, v in question_collection.items():
            print(
                f"qa (before filtering): {k} has {len(v['data'])} questions, {len(v['all_answers'])} answers."
            )
        question_collection, drop_count = filter_collection(
            question_collection,
            [k for k, v in img_id_to_y.items() if v is not None],
            True,
        )
        img_id_to_y = {
            id: file_name_to_y.get(f"{id}.jpg")
            for id in tqdm(
                all_img_ids(question_collection), desc="qa: select needed embeddings"
            )
        }
        print(f"Now contain only {len(img_id_to_y)} images.")
        if config.get("run_label") == "DEBUG":
            # DBG FOR FAST RUN
            dbg_trim = 100
            print(
                f"[qa_eval] Debug mode. Trimming to {dbg_trim} questions per type.")
            for k in question_collection.keys():
                question_collection[k]["data"] = question_collection[k]["data"][
                    :dbg_trim
                ]
            img_id_to_y = {
                id: file_name_to_y.get(f"{id}.jpg")
                for id in tqdm(
                    all_img_ids(question_collection), desc="qa: select needed embeddings"
                )
            }
            print(f"Now contain only {len(img_id_to_y)} images.")
        print(
            f"Dropped {drop_count} questions.\nFinal QA dataset: {num_questions(question_collection)} questions, on {len(img_id_to_y)} images."
        )
        for k, v in question_collection.items():
            print(
                f"qa preview: {k} has {len(v['data'])} questions, {len(v['all_answers'])} answers."
            )
        total_qa_score = 0
        returndict = {}

        for structural_type, dataset in question_collection.items():
            whitelist = config.get("eval.qa.structure_whitelist", None)
            if whitelist is not None and structural_type not in whitelist:
                print(f"Skipping {structural_type}.")
                continue
            k, v = structural_type, dataset
            s_gen = get_question_to_statements_function(
                structural_type, dataset["all_answers"], config
            )
            if s_gen is None or structural_type in config.get("eval.qa.skip_types", []):
                print(
                    f"WARN: SKIPPING QUESTIONS of structural type {structural_type}")
                continue
            data = dataset["data"]
            print(
                f"qa:{k}. has {len(v['data'])} questions, {len(v['all_answers'])} answers."
            )
            score, parse_f = eval_qa_type(data, config, img_id_to_y, s_gen)
            print(
                f"qa:{structural_type} {parse_f} questions failed. {score} correct. out of {len(data)}"
            )
            total_qa_score += score
            returndict[f"/{structural_type}"] = score / max(
                1, len(data)
            )  # Note: stuff with errors (no image) counts as 0 towards score

        returndict["/total"] = total_qa_score / \
            num_questions(question_collection)
        return returndict

def train_eval_qa_model(config, graphDataset, train_bool, replace, model):
    graphDataset_train, graphDataset_test = graphDataset
    
    categories = ['compare', 'query', 'choose', 'logical', 'verify']
    #categories = ['logical']
    
    config['eval.qa.gqa.question_file'] = "train_balanced_questions.json"
    question_collection_train_pre = qa_collection(config)
    
    question_collection_train_pre_sub = {}
    for cat in categories:
        question_collection_train_pre_sub[cat] = question_collection_train_pre[cat]
    
    model_name = config['dataset.base_embedding.using']
    
    run_config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": 512,
            "lr": 5e-5,
            "epochs": 14,
            "max_len": 32,
            'dataset_name': config['dataset'],
            'model_name': model_name
        }
    
    img_dim = 512
    if model_name == 'SigLIP':
        img_dim = 768
    elif model_name == 'DINOv2':
        img_dim = 384 
    
    names = ["vqa_model_" + run_config['dataset_name'] + "_" + run_config['model_name'] + ".pth"]#, "vqa_model_gqa_DINOv2.pth", "vqa_model_gqa_SigLIP.pth"]
        
    answers = {}
    for k in question_collection_train_pre:
        answers[k] = question_collection_train_pre[k]['all_answers']
    config['eval.qa.gqa.question_file'] = "val_balanced_questions.json"
    question_collection_val_pre = qa_collection(config, answers)
    
    question_collection_val_pre_sub = {}
    for cat in categories:
        question_collection_val_pre_sub[cat] = question_collection_val_pre[cat]
    
    question_collection_val, img_id_to_y_val = get_question_collection(config, graphDataset_test, question_collection_val_pre_sub, replace, model)
    
    question_collection_train, img_id_to_y_train = get_question_collection(config, graphDataset_train, question_collection_train_pre_sub, replace, model)
    
    if train_bool:
        names = train_vqa_model(run_config, question_collection_train, img_id_to_y_train, img_dim, question_collection_val, img_id_to_y_val)
    
    for name in names:
        print('Test eval set for feature vector input')
        results = test_vqa_model(run_config, question_collection_val, img_id_to_y_val, name, img_dim, True)
        print('Test eval set for zero vector input')
        results = test_vqa_model(run_config, question_collection_val, img_id_to_y_val, name, img_dim, False)
        # print('Test train set')
        # results = test_vqa_model(run_config, question_collection_train, img_id_to_y_train, name, img_dim)

def test_vqa_model(config, ques_col, img_id, name, img_dim, feature_vector):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no PAD by default

    # Build dataset + loader
    dataset = VQADataset(ques_col, img_id, tokenizer, max_len=config["max_len"])
    loader = DataLoader(dataset, batch_size=config["batch_size"],
                        shuffle=False, collate_fn=collate_fn)

    # Init model
    model = VQA_model(
        img_feature_dim=img_dim,
        vocab_size=tokenizer.vocab_size,
        num_answer_classes=[len(ques_col[a]['all_answers']) for a in ques_col],
        answer_types=list(ques_col.keys()),
        max_len=config['max_len']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load("/home/vquapil/TESA/" + name))
    model.eval()
    
    stats = {atype: {"correct": 0, "total": 0} for atype in ques_col.keys()}

    with torch.no_grad():
        for q_tokens, img_feats, labels, types in tqdm(loader, desc="Evaluating"):
            q_tokens = q_tokens.to(config["device"])
            if feature_vector:
                img_feats = img_feats.to(config["device"])
            else:
                img_feats = torch.zeros_like(img_feats).to(DEVICE)
            labels = labels.to(config["device"])

            # Group by answer_type
            unique_types = set(types)
            for atype in unique_types:
                mask = [i for i, t in enumerate(types) if t == atype]
                if len(mask) == 0:
                    continue

                q_batch = q_tokens[mask]
                img_batch = img_feats[mask]
                y_batch = labels[mask]

                logits = model(q_batch, img_batch, atype)   # (B_sub, num_classes)
                preds = logits.argmax(dim=-1)

                correct = (preds == y_batch).sum().item()
                total = y_batch.size(0)

                stats[atype]["correct"] += correct
                stats[atype]["total"] += total

    # 🔹 Compute accuracies
    results = {atype: (s["correct"] / s["total"] if s["total"] > 0 else 0.0)
               for atype, s in stats.items()}

    # Print per-category accuracy
    for atype, acc in results.items():
        print(f"{atype}: {acc*100:.2f}%")

    # Return dict for further use
    return results

# 🔹 Training loop
def train_vqa_model(config, ques_col, img_id, img_dim, ques_col_val, img_id_val):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no PAD by default

    # Build dataset
    dataset = VQADataset(ques_col, img_id, tokenizer, max_len=config["max_len"])

    # 🔹 Build loaders
    loader = DataLoader(
        dataset, batch_size=config["batch_size"],
        shuffle=True, collate_fn=collate_fn
    )

    # Init model
    model = VQA_model(
        img_feature_dim=img_dim,
        vocab_size=tokenizer.vocab_size,
        num_answer_classes=[len(ques_col[a]['all_answers']) for a in ques_col],
        answer_types=list(ques_col.keys()),
        max_len=config['max_len']
    ).to(DEVICE)

    # Losses for each head (CrossEntropy)
    criterions = {atype: nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1) for atype in ques_col.keys()}

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.15)
    total_steps = config['epochs'] * len(loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    name = "vqa_model_" + config['dataset_name'] + "_" + config['model_name'] + ".pth"
    torch.save(model.state_dict(), "/home/vquapil/TESA/" + name)
    
    # Training loop
    for epoch in range(config["epochs"]):
        if epoch % 3 == 0:
            model.eval()
            print('Test eval set')
            results = test_vqa_model(config, ques_col_val, img_id_val, name, img_dim)
            print('Test train set')
            results = test_vqa_model(config, ques_col, img_id, name, img_dim)
        
        model.train()
        total_loss = 0
        for q_tokens, img_feats, labels, types in (pbar := tqdm(loader)):
            q_tokens = q_tokens.to(config["device"])
            img_feats = img_feats.to(config["device"])
            labels = labels.to(config["device"])

            optimizer.zero_grad()
            batch_losses = []

            # 🔹 Group by answer_type so we can compute in parallel
            unique_types = set(types)
            for atype in unique_types:
                mask = [i for i, t in enumerate(types) if t == atype]
                if len(mask) == 0:
                    continue

                q_batch = q_tokens[mask]
                img_batch = img_feats[mask]
                y_batch = labels[mask]

                logits = model(q_batch, img_batch, atype)   # (B_sub, num_classes)
                loss = criterions[atype](logits, y_batch)
                batch_losses.append(loss)

            # 🔹 Backward once on mean loss
            if batch_losses:
                loss = torch.concat(batch_losses).mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(loader):.4f}")
        
        torch.save(model.state_dict(), "/home/vquapil/TESA/" + name)
        
    return [name]

class VQA_model(nn.Module):
    def __init__(self,
                 vocab_size,               # question tokenizer vocab size
                 img_feature_dim=512,      # CLIP global embedding dim
                 num_layers=6,
                 num_heads=8,
                 max_len=32,
                 num_answer_classes=[],    # e.g., [2, 1542, 12642]
                 answer_types=[]           # e.g., ["yesno", "number", "other"]
                ):
        super().__init__()

        assert len(num_answer_classes) == len(answer_types), \
            "num_answer_classes and answer_types must align"

        # GPT2 embeddings
        gpt2 = GPT2Model.from_pretrained("gpt2")
        self.token_embedding = gpt2.wte
        self.hidden_dim = gpt2.config.n_embd
        self.pos_embedding = gpt2.wpe

        dropout = 0.2

        # Project image embedding
        self.img_proj = nn.Sequential(
            nn.Linear(img_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),  # 🔹 dropout after first activation
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(p=dropout)   # 🔹 optional: dropout after second layer too
        )
        self.img_ln = nn.LayerNorm(self.hidden_dim)

        # GPT2 transformer
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=self.hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            use_cache=False,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.decoder = GPT2Model(config)

        # Add the new pooling layer
        self.pooling = AttentionPooling(self.hidden_dim, num_heads=num_heads)

        # Heads per answer type
        self.heads = nn.ModuleDict({
            atype: nn.Linear(self.hidden_dim, nclass)  # concat [CLS] + IMG
            for atype, nclass in zip(answer_types, num_answer_classes)
        })

        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, question_tokens, img_features, answer_type):
        B, L = question_tokens.shape
        device = question_tokens.device

        # ------------------------------
        # Question embeddings
        # ------------------------------
        tok_emb = self.token_embedding(question_tokens)             # (B,L,H)
        pos_ids = torch.arange(L, device=device) + 2
        pos_emb = self.pos_embedding(pos_ids)[None, :, :]
        q_emb = tok_emb + pos_emb                                   # (B,L,H)

        # ------------------------------
        # Project image embedding (single token)
        # ------------------------------
        img_token = self.img_proj(img_features)
        img_token = self.img_ln(img_token)
        img_token = img_token.unsqueeze(1)                          # (B,1,H)
        img_pos_emb = self.pos_embedding(torch.tensor([1], device=device)).unsqueeze(0)
        img_token = img_token + img_pos_emb                         # (B,1,H)

        # ------------------------------
        # CLS token
        # ------------------------------
        cls_emb = einops.repeat(self.cls_token, 'b n d -> r (b n) d', r=B)
        cls_pos_emb = self.pos_embedding(torch.tensor([0], device=device)).unsqueeze(0)
        cls_emb = cls_emb + cls_pos_emb

        # ------------------------------
        # Concatenate [CLS] + IMG + QUESTION
        # ------------------------------
        x = torch.cat([cls_emb, img_token, q_emb], dim=1)           # (B,L+2,H)

        # ------------------------------
        # Transformer
        # ------------------------------
        hidden_states = self.decoder(inputs_embeds=x).last_hidden_state  # (B,L+2,H)

        # Instead of picking tokens, use the attention pooling layer
        pooled_representation = self.pooling(hidden_states) # (B, H)
        
        # Optional: Add dropout for regularization
        pooled_representation = self.dropout(pooled_representation)

        # ------------------------------
        # Use CLS + global image token for prediction
        # ------------------------------
        #pooled_cls = hidden_states[:, 0, :]         # (B,H)
        #pooled_img = hidden_states[:, 1, :]         # (B,H)
        #concat = torch.cat([pooled_cls, pooled_img], dim=-1) # (B, 2*H)
        #concat = self.dropout(concat)

        logits = self.heads[answer_type](pooled_representation)    # (B,num_classes)
        return logits


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        # The learnable query token
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x):
        # x has shape (B, SeqLen, HiddenDim)
        B = x.shape[0]
        # Repeat the query for each item in the batch
        query = self.pool_query.repeat(B, 1, 1) # (B, 1, HiddenDim)
        
        # The MHA layer returns the attended output and attention weights
        pooled_output, _ = self.attention(query=query, key=x, value=x)
        
        return pooled_output.squeeze(1) # (B, HiddenDim)

# 🔹 Dataset
class VQADataset(Dataset):
    def __init__(self, ques_col, img_id, tokenizer, max_len=32):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.img_id = img_id

        # Flatten ques_col into list of samples
        for category, content in ques_col.items():
            for d in content["data"]:
                self.samples.append({
                    "question": d["question"],
                    "imageId": str(d["imageId"]),
                    "answer_index": d["answer_index"],
                    "category": category
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Encode question
        enc = self.tokenizer(
            sample["question"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        q_tokens = enc["input_ids"].squeeze(0)  # (L,)

        # Image embedding
        img_feat = self.img_id[sample["imageId"]]  # already a tensor

        return {
            "question_tokens": q_tokens,
            "img_features": img_feat,
            "answer_index": torch.tensor(sample["answer_index"]),
            "answer_type": sample["category"]
        }


# 🔹 Collate function (pads batch manually if needed)
def collate_fn(batch):
    q_tokens = torch.stack([b["question_tokens"] for b in batch])
    img_feats = torch.stack([b["img_features"] for b in batch])
    labels = torch.stack([b["answer_index"] for b in batch])
    types = [b["answer_type"] for b in batch]
    return q_tokens, img_feats, labels, types

def get_question_collection(config, graphDataset, question_collection, replace, model):
    # make embeddings
    file_name_to_y = {}
    file_name_to_full_filename = {}
    for d in tqdm(graphDataset):
        parsed_filename = d["file_name"].split("/")[-1].lstrip("0")
        if parsed_filename in file_name_to_y.keys():
            fullname = file_name_to_full_filename[parsed_filename]
            duplname = d["file_name"]
            if fullname != duplname:  # ID clash
                print(
                    f"WARN! Duplicate graph {parsed_filename} found in dataset. Ignoring it.\n Full name: {fullname}\n Dupl name: {duplname}"
                )
            else:
                pass  # duplicate graph annotation for same graph
        else:
            if replace:
                with torch.no_grad():
                    file_name_to_y[parsed_filename] = model(d.to(DEVICE))[0]
            else:
                file_name_to_y[parsed_filename] = d.y
            file_name_to_full_filename[parsed_filename] = d["file_name"]

    img_id_to_y = {
        id: file_name_to_y.get(f"{id}.jpg")
        for id in tqdm(
            all_img_ids(question_collection), desc="qa: select needed embeddings"
        )
    }
    for k, v in question_collection.items():
        print(
            f"qa (before filtering): {k} has {len(v['data'])} questions, {len(v['all_answers'])} answers."
        )
    question_collection, drop_count = filter_collection(
        question_collection,
        [k for k, v in img_id_to_y.items() if v is not None],
        False,
    )
    img_id_to_y = {
        id: file_name_to_y.get(f"{id}.jpg")
        for id in tqdm(
            all_img_ids(question_collection), desc="qa: select needed embeddings"
        )
    }
    print(f"Now contain only {len(img_id_to_y)} images.")
    if config.get("run_label") == "DEBUG":
        # DBG FOR FAST RUN
        dbg_trim = 100
        print(
            f"[qa_eval] Debug mode. Trimming to {dbg_trim} questions per type.")
        for k in question_collection.keys():
            question_collection[k]["data"] = question_collection[k]["data"][
                :dbg_trim
            ]
        img_id_to_y = {
            id: file_name_to_y.get(f"{id}.jpg")
            for id in tqdm(
                all_img_ids(question_collection), desc="qa: select needed embeddings"
            )
        }
        print(f"Now contain only {len(img_id_to_y)} images.")
    print(
        f"Dropped {drop_count} questions.\nFinal QA dataset: {num_questions(question_collection)} questions, on {len(img_id_to_y)} images."
    )
    for k, v in question_collection.items():
        print(
            f"qa preview: {k} has {len(v['data'])} questions, {len(v['all_answers'])} answers."
        )
        
    return question_collection, img_id_to_y