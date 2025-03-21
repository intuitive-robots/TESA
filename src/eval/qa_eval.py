import os

import torch
from tqdm import tqdm

from src.eval.qa_loader import (all_img_ids, filter_collection,
                                num_questions)
from src.eval.qa_statementGen import get_question_to_statements_function
from src.models.component_handler import qa_collection, similarity_from_config
from src.util.device import DEVICE

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
        for d in graphDataset:
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
