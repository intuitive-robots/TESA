import json
import os

import einops
import torch
from tqdm import tqdm

from src.eval.image_generation import eval_image_generation
from src.eval.classification_eval import eval_classification
from src.eval.image_captioning_eval import eval_image_captioning
import src.data_handling.datasets as datasets
from src.data_handling.datasethandler import rie_ds
from src.data_handling.sg_parser import load_raw_train_val_test
from src.eval.retrieval_eval import eval_retrieval
from src.eval.sgg_eval import eval_sgg
from src.eval.sim_eval import eval_sim
from src.eval.statistics import build_prior_knowledge
from src.models.component_handler import similarity_from_config
from src.util.util import (ds_base_embedding, get_split_keys, guarantee_dir,
                           img_to_y, text_to_y)
from src.util.device import DEVICE
from src.train_sgg import test_sgg_model, train_sgg_model

DATA_BASE = os.getenv("DATA_DIR")


def psg_eval_text_dict():
    r"""Image file names to textual descriptions."""
    with open(f"{DATA_BASE}/psg_captions/val2017_text.json") as f:
        return json.load(f)


def psg_eval_text_dict_from_captions(stamp, on_coco_captions):
    coco_train_captions = psg_eval_text_dict_from_captions_file(
        f"{DATA_BASE}/psg_captions/captions_train2017.json", "train2017/"
    )
    coco_val_captions = psg_eval_text_dict_from_captions_file(
        f"{DATA_BASE}/psg_captions/captions_val2017.json", "val2017/"
    )
    # merge dicts.
    all_coco_captions = {**coco_train_captions, **coco_val_captions}
    if on_coco_captions == "all":
        return all_coco_captions
    else:
        assert on_coco_captions in ["train", "test", "val"]
        # filtering for captions from train!
        print(f"Filtering for {on_coco_captions} captions")
        test_keys = get_split_keys(stamp, on_coco_captions)
        return {k: all_coco_captions[k] for k in all_coco_captions if k in test_keys}


def psg_eval_text_dict_from_captions_file(path, return_key_prefix):
    with open(path) as f:
        caption_json = json.load(f)
    annotations = caption_json["annotations"]
    images = caption_json["images"]

    image_id_to_caption = {}

    imgid2filename_suffix = {
        image_filename["id"]: image_filename["file_name"] for image_filename in images
    }
    for annotation in tqdm(annotations, desc=f"Loading captions from {path}"):
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        # find image_filename
        image_filename = imgid2filename_suffix.get(image_id)
        if image_id not in image_id_to_caption and image_filename:
            image_id_to_caption[f"{return_key_prefix}{image_filename}"] = caption
    return image_id_to_caption


def psg_eval_dataset(
    initial_features,
    g_model,
    config,
    text_prefix,
    on_coco_captions=False,  # "val" or "train" or "test" or "all",
    stamp=None,
):
    r"""
    @returns (image_features, text_features, graph_features, graph_list) - where each are in same (correct) order.

    Evaluation Dataset containing images from coco/val2017, SG form PSG,
    on_coco_captions: False (manual created) # "val" or "train" or "test" or "all" (filter coco captions according to current ds)
    """
    assert config.get("dataset") == "psg"
    assert on_coco_captions  # manual not supported anymore.
    if not on_coco_captions:
        descriptions_dict = psg_eval_text_dict()
    else:
        descriptions_dict = psg_eval_text_dict_from_captions(
            stamp, on_coco_captions)

        print(
            f"[psg_eval_ds] loading dataset using {on_coco_captions} captions ({len(descriptions_dict)} images)"
        )

    data = {
        "image_fnames": [],
        "text_inputs": [],
        "graph_inputs": [],
        "image_paths": [],
    }
    for key in descriptions_dict:
        data["image_fnames"].append(key)
        data["text_inputs"].append(descriptions_dict[key])
    data["image_paths"] = [
        f"{DATA_BASE}/raw/coco/{fname}" for fname in data["image_fnames"]
    ]

    # graph inputs:
    nuconfig = config.copy()
    nuconfig.update({"train_split": 1.0})
    (raw_train_data, _, _) = load_raw_train_val_test(
        nuconfig,
        [],  # can be used in test but not during training.
    )
    psg_format_graphs = raw_train_data["data"]
    psg_format_graphs_by_filename = {
        g["file_name"]: g for g in psg_format_graphs}
    obj_classes = raw_train_data["obj_classes"]
    pred_classes = raw_train_data["predicate_classes"]
    f_vec = datasets.build_node_features_rin(
        initial_features, obj_classes, pred_classes
    )  # run clip on all only once.
    (all_obj_vec, all_pred_vec) = torch.split(
        f_vec, [len(obj_classes), len(pred_classes)]
    )

    for fname in tqdm(data["image_fnames"], "[psg_eval_ds] loading PSG graphs"):
        psg_g = psg_format_graphs_by_filename.get(fname)
        if psg_g is not None:
            g = datasets.psg_to_rin_data_object(
                psg_g,
                datasets.get_label(
                    psg_g["file_name"], ds_base_embedding(config), "psg"
                ),
                all_obj_vec,
                all_pred_vec,
            )
            if g is not None:
                data["graph_inputs"].append(g)
            if g is None:
                raise NotImplementedError  # possible improvement: handling
        else:
            data["graph_inputs"].append(None)
            # print(f"Graph not found for {fname}")
            # raise FileNotFoundError  # possible improvement: handling
    # Remove Nones (no graph found)
    data = {
        k: [v for idx, v in enumerate(
            data[k]) if data["graph_inputs"][idx] is not None]
        for k in data
    }
    print(f"[psg_eval_ds] loaded {len(data['graph_inputs'])} PSG graphs")
    eval_bs = config.get("batch_size", 2000)

    with torch.no_grad():
        text_features = []
        for i in tqdm(
            range(0, len(data["text_inputs"]),
                  eval_bs), "[psg eval ds] text features"
        ):
            batch_text_inputs = data["text_inputs"][i: i + eval_bs]
            text_features.append(
                text_to_y(
                    [f"{text_prefix}{t}" for t in batch_text_inputs], config)
            )
        text_features = torch.cat(text_features, dim=0)

        image_features = []
        for i in tqdm(
            range(0, len(data["image_paths"]),
                  eval_bs), "[psg eval ds] image features"
        ):
            batch_image_paths = data["image_paths"][i: i + eval_bs]
            image_features.append(img_to_y(batch_image_paths, config))
        image_features = torch.cat(image_features, dim=0)

        graph_features = torch.cat(
            [
                g_model(g.to(DEVICE))[0]
                for g in tqdm(data["graph_inputs"], "[psg eval ds] graph features")
            ],
            dim=-1,
        )

    graph_features = einops.rearrange(
        graph_features, "(n dim) -> n dim", dim=image_features.shape[1]
    )
    return (image_features, text_features, graph_features, data["graph_inputs"])


def pca_vis_triangle(image_features, text_features, graph_features, perspective):
    r"""
    visualizes using PCA. perspective can be image, graph, text or combined.
    """
    # PCA viz
    from torch_pca import PCA

    all_features = torch.cat((image_features, text_features, graph_features))
    pca_model = PCA(n_components=2, svd_solver="full")
    # fit:
    if perspective == "image":
        pca_model.fit_transform(image_features)
    elif perspective == "text":
        pca_model.fit_transform(text_features)
    elif perspective == "graph":
        pca_model.fit_transform(graph_features)
    elif perspective == "combined":
        pca_model.fit_transform(all_features)
    else:
        raise ValueError(
            "perspective must be one of: image, graph, text, combined")
    # transform:
    all_components = pca_model.transform(all_features)

    l = len(image_features)
    image_c = all_components[:l].cpu()
    text_c = all_components[l: 2 * l].cpu()
    graph_c = all_components[2 * l:].cpu()

    import matplotlib.pyplot as plt

    with torch.no_grad():
        fig, ax = plt.subplots()
        ax.scatter(
            image_c[:, 0],
            image_c[:, 1],
            color="blue",
            marker="o",
            label="Image embeddings",
        )
        ax.scatter(
            text_c[:, 0],
            text_c[:, 1],
            color="black",
            marker="o",
            label="Text embeddings",
        )
        ax.scatter(
            graph_c[:, 0],
            graph_c[:, 1],
            color="red",
            marker="x",
            label="Graph embeddings",
        )
        for i in range(l):
            ax.plot(
                [image_c[i, 0], graph_c[i, 0], text_c[i, 0], image_c[i, 0]],
                [image_c[i, 1], graph_c[i, 1], text_c[i, 1], image_c[i, 1]],
                color="gray",
            )
        ax.legend()  # Add legend to the plot

    return fig


def loss_on_loader(g_model, loader, loss_fn):
    image_features = []
    graph_features = []
    g_model.eval()
    with torch.no_grad():
        for d in loader:
            image_features.append(d.y.cpu())
            graph_features.append(g_model(d.to(DEVICE)).cpu())

    dim = graph_features[0].shape[1]
    image_features = einops.rearrange(
        torch.cat(image_features), "(m dim) -> m dim", dim=dim
    )
    graph_features = torch.cat(graph_features)

    return loss_fn(image_features, graph_features)


def downstream_all(
    model,
    config,
    dataset,
    out_prefix,
    prior_knowledge,
    print_predictions=False,
    replace_y_with_model_output=False,
    encodings_fb=None,
):
    r"""
    [(dataset, logging_prefix)]
    print_predicate: output triples for qualitative evaluation.
    """

    train_bool = False

    returndict = {}
    # TASK: similarity:
    if config.get("eval.sim", False):
        out = eval_sim(dataset, config, model)
        for k, v in out.items():
            returndict[f"output_similarity/{out_prefix}{k}"] = v
    # TASK: QA
    if config.get("eval.qa", False):
        from src.eval.qa_eval import eval_qa, train_eval_qa_model

        #out = eval_qa(config, dataset)  # needs dataset.y for comparisons.
        out = train_eval_qa_model(config, dataset, train_bool, replace_y_with_model_output, model)
        # for k, v in out.items():
        #     returndict[f"qa/{out_prefix}{k}"] = v
    # TASK: retrieval:
    if config.get("eval.retrieval", False):
        retrieval_ks = config.get("eval.retrieval.k", [])
        if retrieval_ks:
            for retrieval_k in config.get("eval.retrieval.k", []):
                out = eval_retrieval(model, dataset, config, retrieval_k)
                for k, v in out.items():
                    returndict[f"retrieval/{out_prefix}{k}"] = v
    # TASK: SGG:
    if config.get("eval.sgg", False):
        modes = config.get("eval.sgg.modes", ["iter"])
        if modes:
            for mode in modes:
                out = eval_sgg(
                    encodings_fb,
                    model,
                    dataset,
                    config,
                    mode,
                    prior_knowledge,
                    print_predictions,
                )
                for k, v in out.items():
                    returndict[f"sgg/{out_prefix}{k}"] = v
    # TASK: Classification
    if config.get("eval.clas", False):
        graph, img = eval_classification(model, dataset, config, config['batch_size'])
        print(f"Classification using graphs: {graph}")
        print(f"Classification using images: {img}")
    # Evaluate Image Captioning
    if config.get("eval.capt", False):
        for i in [10, 50, 100]:
            graph, img = eval_image_captioning(model, dataset, config, i, 5)
            print(f"Captioning using graphs: {graph} for {i}")
            print(f"Captioning using images: {img} for {i}")
    if config.get("eval.gen", False):
        eval_image_generation(model, dataset, config)
    # finish up.
    # if replace_y_with_model_output:
    #     for g, y in zip(dataset, restore_y):
    #         g.y = y

    return returndict


def eval(trainer, on_train_data=False):
    trainer.model.eval()

    prior_knowledge = build_prior_knowledge(
        tqdm(
            rie_ds(trainer.raw_train_data,
                   trainer.config), "Building prior knowledge"
        ),
        trainer.config,
    )
    tt_pref = "train" if on_train_data else "test"
    dataset = trainer.train_ds if on_train_data else trainer.test_ds
    #dataset = list(tqdm(dataset, desc=f"Loading {tt_pref} data for eval."))
    base_embeddings = trainer.config.get(
        "eval.base_embeddings", ["image", "graph"])
    print(f"---- Eval on {'train' if on_train_data else 'test'} data ----")
    
    if trainer.config.get("eval.capt", False):
        dataset = psg_eval_dataset(
            trainer.config.get("initial_features"),
            trainer.model,
            trainer.config,
            "",
            on_coco_captions="val",
            stamp=trainer.load_model_name,
        )
        
    if trainer.config.get("eval.train_sgg", False):
        #train_sgg_model(trainer)
        # test_sgg_model(trainer, "image", 1)
        # test_sgg_model(trainer, "graph", 1)
        test_sgg_model(trainer, "zero", 1)
        # test_sgg_model(trainer, "image", 5)
        # test_sgg_model(trainer, "graph", 5)
        # test_sgg_model(trainer, "zero", 5)
        
    if trainer.config.get("eval.qa", False):
        dataset = trainer.train_ds, trainer.test_ds
    
    returndict = {}
    if "image" in base_embeddings:
        print("--- Eval on Images")
        r2 = downstream_all(
            trainer.model,
            trainer.config,
            dataset,
            f"{tt_pref}/Image",
            prior_knowledge=prior_knowledge,
            encodings_fb=trainer.train_ds.edges_dict_forward_backward(),
        )
        returndict.update(r2)
        # produce output:
    if "graph" in base_embeddings:
        print("--- Eval on Graphs")
        r2 = downstream_all(
            trainer.model,
            trainer.config,
            dataset,
            f"{tt_pref}/Graph",
            prior_knowledge=prior_knowledge,
            replace_y_with_model_output=True,
            encodings_fb=trainer.train_ds.edges_dict_forward_backward(),
        )
        returndict.update(r2)
    return returndict


def clip_trio_eval(trainer, text_prefix, stamp, on_coco_captions=False):

    (image_features, text_features, graph_features, graphs) = trio_ds(
        trainer, text_prefix, stamp, on_coco_captions=on_coco_captions
    )
    print("--- Evaluationg on eval data, on Images, Texts, and Graphs ---")

    trio_ds_embeddings = [
        (image_features, "eval/Image"),
        (text_features, "eval/Text"),
        (graph_features, "eval/Graph"),
    ]
    returndict = {}
    prior_knowledge = build_prior_knowledge(
        trainer.train_ds,
        trainer.config,
    )
    for features, embedding_name in trio_ds_embeddings:
        for i, g in enumerate(graphs):
            g.y = features[i]
        returndict.update(
            downstream_all(
                trainer.model,
                trainer.config,
                graphs,
                embedding_name,
                prior_knowledge,
                True,
                encodings_fb=trainer.train_ds.edges_dict_forward_backward(),
            )
        )
    return returndict


def clip_trio_pca(
    stamp,
    trainer,
    view="combined",
    text_prefix="",
    max_data=None,
    on_coco_captions=False,
    export_plot=False,
):
    MODEL_STAMP = stamp
    (image_features, text_features, graph_features, _) = trio_ds(
        trainer, text_prefix, stamp, on_coco_captions=on_coco_captions
    )
    # normalize features
    if not trainer.config.get("loss" == "MSE"):
        image_features = image_features / \
            image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        graph_features = graph_features / \
            graph_features.norm(dim=1, keepdim=True)

    if max_data:
        image_features = image_features[:max_data]
        text_features = text_features[:max_data]
        graph_features = graph_features[:max_data]
    print(f"PCA with n = {len(image_features)}")
    sim = similarity_from_config(trainer.config)
    sim_it = sim(image_features, text_features)
    sim_ig = sim(image_features, graph_features)
    sim_tg = sim(text_features, graph_features)
    print(f"sim Image - Text: {torch.mean(sim_it).item():.2f}")
    print(f"sim Image - Graph: {torch.mean(sim_ig).item():.2f}")
    print(f"sim Text - Graph: {torch.mean(sim_tg).item():.2f}")
    # PCA plot
    if export_plot:
        if not MODEL_STAMP:
            MODEL_STAMP = trainer.config.get("run_label", "")
        fig = pca_vis_triangle(
            image_features, text_features, graph_features, view)
        guarantee_dir("out/figs")
        # fig.savefig(f"out/figs/{MODEL_STAMP}_PCA_triangle_{view}-view.png")
        p = f"out/figs/{MODEL_STAMP}_PCA_triangle_{view}-view{("_"+str(max_data)) if max_data else ""}-split_{on_coco_captions}.pdf"
        fig.savefig(p)
        print(f"Saved PCA triangle plot to {p}")
    v_i = torch.var(image_features, dim=0, unbiased=False)
    v_t = torch.var(text_features, dim=0, unbiased=False)
    v_g = torch.var(graph_features, dim=0, unbiased=False)

    def sim_on_ds(dataset):
        sim = similarity_from_config(trainer.config)
        sim_list = []
        for a in dataset:
            for b in dataset:
                sim_list.append(sim(a.unsqueeze(0), b.unsqueeze(0)).item())
        return torch.tensor(sim_list).mean().item()

    # rount to 4 decimals
    print("SIM; Variances (min, avg, max over dimensions):")
    istr = f"images: {sim_on_ds(image_features):.6f}; {v_i.min():.6f}, {v_i.mean():.6f}, {v_i.max():.6f}"
    tstr = f"texts:  {sim_on_ds(text_features):.6f}; {v_t.min():.6f}, {v_t.mean():.6f}, {v_t.max():.6f}"
    gstr = f"graphs: {sim_on_ds(graph_features):.6f}; {v_g.min():.6f}, {v_g.mean():.6f}, {v_g.max():.6f}"
    print(istr)
    print(tstr)
    print(gstr)


def trio_ds(trainer, text_prefix, stamp, on_coco_captions=False):
    r"@returns (image_features, text_features, graph_features, graph_list)"
    return psg_eval_dataset(
        trainer.config.get("initial_features"),
        trainer.model,
        trainer.config,
        text_prefix,
        on_coco_captions=on_coco_captions,
        stamp=stamp,
    )
