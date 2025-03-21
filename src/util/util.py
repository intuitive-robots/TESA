import os

import torch.nn.functional as F
import yaml

import wandb

# some config handling.


def get_data_dims(torch_geometric_data_object):
    r"""returns (input_dim, output_dim, edge_attr_dim)"""
    d = torch_geometric_data_object
    edge_count = d.edge_index.shape[1]
    return (d.x.shape[1], d.y.shape[0], d.edge_attr.shape[0] // edge_count)


def ds_base_embedding(config):
    # take every item in config starting with "dataset.base_embedding." and add here.
    prefix = "dataset.base_embedding."
    return {k[len(prefix):]: v for k, v in config.items() if k.startswith(prefix)}


def load_config_yaml(load_model_name=None, yaml_path="config.yaml"):
    if load_model_name is None:
        print(f"Loading config from {yaml_path}")
        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    if load_model_name is not None:
        with open(yaml_path, "r") as f:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        model_config_path = f"out/configs/{load_model_name}"
        print(f"Loading config from {model_config_path}")
        with open(model_config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # overwrite config values:
        overwrite_keys = new_config.get("_overwrite_model_config_keys")
        overwrite_keys = [k for k in overwrite_keys if k in new_config.keys()]
        if overwrite_keys is not None:
            print(
                f"overwriting keys: {overwrite_keys} with current config values.")
            for key in overwrite_keys:
                if key in new_config.keys():
                    config[key] = new_config.get(key, None)
    return config


def wandb_start(config):
    wandb.init(
        project=config.get("wandb-project", "default"),
        config=config,
        mode="disabled" if config["OFFLINE"] else "online",
        name=config.get("run_label", None),
    )
    config = wandb.config.as_dict()  # for sweeps.
    print(config)
    return config


def load_config_yaml_and_wandb(load_model_name):
    config = load_config_yaml(load_model_name)
    return wandb_start(config)


def logme(dict, prefix=""):
    print("--<logme>")
    prefix_dict = {prefix + key: value for key, value in dict.items()}
    for i in prefix_dict:
        print(f"{i}: {(100 * prefix_dict[i]):.2f}%")
    wandb.log(prefix_dict)
    print("--</logme>")


def guarantee_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created dir at {path}")


def get_relation(d, from_node, to_node):
    """
    Helper.

    Returns the edge between from_node and to_node.
    """
    edge_human = d.edge_human
    for a, b, rel in edge_human:
        if a == from_node and b == to_node:
            return rel
    return None


def cosine_similarity_logits(image_features, text_features):
    # normalized features
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    logits_per_first = image_features @ text_features.t()
    logits_per_second = logits_per_first.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_first, logits_per_second


def own_mse_logits(image_features, graph_features):
    input1_expanded = image_features.unsqueeze(1)  # shape: [10, 1, 512]
    input2_expanded = graph_features.unsqueeze(0)  # shape: [1, 10, 512]

    mse_all_to_all = ((input1_expanded - input2_expanded) ** 2).mean(
        dim=2
    )  # Shape: [10, 10]

    return -mse_all_to_all, -mse_all_to_all.t()


def filtergraphs(dataset, config, context_str):
    r"""use only graphs up to a certain node cound for evaluation."""
    do_fast_estimate = config.get("eval.fast_estimate", False)
    do_lowerbound = do_fast_estimate and config.get(
        "eval.fast_estimate.lower_bound", False
    )
    if do_fast_estimate:
        min_nodes = config.get("eval.fast_estimate.min_nodes", 0)
        max_nodes = config.get("eval.fast_estimate.max_nodes", 100)
        max_graphs = config.get("eval.fast_estimate.max_graphs", 1000000)
        # take graphs until max_graphs is reached.
        take_graphs = []
        drop_graphs = []
        for d in dataset:
            if min_nodes <= len(d.x) <= max_nodes:
                take_graphs.append(d)
            else:
                drop_graphs.append(d)
            if len(take_graphs) == max_graphs:
                break
        drop_frac = len(drop_graphs) / (len(drop_graphs) + len(take_graphs))
        trunc_frac = 1 - (len(drop_graphs) + len(take_graphs)) / len(dataset)
        print(
            f"""
            note [{context_str}]: fast estimate eval.
            {drop_frac*100:.1f}% of graphs have been dropped due to
            node count not between {min_nodes} and {max_nodes} nodes.
            """
        )
        print(
            f"{trunc_frac*100:.1f}% of graphs truncated because already more than max_graphs."
        )
        if do_lowerbound:
            if len(dataset) != len(take_graphs) + len(drop_graphs):
                print(
                    f"""
                    WARN [{context_str}]: Lower bound does not account for 'tail' of truncated graphs! 
                    Therefore it is still an estimate.
                    """
                )
        dataset_to_eval = take_graphs
        dataset_to_skip = drop_graphs
    else:
        dataset_to_eval = dataset
        dataset_to_skip = []
        print(f" [{context_str}] Using all graphs.")
    return dataset_to_eval, dataset_to_skip, do_lowerbound


def text_to_y(text_list, config):
    r"""convert text to embedding using same model that is used for the images."""
    model_name = config.get("dataset.base_embedding.using")
    if model_name == "clip":
        from src.util.util_clip import clip_encode_text_list

        return clip_encode_text_list(text_list)
    if model_name == "SigLIP":
        from src.util.util_siglip import siglip_encode_textList

        return siglip_encode_textList(text_list)
    raise ValueError(
        f"Model {model_name} unknown or not usable for embedding text.")


def img_to_y(imageList, config):
    r"""convert images to embedding using same model that is used for the text."""
    model_name = config.get("dataset.base_embedding.using")
    if model_name == "clip":
        from src.util.util_clip import clip_encode_image_list

        return clip_encode_image_list(imageList)
    if model_name == "SigLIP":
        from src.util.util_siglip import siglip_encode_imageList

        return siglip_encode_imageList(imageList)
    raise ValueError(
        f"Model {model_name} unknown or not usable for embedding images.")


def overwrite_config(config, overconfig):
    for key, value in overconfig.items():
        config[key] = value
    return config


def get_split_keys(load_model_name, split_name):
    try:
        SPLIT_PATH = f"out/splits/{load_model_name}_{split_name}"
        print(f"Loading {split_name} split from {SPLIT_PATH}")
        with open(SPLIT_PATH, "r") as f:
            return f.read().splitlines()
    except:
        raise ValueError(
            f"Could not load test split from {SPLIT_PATH}. Perhaps the model {load_model_name} was not archived in a previous run?"
        )
