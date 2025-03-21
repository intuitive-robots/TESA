
import os
import random
import clip
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

# import visual_genome.local as vg
from src.util.util_clip import clip_model
from src.data_handling.sg_parser import GQA, PSG, VG
from src.util.device import DEVICE
from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

COCO_IMGS = "coco"


class RelIsNodeUnifiedJsonDataset(Dataset):
    """
    new normal dataset.

    json in /unified_json/{ds_name}.json should be formatted:
    - "data": list of graphs
    - "obj_classes": list of object classes
    - "pred_classes": list of predicate classes

    """

    def __init__(
        self,
        ds_base_embedding,
        raw_data,
        ds_name=PSG,
        initial_features="clip_embedding",
        for_training_only=False,
    ):
        if not for_training_only:
            print("WARN: sgg on RiN is not supported.")
        self.ds_base_embedding = ds_base_embedding
        self.ds_name = ds_name
        self.initial_features = initial_features
        self.data = raw_data["data"]
        self.obj_classes = raw_data["obj_classes"]
        self.pred_classes = raw_data["predicate_classes"]

        f_vec = build_node_features_rin(
            self.initial_features, self.obj_classes, self.pred_classes
        )  # run clip on all only once.
        (self.all_obj_vec, self.all_pred_vec) = torch.split(
            f_vec, [len(self.obj_classes), len(self.pred_classes)]
        )

    def shuffle(self):
        random.shuffle(self.data)
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self.data)))
            if len(indices) > 5000:
                return [
                    self._process_single_item(i)
                    for i in tqdm(indices, "Getting items...")
                ]
            else:
                return [self._process_single_item(i) for i in indices]
        return self._process_single_item(index)

    def _process_single_item(self, index):
        """Helper function to process a single item."""
        psg_g = self.data[index]
        graph = psg_to_rin_data_object(
            self.data[index],
            get_label(psg_g["file_name"],
                      self.ds_base_embedding, self.ds_name),
            self.all_obj_vec,
            self.all_pred_vec,
        )
        return graph

    def __len__(self):
        return len(self.data)

    def edges_dict_forward_backward(self):
        # TODO sgg for RiN
        return None  # sgg not supported for RiN.


def build_node_features_rin(initial_feature_mode, objects, predicates):
    r"""
    Build node features for all objects and predicate.
    @return (object_node_features, predicate_node_features) as one tensor
    """
    if initial_feature_mode == "one_hot":
        outer_dim = len(objects) + len(predicates)
        identity_matrix = torch.eye(outer_dim)
        return identity_matrix

    def pre_token_512(txt_list, prefix):
        return clip.tokenize(
            [f"{prefix} {o}" for o in txt_list], context_length=512
        ).to(DEVICE)

    def pre_token(txt_list, prefix):
        return clip.tokenize([f"{prefix} {o}" for o in txt_list]).to(DEVICE)

    with torch.no_grad():
        if initial_feature_mode == "clip_embedding":
            tokens = torch.cat(
                (pre_token(objects, "object: "),
                 pre_token(predicates, "predicate: "))
            )
            return clip_model.encode_text(tokens).cpu()
        elif initial_feature_mode == "clip_token":
            return torch.cat(
                (
                    pre_token_512(objects, "object: "),
                    pre_token_512(predicates, "predicate: "),
                )
            ).cpu()

    raise NotImplementedError(
        f"initial feature mode `{initial_feature_mode}` unknown.")


def get_label(file_name, ds_base_embedding, ds_name):
    r"""
    gets label from processed folder (label: clip of image). 
    ds_base_embedding: {'from': 'image', 'using': 'clip'}
    """
    assert (
        ds_base_embedding.get("from") == "image"
    ), f"Can't get label from sourcetype {ds_base_embedding.get('from')}"
    preprocessor = ds_base_embedding["using"]

    (subfolder, name) = os.path.split(file_name)

    if ds_name == PSG:
        base_folder = COCO_IMGS
    elif ds_name == GQA:
        base_folder = f"{GQA}/images"
    elif ds_name == VG:
        base_folder = f"{VG}/VG_100K"
    else:
        raise NotImplementedError(f"ds_name {ds_name} not supported.")
    preprocessed_name = (
        f"{DATA_DIR}/processed/{base_folder}/{subfolder}/{preprocessor}_out/{name}.npy"
    )
    if os.path.exists(preprocessed_name):
        return torch.tensor(np.load(preprocessed_name)).to(torch.float32)
    else:
        if ds_name == VG:
            base_folder_2 = f"{VG}/VG_100K_2"
            preprocessed_name = f"{DATA_DIR}/processed/{base_folder_2}/{subfolder}/{preprocessor}_out/{name}.npy"
            if os.path.exists(preprocessed_name):
                return torch.tensor(np.load(preprocessed_name)).to(torch.float32)

        print(f"warn: ignoring image: {preprocessed_name} (not found)")
        return None
    # raise FileNotFoundError("should be preproccessed.")


def psg_to_rin_data_object(
    psg_format_graph,
    label,
    all_obj_vec,
    all_pred_vec,
):
    if psg_format_graph is None or label is None:
        return None

    # object_classes = thing_classes + stuff_classes
    # objects = [ object_classes[a["category_id"]] for a in psg_format_graph["annotations"] ]
    relations = psg_format_graph["relations"]
    # predicates = [ predicate_classes[r[2]] for r in relations]
    # # objects and relationships become nodes.
    if "nodes" in psg_format_graph:  # hacky: own dataset has different format.
        object_count = len(psg_format_graph["nodes"])
    else:
        object_count = len(psg_format_graph["annotations"])
    if object_count == 0 or len(relations) == 0:
        return None
    # # relations come after object labels.
    if "nodes" in psg_format_graph:  # hacky: own dataset has different format.
        object_indices = psg_format_graph["nodes"]
    else:
        object_indices = [
            a["category_id"] for a in psg_format_graph["annotations"]
        ]  # which objects to "pick" from "global list"
    rel_indices = [r[2] for r in relations]

    object_labels = all_obj_vec[object_indices]
    rel_labels = all_pred_vec[rel_indices]
    x = torch.cat((object_labels, rel_labels),
                  dim=0).squeeze(1).to(torch.float32)

    edge_index = []
    edge_attr = []

    # collect edges:
    for i, r in enumerate(relations):
        predicate_node_index = object_count + i
        # forward edges
        edge_index.append([r[0], predicate_node_index])
        edge_index.append([predicate_node_index, r[1]])
        # backward edges
        edge_index.append([predicate_node_index, r[0]])
        edge_index.append([r[1], predicate_node_index])
        # use edge_attr to mark backwards edges.
        edge_attr.append([1.0])
        edge_attr.append([1.0])
        edge_attr.append([-1.0])
        edge_attr.append(
            [-1.0]
            # paul: gibt paper, die vergleichen, was da besser ist. (aufsplitten in 2 edges; attribute für rückwärtskanten)
        )

    return Data(
        x=x,
        y=label.flatten(),
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.cat([torch.tensor(a) for a in edge_attr]),
        file_name=psg_format_graph["file_name"],
    )
