# relations encoded as edges.
import clip
import torch
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from src.util.util_clip import clip_model
from src.data_handling.datasets import get_label
from src.data_handling.sg_parser import PSG

from src.util.device import DEVICE


class RieGraphData(Data):
    def __init__(
        self,
        x=None,
        y=None,
        edge_index=None,
        edge_attr=None,
        hyperedge_index_nodes=None,  # first row of hyperedge index
        hyperedge_index_hyperedges=None,  # second row of hyperedge index
        edge_human=None,
        node_human=None,
        file_name=None,
        **kwargs,
    ):
        super().__init__(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr, **kwargs)
        self.hyperedge_index_nodes = hyperedge_index_nodes
        self.hyperedge_index_hyperedges = hyperedge_index_hyperedges
        self.edge_human = edge_human
        self.node_human = node_human
        self.file_name = file_name

    def __inc__(self, key, value, store):
        if key == "hyperedge_index_nodes":
            # just increase by number of nodes.
            return self.x.size(0)
        if key == "hyperedge_index_hyperedges":
            if len(self.hyperedge_index_nodes) == 0:
                return torch.tensor(0)  # no hyperedges.
            else:
                return self.hyperedge_index_hyperedges.max() + 1
        else:
            return super().__inc__(key, value, store)


class RelIsEdgeUnifiedJsonDataset(Dataset):
    """
    relations are edges.

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
        self.ds_base_embedding = ds_base_embedding
        self.ds_name = ds_name
        self.initial_features = initial_features
        self.for_training_only = for_training_only

        self.data = raw_data["data"]
        self.obj_classes = raw_data["obj_classes"]
        self.pred_classes = raw_data["predicate_classes"]

        (self.all_obj_vec, self.all_pred_vec) = build_node_and_edge_features_rie(
            self.initial_features, self.obj_classes, self.pred_classes
        )  # run clip on all only once.
        self._fb_dict = {
            self.pred_classes[i]: (
                torch.cat((self.all_pred_vec[i], torch.tensor([1]))),
                torch.cat((self.all_pred_vec[i], torch.tensor([-1]))),
            )
            for i in range(len(self.pred_classes))
        }
        self.all_pred_emb_forward_backward = [
            self._fb_dict[pred] for pred in self.pred_classes
        ]

    def edges_dict_forward_backward(self):
        return self._fb_dict

    # def shuffle(self):
    #     random.shuffle(self.data)
    #     return self

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
        if not self.for_training_only:
            g = psg_to_rie_object(
                self.data[index],
                get_label(psg_g["file_name"],
                          self.ds_base_embedding, self.ds_name),
                self.all_obj_vec,
                self.all_pred_vec,
                self.obj_classes,
                self.pred_classes,
                psg_g["file_name"],
            )
        else:  # less memory-intensive, less useful.
            g = psg_to_rie_data(
                self.data[index],
                get_label(psg_g["file_name"],
                          self.ds_base_embedding, self.ds_name),
                self.all_obj_vec,
                all_pred_emb_forward_backward=self.all_pred_emb_forward_backward,
            )

        return g

    def __len__(self):
        return len(self.data)


def psg_to_rie_data(
    psg_format_graph, label, all_obj_vec, all_pred_emb_forward_backward
):
    r"""
    Normal pyg data. no human-readable annotations.
    """
    if psg_format_graph is None or label is None:
        raise ValueError("psg_format_graph or label is None")

    relations = psg_format_graph["relations"]
    nodes = psg_format_graph["nodes"]
    if len(nodes) == 0 or len(relations) == 0:
        return None
    edge_index = []
    edge_attr = []
    # collect edges:
    for r in relations:
        # forward edges
        edge_index.append([r[0], r[1]])
        # backward edges
        edge_index.append([r[1], r[0]])
        (f, b) = all_pred_emb_forward_backward[r[2]]
        edge_attr.append(f)
        edge_attr.append(b)

    return Data(
        x=all_obj_vec[nodes],
        y=label.flatten(),
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.cat(edge_attr),
    )


def psg_to_rie_object(
    psg_format_graph,
    label,
    all_obj_vec,
    all_pred_vec,
    obj_classes,
    pred_classes,
    file_name,
):
    r"""
    - relations are encoded as edges!
    """
    if psg_format_graph is None or label is None:
        raise ValueError("psg_format_graph or label is None")

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
    predicates = [
        pred_classes[i] for i in rel_indices
    ]  # possible improvement: indices faster: pred_classes
    x = object_labels
    node_human = [
        obj_classes[i] for i in object_indices
    ]  # possible improvement: indices faster: obj_classes
    edge_index = []
    edge_attr = []
    edge_human = []
    # collect edges:
    for i, r in enumerate(relations):
        # forward edges
        edge_index.append([r[0], r[1]])
        # backward edges
        edge_index.append([r[1], r[0]])
        # use last dimension of edge feature to mark backwards edges.
        # possible improvement: use edges_dict_forward_backward
        edge_attr.append(torch.cat((rel_labels[i], torch.tensor([1]))))
        edge_attr.append(torch.cat((rel_labels[i], torch.tensor([-1]))))
        edge_human.append(
            (r[0], r[1], predicates[i])
        )  # possible improvement: indices faster: rel_indices[i]
    label = label.flatten()

    # hyperedges step one: define hyperedges list of sets.
    hyperedges_as_sets = []  # nodes and corresponding hyperedges
    # hyperedges step two: create hyperedge index.
    hyperedge_index_nodes = []
    hyperedge_index_hyperedges = []
    for i, _ in enumerate(hyperedges_as_sets):
        for j, _ in enumerate(hyperedges_as_sets[i]):
            hyperedge_index_nodes.append(j)
            hyperedge_index_hyperedges.append(i)

    return RieGraphData(
        x=x,
        y=label,
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.cat(edge_attr),
        hyperedge_index_nodes=torch.tensor(hyperedge_index_nodes),
        hyperedge_index_hyperedges=torch.tensor(hyperedge_index_hyperedges),
        edge_human=edge_human,
        node_human=node_human,
        file_name=file_name,
    )


def build_node_and_edge_features_rie(initial_feature_mode, objects, predicates):
    r"""
    Build node features for all objects and predicate.
    @return (object_node_features, predicate_node_features)
    """
    # add prefix to each txt, tokenize.
    if initial_feature_mode == "one_hot":
        return (torch.eye(len(objects)), torch.eye(len(predicates)))

    def pre_token_512(txt_list, prefix):
        return clip.tokenize(
            [f"{prefix} {o}" for o in txt_list], context_length=512
        ).to(DEVICE)

    def pre_token(txt_list, prefix):
        return clip.tokenize([f"{prefix} {o}" for o in txt_list]).to(DEVICE)

    with torch.no_grad():
        if initial_feature_mode == "clip_embedding":
            return (
                clip_model.encode_text(pre_token(objects, "object: ")).cpu(),
                clip_model.encode_text(
                    pre_token(predicates, "predicate: ")).cpu(),
            )
        # possible improvement: (later): use siglip_embeddings.
        elif initial_feature_mode == "clip_token":
            return (
                pre_token_512(objects, "object: "),
                pre_token_512(predicates, "predicate: "),
            )

    raise NotImplementedError(
        f"initial feature mode `{initial_feature_mode}` unknown.")
