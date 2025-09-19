import json
import os
import random
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm, trange

from src.util.util import guarantee_dir

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")


def load_vg_new(split: str):
    r"""@param split: ["train", "val", "test"]"""
    with open(f"{DATA_DIR}/raw/{VG}/rel.json") as f:
        rel_json = json.load(f)
    with open(f"{DATA_DIR}/raw/{VG}/{split}.json") as f:
        vg_json = json.load(f)
    object_classes: List[str] = ["zero"] + [
        # should be identical for all splits..
        c["name"] for c in vg_json["categories"]
    ]  # ids start with 1 => added "zero". 151 for train..
    predicate_classes: List[str] = rel_json["rel_categories"]  # 51 for train

    relations_per_id = rel_json[split]  # 57723 for train
    data_by_image_id = {}
    for a in vg_json["annotations"]:
        image_id = str(a["image_id"])
        if image_id not in data_by_image_id:
            data_by_image_id[image_id] = {
                "node_id_to_category_id": {},
                "relations": relations_per_id[image_id],
                "file_name": f"{image_id}.jpg",
            }
        data_by_image_id[image_id]["node_id_to_category_id"][a["id"]
                                                             ] = a["category_id"]

    # map node id to category id dict to list
    data = []
    for d in data_by_image_id.values():
        nodes = []
        nodes_relabel_dict = {}
        for node_id, category_id in d["node_id_to_category_id"].items():
            nodes.append(category_id)
            nodes_relabel_dict[node_id] = len(nodes) - 1
        n = {
            "relations": [
                # [nodes_relabel_dict[s], nodes_relabel_dict[o], p]
                [s, o, p]
                for [s, o, p] in d["relations"]
            ],
            "file_name": d["file_name"],
            "nodes": nodes,
        }
        data.append(n)
    return {
        "data": data,
        "obj_classes": object_classes,
        "predicate_classes": predicate_classes,
    }


def load_gqa(val: bool):
    with open(f"{DATA_DIR}/raw/{GQA}/scenegraphs/train_sceneGraphs.json") as f:
        train_raw_data = json.load(f)
    with open(f"{DATA_DIR}/raw/{GQA}/scenegraphs/val_sceneGraphs.json") as f:
        val_raw_data = json.load(f)

    object_classes: List[str] = []
    predicate_classes: List[str] = []
    attribute_classes: List[str] = []  # not used.

    def scan(items):
        ret = []
        for imageId, sg in tqdm(items, total=len(items)):
            relations = []  # from id, to id, predicate ds_id
            obj_ids = list(sg["objects"].keys())  # fixed order of objects
            for objId in obj_ids:
                obj = sg["objects"][objId]
                if obj["name"] not in object_classes:
                    object_classes.append(obj["name"])
                # possible improvement: use attributes. we dismiss attributes for now.
                for attr in obj["attributes"]:
                    if attr not in attribute_classes:
                        attribute_classes.append(attr)

                for rel in obj["relations"]:
                    if rel["name"] not in predicate_classes:
                        predicate_classes.append(rel["name"])
                    subj = obj_ids.index(objId)
                    obj = obj_ids.index(rel["object"])
                    pred = predicate_classes.index(rel["name"])
                    relations.append([subj, obj, pred])
            nodes = [object_classes.index(
                sg["objects"][id]["name"]) for id in obj_ids]
            relevant_psg_g = {
                "file_name": f"{imageId}.jpg",
                "nodes": nodes,  # node_ds_ids
                "relations": relations,
            }
            if (
                len(relevant_psg_g["nodes"]) == 0
                or len(relevant_psg_g["relations"]) == 0
            ):
                continue
            ret.append(relevant_psg_g)
        return ret

    # make sure OHE is consistent.
    train_data = scan(train_raw_data.items())
    val_data = scan(val_raw_data.items())
    data = train_data if not val else val_data
    # output:
    return {
        "data": data,
        "obj_classes": object_classes,
        "predicate_classes": predicate_classes,
    }


def load_psg():
    with open(f"{DATA_DIR}/raw/{PSG}/psg.json") as f:
        raw_data = json.load(f)
    psg_format_graphs = raw_data.get("data")
    data = []
    for i in trange(len(psg_format_graphs)):
        psg_g = psg_format_graphs[i]
        relevant_psg_g = {
            "file_name": psg_g["file_name"],
            "nodes": [a["category_id"] for a in psg_g["annotations"]],
            "relations": psg_g["relations"],
        }
        if (
            len(psg_g["annotations"]) == 0 or len(psg_g["relations"]) == 0
        ):  # never triggered. skip empty graphs (there are some!)
            continue
        data.append(relevant_psg_g)
    # output:
    return {
        "data": data,
        "obj_classes": raw_data["thing_classes"] + raw_data["stuff_classes"],
        "predicate_classes": raw_data["predicate_classes"],
    }


GQA = "gqa"
PSG = "psg"
VG = "vg"  # second try for vg...


def shuffle_split(from_data, from_data_name, to_data_name, keep_ratio):
    print(f"shuffle {from_data_name}.")
    random.shuffle(from_data)
    print(f"splitting from {from_data_name} into {to_data_name}.")
    split = int(len(from_data) * keep_ratio)
    moved_data = from_data[split:]
    kept_data = from_data[:split]
    return kept_data, moved_data


def load_raw_train_val_test(config, force_fnames_into_test):
    r"""
    Will return a tuple of (train, val, test) data in unified json format.
    Splitting according to given train_split (train_split, (1-train_split)/2, (1-train_split)/2), if no split was provided in the dataset.
    """
    ds_name = config["dataset"]
    train_split = config.get("train_split", None)

    UNIFIED_JSON_PATH = f"{DATA_DIR}/unified_json/{ds_name}.json"
    # try to load pre-existing:
    if os.path.exists(UNIFIED_JSON_PATH):
        with open(UNIFIED_JSON_PATH, "r") as f:
            store = json.load(f)
    # make.
    else:
        if ds_name == PSG:
            store = {"train": load_psg()}  # 47874
        elif ds_name == GQA:
            store = {
                "train": load_gqa(False),  # 72600
                "val": load_gqa(True),
            }
        elif ds_name == VG:
            store = {
                "train": load_vg_new("train"),  # 57723
                "val": load_vg_new("val"),  # 5000
                "test": load_vg_new("test"),  # 26446
            }
        else:
            raise NotImplementedError("Dataset not supported.")
        # save
        guarantee_dir(f"{DATA_DIR}/unified_json/")
        with open(f"{UNIFIED_JSON_PATH}", "w") as f:
            json.dump(store, f)
        print(
            f"Dataset stored with: {len(store["train"]["data"])} train-items, {len(store["val"]["data"]) if "val" in store else "no"} val-items, {len(store["test"]["data"]) if "test" in store else "no"} test-items."
        )
    print(
        f"{len(store["train"]["data"])} train, {len(store["val"]["data"]) if "val" in store else "no"} val, {len(store["test"]["data"]) if "test" in store else "no"} test items found in store."
    )

    def filter_cb(data):
        remo = [d for d in data if d["file_name"] in force_fnames_into_test]
        count = len(remo)
        if count > 0:
            print(
                f"removing {count} graphs from data (callback) (those will be moved into test)."
            )
            return [d for d in data if d["file_name"] not in force_fnames_into_test], [
                d for d in data if d["file_name"] in force_fnames_into_test
            ]
        else:
            return data, []

    train, rm = filter_cb(store["train"]["data"])
    obj_classes = store["train"]["obj_classes"]
    predicate_classes = store["train"]["predicate_classes"]
    # sanity-check and extract pre-existing val/test:
    if "val" in store:
        assert obj_classes == store["val"]["obj_classes"]
        assert predicate_classes == store["val"]["predicate_classes"]
        val, rm2 = filter_cb(store["val"]["data"])
        rm += rm2
    if "test" in store:
        assert obj_classes == store["test"]["obj_classes"]
        assert predicate_classes == store["test"]["predicate_classes"]
        test, rm3 = filter_cb(store["test"]["data"])
        rm += rm3
    # split
    seed = config["seed"]
    random.seed(seed)
    if not "val" in store and not "test" in store:
        # val and test from train.
        train, val = shuffle_split(train, "train", "val", train_split)
        val, test = shuffle_split(val, "val", "test", 0.3)
    elif not "test" in store:
        if config.get("eval.qa", False):
            print(
                "[sg_parser] qa-evaluation activated on dataset without test data. USING VAL AS TEST."
            )
            test = val
        else:
            val, test = shuffle_split(val, "val", "test", 0.3)
    elif not "val" in store:
        # split train into val,test.
        train, val = shuffle_split(train, "train", "val", 0.05)
    if config.get("eval.qa", False):
        if config.get("dataset") == "psg":
            print(
                "[sg_parser] qa-evaluation activated on PSG. USING TRAIN+VAL+TEST AS TEST."
            )
            test = train + val + test
    test = test + rm

    if config.get("run_label") == "DEBUG":
        debug_graph_count_limit = 100
        print(
            f"[sg_parser] debug mode. only loading {debug_graph_count_limit} graphs.")
        train = train[:debug_graph_count_limit]
        val = val[:debug_graph_count_limit]
        test = test[:debug_graph_count_limit]
    print(f"{len(train)} train, {len(val)} val, {len(test)} test items loaded.")

    if config["eval.train_sgg"]:
        if config['dataset'] == 'vg':
            with open(f"/home/vquapil/masterarbeit-data/raw/vg/train.json", "r") as f:
                bb_data = json.load(f)
            train = get_bb_data(train, bb_data)
            with open(f"/home/vquapil/masterarbeit-data/raw/vg/val.json", "r") as f:
                bb_data = json.load(f)
            val = get_bb_data(val, bb_data)
            with open(f"/home/vquapil/masterarbeit-data/raw/vg/test.json", "r") as f:
                bb_data = json.load(f)
            test = get_bb_data(test, bb_data)
        elif config['dataset'] == 'psg':
            with open(f"/home/vquapil/masterarbeit-data/raw/psg/psg.json", "r") as f:
                bb_data = json.load(f)
            train, val, test = get_bb_data_psg(train + val + test, bb_data, len(train), len(val))
        else:
            print("Dataset for training SGG not implemented")
    
    if config["eval.clas"]:
        with open(f"{DATA_DIR}/psg_captions/classification_coco.json", "r") as f:
            class_data = json.load(f)
        train_full = {
            "data": train,
            "obj_classes": obj_classes,
            "predicate_classes": predicate_classes,
            "classification_data": get_class_data(train, class_data)
        }
    else:
        train_full = {
            "data": train,
            "obj_classes": obj_classes,
            "predicate_classes": predicate_classes,
        }
    val_full = {
        "data": val,
        "obj_classes": obj_classes,
        "predicate_classes": predicate_classes,
    }
    if config["eval.clas"]:
        with open(f"{DATA_DIR}/psg_captions/classification_coco.json", "r") as f:
            class_data = json.load(f)
        test_full = {
            "data": test,
            "obj_classes": obj_classes,
            "predicate_classes": predicate_classes,
            "classification_data": get_class_data(test, class_data)
        }
    else:
        test_full = {
            "data": test,
            "obj_classes": obj_classes,
            "predicate_classes": predicate_classes,
        }
    if config["eval.train_sgg"]:
        train_full['train_sgg'] = []
        val_full['train_sgg'] = []
        test_full['train_sgg'] = []
        
    return train_full, val_full, test_full

def get_class_data(test, class_data):
    classification_data = []
    num = 0
    for obj in test:
        idx = str(int(obj["file_name"].split("/")[-1].split(".")[0]))
        if idx not in class_data:
            class_data[idx] = {
                "category": -1,
                "file_name": "",
                "name": "",
                "description": "",
                "area": 0.0,
                "label": -1
            }
            num += 1
        classification_data.append(class_data[idx])
    print(f"Missing classification labels {num}")
    return classification_data

def get_bb_data(data, bb_info):
    whole_dict = {}
    for i in range(len(data)):
        whole_dict[int(data[i]["file_name"].split(".")[0])] = {
            "relations": data[i]["relations"],
            "nodes": data[i]["nodes"],
            "height": 0,
            "width": 0,
            "bbox": [],
            "relation_bbox_normalized": []
        }
    
    for img in bb_info["images"]:
        whole_dict[img['id']]["height"] = img["height"]
        whole_dict[img["id"]]["width"] = img["width"]
    
    for an in bb_info["annotations"]:
        bbox = an["bbox"]
        whole_dict[an["image_id"]]["bbox"].append(bbox)
        
    for obj in whole_dict:
        temp_list = []
        for rel in whole_dict[obj]["relations"]:
            temp_list.append([whole_dict[obj]["bbox"][rel[0]], whole_dict[obj]["bbox"][rel[1]]])
        whole_dict[obj]['relation_bbox_normalized'] = xywh_to_cxcywh(normalize_boxes_xywh(temp_list, whole_dict[obj]["width"], whole_dict[obj]["height"]))
    
    for i in range(len(data)):
        id = int(data[i]["file_name"].split(".")[0])
        data[i]['relation_bbox_normalized'] = whole_dict[id]['relation_bbox_normalized']
    return data

def get_bb_data_psg(data, bb_info, train, val):
    whole_dict = {}
    index_to_remove = []
    train_sub, val_sub = 0,0
    for i in range(len(data)):
        index = int(data[i]["file_name"].split(".")[0].split('/')[-1])
        if index in whole_dict:
            index_to_remove.append(i)
            if i < train:
                train_sub += 1
            if i >= train and i < train+val:
                val_sub += 1
            continue
        whole_dict[index] = {
            "relations": data[i]["relations"],
            "nodes": data[i]["nodes"],
            "height": 0,
            "width": 0,
            "bbox": [],
            "relation_bbox_normalized": []
        }
        
    data = [v for i, v in enumerate(data) if i not in index_to_remove]
    
    train = train - train_sub
    val = val - val_sub
        
    for obj in bb_info["data"]:
        id = int(obj['coco_image_id'])
        if id not in list(whole_dict.keys()):
            continue
        whole_dict[id]["height"] = obj["height"]
        whole_dict[id]["width"] = obj["width"]
    
        for an in obj["annotations"]:
            bbox = an["bbox"]
            whole_dict[id]["bbox"].append(bbox)
        
    for obj in whole_dict:
        temp_list = []
        for rel in whole_dict[obj]["relations"]:
            temp_list.append([whole_dict[obj]["bbox"][rel[0]], whole_dict[obj]["bbox"][rel[1]]])
        whole_dict[obj]['relation_bbox_normalized'] = normalize_boxes_xyxy_to_cxcywh(temp_list, whole_dict[obj]["width"], whole_dict[obj]["height"])
    
    for i in range(len(data)):
        id = int(data[i]["file_name"].split(".")[0].split('/')[-1])
        data[i]['relation_bbox_normalized'] = whole_dict[id]['relation_bbox_normalized']
    return data[:train], data[train:train+val], data[train+val:]
    
def normalize_boxes_xywh(boxes, img_w, img_h):
    """
    Normalize bounding boxes from [x, y, w, h] to [0,1].
    boxes: list of [x, y, w, h]
    img_w, img_h: image dimensions
    returns: list of [x_norm, y_norm, w_norm, h_norm]
    """
    all = []
    for box in boxes:
        normalized = []
        for x, y, w, h in box:
            x_norm = max(0.0, min(1.0, x / img_w))
            y_norm = max(0.0, min(1.0, y / img_h))
            w_norm = max(0.0, min(1.0, w / img_w))
            h_norm = max(0.0, min(1.0, h / img_h))
            normalized.append([x_norm, y_norm, w_norm, h_norm])
        all.append(normalized)
    return all

def normalize_boxes_xyxy_to_cxcywh(boxes, img_w, img_h):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to normalized [cx, cy, w, h].
    
    boxes: list of lists (each box can be [[x1, y1, x2, y2], ...])
    img_w, img_h: image dimensions
    returns: list of normalized boxes [[cx, cy, w, h], ...] with values in [0,1]
    """
    all_normalized = []
    for box_group in boxes:  # supports nested structure like your original
        normalized_group = []
        for x1, y1, x2, y2 in box_group:
            # width and height in absolute pixels
            w = x2 - x1
            h = y2 - y1
            # center coordinates in absolute pixels
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            # normalize everything
            cx /= img_w
            cy /= img_h
            w /= img_w
            h /= img_h
            # clamp to [0, 1] for safety
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            normalized_group.append([cx, cy, w, h])
        all_normalized.append(normalized_group)
    return all_normalized


def xywh_to_cxcywh(boxes):
    """
    Convert [x, y, w, h] boxes to [cx, cy, w, h].
    boxes: list of normalized [x, y, w, h]
    returns: list of [cx, cy, w, h]
    """
    all = []
    for box in boxes:
        converted = []
        for x, y, w, h in box:
            cx = x + w / 2.0
            cy = y + h / 2.0
            converted.append([cx, cy, w, h])
        all.append(converted)
    return all

