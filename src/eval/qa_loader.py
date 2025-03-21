import copy
import json
import os

from multiset import Multiset
from tqdm import tqdm

type FilteredQTuple = tuple[
    str, str, str, str
]  # question, answer, image_id, structural_type
DATA_BASE = os.getenv("DATA_DIR")
YES_NO_STRUCTURAL_TYPES = ["verify", "logical", "yes/no"]
W_QUESTION_STRUCTURAL_TYPES = ["query", "other"]


def qa_tuples_to_question_collection(qa_tuples):
    r"""
    Tuple:
     (question: str, answer: str, imageId: str (without .jpg), type)
     types:
     - query (unknown answer)
     - compare
     - verify (yes/no)
     - choose
     - logical (yes/no) => answer should be "yes" or "no".
    """
    per_structural_type = {}
    for k in set([f[3] for f in qa_tuples]):
        part_filtered_tuples = [f for f in qa_tuples if f[3] == k]
        # more part-filtering!

        answers = list(set([f[1] for f in part_filtered_tuples]))
        if k in YES_NO_STRUCTURAL_TYPES:
            answers = ["yes", "no"]  # enforce order: yes -> 0, no -> 1
            filtered_tuples = [
                f for f in part_filtered_tuples if f[1] in answers]
            if len(filtered_tuples) < len(part_filtered_tuples):
                print(
                    f"dropped {len(part_filtered_tuples) - len(filtered_tuples)} questions with answers not in {answers}."
                )
                print([f[1]
                      for f in filtered_tuples if f[1] not in ["yes", "no"]])
        else:
            filtered_tuples = part_filtered_tuples
        data = [
            {
                "question": f[0],
                "imageId": f[2],
                "answer_index": answers.index(f[1]),
            }
            for f in tqdm(filtered_tuples, "qa: load question tuples")
        ]
        per_structural_type[k] = {
            "all_answers": answers,
            "data": data,
        }

    return per_structural_type


def vqa_load_collection():
    question_f = f"{DATA_BASE}/raw/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
    answer_f = f"{DATA_BASE}/raw/vqa/v2_mscoco_val2014_annotations.json"
    with open(question_f, "r") as f:
        questions = json.load(f)
    with open(answer_f, "r") as f:
        answers = json.load(f)
    ans_by_qid = {
        a["question_id"]: a["multiple_choice_answer"] for a in answers["annotations"]
    }
    q_type_by_qid = {a["question_id"]: a["answer_type"]
                     for a in answers["annotations"]}
    qa_tuples = [
        (
            q["question"],
            ans_by_qid[q["question_id"]],
            q["image_id"],
            q_type_by_qid[q["question_id"]],
        )
        for q in questions["questions"]
    ]
    return qa_tuples_to_question_collection(qa_tuples)


def gqa_load_collection(fname):
    r"""
    Load questions from GQA dataset.
    @param fname: filename in raw/gqa/questions
    @return: {structural_type: {"all_answers": List[str], "data": List[{"question": str, "imageId": str, "answer_index": int}]}}
    """
    with open(f"{DATA_BASE}/raw/gqa/questions/{fname}", "r") as f:
        qa_dicts = json.load(f)
    qa_tuples: list[FilteredQTuple] = [
        (qa["question"], qa["answer"], qa["imageId"], qa["types"]["structural"])
        for qa in qa_dicts.values()
    ]
    # question_statistics(filtered, fname)

    return qa_tuples_to_question_collection(qa_tuples)


def filter_collection(question_collection, img_ids, filter_all_answers=True):
    r"""
    Filter question collection by image ids.
    """
    c = copy.deepcopy(question_collection)
    dropcount = 0
    for k, v in c.items():
        oldlen = len(v["data"])
        v["data"] = [d for d in v["data"] if d["imageId"] in img_ids]
        dropcount += oldlen - len(v["data"])

    if filter_all_answers:
        for k, v in tqdm(c.items(), "filter all answers (tqdm categories)"):
            if k not in YES_NO_STRUCTURAL_TYPES:
                answers = list(
                    set([v["all_answers"][d["answer_index"]]
                        for d in v["data"]])
                )
                v["data"] = [
                    {
                        "question": d["question"],
                        "imageId": d["imageId"],
                        "answer_index": answers.index(
                            v["all_answers"][d["answer_index"]]
                        ),
                    }
                    for d in v["data"]
                ]
                v["all_answers"] = answers
            # don't filter yes/no questions

    return c, dropcount


def all_img_ids(question_dataset):
    return list(
        set(
            [
                d["imageId"]
                for dataset in question_dataset.values()
                for d in dataset["data"]
            ]
        )
    )


def num_questions(question_dataset):
    return sum([len(d["data"]) for d in question_dataset.values()])


# raw/gqa/questions/testdev_balanced_questions.json: 12578
# all: 172174
