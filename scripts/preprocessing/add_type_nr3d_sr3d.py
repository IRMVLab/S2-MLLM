# modified from https://github.com/iris0329/SeeGround#

import os
import json
import argparse
import jsonlines
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import ast
def is_explicitly_view_dependent(tokens):
    target_words = {
        "front",
        "behind",
        "back",
        "right",
        "left",
        "facing",
        "leftmost",
        "rightmost",
        "looking",
        "across",
    }
    processed_tokens= ast.literal_eval(tokens)
    return len(set(processed_tokens).intersection(target_words)) > 0


def decode_stimulus_string(s):
    parts = s.split("-", maxsplit=4)
    if len(parts) == 4:
        scene_id, instance_label, n_objects, target_id = parts
        distractor_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractor_ids = parts

    instance_label = instance_label.replace("_", " ")
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractor_ids = [int(i) for i in distractor_ids.split("-") if i]

    assert len(distractor_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractor_ids


def load_ref_data(anno_file, scan_ids):
    ref_data = []
    df = pd.read_csv(anno_file, encoding="utf-8")
    data = df.to_dict(orient="records") 
    for i, item in enumerate(tqdm(data)):
        scene_id = item['scan_id']
        if scene_id in scan_ids:
            ref_data.append(item)
    return ref_data


def load_csv_data(anno_file):
    df = pd.read_csv(anno_file, encoding="utf-8")
    data = df.to_dict(orient="records") 
    return data

def process_reference_item(ref, args):
    hardness = decode_stimulus_string(ref["stimulus_id"])[2]
    easy_context_mask = hardness <= 2
    view_dep_mask = is_explicitly_view_dependent(ref["tokens"])
    if easy_context_mask:
        ref['type'] = 'easy'
    else:
        ref['type'] = 'hard'
    ref["view_dep"] = view_dep_mask
    return ref


def save_processed_data(new_data, save_dir, scan_id):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/{scan_id}.json", "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved scan {scan_id} data to {save_dir}/{scan_id}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="add grounding type")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="source_data",
        help="Directory to save data.",
    )
    parser.add_argument(
        "--anno_file",
        type=str,
        default="source_data/sr3d.csv",
        help="Path to the annotation file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nr3d",
        help="dataset name",
    )
    args = parser.parse_args()
    dataset = args.dataset
    ref_data_train = load_csv_data(f"source_data/{dataset}_train.csv")
    ref_data_val = load_csv_data(f"source_data/{dataset}_test.csv")
    new_data_train = []
    new_data_val = []
    # Process each reference item
    for ref in tqdm(ref_data_train):
        new_data_train.append(
            process_reference_item(
                ref,
                args,
            )
        )
    for ref in tqdm(ref_data_val):
        new_data_val.append(
            process_reference_item(
                ref,
                args,
            )
        )
    
    df_train = pd.DataFrame(new_data_train)
    df_train.to_csv(f"source_data/{dataset}_train_viewtype.csv", index=False, encoding="utf-8")
    df_val = pd.DataFrame(new_data_val)
    df_val.to_csv(f"source_data/{dataset}_val_viewtype.csv", index=False, encoding="utf-8")