import os
import json
import torch
import argparse
from tqdm import tqdm
import pandas as pd

# modified from https://github.com/3dlg-hcvc/M3DRef-CLIP/blob/main/dataset/scanrefer/add_evaluation_labels.py
def load_scene(filename):
    d = torch.load(filename)
    object_ids = d['aabb_obj_ids'].tolist()
    corner_xyz = d['aabb_corner_xyz'].tolist()

    ret = {}
    for i in range(len(object_ids)):
        object_id = str(object_ids[i])

        xs, ys, zs = zip(*corner_xyz[i])
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        l = z_max - z_min

        ret[object_id] = (x_center, y_center, z_center, w, h, l)

    return ret

# modified from https://github.com/LaVi-Lab/Video-3D-LLM/blob/main/scripts/3d/preprocessing/process_scanrefer.py
def main(args):

    if args.template_type == "gen":
        template = "<image>Localize the object according to the following description.\n{desc}\nOutput the answer in the format x_min,y_min,z_min,x_max,y_max,z_max",
    elif args.template_type == "cls":
        template = "<image>Identify the object according to the following description.\n{desc}\nThere may be no corresponding object, or there may be one or more objects."

    all_data = []
    scan2box = {}
    for split in ['train','val']:
        df = pd.read_csv(f"nr3d_{split}_viewtype.csv", encoding="utf-8")
        datas = df.to_dict(orient="records") 
        for i, item in enumerate(tqdm(datas)):
            scene_id = item['scan_id']
            if scene_id not in scan2box:
                scan2box[scene_id] = load_scene(os.path.join('/data', "pcd_with_object_aabbs", f'{split}', f"{scene_id}.pth"))
            box = scan2box[scene_id][str(item['target_id'])]
            desc = item['utterance'].capitalize()
            object_name = item["instance_type"]
            all_data.append({
                "id": i,
                "video": f"scannet/{item['scan_id']}",
                "conversations": [
                    {
                        "value": template.format(desc=desc),
                        "from": "human",
                    },
                    {
                        "value": f"The {object_name} is located at <ground> in global coordinates.",
                        "from": "gpt",
                    },
                ],
                "box": box,
                "metadata": {
                    "dataset": "nr3d",
                    "question_type": item["type"], 
                    "view_dep": item['view_dep'],
                    "object_id": item["target_id"],
                }
            })
    
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'nr3d_{split}_llava_style.json'), 'w') as f:
            json.dump(all_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/dataprocessed")
    parser.add_argument("--template_type", type=str, default="cls")
    args = parser.parse_args()

    main(args)