import json
import os
import random
import numpy as np  
import sys
from tqdm import tqdm

def load_json(pred_file):
    with open(pred_file, 'r') as f:
        return json.load(f)


def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """

    if box_b is None:
        return 0
    box_a = np.array(box_a)
    box_b = np.array(box_b)

    try:
        max_a = box_a[0:3] + box_a[3:6] / 2
        max_b = box_b[0:3] + box_b[3:6] / 2
    except:
        import pdb
        pdb.set_trace()
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def find_view_dep(questions,qs_id):
    for line in questions:
        idx = line["id"]
        if idx == qs_id:
            return line['metadata']['view_dep']
        else:
            continue


def calculate_accuracy(preds):
    correct_predictions = 0
    total_predictions = len(preds)
    
    for pred_entry in preds:
        if pred_entry['gt'] == pred_entry['predicted_id']:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def main(pred_dir,question_file):
    """
    Evaluate the accuracy of prediction files in a directory. The directory should contain JSON files with prediction data.
    """
    with open(os.path.expanduser(question_file)) as f:
        questions = json.load(f)

    with open(pred_dir) as f:
        pred_files = [json.loads(line.strip()) for line in f.readlines()]
    total_correct_predictions = 0
    total_predictions = 0
    unique_total = 0
    easy_total = 0
    dep_total = 0
    correct_easy = 0
    correct_dep = 0

    correct_25 = 0
    unique_25 = 0
    correct_50 = 0
    

    for item in tqdm(pred_files):
        qs_id = item['sample_id']
        gt = item['gt_response']
        pred = item['pred_response']
        view_dep = item['view_dep']
        iou = calc_iou(gt, pred)

        if item['question_type'] == 'easy':
            easy_total += 1
        if view_dep:
            dep_total += 1

        if iou >= 0.25:
            if item['question_type'] == 'easy':
                correct_easy += 1
            if view_dep:
                correct_dep += 1

        if iou >= 0.25:
            correct_25 += 1

        if iou >= 0.5:
            correct_50 += 1

    total_predictions += len(pred_files)

    print()
    print()
    print('Easy     {:.2%}   {} / {}'.format(correct_easy / easy_total, correct_easy, easy_total))
    print('Hard     {:.2%}   {} / {}'.format((correct_25 - correct_easy) / (total_predictions - easy_total),
                                correct_25 - correct_easy,
                                total_predictions - easy_total))
    print('Dep      {:.2%}   {} / {}'.format(correct_dep / dep_total, correct_dep, dep_total))
    print('Indep    {:.2%}   {} / {}'.format((correct_25 - correct_dep) / (total_predictions - dep_total),
                                        correct_25 - correct_dep,
                                        total_predictions - dep_total))
    print()
    print('Acc@25           {:.2%}   {} / {}'.format(correct_25 / total_predictions, correct_25, total_predictions))
    print('Acc@50           {:.2%}   {} / {}'.format(correct_50 / total_predictions, correct_50, total_predictions))


if __name__ == '__main__':
    pred_dir = 'results/n3rd_val.jsonl'
    question_file = 'n3rd_val_llava_style_viewtype.json' 
    main(pred_dir,question_file)
