import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from prettytable import PrettyTable

# def calculate_iou_per_class(pred, gt, class_id,  fileNum="", modelName="", fold="", metric="IOU"):
#     pred_mask = (pred == class_id)
#     gt_mask = (gt == class_id)
#
#     pred_mask1 = pred_mask.ravel()
#     gt_mask1 = gt_mask.ravel()
#
#     intersection = np.logical_and(pred_mask, gt_mask).sum()
#     union = np.logical_or(pred_mask, gt_mask).sum()
#     truth = np.logical_and(gt_mask, gt_mask).sum()
#     predicted = np.logical_and(pred_mask, pred_mask).sum()
#
#     iou = round(intersection / union, 4) if union != 0 else 0.0
#     recall = round(intersection / truth, 4) if union != 0 else 0.0
#     precision = round(intersection / predicted, 4) if union != 0 else 0.0
#     f1 = round( 2*(recall * precision) / (recall + precision), 4) if union != 0 else 0.0
#     fmindex = round(np.sqrt(recall * precision), 4) if union != 0 else 0.0
#
#     precision_sk = round(precision_score(gt_mask1, pred_mask1), 4)
#     recall_sk = round(recall_score(gt_mask1, pred_mask1), 4)
#     f1_sk = round(f1_score(gt_mask1, pred_mask1), 4)
#
#     table = PrettyTable()
#     table.field_names = ["fold", "modelName", "fileNum", "classID", "iou", "recall", "precision", "f1", "fmindex", "precision_sk", "recall_sk", "f1_sk"]
#     table.add_row([fold, modelName, fileNum, class_id, iou, recall, precision, f1, fmindex, precision_sk, recall_sk, f1_sk])
#
#     print(table)
#     return table, iou
    # if metric == "IOU":
    #     return iou
    # elif metric == "recall":
    #     return recall
    # elif metric == "precision":
    #     return precision
    # elif metric == "f1":
    #     return f1
    # elif metric == "fmindex":
    #     return fmindex
    # else:
    #     return "Other"




def compute_ious(pred, gt, class_ids, fileNum="", modelName="", fold="", metric="IOU", isChosen=0):
    bones = {1: "Femur", 2: "Tibia"}
    ious = {}
    table = PrettyTable()
    table.field_names = ["Category", "fold", "modelName", "fileNum", "isChosen", "classID", "boneName", "iou", "recall", "precision", "f1", "fmindex"] +   ["intersection",
        "union", "truth", "predicted", "truth_neg"]
    for class_id in class_ids:
        thisBone = bones[class_id]
        pred_mask = (pred == class_id)
        gt_mask = (gt == class_id)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        truth_neg = np.sum(gt_mask == False)
        truth_pos = np.sum(gt_mask == True)
        predicted = np.sum(pred_mask == True)

        iou = round(intersection / union, 4) if union != 0 else 0.0
        recall = round(intersection / truth_pos, 4) if union != 0 else 0.0
        precision = round(intersection / predicted, 4) if union != 0 else 0.0
        f1 = round(2 * (recall * precision) / (recall + precision), 4) if union != 0 else 0.0
        fmindex = round(np.sqrt(recall * precision), 4) if union != 0 else 0.0

        if np.random.random(1)>1.99:
            pred_mask1 = pred_mask.ravel()
            gt_mask1 = gt_mask.ravel()
            precision_sk = round(precision_score(gt_mask1, pred_mask1), 4)
            recall_sk = round(recall_score(gt_mask1, pred_mask1), 4)
            f1_sk = round(f1_score(gt_mask1, pred_mask1), 4)
        else:
            precision_sk = -1
            recall_sk = -1
            f1_sk = -1

        table.add_row(["FrontSeg", fold, modelName, fileNum, isChosen, class_id, thisBone, iou, recall, precision, f1, fmindex] + #, precision_sk, recall_sk, f1_sk]+
                      [intersection, union, truth_pos, predicted, truth_neg])

        # iou = calculate_iou_per_class(pred_mask, gt_mask, class_id,  fileNum, modelName, fold, metric)
        ious[class_id] = iou

    print(table)
    return table, ious #, [intersection, union, truth_pos, predicted, truth_neg]

def save_iou_results(filename, img_name, ious, palette):
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(f"{img_name} - ")
        iou_str = "-".join([f"{palette[class_id][0]}类 - {iou:.4f}" for class_id, iou in ious.items()])
        f.write(iou_str + "\n")

# breakpoint()
opacity=0.5
palette = [
        ['background', [0, 0, 0]],
        ['Femur', [0, 0, 255]],
        ['Tibia', [0, 255, 0]]
    ]

    # 类别映射：标注图像类别值 -> 预测图像类别值
gt_mapping = {
    0: 0,  # background
    38: 1,  # Femur
    75: 2  # Tibia
}

# 调色板字典
palette_dict = {idx: each[1] for idx, each in enumerate(palette)}
