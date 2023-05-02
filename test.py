"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    plot_image
)
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    train_loader, test_loader, train_eval_loader, sample_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv", sample_csv_path=config.DATASET + "/test_sample.csv"
    )


    load_checkpoint(
        config.TEST_CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    )

    pred_boxes, true_boxes = get_evaluation_bboxes(
        sample_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )
    true_boxes_idx_wise = [[]]
    for box in true_boxes:
        img_idx = box[0]
        if len(true_boxes_idx_wise) - 1 < img_idx:
            true_boxes_idx_wise.append([box[1:]])
        else:
            true_boxes_idx_wise[-1].append(box[1:])

    pred_boxes_idx_wise = [[]]
    for box in pred_boxes:
        img_idx = box[0]
        if len(pred_boxes_idx_wise) - 1 < img_idx:
            pred_boxes_idx_wise.append([box[1:]])
        else:
            pred_boxes_idx_wise[-1].append(box[1:])

    train_idx = 0
    for (x, y) in sample_loader:
        for i in range(len(x)):
            plot_image(x[i].permute(1, 2, 0).detach().cpu(), pred_boxes_idx_wise[train_idx])
            train_idx += 1


if __name__ == "__main__":
    main()