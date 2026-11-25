# evaluation_recall.py
"""This script measures how well pretrained YOLO finds people in a dataset of images.

It runs the unchanged yolo11n model on every image, compares its person boxes against our hand-labeled boxes, and
counts a person as found when the boxes overlap by at least 50%. It then reports recall (the share of labeled people
YOLO found) and false detections, using the same settings for every game and for the real-person baseline.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou_matrix(a, b):
    """IoU between every box in a (N, 4) and b (M, 4), both xyxy."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_labels(path, w, h):
    """YOLO normalized xywh label file -> (N, 4) pixel xyxy array of person boxes."""
    if not path.exists():
        return np.zeros((0, 4))
    boxes = [list(map(float, line.split()[1:5])) for line in path.read_text().splitlines() if line.split()[:1] == ["0"]]
    if not boxes:
        return np.zeros((0, 4))
    xywh = np.array(boxes) * [w, h, w, h]
    return np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)


def match(gt, pred, iou_thr):
    """Greedy one-to-one matching by IoU. Returns a bool array: which gt boxes were detected."""
    detected = np.zeros(len(gt), dtype=bool)
    if len(gt) and len(pred):
        ious = iou_matrix(gt, pred)
        used = np.zeros(len(pred), dtype=bool)
        for g, p in sorted(zip(*np.nonzero(ious >= iou_thr)), key=lambda gp: -ious[gp]):
            if not detected[g] and not used[p]:
                detected[g] = used[p] = True
    return detected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="dataset name used for output files, e.g. valorant")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a labeled person as detected")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    model = YOLO(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_gt = n_tp = n_pred = 0
    rows = []
    for result in model.predict(images, classes=[0], conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False):
        path = Path(result.path)
        h, w = result.orig_shape
        gt = load_labels(Path(args.labels) / f"{path.stem}.txt", w, h)
        pred = result.boxes.xyxy.cpu().numpy()
        detected = match(gt, pred, args.iou)
        n_gt += len(gt)
        n_tp += int(detected.sum())
        n_pred += len(pred)
        for box, hit in zip(gt, detected):
            bw, bh = box[2] - box[0], box[3] - box[1]
            rows.append(
                [path.name, round(bw, 1), round(bh, 1), round(bh / h, 4), round(bw * bh / (w * h), 5), int(hit)]
            )

    summary = {
        "dataset": args.name,
        "images": len(images),
        "labeled_people": n_gt,
        "detected": n_tp,
        "missed": n_gt - n_tp,
        "false_detections": n_pred - n_tp,
        "recall": round(n_tp / n_gt, 4) if n_gt else None,
        "precision": round(n_tp / n_pred, 4) if n_pred else None,
        "false_detections_per_image": round((n_pred - n_tp) / len(images), 3) if images else None,
        "settings": {
            "model": args.model,
            "conf": args.conf,
            "iou_match": args.iou,
            "imgsz": args.imgsz,
            "classes": [0],
        },
    }
    (out / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"{args.name}_boxes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "box_w_px", "box_h_px", "box_h_frac", "box_area_frac", "detected"])
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
