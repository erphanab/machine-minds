"""
Train the YOLO-seg detector and write its model card (Machine Minds step A5).

Runs in the YOLO environment (CUDA PyTorch + Ultralytics), not Isaac's or the
host .venv. Defaults are the guide's recipe:

    $YOLO_PY mm/perception/train_yolo.py --name yolo_seg_v1
    # = yolo segment train model=yolo11s-seg.pt data=configs/dataset.yaml
    #   imgsz=1280 epochs=100 batch=8 hsv_h=0.0 hsv_s=0.3 hsv_v=0.4
    #   degrees=180 fliplr=0.5 flipud=0.5 mosaic=1.0 seed=0 project=models

hsv_h stays 0: hue shifts would give objects colours they never have in the
real cell. Top-down views make full rotation and both flips safe.

Writes models/<name>/card.json: library versions, dataset hash, seed, image
size, epochs and final val/test metrics. Weights stay git-ignored.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_dataset(path):
    """Copy of dataset.yaml with `path` made absolute (repo-relative in the file)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not os.path.isabs(cfg["path"]):
        cfg["path"] = os.path.normpath(os.path.join(ROOT, cfg["path"]))
    return cfg


def dataset_hash(cfg, splits=("train", "val", "test")):
    """SHA-256 over every image and label file in the given splits, in order."""
    h = hashlib.sha256()
    for split in splits:
        for kind in ("images", "labels"):
            d = os.path.join(cfg["path"], kind, split)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                h.update(f"{kind}/{split}/{name}".encode())
                with open(os.path.join(d, name), "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
    return h.hexdigest()


def seg_metrics(result):
    """Mask and box mAP from an Ultralytics SegmentMetrics object, per class too."""
    out = {
        "mask_map50": float(result.seg.map50),
        "mask_map50_95": float(result.seg.map),
        "box_map50": float(result.box.map50),
        "box_map50_95": float(result.box.map),
        "mask_precision": float(result.seg.mp),
        "mask_recall": float(result.seg.mr),
    }
    names = result.names
    out["per_class_mask_map50_95"] = {
        names[int(c)]: float(result.seg.maps[int(c)]) for c in result.ap_class_index
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--data", default=os.path.join(ROOT, "configs", "dataset.yaml"))
    parser.add_argument("--name", default="yolo_seg_v1")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0, help="train on this share")
    parser.add_argument("--device", default="0")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--card-only", action="store_true",
        help="no training: evaluate models/<name>/weights/best.pt and write card.json "
        "(for a run stopped early, e.g. v3 at epoch 24 once converged)",
    )
    parser.add_argument("--note", default=None, help="free text recorded in card.json")
    parser.add_argument("--optimizer", default=None, help="fine-tuning: e.g. AdamW (Ultralytics' auto ignores --lr0)")
    parser.add_argument("--lr0", type=float, default=None)
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    cfg = resolve_dataset(args.data)
    project = os.path.join(ROOT, "models")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        data_yaml = tmp.name

    hparams = dict(
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.4,
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        seed=args.seed,
        deterministic=True,
        fraction=args.fraction,
    )
    if args.optimizer:
        hparams["optimizer"] = args.optimizer
    if args.lr0 is not None:
        hparams["lr0"] = args.lr0
    run_dir = os.path.join(project, args.name)
    t0 = time.time()
    if args.card_only:
        train_s = None
    else:
        model = YOLO(args.model)
        model.train(
            data=data_yaml,
            project=project,
            name=args.name,
            exist_ok=args.exist_ok,
            workers=args.workers,
            device=args.device,
            **hparams,
        )
        train_s = time.time() - t0
    best = os.path.join(run_dir, "weights", "best.pt")
    trained = YOLO(best)
    metrics = {}
    for split in ("val", "test"):
        if cfg.get(split):
            r = trained.val(
                data=data_yaml,
                split=split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=os.path.join(run_dir, "eval"),
                name=split,
                exist_ok=True,
                plots=False,
            )
            metrics[split] = seg_metrics(r)

    with open(os.path.join(ROOT, "configs", "roi.yaml")) as f:
        roi = yaml.safe_load(f)
    card = {
        "name": args.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights": os.path.relpath(best, ROOT).replace(os.sep, "/"),
        "base_model": args.model,
        "classes": cfg["names"],
        "dataset": {
            "config": os.path.relpath(args.data, ROOT).replace(os.sep, "/"),
            "sha256": dataset_hash(cfg),
            "counts": {
                s: len(os.listdir(os.path.join(cfg["path"], "images", s)))
                for s in ("train", "val", "test")
                if os.path.isdir(os.path.join(cfg["path"], "images", s))
            },
        },
        "roi": roi,
        "hyperparameters": hparams,
        "train_seconds": None if train_s is None else round(train_s),
        "note": args.note,
        "metrics": metrics,
        "versions": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
    }
    with open(os.path.join(run_dir, "card.json"), "w") as f:
        json.dump(card, f, indent=2)
    os.unlink(data_yaml)
    print(f"[train_yolo] card -> {os.path.join(run_dir, 'card.json')}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # required on Windows: dataloader workers re-import
    sys.exit(main())
