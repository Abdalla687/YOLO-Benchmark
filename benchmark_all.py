"""
Unified YOLO benchmark (Track A, Windows, CPU).
One script for ALL configs: 4 models x {PyTorch, ONNX, NCNN} x {320, 416, 640}.

Per config it records: Avg ms, P95 ms, FPS (end-to-end predict call),
Pre/Inf/Post ms, CPU %, RAM MB, Size MB, and mAP50 / mAP50-95 / Precision / Recall
(person class only) via model.val().

Usage:
  python benchmark_all.py --all                       # every config, skips ones already in results_all.csv
  python benchmark_all.py --model yolo11s --fmt ncnn --imgsz 320
  python benchmark_all.py --missing                   # only the 11 configs not measured yet
Requirements: pip install ultralytics psutil ncnn onnx onnxruntime pnnx
"""
import argparse, csv, os, shutil, time, glob
import numpy as np, cv2, psutil
from ultralytics import YOLO

MODELS = {"yolov8n": "YOLOv8n", "yolov8s": "YOLOv8s", "yolo11n": "YOLO11n", "yolo11s": "YOLO11s"}
FORMATS = ["pytorch", "onnx", "ncnn"]
SIZES = [320, 416, 640]
OUT = "results_all.csv"
COLS = ["Model", "Format", "Imgsz", "mAP50", "mAP50-95", "Precision", "Recall", "FPS", "Avg ms", "P95 ms",
        "Pre ms", "Inf ms", "Post ms", "CPU %", "RAM MB", "Size MB", "Notes"]

# Configs that had no measurement in the submitted results
MISSING = [("yolov8n", "pytorch", 416), ("yolov8n", "pytorch", 640),
           ("yolov8s", "pytorch", 320), ("yolov8s", "pytorch", 416), ("yolov8s", "pytorch", 640),
           ("yolo11n", "pytorch", 320),
           ("yolo11n", "ncnn", 416), ("yolo11n", "ncnn", 640),
           ("yolo11s", "ncnn", 320), ("yolo11s", "ncnn", 416), ("yolo11s", "ncnn", 640)]


def model_path(name, fmt, imgsz):
    """Export on first use. Fixed-size exports so every file matches its imgsz."""
    pt = f"{name}.pt"
    if fmt == "pytorch":
        return pt, os.path.getsize(pt) / 1048576
    base = f"{name}_{imgsz}"
    if fmt == "onnx":
        target = f"{base}.onnx"
        if not os.path.exists(target):
            exp = YOLO(pt).export(format="onnx", imgsz=imgsz, simplify=True)
            shutil.move(exp, target)
        return target, os.path.getsize(target) / 1048576
    target = f"{base}_ncnn_model"
    if not os.path.isdir(target):
        exp = YOLO(pt).export(format="ncnn", imgsz=imgsz)
        shutil.move(exp, target)
    size = sum(os.path.getsize(os.path.join(target, f)) for f in ("model.ncnn.bin", "model.ncnn.param"))
    return target, size / 1048576


def load_images(folder):
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    imgs = [cv2.imread(p) for p in paths]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise FileNotFoundError(f"No images in {folder}")
    return imgs


def bench(model, imgs, imgsz, conf, warmup, iters):
    proc = psutil.Process()
    kw = dict(imgsz=imgsz, conf=conf, classes=[0], device="cpu", verbose=False)
    for i in range(warmup):
        model.predict(source=imgs[i % len(imgs)], **kw)
    proc.cpu_percent(None)                          # reset counter
    lat, pre, inf, post, rss = [], [], [], [], []
    t_wall = time.perf_counter()
    for i in range(iters):
        t0 = time.perf_counter()
        r = model.predict(source=imgs[i % len(imgs)], **kw)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
        rss.append(proc.memory_info().rss / 1048576)
    wall = time.perf_counter() - t_wall
    cpu = proc.cpu_percent(None) / psutil.cpu_count()   # % of total CPU capacity over the run
    return dict(avg=np.mean(lat), p95=np.percentile(lat, 95), fps=1000 / np.mean(lat),
                pre=np.mean(pre), inf=np.mean(inf), post=np.mean(post), cpu=cpu, ram=max(rss), wall=wall)


def accuracy(path, imgsz, data):
    m = YOLO(path, task="detect")
    r = m.val(data=data, imgsz=imgsz, classes=[0], device="cpu", batch=1, verbose=False, plots=False)
    return r.box.map50, r.box.map, r.box.mp, r.box.mr


def already_done():
    if not os.path.exists(OUT):
        return set()
    return {(r["Model"], r["Format"], int(r["Imgsz"])) for r in csv.DictReader(open(OUT))}


def run_one(name, fmt, imgsz, a, imgs):
    label = (MODELS[name], {"pytorch": "PyTorch", "onnx": "ONNX", "ncnn": "NCNN"}[fmt], imgsz)
    if label in already_done() and not a.force:
        print("skip (already measured):", label); return
    print("=== ", label)
    path, size = model_path(name, fmt, imgsz)
    model = YOLO(path, task="detect")
    b = bench(model, imgs, imgsz, a.conf, a.warmup, a.iters)
    acc = accuracy(path, imgsz, a.data) if not a.no_acc else ("", "", "", "")
    notes = []
    if b["p95"] > 100: notes.append("P95>100ms")
    if b["fps"] < 10: notes.append("FPS<10")
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(COLS)
        w.writerow([label[0], label[1], imgsz,
                    *[round(x, 4) if x != "" else "" for x in acc],
                    round(b["fps"], 2), round(b["avg"], 2), round(b["p95"], 2),
                    round(b["pre"], 2), round(b["inf"], 2), round(b["post"], 2),
                    round(b["cpu"], 1), round(b["ram"], 1), round(size, 1), "; ".join(notes)])
    print(f"   FPS {b['fps']:.1f} | avg {b['avg']:.1f} ms | P95 {b['p95']:.1f} ms | CPU {b['cpu']:.0f}% | RAM {b['ram']:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--fmt", choices=FORMATS)
    ap.add_argument("--imgsz", type=int, choices=SIZES)
    ap.add_argument("--source", default="test_images", help="folder of test images")
    ap.add_argument("--data", default="coco128.yaml", help="dataset yaml for mAP (use a person-only COCO val subset for the final numbers)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-acc", action="store_true", help="skip mAP (latency only)")
    ap.add_argument("--force", action="store_true", help="re-run even if already in results_all.csv")
    a = ap.parse_args()

    if a.all:
        cfgs = [(m, f, s) for m in MODELS for f in FORMATS for s in SIZES]
    elif a.missing:
        cfgs = MISSING
    elif a.model and a.fmt and a.imgsz:
        cfgs = [(a.model, a.fmt, a.imgsz)]
    else:
        ap.error("use --all, --missing, or --model/--fmt/--imgsz")
    imgs = load_images(a.source)
    for c in cfgs:
        run_one(*c, a, imgs)
