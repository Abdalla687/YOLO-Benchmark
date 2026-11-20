# YOLO Person Detection Benchmark

## Project Overview

This project benchmarks different YOLO models for person detection on a Windows CPU environment.

The goal is to compare model performance across different formats and image sizes using speed and accuracy metrics.

## Models Tested

- YOLOv8n
- YOLOv8s
- YOLO11n
- YOLO11s

## Formats Tested

- PyTorch
- ONNX
- NCNN

## Image Sizes

- 320 × 320
- 416 × 416
- 640 × 640

A total of **36 configurations** were benchmarked.

## Experimental Environment

- Operating System: Windows
- CPU: Intel Core i7-6700HQ @ 2.60 GHz
- Python: 3.13.14
- Ultralytics: 8.4.215
- PyTorch: 2.15.0 (CPU)
- ONNX Runtime: 1.32.0
- Execution Device: CPU

## Evaluation Metrics

The benchmark records:

- Average Latency (ms)
- P95 Latency (ms)
- FPS
- Preprocessing Time
- Inference Time
- Postprocessing Time
- Precision
- Recall
- mAP50
- mAP50-95
- CPU Usage
- RAM Usage

## Dataset

COCO128 was used for validation and accuracy evaluation.

The results provide a baseline comparison of the tested models and configurations.

## Project Structure

```text
YOLO-Benchmark/
│
├── Notepad/
│   ├── benchmarkall.py
│   └── make_chart.py
│
├── charts/
│   ├── latency_bar.png
│   └── accuracy_vs_fps.png
│
├── results_all.csv
├── requirements.txt
├── README.md
└── YOLO_Windows_Benchmark_Report_BW.pdf
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOLO-Benchmark.git
cd YOLO-Benchmark
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Benchmark

Run the benchmark script:

```bash
python Notepad/benchmarkall.py
```

## Generating Charts

Generate the performance charts using:

```bash
python Notepad/make_chart.py results_all.csv
```

The generated charts include:

- `latency_bar.png`: Average and P95 latency comparison.
- `accuracy_vs_fps.png`: Accuracy versus FPS comparison.

## Results

All benchmark results are stored in `results_all.csv`.

The charts and PDF report provide visual and detailed summaries of the experimental results.

## Limitations and Future Work

- Tests were conducted on a Windows CPU environment.
- Raspberry Pi 5 deployment has not been evaluated in this benchmark.
- Further testing with the actual wheelchair camera footage is required.
- Real-world performance must be evaluated under the target deployment conditions.

## Author

Abdullah Emad Ebrahim

**Project:** YOLO Person Detection Benchmark
