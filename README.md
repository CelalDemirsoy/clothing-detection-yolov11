# 👕 Real-Time Clothing Detection with YOLOv11

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-00FFFF?style=flat)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)

> Graduation project — Real-time clothing classification system from video streams using YOLOv11.

## 🏆 Results

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **92%** |
| **Inference Speed** | **30 FPS** |
| Framework | YOLOv11 + PyTorch |
| Export | ONNX (web deployment) |

## 📁 Project Structure

```
clothing-detection-yolov11/
├── run.py              # Main detection script (webcam/video)
├── train_grafs.py      # Training graphs & visualization
├── grafikciz.py        # Plot utilities
├── pttoonnx.py         # PyTorch → ONNX model converter
├── results.csv         # Training metrics log
└── website.html        # Web-based demo interface
```

## 🚀 Usage

**Install dependencies:**
```bash
pip install ultralytics opencv-python torch
```

**Run detection:**
```bash
python run.py
```

**Convert model to ONNX:**
```bash
python pttoonnx.py
```

## 🧠 Model

- Architecture: **YOLOv11** (Ultralytics)
- Training: Google Colab
- Dataset: Custom annotated clothing dataset
- Model weights: `best.pt` (available on request)

## 📊 Training

Training metrics available in `results.csv`.
Visualizations generated via `train_grafs.py` and `grafikciz.py`.

## 👤 Author

**Celal Demirsoy**
🌐 [celaldemirsoy.vercel.app](https://celaldemirsoy.vercel.app)
💻 [github.com/CelalDemirsoy](https://github.com/CelalDemirsoy)
