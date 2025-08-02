📫 Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/antonio-montemurro-2b3107287/)

# YOLO Person Tracker

A real-time person detection and tracking application using YOLOv8 and DeepSort with flexible video input support (webcam, IP camera, video file).

---

## Features

- Detects and tracks unique persons in video streams
- Supports multiple YOLOv8 models (nano, small, medium, large, extra large)
- Supports input from webcam, IP camera, or video files
- Utilizes GPU acceleration if available and enabled
- Simple interactive menu for configuration

---

## Requirements

- Python 3.8+
- CUDA-enabled GPU (optional, for faster inference)
- Dependencies listed in `requirements.txt`

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/yolo-person-tracker.git
cd yolo-person-tracker
```

2. (Recommanded) Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
