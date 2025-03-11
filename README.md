# Vehicle detector system

The vehicle detector system for intelligent toll stations that can detect vehicles and track lane usage.

## Project Overview

This system uses computer vision techniques to:
- Detect vehicles in video streams
- Track vehicles across frames
- Identify lane usage
- Count vehicles by lane
- Generate analytics reports

## Run Locally

### Prerequisites
- Python 3.6+
- pip package manager

### Installation

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Reinstall OpenCV (if you encounter any issues)

```bash
pip uninstall opencv-python
pip install opencv-python
```

### Usage

Run the main detection script:

```bash
python main.py
```

### Configuration for train yolo model

You can modify parameters in the `.env` file to customize the detection system.

## Project Structure

- `main.py`: Entry point for the application
- `detectors/`: Vehicle and lane detection algorithms
- `constants/`: Configuration and constant values
- `models/`: Pre-trained ML models for vehicle tracking
- `trackers/`: Vehicle tracking algorithms
- `utils/`: Helper functions
- `input_videos/`: Sample videos for testing
- `outputs/`: Detection results and analytics

## Evaluation

Check the Jupyter notebooks in the `outputs/` directory to see evaluation results.


