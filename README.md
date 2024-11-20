# Thesis Project

This project implements a machine learning model for image processing and segmentation using PyTorch.

## Project Structure

```
├── src/               # Source code
│   ├── models/        # Neural network models
│   ├── data/         # Data loading and processing
│   ├── utils/        # Utility functions
│   └── configs/      # Configuration files
├── tests/            # Unit tests
├── data/             # Dataset files
├── experiments/      # Experiment results
├── models/           # Saved model checkpoints
└── requirements.txt  # Project dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

- Training: `python src/train.py`
- Inference: `python src/inference.py`

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- NumPy >= 1.19.5
- OpenCV >= 4.5.3
- Albumentations >= 1.0.3
- And more (see requirements.txt)
