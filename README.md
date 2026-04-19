# License Plate Deblurring & OCR Pipeline

Pipeline for deblurring license plates and extracting text using classical and CNN-based methods.

## Project Structure

```
pipeline/
├── deblurring_cnn.py       # CNN-based deblurring
├── deblurring_input.py     # Classical deblurring
├── ocr_plate.py            # License plate OCR
├── paths.py                # Path configuration
├── models/                 # Model checkpoints
├── data/
│   ├── input/             # Input images
│   ├── output/            # Deblurred results
│   └── debug/             # Debug artifacts
└── archive/               # Previous versions
```

## Path Management

All paths are centralized in `paths.py`. Do not hardcode paths in scripts.

```python
from paths import DATA_INPUT, DATA_OUTPUT, CHECKPOINT
```

Paths:
- Input: `data/input/`
- Output: `data/output/`
- Model: `models/plate_deblur_cnn.pth`
- Debug: `data/debug/`

## Usage

Classical method (no training required):
```bash
python deblurring_input.py data/input/image.jpg data/output/result.jpg
```

Train CNN model:
```bash
python deblurring_cnn.py train
```

Deblur with CNN:
```bash
python deblurring_cnn.py infer data/input/image.jpg
```

Extract license plate text:
```bash
python ocr_plate.py data/output/deblurred.jpg
```

## Dependencies

- Python 3.7+
- torch, torchvision
- PIL, numpy, scipy
- matplotlib
- Tesseract (for OCR)

## Ignored Files

The `.gitignore` excludes:
- `__pycache__/`, `.pyc` files
- Large data files in `data/`
- Virtual environment (`.venv/`)
- Model checkpoints (exclude if too large)
