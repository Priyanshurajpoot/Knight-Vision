# Knight Vision 🔍♟️

<img width="150" height="150" alt="Knight Vision" src="https://github.com/user-attachments/assets/9bf60540-f9ac-456a-8eb7-1c0fbfed5b3d" />


A real-time chess position recognition and analysis system using computer vision and deep learning. Knight Vision captures chess positions from a camera feed, recognizes pieces using a trained neural network, generates FEN notation, and analyzes positions using Stockfish.

## Features

- **Real-time Board Detection**: Automatically detects and captures chessboards from camera feed
- **Quality Validation**: Ensures captured images are sharp, well-lit, and properly aligned
- **Piece Recognition**: Uses a trained CNN to identify chess pieces (13 classes: empty + 12 piece types)
- **FEN Generation**: Converts recognized positions to standard FEN notation
- **Position Analysis**: Integrates with Stockfish for move recommendations and evaluation
- **Dual Recognition Methods**: Supports both neural network and template matching fallback
- **User-Friendly GUI**: PyQt5 interface with three tabs for live capture, recognition, and analysis

## System Requirements

### Hardware
- Webcam or external camera
- CPU: Modern multi-core processor (GPU optional but recommended for training)
- RAM: 4GB minimum, 8GB recommended
- Storage: 2GB for application and models

### Software
- Python 3.8+
- OpenCV 4.5+
- PyTorch 1.10+
- PyQt5 5.15+
- Stockfish chess engine

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Priyanshurajpoot/knight-vision.git
cd knight-vision
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Stockfish

Download Stockfish from [official website](https://stockfishchess.org/download/) and note the executable path.

### 5. Project Structure

```
knight-vision/
├── app.py                          # Application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── DOCUMENTATION.md                # Detailed documentation
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py             # Main GUI window
│   └── dialog.py                  # Settings dialog
│
├── core/
│   ├── __init__.py
│   ├── detection.py               # Board detection and square extraction
│   ├── classifier.py              # Neural network piece classifier
│   ├── fen.py                     # FEN notation utilities
│   ├── engine.py                  # Stockfish integration
│   ├── models.py                  # Neural network architectures
│   ├── utils.py                   # Helper functions
│   ├── synth_dataset.py           # Synthetic data generation
│   ├── train_classifier.py        # Model training script
│   └── convert_to_torchscript.py  # Model conversion utility
│
├── templates/                     # Piece template images (for fallback)
│   ├── wP.png, wN.png, wB.png, wR.png, wQ.png, wK.png
│   └── bP.png, bN.png, bB.png, bR.png, bQ.png, bK.png
│
├── data/
│   └── synth/                    # Generated synthetic training data
│       ├── empty/
│       ├── wP/, wN/, wB/, wR/, wQ/, wK/
│       └── bP/, bN/, bB/, bR/, bQ/, bK/
│
└── models/                       # Trained model files
    ├── best_state_dict.pth
    ├── final_state_dict.pth
    ├── model_scripted.pt
    └── meta.json
```

## Quick Start

### 1. Prepare Piece Templates (Optional)

If you want to use template matching as fallback, place 64x64 PNG images of chess pieces in the `templates/` directory. Name them as:
- White pieces: `wP.png`, `wN.png`, `wB.png`, `wR.png`, `wQ.png`, `wK.png`
- Black pieces: `bP.png`, `bN.png`, `bB.png`, `bR.png`, `bQ.png`, `bK.png`

### 2. Generate Synthetic Training Data

```bash
python core/synth_dataset.py --out_dir data/synth --samples_per_class 2000 --size 64
```

This creates 2000 augmented samples per class (26,000 total images).

### 3. Train the Classifier

```bash
python core/train_classifier.py --data_dir data/synth --epochs 12 --batch_size 128 --out_dir models
```

Training typically takes 10-30 minutes depending on your hardware.

### 4. Run the Application

```bash
python app.py
```

## Usage Guide

### Live Tab - Board Capture

1. Click **"Start Camera & Capture Board"**
2. Position your chessboard in the camera view
3. The system detects the board automatically
4. Once a high-quality capture is detected, it switches to the Image tab

**Tips for Best Capture:**
- Ensure good, even lighting (avoid shadows)
- Position camera perpendicular to board
- Fill the frame with the chessboard
- Use a contrasting background
- Keep the camera steady

### Image Tab - Piece Recognition

1. Review the captured board image
2. Click **"Start Recognition"** to identify pieces
3. Pieces are labeled on the board
4. Click **"Re-recognize Pieces"** if needed to try again
5. Click **"Confirm & Generate FEN"** to finalize

The FEN notation appears in the text box below.

### Analysis Tab - Engine Analysis

1. Click **"Set Stockfish Path & Initialize Engine"** (first time only)
2. Browse to your Stockfish executable
3. Optionally click **"Set Model Path & Load Model"** to switch models
4. Click **"Start Analysis"** to get the best move and evaluation

## Training Your Own Model

### Prepare Your Dataset

Option A: Use synthetic data (recommended for quick start):
```bash
python core/synth_dataset.py --samples_per_class 2000
```

Option B: Collect real images:
- Organize in ImageFolder structure: `data/real/{class_name}/*.png`
- Classes: empty, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK
- Minimum 500 images per class recommended

### Train the Model

```bash
python core/train_classifier.py \
    --data_dir data/synth \
    --out_dir models \
    --epochs 12 \
    --batch_size 128 \
    --lr 0.001 \
    --img_size 64
```

### Monitor Training

Watch for:
- Decreasing training loss
- Validation accuracy > 95%
- Best model saved when validation loss improves

### Convert to TorchScript (if needed)

```bash
python core/convert_to_torchscript.py \
    --state models/best_state_dict.pth \
    --out models/model_scripted.pt
```

## Configuration

### Model Parameters

Edit in `core/classifier.py`:
```python
DEFAULT_CLASS_MAP = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK"
]
```

### Detection Parameters

Edit in `core/detection.py`:
```python
stable_threshold = 5  # Frames needed for stability
template_dir = "templates"  # Template location
```

### Quality Thresholds

Edit in `core/utils.py`:
```python
aspect_ratio_tolerance = 0.2
sharpness_threshold = 100
brightness_range = (50, 200)
contrast_threshold = 30
```

## Troubleshooting

### Board Not Detected
- Improve lighting conditions
- Ensure board has clear edges
- Use a contrasting background
- Reduce camera distance

### Poor Recognition Accuracy
- Retrain model with more data
- Check piece template quality
- Ensure good image quality (sharp, well-lit)
- Verify correct class mapping

### Engine Errors
- Verify Stockfish path is correct
- Check Stockfish version compatibility (11+)
- Ensure executable has proper permissions

### Camera Issues
- Check camera permissions
- Try different camera index: `cv2.VideoCapture(1)`
- Update camera drivers
- Close other applications using the camera

## Performance Tips

1. **GPU Acceleration**: Install PyTorch with CUDA support for faster inference
2. **Model Optimization**: Use TorchScript traced models for production
3. **Batch Processing**: Process multiple squares simultaneously
4. **Image Resolution**: Use 64x64 square images for optimal speed/accuracy balance

## Advanced Features

### Custom Piece Templates

Create your own templates by:
1. Capturing clear images of each piece type
2. Cropping to 64x64 pixels
3. Ensuring transparent background (RGBA format)
4. Saving in `templates/` directory

### Multiple Model Support

Load different models for different board styles:
```python
classifier = PieceClassifier("models/wood_board_model.pt")
detector.classifier = classifier
```

### FEN Validation

The system validates FEN strings for:
- Correct row count (8)
- Proper square count per row (8)
- Presence of both kings
- Non-empty board



## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request



## Acknowledgments

- Stockfish chess engine team
- PyTorch and OpenCV communities
- Chess.com for piece design inspiration

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: priyanshux5xraj@gmail.com


## Version History

- v1.0.0 (2025-01): Initial release
  - Real-time board detection
  - CNN-based piece recognition
  - Stockfish integration
  - Quality validation




AUTHOR
- Priyanshu Rajpoot
