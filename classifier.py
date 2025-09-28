# core/classifier.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class PieceClassifier:
    """
    Wrapper for PyTorch piece classifier.
    - Expect a model saved with torch.jit.trace or state_dict (we try both).
    - The model should output logits or probabilities of shape (N, C).
    """

    # Default mapping: index -> piece label used by detection (13 classes)
    # 0: empty, 1-6: white P,N,B,R,Q,K, 7-12: black P,N,B,R,Q,K
    DEFAULT_CLASS_MAP = [
        "empty",
        "wP", "wN", "wB", "wR", "wQ", "wK",
        "bP", "bN", "bB", "bR", "bQ", "bK"
    ]

    def __init__(self, model_path, device=None, input_size=64, class_map=None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.class_map = class_map if class_map is not None else self.DEFAULT_CLASS_MAP
        self.model = None
        self._load_model()
        # Preprocessing transform: convert square (numpy BGR) -> tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),  # converts to [0,1] and channels-first
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        try:
            # Try loading a scripted/traced model first
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
        except Exception:
            # Try loading state_dict into a common small model architecture (user should adapt)
            # We'll attempt to load generic architectures by letting the user provide a traced model ideally.
            # As fallback, attempt to load a state_dict into a simple CNN defined inline if shapes match.
            state = torch.load(self.model_path, map_location=self.device)
            # If it's a state_dict (dict of tensors) we try to build a small net
            if isinstance(state, dict):
                try:
                    from torch import nn
                    class SmallCNN(nn.Module):
                        def __init__(self, num_classes=len(self.class_map)):
                            super().__init__()
                            self.features = nn.Sequential(
                                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
                            )
                            self.fc = nn.Linear(128, num_classes)

                        def forward(self, x):
                            x = self.features(x)
                            x = x.view(x.size(0), -1)
                            x = self.fc(x)
                            return x

                    model = SmallCNN(num_classes=len(self.class_map))
                    model.load_state_dict(state)
                    model.to(self.device)
                    model.eval()
                    self.model = model
                except Exception as e:
                    raise RuntimeError(f"Failed to load state_dict model: {e}")
            else:
                raise RuntimeError("Unknown model file format. Please provide a torchscript model or state_dict.")

    def _preprocess_square(self, square_bgr):
        """
        square_bgr: numpy array (H,W,3) in BGR (OpenCV)
        returns: torch tensor (C,H,W) on device
        """
        # Convert BGR -> RGB, then to PIL Image for transforms
        rgb = square_bgr[..., ::-1]
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).to(self.device)
        return tensor

    def predict_batch(self, squares):
        """
        squares: list of numpy arrays (H,W,3) BGR
        Returns: list of labels (e.g. 'wP', 'bK', 'empty')
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # Preprocess and batch
        tensors = [self._preprocess_square(sq) for sq in squares]
        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
            # If model outputs logits, convert to probabilities (optional)
            if out.dim() == 1 or out.dim() == 2:
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                probs = F.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
            else:
                # unexpected shape
                preds = torch.argmax(out, dim=1).cpu().numpy()

        labels = [self.class_map[int(p)] for p in preds]
        return labels

    def predict(self, square):
        """Single-square prediction convenience"""
        return self.predict_batch([square])[0]
