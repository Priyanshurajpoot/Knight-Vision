# core/utils.py
import cv2
import numpy as np

def check_board_quality(board_img):
    """
    Returns (ok: bool, message: str) if image is sharp, well-lit,
    square-shaped and suitable for recognition.
    """
    if board_img is None:
        return False, "No board detected"

    # --- Aspect ratio check (should be ~1.0) ---
    h, w = board_img.shape[:2]
    aspect_ratio = w / float(h)
    if abs(aspect_ratio - 1.0) > 0.2:
        return False, "Board not square enough"

    # --- Sharpness (variance of Laplacian) ---
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 100:
        return False, "Image too blurry"

    # --- Brightness & contrast ---
    mean, std = cv2.meanStdDev(gray)
    if mean[0][0] < 50 or mean[0][0] > 200:
        return False, "Lighting not suitable"
    if std[0][0] < 30:
        return False, "Low contrast"

    return True, "Good quality"


# (keep your existing helpers below if you had any)
