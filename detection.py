# core/detection.py
import cv2
import numpy as np
import os

class BoardDetector:
    def __init__(self, classifier=None, stable_threshold=5, template_dir="templates"):
        """
        classifier: optional PieceClassifier instance (from core.classifier)
        stable_threshold: frames to consider board stable
        template_dir: fallback templates directory
        """
        self.prev_fen = None
        self.stable_count = 0
        self.stable_threshold = stable_threshold
        self.loaded_templates = None
        self.classifier = classifier
        self.template_dir = template_dir

    def detect_board(self, frame):
        """Detect the chessboard and return cropped board + rectangle coordinates."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(thresh, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        board_rect = None
        board_img = None

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and area > 10000:  # ensure large-enough square
                board_rect = approx.reshape(4, 2).astype("float32")
                # Order points (tl, tr, br, bl)
                s = board_rect.sum(axis=1)
                diff = np.diff(board_rect, axis=1)
                tl = board_rect[np.argmin(s)]
                br = board_rect[np.argmax(s)]
                tr = board_rect[np.argmin(diff)]
                bl = board_rect[np.argmax(diff)]
                src = np.array([tl, tr, br, bl], dtype="float32")

                width, height = 480, 480
                dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src, dst)
                board_img = cv2.warpPerspective(frame, M, (width, height))
                print(f"[Detection] Board detected, area={area}")
                return board_img, src

        print("[Detection] No board detected")
        return None, None

    def extract_squares(self, board_img):
        """
        Divide the board into 8x8 squares.
        Returns list of 64 square images (row-major order) or None if invalid.
        """
        if board_img is None:
            print("[Detection] No valid board image to extract squares")
            return None

        h, w = board_img.shape[:2]
        sq_h, sq_w = h // 8, w // 8
        if sq_h <= 0 or sq_w <= 0:
            print("[Detection] Invalid board dimensions for squares")
            return None

        squares = []
        for r in range(8):
            for c in range(8):
                y1, y2 = r * sq_h, (r + 1) * sq_h
                x1, x2 = c * sq_w, (c + 1) * sq_w
                square = board_img[y1:y2, x1:x2]
                if square is None or square.size == 0:
                    print(f"[Detection] Empty square at {r},{c}")
                    return None
                squares.append(square)

        if len(squares) != 64:
            print(f"[Detection] Invalid square count: {len(squares)} (expected 64)")
            return None

        return squares

    def _load_templates(self):
        templates = {}
        if not os.path.isdir(self.template_dir):
            print(f"[Detection] Template dir not found: {self.template_dir}")
            self.loaded_templates = {}
            return
        for fname in os.listdir(self.template_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(self.template_dir, fname)
            piece_name = os.path.splitext(fname)[0]  # wP, bQ, etc.
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            templates[piece_name] = gray
            print(f"[Detection] Loaded template {piece_name} size {gray.shape}")
        self.loaded_templates = templates

    def _template_match_squares(self, squares, threshold=0.6):
        """Template-matching fallback."""
        if self.loaded_templates is None:
            self._load_templates()
        board = [["" for _ in range(8)] for _ in range(8)]
        templates = self.loaded_templates or {}
        for idx, square in enumerate(squares):
            gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            best_match = None
            max_val = 0
            for piece, tmpl in templates.items():
                try:
                    tmpl_resized = cv2.resize(tmpl, (gray_square.shape[1], gray_square.shape[0]))
                except Exception:
                    continue
                res = cv2.matchTemplate(gray_square, tmpl_resized, cv2.TM_CCOEFF_NORMED)
                _, val, _, _ = cv2.minMaxLoc(res)
                if val > max_val and val > threshold:
                    max_val = val
                    best_match = piece
            row, col = idx // 8, idx % 8
            if best_match:
                board[row][col] = best_match[1] if best_match[0] == 'w' else best_match[1].lower()
            else:
                board[row][col] = ""
        return board

    def recognize_pieces(self, squares):
        """
        Detect pieces using classifier if present; otherwise fallback to template matching.
        squares: list of 64 images (8x8), row-major
        returns: 8x8 board array in FEN letters
        """
        if squares is None or len(squares) != 64:
            print("[Detection] Invalid squares for recognition.")
            return None

        # If classifier present, use it
        if self.classifier is not None:
            try:
                labels = self.classifier.predict_batch(squares)
                board = [["" for _ in range(8)] for _ in range(8)]
                for idx, lab in enumerate(labels):
                    row, col = idx // 8, idx % 8
                    if lab is None or lab == "empty":
                        board[row][col] = ""
                    else:
                        if lab[0] == 'w':
                            board[row][col] = lab[1]  # uppercase
                        else:
                            board[row][col] = lab[1].lower()
                return board
            except Exception as e:
                print(f"[Detection] Classifier error, falling back to templates: {e}")

        # Fallback
        return self._template_match_squares(squares)

    def is_valid_board(self, board_state):
        """Check if board_state is valid (8x8 array)."""
        if board_state is None:
            return False
        return len(board_state) == 8 and all(len(row) == 8 for row in board_state)

    def is_stable(self, fen):
        """Compare with previous FEN to determine stability."""
        if fen == self.prev_fen:
            self.stable_count += 1
        else:
            self.stable_count = 0
        self.prev_fen = fen
        return self.stable_count >= self.stable_threshold
