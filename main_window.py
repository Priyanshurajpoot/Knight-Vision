# ui/main_window.py
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTabWidget, QTextEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from core.detection import BoardDetector
from core.fen import FENUtils
from core.engine import Engine
from core.classifier import PieceClassifier
from core.utils import check_board_quality   # NEW


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knight Vision")
        self.setGeometry(100, 100, 900, 700)

        self.engine = None
        self.engine_path = None
        self.board_detector = BoardDetector()
        self.current_frame = None
        self.current_board_img = None
        self.current_fen = ""
        self.recognized_board_state = None

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Live Tab ---
        self.live_tab = QWidget()
        self.tabs.addTab(self.live_tab, "Live")
        live_layout = QVBoxLayout()
        self.live_tab.setLayout(live_layout)

        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setFixedSize(560, 560)
        self.camera_label.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(self.camera_label)

        self.start_camera_btn = QPushButton("Start Camera & Capture Board")
        self.start_camera_btn.clicked.connect(self.start_camera)
        live_layout.addWidget(self.start_camera_btn)

        # --- Image Tab ---
        self.image_tab = QWidget()
        self.tabs.addTab(self.image_tab, "Image")
        image_layout = QVBoxLayout()
        self.image_tab.setLayout(image_layout)

        self.image_label = QLabel("Captured Board")
        self.image_label.setFixedSize(560, 560)
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()
        self.start_recognition_btn = QPushButton("Start Recognition")
        self.start_recognition_btn.clicked.connect(self.start_recognition)
        btn_layout.addWidget(self.start_recognition_btn)

        self.rerecognize_btn = QPushButton("Re-recognize Pieces")
        self.rerecognize_btn.clicked.connect(self.start_recognition)
        self.rerecognize_btn.setEnabled(False)
        btn_layout.addWidget(self.rerecognize_btn)

        self.confirm_fen_btn = QPushButton("Confirm & Generate FEN")
        self.confirm_fen_btn.clicked.connect(self.confirm_fen)
        self.confirm_fen_btn.setEnabled(False)
        btn_layout.addWidget(self.confirm_fen_btn)

        image_layout.addLayout(btn_layout)

        self.image_fen_text = QTextEdit()
        self.image_fen_text.setReadOnly(True)
        image_layout.addWidget(self.image_fen_text)

        # --- Analysis Tab ---
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Analysis")
        analysis_layout = QVBoxLayout()
        self.analysis_tab.setLayout(analysis_layout)

        self.engine_status_label = QLabel("Engine not initialized")
        analysis_layout.addWidget(self.engine_status_label)

        self.set_engine_btn = QPushButton("Set Stockfish Path & Initialize Engine")
        self.set_engine_btn.clicked.connect(self.set_engine_path)
        analysis_layout.addWidget(self.set_engine_btn)

        self.set_model_btn = QPushButton("Set Model Path & Load Model")
        self.set_model_btn.clicked.connect(self.set_model_path)
        analysis_layout.addWidget(self.set_model_btn)

        self.start_analysis_btn = QPushButton("Start Analysis")
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        analysis_layout.addWidget(self.start_analysis_btn)

        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_result_text)

    # --- Camera Capture ---
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame.copy()
        board_img, rect = self.board_detector.detect_board(frame)

        if board_img is not None:
            ok, msg = check_board_quality(board_img)
            if ok:
                # ✅ Good capture → move to Image Tab
                self.current_board_img = board_img.copy()
                rgb = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)
                qt_image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(qt_image))

                self.tabs.setCurrentWidget(self.image_tab)
                self.timer.stop()
                self.cap.release()
                print("[MainWindow] Board captured and camera stopped.")
            else:
                # ❌ Bad capture → show feedback in Live Tab
                self.camera_label.setText(f"⚠ {msg}")
        else:
            self.camera_label.setText("No board detected.")

        # Show live feed in background
        if board_img is None:  # only show live if not locked
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, w*ch, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    # --- Recognition ---
    def start_recognition(self):
        if self.current_board_img is None:
            self.image_fen_text.setPlainText("No captured board available.")
            return

        squares = self.board_detector.extract_squares(self.current_board_img)
        if squares:
            board_state = self.board_detector.recognize_pieces(squares)
            self.recognized_board_state = board_state

            # Draw labels
            annotated = self.current_board_img.copy()
            square_size = annotated.shape[0] // 8
            for r in range(8):
                for c in range(8):
                    piece = board_state[r][c]
                    if piece != "":
                        x = c * square_size + 10
                        y = (r + 1) * square_size - 10
                        cv2.putText(
                            annotated, piece, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2, cv2.LINE_AA
                        )
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))

            self.rerecognize_btn.setEnabled(True)
            self.confirm_fen_btn.setEnabled(True)
            self.image_fen_text.setPlainText("Recognition complete. Please confirm.")
        else:
            self.image_fen_text.setPlainText("Failed to extract squares.")

    def confirm_fen(self):
        if self.recognized_board_state is None:
            self.image_fen_text.setPlainText("No recognition result to confirm.")
            return
        fen = FENUtils.board_to_fen(self.recognized_board_state)
        if not FENUtils.is_valid_fen(fen):
            self.image_fen_text.setPlainText("Invalid recognition result. Try re-recognizing.")
            return
        self.current_fen = fen
        self.image_fen_text.setPlainText(f"Final FEN:\n{fen}")

    # --- Engine ---
    def set_engine_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Stockfish Executable')
        if fname:
            self.engine_path = fname
            try:
                self.engine = Engine(self.engine_path)
                self.engine_status_label.setText(f"Engine Ready: {os.path.basename(self.engine_path)}")
            except Exception as e:
                self.engine_status_label.setText(f"Engine Error: {e}")

    def set_model_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Piece Classifier (.pt/.pth)')
        if fname:
            try:
                classifier = PieceClassifier(fname)
                self.board_detector.classifier = classifier
                self.analysis_result_text.setPlainText(f"Model loaded: {os.path.basename(fname)}")
            except Exception as e:
                self.analysis_result_text.setPlainText(f"Failed to load model: {e}")

    def start_analysis(self):
        if not self.engine:
            self.analysis_result_text.setPlainText("Engine not initialized.")
            return
        if not FENUtils.is_valid_fen(self.current_fen):
            self.analysis_result_text.setPlainText("Invalid or no FEN available.")
            return
        try:
            best_move, score = self.engine.analyze_fen(self.current_fen)
            self.analysis_result_text.setPlainText(f"Best move: {best_move}\nScore: {score}")
        except Exception as e:
            self.analysis_result_text.setPlainText(f"Analysis failed: {e}")

    def closeEvent(self, event):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if self.engine:
            self.engine.close()
        event.accept()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
