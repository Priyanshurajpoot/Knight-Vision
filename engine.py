# core/engine.py
import chess
import chess.engine

class Engine:
    def __init__(self, path=None):
        """
        Engine class to integrate Stockfish.
        path: path to the Stockfish executable
        """
        self.engine = None
        self.path = path
        if self.path:
            self.init_engine(self.path)

    def init_engine(self, path=None):
        """
        Initialize the Stockfish engine with the given path.
        """
        if path:
            self.path = path
        if self.path is None:
            raise ValueError("Stockfish path not provided")
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        except Exception as e:
            raise RuntimeError(f"Failed to start Stockfish: {e}")

    def analyze_fen(self, fen, depth=15):
        """
        Analyze a FEN string using Stockfish.
        Returns: best_move, score
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        board = chess.Board(fen)
        info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
        best_move = info.get("pv")[0] if info.get("pv") else None
        score = info.get("score")
        return best_move, score

    def close(self):
        """
        Close the engine safely.
        """
        if self.engine:
            self.engine.quit()
            self.engine = None
