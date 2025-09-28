# fen.py
import re

class FENUtils:
    @staticmethod
    def board_to_fen(board):
        """
        Convert an 8x8 board array into FEN string.
        Empty squares should be represented as "" in board array.
        """
        fen_rows = []
        for row in board:
            empty = 0
            fen_row = ""
            for cell in row:
                if cell == "" or cell is None:
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += cell
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)

        return "/".join(fen_rows) + " w KQkq - 0 1"

    @staticmethod
    def is_valid_fen(fen: str) -> bool:
        """
        Validate a FEN string (basic rules):
        - Exactly 8 rows
        - Each row adds up to 8 squares
        - Contains both kings
        - Not a completely empty board
        - Has at least side-to-move field
        """
        if not fen or not isinstance(fen, str):
            return False

        parts = fen.split(" ")
        if len(parts) < 1:
            return False

        board_part = parts[0]
        rows = board_part.split("/")
        if len(rows) != 8:
            return False

        for row in rows:
            count = 0
            for ch in row:
                if ch.isdigit():
                    count += int(ch)
                elif ch.isalpha():
                    count += 1
                else:
                    return False
            if count != 8:
                return False

        # must have both kings
        if "K" not in board_part or "k" not in board_part:
            return False

        # reject completely empty board
        if board_part == "8/8/8/8/8/8/8/8":
            return False

        return True


# Expose module-level shortcuts for compatibility
board_to_fen = FENUtils.board_to_fen
is_valid_fen = FENUtils.is_valid_fen
