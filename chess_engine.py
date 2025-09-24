# chess_engine.py
import chess
import torch
import torch.nn as nn
import numpy as np
import os

# ---------------------------
# Board Encoding
# ---------------------------
def encode_board(board: chess.Board) -> np.ndarray:
    """Encode chess board into 12x8x8 binary planes, flattened."""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    piece_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            offset = 0 if piece.color else 6
            plane = piece_to_plane[piece.piece_type] + offset
            row = 7 - (square // 8)
            col = square % 8
            planes[plane, row, col] = 1.0
    return planes.flatten()

# ---------------------------
# Move Encoding Helpers
# ---------------------------
def move_to_index(move: chess.Move) -> int:
    """Encode a chess.Move to an integer index for DQN output."""
    idx = move.from_square * 64 + move.to_square
    if move.promotion:
        promotion_offset = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
        idx += 4096 + promotion_offset[move.promotion]
    return idx

def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Decode an integer index back to a chess.Move."""
    if idx < 4096:
        return chess.Move(idx // 64, idx % 64)
    else:
        promo_idx = idx - 4096
        from_sq = promo_idx // 64
        to_sq = promo_idx % 64
        promotion_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promo_idx // 64]
        return chess.Move(from_sq, to_sq, promotion=promotion_piece)

# ---------------------------
# DQN Network
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_size=768, output_size=4672):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------
# Load DQN Weights
# ---------------------------
def load_dqn(weights_file: str, input_size=768, output_size=4672) -> DQN:
    """Load a DQN model with weights. If not found, returns random DQN."""
    dqn = DQN(input_size, output_size)
    if os.path.exists(weights_file):
        dqn.load_state_dict(torch.load(weights_file))
        dqn.eval()
        print(f"Loaded trained DQN weights from {weights_file}!")
    else:
        print("No trained weights found, using random DQN.")
    return dqn
