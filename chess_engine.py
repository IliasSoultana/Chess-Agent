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
    """
    Encodes a chess board state into a numerical representation for the neural network.

    The board is represented as a set of 12 binary "planes," each of size 8x8.
    Each plane corresponds to a specific piece type and color.

    - Planes 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    - Planes 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)

    A '1' in a plane at a specific position indicates the presence of that piece
    at that square. The final array is flattened to a 1D vector of size 768 (12*8*8).

    Args:
        board (chess.Board): The current board state from the `python-chess` library.

    Returns:
        np.ndarray: A flattened 1D numpy array of size 768 representing the board state.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    piece_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            offset = 0 if piece.color == chess.WHITE else 6
            plane = piece_to_plane[piece.piece_type] + offset
            row = 7 - (square // 8)  # Flip board for standard orientation
            col = square % 8
            planes[plane, row, col] = 1.0
    return planes.flatten()

# ---------------------------
# Move Encoding Helpers
# ---------------------------
def move_to_index(move: chess.Move) -> int:
    """
    Encodes a `chess.Move` object into a unique integer index.

    This is used to map the DQN's output neurons to specific chess moves.
    The encoding scheme is as follows:

    - Standard moves: ``from_square * 64 + to_square``. This maps to indices 0-4095.
    - Promotions: Special indices are used to denote promotions to Queen, Rook,
      Bishop, or Knight. These are mapped to indices 4096 and above.

    Args:
        move (chess.Move): The move to encode.

    Returns:
        int: A unique integer index representing the move.
    """
    idx = move.from_square * 64 + move.to_square
    if move.promotion:
        promotion_offset = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
        # 4096 is the base index for promotion moves
        idx += 4096 + promotion_offset[move.promotion]
    return idx

def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """
    Decodes a unique integer index back into a `chess.Move` object.

    This is the inverse of `move_to_index`. It reconstructs the move from the
    integer, handling both standard moves and promotions.

    Args:
        idx (int): The integer index to decode.
        board (chess.Board): The current board state, needed to validate the move.

    Returns:
        chess.Move: The decoded chess move.
    """
    if idx < 4096:
        from_sq = idx // 64
        to_sq = idx % 64
        return chess.Move(from_sq, to_sq)
    else:
        # Handle promotions
        promo_base_idx = idx - 4096
        # This part of the logic needs to be carefully mapped from move_to_index
        # Assuming a simple mapping for now.
        # This logic might need refinement based on the exact output space.
        from_sq = (promo_base_idx % 4096) // 64
        to_sq = (promo_base_idx % 4096) % 64
        promo_type = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promo_base_idx // 4096]
        return chess.Move(from_sq, to_sq, promotion=promo_type)

# ---------------------------
# DQN Network
# ---------------------------
class DQN(nn.Module):
    """
    A simple Deep Q-Network (DQN) for playing chess.

    This network takes a flattened board representation as input and outputs
    Q-values for each possible move. It consists of three fully connected layers.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The output layer.
    """
    def __init__(self, input_size=768, output_size=4672):
        """
        Initializes the DQN model.

        Args:
            input_size (int): The size of the input vector (12*8*8 = 768).
            output_size (int): The number of possible moves to output Q-values for.
                               This includes standard moves and promotions.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor representing the board state.

        Returns:
            torch.Tensor: The output tensor of Q-values for each move.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------
# Load DQN Weights
# ---------------------------
def load_dqn(weights_file: str, input_size=768, output_size=4672) -> DQN:
    """
    Loads a DQN model from a saved weights file.

    If the specified file exists, it loads the model's state dictionary.
    If the file does not exist, it returns a new DQN model with random weights.

    Args:
        weights_file (str): The path to the .pth file containing the model weights.
        input_size (int): The input size of the DQN model.
        output_size (int): The output size of the DQN model.

    Returns:
        DQN: The loaded or newly initialized DQN model.
    """
    dqn = DQN(input_size, output_size)
    if os.path.exists(weights_file):
        dqn.load_state_dict(torch.load(weights_file))
        dqn.eval()  # Set the model to evaluation mode
        print(f"Loaded trained DQN weights from {weights_file}!")
    else:
        print("No trained weights found, using a new DQN model with random weights.")
    return dqn
