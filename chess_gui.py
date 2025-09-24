# chess_gui.py
"""
This script implements a graphical user interface (GUI) for playing chess against
the DQN-based AI. It uses the Pygame library to render the chessboard and pieces
and handles user input for making moves. The AI's moves are determined by the
trained DQN model loaded from `chess_engine.py`.
"""
import pygame
import chess
import torch
import os
from chess_engine import DQN, encode_board, move_to_index, load_dqn

# ---------------------------
# GUI Constants
# ---------------------------
WIDTH, HEIGHT = 480, 480
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
WHITE_COLOR = (245, 245, 220)  # Beige for light squares
GRAY_COLOR = (119, 136, 153)   # Slate gray for dark squares
HIGHLIGHT = (186, 202, 68)     # Yellowish green for selected square

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "dqn_trained_stockfish.pt")
IMAGES_PATH = os.path.join(BASE_DIR, "images")

# ---------------------------
# Load piece images
# ---------------------------
piece_images = {}
pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']
for piece in pieces:
    path = os.path.join(IMAGES_PATH, f'{piece}.png')
    if os.path.exists(path):
        img = pygame.image.load(path)
        piece_images[piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
    else:
        print(f"Warning: Image for piece '{piece}' not found at '{path}'!")

# ---------------------------
# Draw board and pieces
# ---------------------------
def draw_board(win: pygame.Surface, selected_square: tuple = None):
    """
    Draws the chessboard squares on the Pygame window.

    Args:
        win (pygame.Surface): The Pygame window surface to draw on.
        selected_square (tuple, optional): A tuple (row, col) representing the
                                           currently selected square, which will be
                                           highlighted. Defaults to None.
    """
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE_COLOR if (row + col) % 2 == 0 else GRAY_COLOR
            pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            if selected_square == (row, col):
                pygame.draw.rect(win, HIGHLIGHT, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3) # Draw border

def draw_pieces(win: pygame.Surface, board: chess.Board):
    """
    Draws the chess pieces on the board.

    It iterates through all squares on the board and draws the corresponding
    piece image if a piece is present.

    Args:
        win (pygame.Surface): The Pygame window surface to draw on.
        board (chess.Board): The current board state from the `python-chess` library.
    """
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            key = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().lower()}"
            if key in piece_images:
                win.blit(piece_images[key], (col * SQUARE_SIZE, row * SQUARE_SIZE))

# ---------------------------
# Main GUI loop
# ---------------------------
def main():
    """
    The main function to run the chess GUI.

    Initializes Pygame, loads the DQN model, and enters the main game loop.
    It handles user input for the player's moves and calls the DQN model to
    generate moves for the AI opponent. The game state is updated and rendered
    continuously.
    """
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Chess")
    clock = pygame.time.Clock()

    board = chess.Board()
    selected_square = None
    player_color = chess.WHITE  # Human plays as White
    dqn = load_dqn(WEIGHTS_PATH)

    run = True
    while run:
        clock.tick(60)

        # Drawing
        draw_board(win, selected_square)
        draw_pieces(win, board)
        pygame.display.update()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Player move
            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == player_color:
                x, y = pygame.mouse.get_pos()
                col, row = x // SQUARE_SIZE, y // SQUARE_SIZE
                sq_index = chess.square(col, 7 - row)
                clicked_square = (row, col)

                if selected_square is None:
                    # First click: select a piece
                    piece = board.piece_at(sq_index)
                    if piece and piece.color == player_color:
                        selected_square = clicked_square
                else:
                    # Second click: attempt to make a move
                    from_sq = chess.square(selected_square[1], 7 - selected_square[0])
                    to_sq = sq_index
                    
                    # Handle promotion automatically to Queen for simplicity
                    move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN if chess.square_rank(to_sq) in [0, 7] and board.piece_at(from_sq).piece_type == chess.PAWN else None)
                    
                    if move in board.legal_moves:
                        board.push(move)
                    selected_square = None

        # AI's turn
        if board.turn != player_color and not board.is_game_over():
            with torch.no_grad():
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    state_tensor = torch.FloatTensor(encode_board(board)).unsqueeze(0)
                    q_values = dqn(state_tensor)
                    
                    # Find the best legal move according to the DQN's Q-values
                    best_move = max(legal_moves, key=lambda m: q_values[0, move_to_index(m)].item())
                    board.push(best_move)

        # Check for game over
        if board.is_game_over():
            result = board.result()
            print(f"Game over: {result}")
            # Optionally, you can display the result on the screen
            # and wait for a restart command.
            run = False # End the game for now

    pygame.quit()

if __name__ == "__main__":
    main()
