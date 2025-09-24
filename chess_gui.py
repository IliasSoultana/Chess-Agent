# chess_gui.py
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
WHITE_COLOR = (245, 245, 220)
GRAY_COLOR = (119, 136, 153)
HIGHLIGHT = (186, 202, 68)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "dqn_trained.pt")
IMAGES_PATH = os.path.join(BASE_DIR, "images")

# ---------------------------
# Load piece images
# ---------------------------
piece_images = {}
pieces = ['wp','wn','wb','wr','wq','wk','bp','bn','bb','br','bq','bk']
for piece in pieces:
    path = os.path.join(IMAGES_PATH, f'{piece}.png')
    if os.path.exists(path):
        img = pygame.image.load(path)
        piece_images[piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
    else:
        print(f"Warning: image '{path}' not found!")

# ---------------------------
# Draw board and pieces
# ---------------------------
def draw_board(win, selected_square=None):
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE_COLOR if (row + col) % 2 == 0 else GRAY_COLOR
            pygame.draw.rect(win, color, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            if selected_square == (row, col):
                pygame.draw.rect(win, HIGHLIGHT, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(win, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            key = f"{'w' if piece.color else 'b'}{piece.symbol().lower()}"
            if key in piece_images:
                win.blit(piece_images[key], (col*SQUARE_SIZE, row*SQUARE_SIZE))

# ---------------------------
# Main GUI loop
# ---------------------------
def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Play DQN Chess")
    clock = pygame.time.Clock()

    board = chess.Board()
    selected_square = None
    player_color = chess.WHITE

    # Load trained DQN
    print("Weights file exists:", os.path.exists(WEIGHTS_PATH))
    dqn = load_dqn(WEIGHTS_PATH)

    run = True
    while run:
        clock.tick(60)
        draw_board(win, selected_square)
        draw_pieces(win, board)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == player_color:
                x, y = pygame.mouse.get_pos()
                col, row = x // SQUARE_SIZE, y // SQUARE_SIZE
                sq_index = chess.square(col, 7 - row)
                clicked_square = (row, col)

                if selected_square is None:
                    if board.piece_at(sq_index) and board.piece_at(sq_index).color == player_color:
                        selected_square = clicked_square
                else:
                    from_sq = chess.square(selected_square[1], 7 - selected_square[0])
                    to_sq = sq_index
                    piece = board.piece_at(from_sq)
                    if piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
                        move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                    else:
                        move = chess.Move(from_sq, to_sq)
                    if move in board.legal_moves:
                        board.push(move)
                    selected_square = None

        # Engine move
        if board.turn != player_color and not board.is_game_over():
            legal_moves = list(board.legal_moves)
            state_tensor = torch.FloatTensor(encode_board(board)).unsqueeze(0)
            q_values = dqn(state_tensor)
            best_move = max(legal_moves, key=lambda m: q_values[0, move_to_index(m)].item())
            piece = board.piece_at(best_move.from_square)
            if piece.piece_type == chess.PAWN and chess.square_rank(best_move.to_square) in [0, 7]:
                best_move = chess.Move(best_move.from_square, best_move.to_square, promotion=chess.QUEEN)
            board.push(best_move)

    pygame.quit()

if __name__ == "__main__":
    main()
