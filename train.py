# train_stockfish_teacher.py
import os
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from chess_engine import DQN, encode_board, move_to_index
from stockfish import Stockfish
import numpy as np
import random

# ---------------------------
# Hyperparameters
# ---------------------------
EPISODES = 1000
MAX_MOVES = 20          # limit moves per game
INPUT_SIZE = 768
OUTPUT_SIZE = 4672
LR = 0.001
EPSILON = 0.1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
SAVE_INTERVAL = 50

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
TRAINED_WEIGHTS = os.path.join(WEIGHTS_DIR, "dqn_trained_teacher.pt")
STOCKFISH_PATH = r"C:\Users\User\Desktop\chess_ai\DQNChess\stockfish\stockfish-windows-x86-64-avx2.exe"

# ---------------------------
# Initialize DQN & Stockfish
# ---------------------------
dqn = DQN(INPUT_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(dqn.parameters(), lr=LR)
loss_fn = nn.MSELoss()

if os.path.exists(TRAINED_WEIGHTS):
    dqn.load_state_dict(torch.load(TRAINED_WEIGHTS))
    print("Loaded existing weights.")
else:
    print("Starting fresh.")

stockfish = Stockfish(path=STOCKFISH_PATH, depth=15)

# ---------------------------
# AI Teacher function
# ---------------------------
def teacher_guided_move(board, stockfish, epsilon=0.1):
    """
    Returns a move and a reward.
    - Stockfish picks candidate moves
    - Teacher evaluates moves (simulated here as simple heuristics)
    """
    # Stockfish top move
    stockfish.set_fen_position(board.fen())
    best_move_str = stockfish.get_best_move()
    
    legal_moves = list(board.legal_moves)
    
    # If Stockfish fails, pick random
    if best_move_str:
        sf_move = chess.Move.from_uci(best_move_str)
    else:
        sf_move = random.choice(legal_moves)

    # Epsilon-greedy: sometimes pick random move
    if random.random() < epsilon:
        chosen_move = random.choice(legal_moves)
    else:
        chosen_move = sf_move

    # Teacher reward function (simple example)
    reward = 0.0
    # reward if move captures piece
    if board.is_capture(chosen_move):
        reward += 0.1
    # reward if move matches Stockfish top move
    if chosen_move == sf_move:
        reward += 0.5
    # small penalty if move leaves a piece attacked (simple heuristic)
    if any(board.is_attacked_by(not board.turn, sq) for sq in [chosen_move.to_square]):
        reward -= 0.05

    return chosen_move, reward

# ---------------------------
# Training loop
# ---------------------------
for episode in range(1, EPISODES + 1):
    board = chess.Board()
    states, targets = [], []
    moves_played = 0
    loss = None

    while not board.is_game_over() and moves_played < MAX_MOVES:
        # Encode board
        state = encode_board(board)
        states.append(state)

        # Teacher-guided move
        move, reward = teacher_guided_move(board, stockfish, EPSILON)

        # Convert move to index safely
        move_idx = move_to_index(move)
        if move_idx >= OUTPUT_SIZE:
            board.push(random.choice(list(board.legal_moves)))
            moves_played += 1
            continue

        # Target vector (teacher guidance)
        target = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        target[move_idx] = reward
        targets.append(target)

        board.push(move)
        moves_played += 1

    # Train DQN after the episode
    if len(states) > 0 and len(states) == len(targets):
        states_tensor = torch.from_numpy(np.array(states, dtype=np.float32))
        targets_tensor = torch.from_numpy(np.array(targets, dtype=np.float32))

        q_values = dqn(states_tensor)
        loss = loss_fn(q_values, targets_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Decay epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    # Save periodically
    if episode % SAVE_INTERVAL == 0:
        torch.save(dqn.state_dict(), os.path.join(WEIGHTS_DIR, f"dqn_teacher_ep{episode}.pt"))

    # Print progress
    if loss is not None:
        print(f"Episode {episode}/{EPISODES} finished. Loss: {loss.item():.4f}  Epsilon: {EPSILON:.3f}")
    else:
        print(f"Episode {episode}/{EPISODES} finished. Loss: N/A  Epsilon: {EPSILON:.3f}")

# Save final weights
torch.save(dqn.state_dict(), TRAINED_WEIGHTS)
print("Training complete. Final weights saved!")
