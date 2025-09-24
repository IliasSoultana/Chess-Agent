# train.py
import random
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os

from chess_engine import DQN, encode_board, move_to_index

# ---------------------------
# Hyperparameters
# ---------------------------
EPISODES = 1000  # adjust for testing
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
GAMMA = 0.99
BATCH_SIZE = 64
LR = 0.001
MAX_MOVES = 30
SAVE_INTERVAL = 50

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
TRAINED_WEIGHTS = os.path.join(WEIGHTS_DIR, "dqn_trained.pt")

# ---------------------------
# Replay buffer & network
# ---------------------------
replay_buffer = deque(maxlen=10000)
OUTPUT_SIZE = 4672
input_size = 768
dqn = DQN(input_size, OUTPUT_SIZE)

if os.path.exists(TRAINED_WEIGHTS):
    dqn.load_state_dict(torch.load(TRAINED_WEIGHTS))
    print("Loaded existing weights.")
else:
    print("Starting fresh.")

optimizer = optim.Adam(dqn.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ---------------------------
# Select move
# ---------------------------
def select_move(board, epsilon):
    legal_moves = list(board.legal_moves)
    if random.random() < epsilon:
        return random.choice(legal_moves)
    else:
        state = torch.FloatTensor(encode_board(board)).unsqueeze(0)
        q_values = dqn(state)
        return max(legal_moves, key=lambda m: q_values[0, move_to_index(m)].item())

# ---------------------------
# Training step
# ---------------------------
def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(np.array(states))
    next_states = torch.FloatTensor(np.array(next_states))
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    q_values = dqn(states)
    next_q_values = dqn(next_states)
    q_targets = q_values.clone()

    for i, action in enumerate(actions):
        idx = move_to_index(action)
        q_targets[i, idx] = rewards[i] + (1 - dones[i]) * GAMMA * torch.max(next_q_values[i])

    loss = loss_fn(q_values, q_targets.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------------------------
# Main training loop
# ---------------------------
for episode in range(1, EPISODES + 1):
    board = chess.Board()
    moves_played = 0

    while not board.is_game_over() and moves_played < MAX_MOVES:
        state = encode_board(board)
        move = select_move(board, EPSILON)
        board.push(move)

        reward = 0.0
        if board.is_capture(move):
            reward += 0.1
        if board.is_checkmate():
            reward = 1.0 if board.turn == chess.BLACK else -1.0
        if board.is_attacked_by(not board.turn, move.to_square):
            reward -= 0.1

        done = board.is_game_over() or moves_played >= MAX_MOVES
        next_state = encode_board(board)
        replay_buffer.append((state, move, reward, next_state, done))
        train_step()
        moves_played += 1

    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    if episode % SAVE_INTERVAL == 0:
        torch.save(dqn.state_dict(), os.path.join(WEIGHTS_DIR, f"dqn_episode_{episode}.pt"))
    print(f"Episode {episode}/{EPISODES} finished. Epsilon: {EPSILON:.3f}")

torch.save(dqn.state_dict(), TRAINED_WEIGHTS)
print("Trained weights saved!")
