# train.py
"""
This script trains the DQN chess agent using a "teacher-student" learning approach.
Stockfish, a powerful traditional chess engine, acts as the "teacher," providing
guidance on good moves. The DQN model is the "student," learning to imitate the
teacher's choices.

The training process is structured as follows:

1.  For each game (episode), the agent plays a series of moves.
2.  In each position, the `teacher_guided_move` function is called. It uses
    Stockfish's best move as a guide but incorporates an epsilon-greedy strategy
    to encourage exploration.
3.  The teacher also provides a simple reward for the chosen move.
4.  The state (board position) and a target vector are stored. The target vector
    is sparse, with the reward placed at the index of the move chosen by the teacher.
5.  After the episode concludes, the collected states and targets are used to
    train the DQN in a supervised manner. The model learns to output high values
    for the moves the teacher would have made.
6.  The model's weights are saved periodically and at the end of training.
"""
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
EPISODES = 1000              # Total number of games to play for training
MAX_MOVES = 20               # Maximum number of moves per game to keep episodes short
INPUT_SIZE = 768             # Size of the encoded board state vector (12x8x8)
OUTPUT_SIZE = 4672           # Number of possible moves (action space size)
LR = 0.001                   # Learning rate for the Adam optimizer
EPSILON_START = 1.0          # Starting value for epsilon (exploration rate)
EPSILON_DECAY = 0.995        # Multiplicative factor for decaying epsilon
EPSILON_MIN = 0.05           # Minimum value for epsilon
SAVE_INTERVAL = 50           # Save model weights every N episodes

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
TRAINED_WEIGHTS = os.path.join(WEIGHTS_DIR, "dqn_trained_teacher.pt")
# IMPORTANT: Update this path to your Stockfish executable
STOCKFISH_PATH = r"C:\Users\User\Desktop\chess_ai\DQNChess\stockfish\stockfish-windows-x86-64-avx2.exe"

def main():
    """
    The main function to run the training loop.

    It initializes the DQN model, optimizer, and Stockfish. Then, it enters the
    main training loop, playing games, collecting data, and training the model.
    """
    # ---------------------------
    # AI Teacher function
    # ---------------------------
    def teacher_guided_move(board: chess.Board, stockfish: Stockfish, epsilon: float) -> tuple[chess.Move, float]:
        """
        Selects a move using guidance from Stockfish (the "teacher").

        This function implements an epsilon-greedy strategy:
        - With probability (1 - epsilon), it chooses the best move suggested by Stockfish.
        - With probability epsilon, it chooses a random legal move to encourage exploration.

        It also calculates a simple reward based on the chosen move.

        Args:
            board (chess.Board): The current board state.
            stockfish (Stockfish): The Stockfish engine instance.
            epsilon (float): The current exploration rate.

        Returns:
            tuple[chess.Move, float]: A tuple containing the chosen move and the calculated reward.
        """
        legal_moves = list(board.legal_moves)
        
        # Get Stockfish's best move
        stockfish.set_fen_position(board.fen())
        best_move_str = stockfish.get_best_move()
        
        # If Stockfish provides a move, use it; otherwise, fall back to a random move.
        sf_move = chess.Move.from_uci(best_move_str) if best_move_str and chess.Move.from_uci(best_move_str) in legal_moves else random.choice(legal_moves)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            chosen_move = random.choice(legal_moves)
        else:
            chosen_move = sf_move

        # Simple reward function (can be made more sophisticated)
        reward = 0.0
        if board.is_capture(chosen_move):
            reward += 0.1  # Small reward for any capture
        if chosen_move == sf_move:
            reward += 0.5  # Larger reward for matching the teacher's move
        
        # Create a temporary board to check the consequences of the move
        temp_board = board.copy()
        temp_board.push(chosen_move)
        if temp_board.is_checkmate():
            reward += 1.0 # High reward for checkmate
        elif temp_board.is_check():
            reward += 0.2 # Reward for putting opponent in check

        return chosen_move, reward

    dqn = DQN(INPUT_SIZE, OUTPUT_SIZE)
    optimizer = optim.Adam(dqn.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    if os.path.exists(TRAINED_WEIGHTS):
        dqn.load_state_dict(torch.load(TRAINED_WEIGHTS))
        print(f"Loaded existing weights from {TRAINED_WEIGHTS}.")
    else:
        print("No existing weights found. Starting training from scratch.")

    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=15)
    except Exception as e:
        print(f"Error initializing Stockfish: {e}")
        print("Please ensure the STOCKFISH_PATH is correct and the executable is working.")
        return

    epsilon = EPSILON_START
    for episode in range(1, EPISODES + 1):
        board = chess.Board()
        states, targets = [], []
        moves_played = 0
        total_reward = 0

        # --- Play one full game (episode) ---
        while not board.is_game_over() and moves_played < MAX_MOVES:
            state = encode_board(board)
            
            # Get a move and reward from the teacher
            move, reward = teacher_guided_move(board, stockfish, epsilon)
            total_reward += reward
            
            # Prepare the training data for this step
            states.append(state)
            target = np.zeros(OUTPUT_SIZE, dtype=np.float32)
            target[move_to_index(move)] = reward  # The "correct" output is the reward at the chosen move's index
            targets.append(target)

            board.push(move)
            moves_played += 1

        # --- Train the DQN with data from the completed episode ---
        if len(states) > 0:
            states_tensor = torch.from_numpy(np.array(states, dtype=np.float32))
            targets_tensor = torch.from_numpy(np.array(targets, dtype=np.float32))

            # Forward pass: get Q-values from the DQN
            q_values = dqn(states_tensor)
            
            # Calculate loss between the DQN's output and the teacher's targets
            loss = loss_fn(q_values, targets_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Episode {episode}/{EPISODES} | Moves: {moves_played} | Loss: {loss.item():.4f} | Epsilon: {epsilon:.3f} | Total Reward: {total_reward:.2f}")
        else:
            print(f"Episode {episode}/{EPISODES} | No moves made.")

        # Decay epsilon for the next episode
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Save model weights periodically
        if episode % SAVE_INTERVAL == 0:
            save_path = os.path.join(WEIGHTS_DIR, f"dqn_teacher_ep{episode}.pt")
            torch.save(dqn.state_dict(), save_path)
            print(f"--- Saved weights to {save_path} ---")

    # --- Final save ---
    torch.save(dqn.state_dict(), TRAINED_WEIGHTS)
    print(f"\nTraining complete. Final weights saved to {TRAINED_WEIGHTS}!")

if __name__ == "__main__":
    main()
