import os
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from chess_engine import DQN, encode_board, move_to_index
from stockfish import Stockfish
import numpy as np
import random


EPISODES = 1000             
MAX_MOVES = 20               
INPUT_SIZE = 768             
OUTPUT_SIZE = 4672         
LR = 0.001                   
EPSILON_START = 1.0        
EPSILON_DECAY = 0.995       
EPSILON_MIN = 0.05           
SAVE_INTERVAL = 50          


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
TRAINED_WEIGHTS = os.path.join(WEIGHTS_DIR, "dqn_trained_teacher.pt")

STOCKFISH_PATH = r"C:\Users\User\Desktop\chess_ai\DQNChess\stockfish\stockfish-windows-x86-64-avx2.exe"

def main():
 

    def teacher_guided_move(board: chess.Board, stockfish: Stockfish, epsilon: float) -> tuple[chess.Move, float]:
       
        legal_moves = list(board.legal_moves)
        
       
        stockfish.set_fen_position(board.fen())
        best_move_str = stockfish.get_best_move()
        
       
        sf_move = chess.Move.from_uci(best_move_str) if best_move_str and chess.Move.from_uci(best_move_str) in legal_moves else random.choice(legal_moves)

        
        if random.random() < epsilon:
            chosen_move = random.choice(legal_moves)
        else:
            chosen_move = sf_move

        reward = 0.0
        if board.is_capture(chosen_move):
            reward += 0.1  
        if chosen_move == sf_move:
            reward += 0.5  
        
        temp_board = board.copy()
        temp_board.push(chosen_move)
        if temp_board.is_checkmate():
            reward += 1.0 
        elif temp_board.is_check():
            reward += 0.2 

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

        
        while not board.is_game_over() and moves_played < MAX_MOVES:
            state = encode_board(board)
            
            
            move, reward = teacher_guided_move(board, stockfish, epsilon)
            total_reward += reward
            
           
            states.append(state)
            target = np.zeros(OUTPUT_SIZE, dtype=np.float32)
            target[move_to_index(move)] = reward  
            targets.append(target)

            board.push(move)
            moves_played += 1

       
        if len(states) > 0:
            states_tensor = torch.from_numpy(np.array(states, dtype=np.float32))
            targets_tensor = torch.from_numpy(np.array(targets, dtype=np.float32))

        
            q_values = dqn(states_tensor)
            
           
            loss = loss_fn(q_values, targets_tensor)

           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Episode {episode}/{EPISODES} | Moves: {moves_played} | Loss: {loss.item():.4f} | Epsilon: {epsilon:.3f} | Total Reward: {total_reward:.2f}")
        else:
            print(f"Episode {episode}/{EPISODES} | No moves made.")

      
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

      
        if episode % SAVE_INTERVAL == 0:
            save_path = os.path.join(WEIGHTS_DIR, f"dqn_teacher_ep{episode}.pt")
            torch.save(dqn.state_dict(), save_path)
            print(f"--- Saved weights to {save_path} ---")

   
    torch.save(dqn.state_dict(), TRAINED_WEIGHTS)
    print(f"\nTraining complete. Final weights saved to {TRAINED_WEIGHTS}!")

if __name__ == "__main__":
    main()
