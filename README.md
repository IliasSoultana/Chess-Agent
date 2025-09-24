# DQN Chess AI

This project is a chess-playing AI that uses a Deep Q-Network (DQN) to learn and play the game of chess. It includes a chess engine, a graphical user interface (GUI) for playing against the AI, and scripts for training the model.

## Overview

The core of this project is a reinforcement learning agent trained to play chess. It learns by playing against itself and other engines (like Stockfish), using a DQN to estimate the value of different board states and moves.

## Features

- **DQN-based Chess AI**: A neural network trained to play chess.
- **Chess Engine**: Handles game logic, move generation, and validation.
- **GUI**: A simple graphical interface to play against the trained AI.
- **Training Pipeline**: Scripts to train the DQN model from scratch.
- **Stockfish Integration**: Can be configured to use Stockfish for training or evaluation.

## Project Structure

```
DQNChess/
├── .gitignore
├── chess_ai/         # Core AI and model-related files
├── chess_engine.py   # Main chess game logic
├── chess_gui.py      # GUI to play against the AI
├── images/           # Images for the GUI
├── stockfish/        # Stockfish engine binaries
└── train.py          # Script for training the AI model
```

- **`chess_engine.py`**: Implements the rules of chess, board state, and move validation.
- **`chess_gui.py`**: Launches a Pygame-based GUI where you can play against the AI.
- **`train.py`**: The main script to start the training process for the DQN agent.
- **`chess_ai/`**: Contains the DQN model definition, memory buffer, and agent logic.
- **`stockfish/`**: Contains the Stockfish engine, which can be used for evaluation or as a training opponent.
- **`images/`**: Contains the chess piece and board images used by the GUI.

## Getting Started

### Prerequisites

- Python 3.8+
- Pygame
- PyTorch or TensorFlow
- `python-chess`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/IliasSoultana/Chess-Agent.git
    cd Chess-Agent
    ```

2.  **Install dependencies:**
    (You might need to create a `requirements.txt` file for this)
    ```bash
    pip install pygame torch python-chess
    ```

3.  **Setup Stockfish:**
    Download the appropriate Stockfish binary for your system and place it in the `stockfish/` directory.

### Usage

1.  **Train the model:**
    ```bash
    python train.py
    ```

2.  **Play against the AI:**
    ```bash
    python chess_gui.py
    ```

## Contributing

Contributions are welcome! If you have ideas for improvements or find any bugs, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
