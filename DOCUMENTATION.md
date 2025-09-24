# DQNChess: Project Documentation

This document provides a comprehensive overview of the DQNChess project, from its core philosophy to the technical details of its implementation. It is intended to be a living document that evolves with the project.

## 1. Project Philosophy and Goals

The primary goal of DQNChess is to explore the application of Deep Q-Networks (DQN) to the game of chess. Unlike traditional chess engines that rely on brute-force search and handcrafted evaluation functions, this project aims to create an AI that learns to play chess through self-play and guided learning, much like a human would.

### My Thought Process

> **[Your Thought Process Here]**
>
> *This is the space for you to elaborate on your personal vision for this project. For example:*
>
> - *What inspired you to start this project?*
> - *What are the key questions you are trying to answer? (e.g., "Can a simple DQN outperform basic heuristics?")*
> - *What are your high-level goals? (e.g., "To create an agent that can beat me in a game," or "To explore the limitations of DQNs in a complex game like chess.")*

---

## 2. Architecture Overview

The project is designed to be modular, separating the core chess logic, the AI model, the training process, and the user interface. This makes it easier to experiment with different components without affecting the others.

> **[Insert Architecture Diagram Here]**
>
> *You can create a simple diagram (even using text or an online tool) that shows how the main files interact. For example:*
>
> `[chess_gui.py] <--> [chess_engine.py] <--> [train.py]`

The main components are:

-   **`chess_engine.py`**: The heart of the project. It defines the neural network architecture (DQN) and contains the crucial functions for encoding the board state and mapping moves to network outputs.
-   **`train.py`**: This script orchestrates the training process. It uses a "teacher-student" model where Stockfish acts as the teacher, guiding the DQN (the student) to learn good moves.
-   **`chess_gui.py`**: A graphical interface built with Pygame that allows a human player to play against the trained AI.
-   **`gui_test.py`**: A simple utility script to test if the Pygame environment is set up correctly.

---

## 3. Code Deep Dive

This section provides a detailed, function-by-function breakdown of the project's source code.

### `chess_engine.py`

This script contains the core components of the chess AI, including the board encoding logic and the DQN model itself.

-   **`encode_board(board)`**: Encodes a chess board state into a numerical representation for the neural network. The board is represented as a set of 12 binary "planes" (8x8), one for each piece type and color, which are then flattened into a single vector.
-   **`move_to_index(move)`**: Encodes a `chess.Move` object into a unique integer index. This is essential for mapping the DQN's output neurons to specific chess moves.
-   **`index_to_move(idx, board)`**: The inverse of `move_to_index`. It decodes an integer index back into a `chess.Move` object.
-   **`class DQN(nn.Module)`**: A simple Deep Q-Network with three fully connected layers. It takes the encoded board state as input and outputs Q-values for each possible move.
-   **`load_dqn(weights_file)`**: A helper function to load a pre-trained DQN model from a file. If no file is found, it initializes a new model with random weights.

### `train.py`

This script trains the DQN agent using a "teacher-student" learning approach with Stockfish as the teacher.

-   **`teacher_guided_move(board, stockfish, epsilon)`**: Selects a move using an epsilon-greedy strategy. With probability `(1 - epsilon)`, it chooses the best move suggested by Stockfish. With probability `epsilon`, it chooses a random legal move to encourage exploration. It also assigns a simple reward to the move.
-   **`main()`**: The main training loop. For each episode, it plays a game, collecting states, moves, and rewards. At the end of the episode, it uses this data to train the DQN in a supervised manner, teaching it to prefer the moves chosen by the teacher.

### `chess_gui.py`

This script implements a Pygame-based GUI for playing against the trained AI.

-   **`draw_board(win, selected_square)`**: Draws the chessboard squares.
-   **`draw_pieces(win, board)`**: Draws the pieces on the board using the loaded images.
-   **`main()`**: The main game loop. It handles user input (mouse clicks to select and move pieces) and, when it's the AI's turn, it uses the loaded DQN to select the best move.

---

## 4. Training Methodology

The training process is based on a "teacher-student" paradigm, which is a form of supervised learning designed to bootstrap the AI's knowledge.

-   **The Teacher**: Stockfish, a powerful and traditional chess engine, serves as the teacher. In any given position, it can provide a very strong "best move."
-   **The Student**: Our DQN model is the student. It starts with no knowledge of chess strategy.
-   **The Process**: During training, the student plays a game. At each step, it asks the teacher for advice. The teacher provides a good move, and the student learns to associate that move with the current board state. We use an epsilon-greedy approach, meaning sometimes the student ignores the teacher and tries a random move to see what happens (exploration).

### My Thought Process on Training

> **[Your Thought Process Here]**
>
> *This is a great place to explain your reasoning behind the training design.*
>
> - *Why did you choose a teacher-student model instead of pure reinforcement learning (e.g., Q-learning from scratch)? (e.g., "Pure RL in chess is very sample-inefficient, so I used a teacher to speed up learning.")*
> - *How did you design the reward function? What other reward structures did you consider?*
> - *What challenges did you face during training?*

### Training Performance

> **[Insert Training Loss Graph Here]**
>
> *After you run the training for a while, you can plot the loss over time. A decreasing loss graph is a great sign that the model is learning. You can add a screenshot of that graph here.*

---

## 5. Future Ideas and Improvements

This project is just the beginning. Here are some ideas for future development:

> **[Your Ideas Here]**
>
> *Jot down any ideas you have for the future. This is your personal roadmap.*
>
> - *e.g., "Implement a more sophisticated reward function that considers material advantage and positional control."*
> - *e.g., "Switch from a simple DQN to a Double DQN or Dueling DQN architecture."*
> - *e.g., "Experiment with pure reinforcement learning once the agent has a basic understanding of the game."*
> - *e.g., "Create a more advanced GUI with features like move history and game analysis."*
