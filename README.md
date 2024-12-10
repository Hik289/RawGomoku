# AlphaZero Gomoku Strategies and MCTS Implementations

This repository contains Python implementations of various strategies and algorithms for playing Gomoku, including AlphaZero-based learning and Monte Carlo Tree Search (MCTS).

## File Overview

1. **`alpha0v2_strategy.py`**
   - Implements the AlphaZero framework for training and playing Gomoku.
   - Contains functions for model training, evaluation, and self-play using an MCTS-based policy-value network.

2. **`alpha0v2_strategy_v2.py`**
   - An updated version of `alpha0v2_strategy.py` with optimizations and improvements in MCTS and self-play.

3. **`mcts_strategy.py`**
   - Implements a pure MCTS-based strategy for Gomoku.
   - Suitable for comparison with learning-based strategies like AlphaZero.

4. **`mctsv2_strategy.py`**
   - A second version of the MCTS strategy with additional heuristic features for move evaluation and selection.

5. **`mctsv4_strategy.py`**
   - A more advanced version of MCTS with further improvements and enhancements.

6. **`random_strategy.py`**
   - Implements a simple random strategy for Gomoku.
   - Useful as a baseline for evaluating other strategies.

# AlphaZero Gomoku Training and Strategies

This repository contains scripts and notebooks for training an AlphaZero-based AI for playing Gomoku, along with various strategies implemented using Monte Carlo Tree Search (MCTS) and random strategies.

## File Structure

### Training Scripts and Notebooks
- **`alphazero_training.ipynb`**: Initial notebook for training AlphaZero on Gomoku.
- **`alphazero_training_v2.ipynb`**: Updated version of the training notebook with improved strategies.
- **`alphazero_training_v3.ipynb`**: Further optimized training notebook.
- **`alphazero_training_v3_copy*.ipynb`**: Backups and variations of the training scripts.

### Strategy Implementations
- **`alpha0v2_strategy.py`**: Implements AlphaZero MCTS strategy for playing Gomoku.
- **`alpha0v2_strategy_v2.py`**: Improved version of AlphaZero strategy.
- **`mcts_strategy.py`**: Classic MCTS-based strategy.
- **`mctsv2_strategy.py`**: Enhanced MCTS strategy with additional optimizations.
- **`mctsv4_strategy.py`**: Further improved MCTS strategy.
- **`random_strategy.py`**: Simple random strategy for Gomoku.

## How to Use

### Training
1. Open any of the training notebooks (e.g., `alphazero_training_v3.ipynb`) in Jupyter Notebook or JupyterLab.
2. Configure parameters such as board size, number of simulations, and learning rate.
3. Run the cells sequentially to train the AlphaZero model.

### Playing
1. Use one of the strategy scripts (e.g., `alpha0v2_strategy.py`) to load the trained AlphaZero model.
2. Play against various strategies (random or MCTS) by running the provided `play_with_model_and_strategy` function.

### Evaluation
- Use `play_with_model_and_strategy` in conjunction with different strategies to evaluate AlphaZero's performance.


## Getting Started

### Prerequisites

- Python 3.7+
- Required libraries:
  - `numpy`
  - `torch` (for AlphaZero-based strategies)
  - `matplotlib` (for visualizations)
  - `turtle` (optional for graphical board rendering)

### How to Use

1. **Training AlphaZero**:
   - Use `alpha0v2_strategy.py` or `alpha0v2_strategy_v2.py` to train the AlphaZero model.
   - Customize parameters like board size, number of iterations, and self-play simulations.

2. **Testing Against Other Strategies**:
   - Load pre-trained AlphaZero weights.
   - Use the `play_with_model_and_strategy` function to test AlphaZero against `mcts_strategy`, `mctsv2_strategy`, `mctsv4_strategy`, or `random_strategy`.

3. **Visualizing the Board**:
   - Use the `turtle`-based board rendering functions provided in the scripts for a graphical representation of the game state.

### Example: Playing Against a Random Strategy

```python
from alpha0v2_strategy_v2 import AlphaZeroNet, play_with_model_and_strategy
from random_strategy import play as random_play

board_size = 7
model = AlphaZeroNet(board_size)
alpha_zero_wins, random_strategy_wins, draws = play_with_model_and_strategy(
    model, random_play, board_size=board_size, rounds=10
)

print(f"AlphaZero Wins: {alpha_zero_wins}, Random Strategy Wins: {random_strategy_wins}, Draws: {draws}")