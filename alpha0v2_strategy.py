import torch
import torch.nn as nn
import numpy as np
import random
from collections import defaultdict

# ------------------------------
# AlphaZero Neural Network
# ------------------------------
class AlphaZeroNet(nn.Module):
    def __init__(self, board_size):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size

        # Define a simple convolutional network
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Policy head
        self.policy_head = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(board_size * board_size * 2, board_size * board_size)

        # Value head
        self.value_head = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc = nn.Linear(board_size * board_size, 1)

    def forward(self, board):
        x = torch.relu(self.conv1(board))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Policy head
        policy = torch.relu(self.policy_head(x))
        policy = policy.view(policy.size(0), -1)
        policy = torch.softmax(self.policy_fc(policy), dim=1)

        # Value head
        value = torch.relu(self.value_head(x))
        value = value.view(value.size(0), -1)
        value = torch.tanh(self.value_fc(value))

        return policy, value


# ------------------------------
# MCTS Node
# ------------------------------
class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_probs):
        """Expand the node with the given action probabilities."""
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prob)

    def update(self, value):
        """Update the node with the given simulation result."""
        self.visits += 1
        self.value_sum += value

    def value(self, exploration_weight=1.0):
        """Compute the node's value with UCT."""
        if self.visits == 0:
            return float('inf')  # Prioritize unexplored nodes
        return (self.value_sum / self.visits) + exploration_weight * self.prior * np.sqrt(np.log(self.parent.visits) / (1 + self.visits))


# ------------------------------
# MCTS with AlphaZero
# ------------------------------
class MCTS:
    def __init__(self, model, board_size):
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move model to GPU if available
        self.board_size = board_size

    def run(self, board, player, simulations=100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
        root = MCTSNode()

        for _ in range(simulations):
            node = root
            sim_board = np.copy(board)
            sim_player = player

            # Selection
            while not node.is_leaf():
                actions = list(node.children.keys())
                values = [child.value() for child in node.children.values()]
                best_action = actions[np.argmax(values)]
                sim_board = self.make_move(sim_board, best_action, sim_player)
                sim_player = 'O' if sim_player == 'X' else 'X'
                node = node.children[best_action]

            # Expansion and Evaluation
            legal_moves = self.get_legal_moves(sim_board)
            if not legal_moves:
                value = 0  # Draw
            else:
                # Move tensor to GPU
                board_tensor = self.board_to_tensor(sim_board, sim_player).to(device)
                policy, value = self.model(board_tensor)
                policy = policy.detach().cpu().numpy().flatten()  # Move back to CPU for numpy processing
                action_probs = [(move, policy[move[0] * self.board_size + move[1]]) for move in legal_moves]
                node.expand(action_probs)
                value = value.item()

            # Backpropagation
            while node is not None:
                node.update(value)
                node = node.parent

        # Choose the best action
        best_action = max(root.children.items(), key=lambda item: item[1].value_sum)[0]
        return best_action, root.children

    def check_winner(self, board, player):
        """Check if the given player has won on the board."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, vertical, two diagonals
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == player:
                    for dx, dy in directions:
                        count = 1
                        for step in range(1, 5):  # Check in the positive direction
                            nx, ny = x + step * dx, y + step * dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx][ny] == player:
                                count += 1
                            else:
                                break
                        for step in range(1, 5):  # Check in the negative direction
                            nx, ny = x - step * dx, y - step * dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx][ny] == player:
                                count += 1
                            else:
                                break
                        if count >= 5:
                            return True
        return False

    def board_to_tensor(self, board, player):
        """Convert board to tensor for the neural network."""
        board_tensor = np.zeros((2, self.board_size, self.board_size))
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == player:
                    board_tensor[0, x, y] = 1
                elif board[x][y] != '.':
                    board_tensor[1, x, y] = 1
        return torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def get_legal_moves(board):
        """Get all legal moves for the current board."""
        size = len(board)
        return [(x, y) for x in range(size) for y in range(size) if board[x][y] == '.']

    @staticmethod
    def make_move(board, move, player):

        """Simulate a move on the board."""
        new_board = np.copy(board)
        x, y = move
        new_board[x][y] = player
        return new_board


# ------------------------------
# Strategy Function
# ------------------------------
def play(board, player):
    """AlphaZero strategy with MCTS."""
    board_size = len(board)
    model = AlphaZeroNet(board_size)

    # Load pretrained weights if available
    model_path = "alphazero_weights.pth"
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print("Warning: No pretrained weights found. Using untrained model.")

    mcts = MCTS(model, board_size)
    best_move = mcts.run(board, player, simulations=100)
    return best_move