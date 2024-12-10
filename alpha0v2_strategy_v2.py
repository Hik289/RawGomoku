import torch
import torch.nn as nn
import numpy as np
import random
from collections import defaultdict

# ------------------------------
# AlphaZero Neural Network
# ------------------------------
class AlphaZeroNet(nn.Module):
    """Policy-value network module based on Net structure"""
    def __init__(self, board_size):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size

        # Common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board):
        # Common layers
        x = torch.relu(self.conv1(board))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Policy head
        x_policy = torch.relu(self.policy_conv(x))
        x_policy = x_policy.view(-1, 4 * self.board_size * self.board_size)
        policy = torch.log_softmax(self.policy_fc(x_policy), dim=1)

        # Value head
        x_value = torch.relu(self.value_conv(x))
        x_value = x_value.view(-1, 2 * self.board_size * self.board_size)
        x_value = torch.relu(self.value_fc1(x_value))
        value = torch.tanh(self.value_fc2(x_value))

        return policy, value


# ------------------------------
# MCTS Node
# ------------------------------
class MCTSNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to MCTSNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = MCTSNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


# ------------------------------
# MCTS with AlphaZero
# ------------------------------
class MCTS:
    def __init__(self, model, board_size):
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move model to GPU if available
        self.board_size = board_size


    def run(self, board, player,c_puct=5,simulations=100, last_move=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
        root = MCTSNode(None, 1.0)

        for _ in range(simulations):
            node = root
            sim_board = np.copy(board)
            sim_player = player
            sim_last_move = last_move

            # Selection
            while not node.is_leaf():
#                 actions = list(node._children.keys())
#                 values = [child.value() for child in node._children.values()]
                best_action, _ = node.select(c_puct)

                sim_board = self.make_move(sim_board, best_action, sim_player)

                sim_player = 'O' if sim_player == 'X' else 'X'
                node = node._children[best_action]

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
                node = node._parent

        # Choose the best action
        if root._children:
            # 正常选择访问次数最多的动作
            best_action = max(root._children.items(), key=lambda item: item[1]._n_visits)[0]
        else:
            # 如果没有扩展任何子节点，则随机选择一个合法动作
            legal_moves = self.get_legal_moves(board)
            if legal_moves:
                best_action = random.choice(legal_moves)
        return best_action, root._children

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

    def board_to_tensor(self, board, player, last_move=None):
        """Convert board to tensor for the neural network."""
        board_tensor = np.zeros((4, self.board_size, self.board_size))
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == player:  # 当前玩家的棋子位置
                    board_tensor[0, x, y] = 1
                elif board[x][y] != '.':  # 对手玩家的棋子位置
                    board_tensor[1, x, y] = 1
        if last_move:  # 最近一次落子的位置
            board_tensor[2, last_move[0], last_move[1]] = 1
        if player == 'O':  # 当前轮到谁下棋
            board_tensor[3, :, :] = 1
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