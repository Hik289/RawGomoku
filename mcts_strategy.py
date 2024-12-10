import random
import copy
import math


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # 当前棋盘状态
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visits = 1  # 初始化访问次数为 1，避免零除
        self.wins = 0  # 节点的获胜次数

    def is_fully_expanded(self):
        """检查节点是否已完全扩展"""
        return len(self.children) > 0 and all(child.visits > 0 for child in self.children)

    def best_child(self, exploration_weight=1.41):
        """根据 UCT 公式选择最佳子节点"""
        def uct_value(child):
            if child.visits == 0:
                return float('inf')  # 未访问的节点优先选择
            return (child.wins / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)

        return max(self.children, key=uct_value)


def get_legal_moves(board):
    """获取当前棋盘的所有合法落子位置"""
    size = len(board)
    moves = []
    for x in range(size):
        for y in range(size):
            if board[x][y] == '.':
                moves.append((x, y))
    return moves


def make_move(board, move, player):
    """模拟在棋盘上执行一步棋"""
    new_board = copy.deepcopy(board)
    x, y = move
    new_board[x][y] = player
    return new_board


def check_winner(board, player):
    """检查某一玩家是否获胜"""
    size = len(board)
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for x in range(size):
        for y in range(size):
            if board[x][y] == player:
                for dx, dy in directions:
                    count = 1
                    for step in range(1, 5):
                        nx, ny = x + step * dx, y + step * dy
                        if 0 <= nx < size and 0 <= ny < size and board[nx][ny] == player:
                            count += 1
                        else:
                            break
                    if count >= 5:
                        return True
    return False


def simulate_game(board, player):
    """随机模拟一局游戏"""
    current_player = player
    moves = get_legal_moves(board)
    while moves:
        move = random.choice(moves)
        board = make_move(board, move, current_player)
        if check_winner(board, current_player):
            return current_player
        current_player = 'O' if current_player == 'X' else 'X'
        moves = get_legal_moves(board)
    return None  # 平局


def mcts_search(initial_board, player, iterations=1000):
    """执行蒙特卡罗树搜索"""
    root = MCTSNode(initial_board)
    root.visits = 1  # 确保根节点的访问次数非零

    for _ in range(iterations):
        node = root
        state = initial_board

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            state = node.state

        # Expansion
        if not node.is_fully_expanded():
            legal_moves = get_legal_moves(state)
            for move in legal_moves:
                new_state = make_move(state, move, player)
                child_node = MCTSNode(new_state, node)
                node.children.append(child_node)

        # Simulation
        selected_node = random.choice(node.children) if node.children else node
        winner = simulate_game(selected_node.state, player)

        # Backpropagation
        while selected_node:
            selected_node.visits += 1
            if winner == player:
                selected_node.wins += 1
            selected_node = selected_node.parent

    # 选择最佳移动
    if root.children:
        best_child = root.best_child(0)  # Exploration weight = 0 to exploit only
        for move in get_legal_moves(initial_board):
            if make_move(initial_board, move, player) == best_child.state:
                return move

    # 兜底策略：随机选择一个合法的落子位置
    legal_moves = get_legal_moves(initial_board)
    if legal_moves:
        return random.choice(legal_moves)

    # 如果仍然没有合法位置，返回 None
    return None


def play(board, player, device=None):
    """基于 MCTS 的策略"""
    return mcts_search(board, player, iterations=100)