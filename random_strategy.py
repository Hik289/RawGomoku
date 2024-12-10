import torch
import random

def play(board, player, device=None):
    """
    简单的随机策略，支持在 GPU 上运行
    Args:
        board: 当前棋盘
        player: 当前玩家 ('X' 或 'O')
        device: 指定运行的设备（'cuda' 或 'cpu'）
    Returns:
        随机选取的合法落子位置 (x, y)
    """
    # 确保棋盘在正确的设备上运行
    if device is not None:
        board = torch.tensor(board, dtype=torch.float32).to(device)
        size = board.shape[0]
    else:
        size = len(board)

    while True:
        # 随机生成一个合法的位置
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        if device is not None:
            if board[x, y].item() == 0:  # 0 表示空位
                return x, y
        else:
            if board[x][y] == '.':  # '.' 表示空位
                return x, y