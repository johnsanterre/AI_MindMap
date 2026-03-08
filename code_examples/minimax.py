"""Minimax — optimal play in two-player zero-sum games."""

import numpy as np

def minimax(board, depth, is_maximizing, evaluate_fn, get_moves_fn, make_move_fn):
    score = evaluate_fn(board)
    if score is not None or depth == 0:
        return score if score is not None else 0, None

    moves = get_moves_fn(board)
    if not moves:
        return 0, None

    best_move = moves[0]
    if is_maximizing:
        best_val = -np.inf
        for move in moves:
            new_board = make_move_fn(board, move, 'X')
            val, _ = minimax(new_board, depth-1, False, evaluate_fn, get_moves_fn, make_move_fn)
            if val > best_val:
                best_val = val
                best_move = move
    else:
        best_val = np.inf
        for move in moves:
            new_board = make_move_fn(board, move, 'O')
            val, _ = minimax(new_board, depth-1, True, evaluate_fn, get_moves_fn, make_move_fn)
            if val < best_val:
                best_val = val
                best_move = move
    return best_val, best_move

# Tic-tac-toe implementation
def evaluate(board):
    lines = [board[0], board[1], board[2],
             [board[i][0] for i in range(3)],
             [board[i][1] for i in range(3)],
             [board[i][2] for i in range(3)],
             [board[i][i] for i in range(3)],
             [board[i][2-i] for i in range(3)]]
    for line in lines:
        if line == ['X','X','X']: return 1
        if line == ['O','O','O']: return -1
    if all(board[i][j] != '.' for i in range(3) for j in range(3)):
        return 0
    return None

def get_moves(board):
    return [(i,j) for i in range(3) for j in range(3) if board[i][j] == '.']

def make_move(board, move, player):
    new = [row[:] for row in board]
    new[move[0]][move[1]] = player
    return new

def print_board(board):
    for row in board:
        print("  " + " ".join(row))

# --- demo ---
board = [['.','.','.'],['.','.','.'],['.','.','.']],
board = [['.','.','.'],['.','.','.'],['.','.','.']]

print("=== Minimax: Tic-Tac-Toe ===\n")
print("X plays optimally using minimax:\n")

current = board
is_x = True
while evaluate(current) is None:
    if is_x:
        val, move = minimax(current, 9, True, evaluate, get_moves, make_move)
        current = make_move(current, move, 'X')
        print(f"X plays {move} (value={val:+d})")
    else:
        # O also plays optimally
        val, move = minimax(current, 9, False, evaluate, get_moves, make_move)
        current = make_move(current, move, 'O')
        print(f"O plays {move} (value={val:+d})")
    print_board(current)
    print()
    is_x = not is_x

result = evaluate(current)
print(f"Result: {'Draw' if result == 0 else 'X wins' if result == 1 else 'O wins'}")
print("(With optimal play from both sides, tic-tac-toe is always a draw.)")
