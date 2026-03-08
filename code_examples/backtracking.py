"""Backtracking — N-Queens and Sudoku solver."""

def n_queens(n):
    """Find all solutions to the N-Queens problem."""
    solutions = []
    def solve(queens, row):
        if row == n:
            solutions.append(queens[:])
            return
        for col in range(n):
            if all(col != c and abs(col - c) != row - r for r, c in enumerate(queens)):
                queens.append(col)
                solve(queens, row + 1)
                queens.pop()
    solve([], 0)
    return solutions

def solve_sudoku(board):
    """Solve a 9x9 Sudoku board in-place. 0 = empty."""
    def valid(r, c, num):
        for i in range(9):
            if board[r][i] == num or board[i][c] == num:
                return False
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for i in range(br, br+3):
            for j in range(bc, bc+3):
                if board[i][j] == num:
                    return False
        return True

    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for num in range(1, 10):
                    if valid(r, c, num):
                        board[r][c] = num
                        if solve_sudoku(board):
                            return True
                        board[r][c] = 0
                return False
    return True

# --- demo ---
print(f"4-Queens: {len(n_queens(4))} solutions")
for sol in n_queens(4):
    print(" ", sol)

print(f"\n8-Queens: {len(n_queens(8))} solutions (first: {n_queens(8)[0]})")
