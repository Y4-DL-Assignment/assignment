import numpy as np
import tkinter as tk
import math

# CONFIGURATION: Change these settings
GRID_SIZE = 4  # Board size (4x4)
WIN_LENGTH = 4  # How many in a row to win (set to 3 for easier wins)


class TicTacToe:
    def __init__(self, size=GRID_SIZE, win_length=WIN_LENGTH):
        self.size = size
        self.win_length = win_length
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return np.append(self.board.flatten(), self.current_player)

    def get_available_actions(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    def step(self, action):
        i, j = action
        if self.board[i, j] != 0 or self.done:
            return self.get_state(), -10, True, {}
        self.board[i, j] = self.current_player
        winner = self.check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            return self.get_state(), 1 if winner == self.current_player else -1, True, {}
        elif not self.get_available_actions():
            self.done = True
            self.winner = 0
            return self.get_state(), 0, True, {}
        else:
            self.current_player *= -1
            return self.get_state(), 0, False, {}

    def check_winner(self):
        return self.check_winner_on_board(self.board)

    def check_winner_on_board(self, board):
        # Check rows
        for i in range(self.size):
            for j in range(self.size - self.win_length + 1):
                window = board[i, j:j + self.win_length]
                if abs(sum(window)) == self.win_length and len(set(window)) == 1 and window[0] != 0:
                    return int(np.sign(sum(window)))

        # Check columns
        for i in range(self.size - self.win_length + 1):
            for j in range(self.size):
                window = board[i:i + self.win_length, j]
                if abs(sum(window)) == self.win_length and len(set(window)) == 1 and window[0] != 0:
                    return int(np.sign(sum(window)))

        # Check diagonals (top-left to bottom-right)
        for i in range(self.size - self.win_length + 1):
            for j in range(self.size - self.win_length + 1):
                window = [board[i + k, j + k] for k in range(self.win_length)]
                if abs(sum(window)) == self.win_length and len(set(window)) == 1 and window[0] != 0:
                    return int(np.sign(sum(window)))

        # Check diagonals (top-right to bottom-left)
        for i in range(self.size - self.win_length + 1):
            for j in range(self.win_length - 1, self.size):
                window = [board[i + k, j - k] for k in range(self.win_length)]
                if abs(sum(window)) == self.win_length and len(set(window)) == 1 and window[0] != 0:
                    return int(np.sign(sum(window)))

        return None


class PerfectAgent:
    """Unbeatable minimax agent with alpha-beta pruning and move ordering"""

    def __init__(self, env):
        self.env = env
        self.size = env.size
        self.win_length = env.win_length
        self.transposition_table = {}

    def minimax(self, board, player, alpha, beta, is_maximizing, depth):
        # Check transposition table
        board_key = (board.tobytes(), player)
        if board_key in self.transposition_table:
            return self.transposition_table[board_key]

        winner = self.env.check_winner_on_board(board)
        if winner == -1:  # AI wins
            score = 100 - depth  # Prefer faster wins
            self.transposition_table[board_key] = score
            return score
        elif winner == 1:  # Human wins
            score = -100 + depth
            self.transposition_table[board_key] = score
            return score

        available = self.get_available_on_board(board)
        if not available:
            self.transposition_table[board_key] = 0
            return 0

        # Move ordering: prioritize center and near-filled positions
        available = self.order_moves(board, available)

        if is_maximizing:  # AI's turn (O = -1)
            best_score = -math.inf
            for action in available:
                i, j = action
                board[i, j] = -1
                score = self.minimax(board, 1, alpha, beta, False, depth + 1)
                board[i, j] = 0
                best_score = max(best_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            self.transposition_table[board_key] = best_score
            return best_score
        else:  # Human's turn (X = 1)
            best_score = math.inf
            for action in available:
                i, j = action
                board[i, j] = 1
                score = self.minimax(board, -1, alpha, beta, True, depth + 1)
                board[i, j] = 0
                best_score = min(best_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            self.transposition_table[board_key] = best_score
            return best_score

    def order_moves(self, board, moves):
        """Order moves to improve alpha-beta pruning efficiency"""

        def move_priority(move):
            i, j = move
            # Prioritize center positions
            center = self.size / 2
            dist_from_center = abs(i - center) + abs(j - center)

            # Count adjacent pieces (prefer moves near existing pieces)
            adjacent_count = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        if board[ni, nj] != 0:
                            adjacent_count += 1

            return (-adjacent_count, dist_from_center)

        return sorted(moves, key=move_priority)

    def choose_action(self):
        best_score = -math.inf
        best_action = None
        board = self.env.board.copy()

        # Clear cache for new move
        self.transposition_table.clear()

        available = self.env.get_available_actions()
        available = self.order_moves(board, available)

        for action in available:
            i, j = action
            board[i, j] = -1
            score = self.minimax(board, 1, -math.inf, math.inf, False, 0)
            board[i, j] = 0

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def get_available_on_board(self, board):
        return [(i, j) for i in range(self.size) for j in range(self.size) if board[i, j] == 0]


class TicTacToeVisualizer(tk.Tk):
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self.title(f"Perfect {env.size}x{env.size} Tic-Tac-Toe (Connect {env.win_length})")
        self.cell_size = 70
        self.canvas = tk.Canvas(self, width=env.size * self.cell_size, height=env.size * self.cell_size)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self.status = tk.Label(self, text=f"You are X. Connect {env.win_length} to win. AI is unbeatable!",
                               font=("Arial", 11, "bold"))
        self.status.pack(pady=10)

        self.reset_btn = tk.Button(self, text="New Game", command=self.reset_game,
                                   font=("Arial", 10))
        self.reset_btn.pack(pady=5)

        self.stats_label = tk.Label(self, text="Wins: 0 | Draws: 0 | Losses: 0",
                                    font=("Arial", 10))
        self.stats_label.pack()

        self.wins = 0
        self.draws = 0
        self.losses = 0

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(self.env.size):
            for j in range(self.env.size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black", width=2)

                if self.env.board[i, j] == 1:
                    # Draw X
                    offset = 12
                    self.canvas.create_line(x0 + offset, y0 + offset, x1 - offset, y1 - offset,
                                            fill="blue", width=3)
                    self.canvas.create_line(x1 - offset, y0 + offset, x0 + offset, y1 - offset,
                                            fill="blue", width=3)
                elif self.env.board[i, j] == -1:
                    # Draw O
                    center_x = x0 + self.cell_size // 2
                    center_y = y0 + self.cell_size // 2
                    radius = self.cell_size // 2 - 12
                    self.canvas.create_oval(center_x - radius, center_y - radius,
                                            center_x + radius, center_y + radius,
                                            outline="red", width=3)
        self.update()

    def on_click(self, event):
        if self.env.done:
            return

        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if row >= self.env.size or col >= self.env.size:
            return

        if (row, col) in self.env.get_available_actions():
            # Human move
            _, _, done, _ = self.env.step((row, col))
            self.draw_board()

            if done:
                self.show_result()
            else:
                self.status.config(text="AI is thinking...")
                self.update()
                self.after(500, self.agent_move)

    def agent_move(self):
        if self.env.done:
            return

        action = self.agent.choose_action()
        _, _, done, _ = self.env.step(action)
        self.draw_board()

        if done:
            self.show_result()
        else:
            self.status.config(text="Your turn (X)")

    def show_result(self):
        if self.env.winner == 1:
            self.status.config(text="🎉 You won! (Impossible!)", fg="green")
            self.wins += 1
        elif self.env.winner == -1:
            self.status.config(text="AI wins!", fg="red")
            self.losses += 1
        else:
            self.status.config(text="Draw! Well played!", fg="orange")
            self.draws += 1

        self.stats_label.config(text=f"Wins: {self.wins} | Draws: {self.draws} | Losses: {self.losses}")

    def reset_game(self):
        self.env.reset()
        self.draw_board()
        self.status.config(text="Your turn (X)", fg="black")


if __name__ == "__main__":
    env = TicTacToe(size=GRID_SIZE, win_length=WIN_LENGTH)
    agent = PerfectAgent(env)
    app = TicTacToeVisualizer(env, agent)
    app.mainloop()