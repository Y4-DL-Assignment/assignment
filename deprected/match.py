import tkinter as tk
from tkinter import messagebox, ttk
import time
import logging

# Configure logging
logging.basicConfig(
    filename='go_game.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class GoGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Go Game (9x9)")
        self.root.configure(bg='#f0e6d2')

        self.BOARD_SIZE = 9
        self.CELL_SIZE = 50
        self.PADDING = 30
        self.STONE_RADIUS = 20

        self.board = [[None for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = 'black'
        self.captures = {'black': 0, 'white': 0}
        self.last_move = None
        self.game_over = False
        self.move_count = 0

        # Timer settings
        self.time_limit = {'black': 600, 'white': 600}  # 10 minutes default
        self.time_remaining = {'black': 600, 'white': 600}
        self.last_time = None
        self.timer_running = False

        self.create_widgets()
        logging.info("=" * 60)
        logging.info("Game initialized - New Go Game (9x9) started")
        logging.info("Initial time: 10 minutes per player")
        logging.info("=" * 60)

    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Deep Q-Learn / Deep SARSA Go Game",
                               font=('Arial', 20, 'bold'), bg='#f0e6d2', fg='black')
        title_label.pack(pady=(15, 5))

        # Time control frame
        time_frame = tk.Frame(self.root, bg='#f0e6d2')
        time_frame.pack(pady=10)

        tk.Label(time_frame, text="Time per player (minutes):", font=('Arial', 11, 'bold'),
                 bg='#f0e6d2', fg='black').pack(side=tk.LEFT, padx=5)

        self.time_var = tk.StringVar(value="10")
        time_options = ['5', '10', '15', '20', '30']
        time_dropdown = ttk.Combobox(time_frame, textvariable=self.time_var,
                                     values=time_options, width=8, state='readonly')
        time_dropdown.pack(side=tk.LEFT, padx=5)

        tk.Button(time_frame, text="Set Time", command=self.set_time,
                  font=('Arial', 10, 'bold'), bg='#2196F3', fg='black',
                  padx=10, pady=3).pack(side=tk.LEFT, padx=5)

        # Top frame for player info
        top_frame = tk.Frame(self.root, bg='#f0e6d2')
        top_frame.pack(pady=10)

        # Black player info
        black_frame = tk.Frame(top_frame, bg='#f0e6d2', relief=tk.RAISED, borderwidth=2)
        black_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(black_frame, text="⚫ Black Player", font=('Arial', 14, 'bold'),
                 bg='#f0e6d2', fg='black').pack(pady=5)
        self.black_captures_label = tk.Label(black_frame, text="Captures: 0",
                                             font=('Arial', 11), bg='#f0e6d2', fg='black')
        self.black_captures_label.pack()
        self.black_timer_label = tk.Label(black_frame, text="Time: 10:00",
                                          font=('Arial', 12, 'bold'), bg='#f0e6d2', fg='black')
        self.black_timer_label.pack(pady=3)
        tk.Button(black_frame, text="Resign", command=lambda: self.resign('black'),
                  font=('Arial', 10, 'bold'), bg='#f44336', fg='black',
                  padx=15, pady=3).pack(pady=5)

        # Control buttons
        control_frame = tk.Frame(top_frame, bg='#f0e6d2')
        control_frame.pack(side=tk.LEFT, padx=20)

        tk.Button(control_frame, text="New Game", command=self.reset_game,
                  font=('Arial', 12, 'bold'), bg='#4CAF50', fg='black',
                  padx=20, pady=8).pack(pady=5)

        tk.Button(control_frame, text="Pass", command=self.pass_turn,
                  font=('Arial', 11, 'bold'), bg='#FF9800', fg='black',
                  padx=25, pady=5).pack(pady=5)

        # White player info
        white_frame = tk.Frame(top_frame, bg='#f0e6d2', relief=tk.RAISED, borderwidth=2)
        white_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(white_frame, text="⚪ White Player", font=('Arial', 14, 'bold'),
                 bg='#f0e6d2', fg='black').pack(pady=5)
        self.white_captures_label = tk.Label(white_frame, text="Captures: 0",
                                             font=('Arial', 11), bg='#f0e6d2', fg='black')
        self.white_captures_label.pack()
        self.white_timer_label = tk.Label(white_frame, text="Time: 10:00",
                                          font=('Arial', 12, 'bold'), bg='#f0e6d2', fg='black')
        self.white_timer_label.pack(pady=3)
        tk.Button(white_frame, text="Resign", command=lambda: self.resign('white'),
                  font=('Arial', 10, 'bold'), bg='#f44336', fg='black',
                  padx=15, pady=3).pack(pady=5)

        # Current turn label
        self.turn_label = tk.Label(self.root, text="Current Turn: Black",
                                   font=('Arial', 16, 'bold'), bg='#f0e6d2', fg='black')
        self.turn_label.pack(pady=5)

        # Canvas for the board
        canvas_width = self.CELL_SIZE * (self.BOARD_SIZE - 1) + self.PADDING * 2
        canvas_height = self.CELL_SIZE * (self.BOARD_SIZE - 1) + self.PADDING * 2

        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height,
                                bg='#daa520', highlightthickness=2, highlightbackground='#8B4513')
        self.canvas.pack(pady=10)

        self.draw_board()
        self.canvas.bind('<Button-1>', self.handle_click)

        # Instructions
        instructions = tk.Label(self.root,
                                text="Click on intersections to place stones. Capture opponent stones by surrounding them.\nUse 'Pass' to skip your turn. Use 'Resign' to forfeit the game.\n\n©2025 SLIIT - DL Assignment Group",
                                font=('Arial', 10), bg='#f0e6d2', fg='black', wraplength=600)
        instructions.pack(pady=10)

        # Start timer
        self.start_timer()

    def set_time(self):
        try:
            minutes = int(self.time_var.get())
            seconds = minutes * 60
            self.time_limit = {'black': seconds, 'white': seconds}
            self.time_remaining = {'black': seconds, 'white': seconds}
            self.update_timer_display()
            messagebox.showinfo("Time Set", f"Timer set to {minutes} minutes per player")
            logging.info(f"Timer configuration changed: {minutes} minutes per player ({seconds} seconds)")
            logging.info(f"Time remaining - Black: {seconds}s, White: {seconds}s")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid time")
            logging.error("Failed to set time - Invalid input value")

    def start_timer(self):
        self.timer_running = True
        self.last_time = time.time()
        logging.info("Game timer started")
        self.update_timer()

    def update_timer(self):
        if not self.timer_running or self.game_over:
            return

        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time

        # Deduct time from current player
        self.time_remaining[self.current_player] -= elapsed

        # Check for time out
        if self.time_remaining[self.current_player] <= 0:
            self.time_remaining[self.current_player] = 0
            self.game_over = True
            self.timer_running = False
            opponent = 'white' if self.current_player == 'black' else 'black'

            logging.warning(f"TIME OUT: {self.current_player.capitalize()} ran out of time!")
            logging.info(f"GAME OVER: {opponent.capitalize()} wins by timeout")
            logging.info(
                f"Final score - Black captures: {self.captures['black']}, White captures: {self.captures['white']}")
            logging.info("=" * 60)

            messagebox.showinfo("Time Out!",
                                f"{self.current_player.capitalize()} ran out of time!\n{opponent.capitalize()} wins!")
            return

        self.update_timer_display()
        self.root.after(100, self.update_timer)

    def update_timer_display(self):
        for player in ['black', 'white']:
            minutes = int(self.time_remaining[player] // 60)
            seconds = int(self.time_remaining[player] % 60)
            time_str = f"Time: {minutes:02d}:{seconds:02d}"

            if player == 'black':
                self.black_timer_label.config(text=time_str)
            else:
                self.white_timer_label.config(text=time_str)

    def resign(self, player):
        if self.game_over:
            logging.warning(f"Resign attempt by {player} after game already ended")
            return

        result = messagebox.askyesno("Resign",
                                     f"Are you sure {player.capitalize()} wants to resign?")
        if result:
            self.game_over = True
            self.timer_running = False
            opponent = 'white' if player == 'black' else 'black'

            logging.warning(f"RESIGNATION: {player.capitalize()} has resigned from the game")
            logging.info(f"GAME OVER: {opponent.capitalize()} wins by resignation")
            logging.info(f"Game ended at move {self.move_count}")
            logging.info(
                f"Final score - Black captures: {self.captures['black']}, White captures: {self.captures['white']}")
            logging.info(
                f"Time remaining - Black: {int(self.time_remaining['black'])}s, White: {int(self.time_remaining['white'])}s")
            logging.info("=" * 60)

            messagebox.showinfo("Game Over",
                                f"{player.capitalize()} resigned.\n{opponent.capitalize()} wins!")

    def pass_turn(self):
        if self.game_over:
            logging.warning(f"Pass attempt by {self.current_player} after game ended")
            return

        opponent = 'white' if self.current_player == 'black' else 'black'
        logging.info(f"Move {self.move_count + 1}: {self.current_player.capitalize()} PASSED their turn")
        logging.info(f"Turn switched to {opponent.capitalize()}")

        messagebox.showinfo("Pass", f"{self.current_player.capitalize()} passed their turn.")

        self.current_player = opponent
        self.turn_label.config(text=f"Current Turn: {self.current_player.capitalize()}")
        self.last_move = None
        self.move_count += 1
        self.draw_board()

    def draw_board(self):
        self.canvas.delete('all')

        # Draw grid lines
        for i in range(self.BOARD_SIZE):
            # Horizontal lines
            x1 = self.PADDING
            y = self.PADDING + i * self.CELL_SIZE
            x2 = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE
            self.canvas.create_line(x1, y, x2, y, fill='#8B4513', width=2)

            # Vertical lines
            x = self.PADDING + i * self.CELL_SIZE
            y1 = self.PADDING
            y2 = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE
            self.canvas.create_line(x, y1, x, y2, fill='#8B4513', width=2)

        # Draw star points
        star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        for row, col in star_points:
            x = self.PADDING + col * self.CELL_SIZE
            y = self.PADDING + row * self.CELL_SIZE
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill='#8B4513', outline='#8B4513')

        # Draw stones
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row][col]:
                    self.draw_stone(row, col, self.board[row][col])

        # Draw last move marker
        if self.last_move:
            row, col = self.last_move
            x = self.PADDING + col * self.CELL_SIZE
            y = self.PADDING + row * self.CELL_SIZE
            marker_color = 'white' if self.board[row][col] == 'black' else 'black'
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=marker_color, outline=marker_color)

    def draw_stone(self, row, col, color):
        x = self.PADDING + col * self.CELL_SIZE
        y = self.PADDING + row * self.CELL_SIZE

        fill_color = 'black' if color == 'black' else 'white'
        outline_color = '#333' if color == 'black' else '#666'

        self.canvas.create_oval(x - self.STONE_RADIUS, y - self.STONE_RADIUS,
                                x + self.STONE_RADIUS, y + self.STONE_RADIUS,
                                fill=fill_color, outline=outline_color, width=2)

    def handle_click(self, event):
        if self.game_over:
            logging.warning(f"Move attempt after game ended at position ({event.x}, {event.y})")
            return

        # Convert click coordinates to board position
        col = round((event.x - self.PADDING) / self.CELL_SIZE)
        row = round((event.y - self.PADDING) / self.CELL_SIZE)

        # Check if click is within board bounds
        if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
            self.place_stone(row, col)
        else:
            logging.debug(f"Click outside board bounds: ({row}, {col})")

    def place_stone(self, row, col):
        # Check if position is already occupied
        if self.board[row][col] is not None:
            logging.warning(
                f"ILLEGAL MOVE: {self.current_player.capitalize()} attempted to place stone at occupied position ({row}, {col})")
            return

        # Place stone temporarily
        self.board[row][col] = self.current_player
        self.move_count += 1

        logging.info(
            f"Move {self.move_count}: {self.current_player.capitalize()} placed stone at position ({row}, {col})")

        # Check for captures
        opponent = 'white' if self.current_player == 'black' else 'black'
        captured_count = self.remove_captures(opponent)

        # Check for suicide (illegal move)
        if not self.has_liberties(row, col) and captured_count == 0:
            self.board[row][col] = None
            self.move_count -= 1

            logging.warning(
                f"ILLEGAL MOVE (SUICIDE): {self.current_player.capitalize()} attempted suicide move at ({row}, {col}) - Move rejected")
            messagebox.showwarning("Illegal Move", "Suicide is not allowed!")
            return

        # Update captures
        if captured_count > 0:
            self.captures[self.current_player] += captured_count
            self.update_captures_display()
            logging.info(f"CAPTURE: {self.current_player.capitalize()} captured {captured_count} {opponent} stone(s)")
            logging.info(f"Total captures - Black: {self.captures['black']}, White: {self.captures['white']}")

        # Log liberties information
        liberties_count = self.count_liberties(row, col)
        logging.debug(f"Stone at ({row}, {col}) has {liberties_count} liberties")

        # Update last move
        self.last_move = (row, col)

        # Log time remaining
        black_time = int(self.time_remaining['black'])
        white_time = int(self.time_remaining['white'])
        logging.debug(f"Time remaining - Black: {black_time}s, White: {white_time}s")

        # Switch player
        self.current_player = opponent
        self.turn_label.config(text=f"Current Turn: {self.current_player.capitalize()}")
        logging.info(f"Turn switched to {self.current_player.capitalize()}")
        logging.info("-" * 40)

        # Redraw board
        self.draw_board()

    def get_neighbors(self, row, col):
        neighbors = []
        if row > 0:
            neighbors.append((row - 1, col))
        if row < self.BOARD_SIZE - 1:
            neighbors.append((row + 1, col))
        if col > 0:
            neighbors.append((row, col - 1))
        if col < self.BOARD_SIZE - 1:
            neighbors.append((row, col + 1))
        return neighbors

    def get_group(self, row, col, visited=None):
        if visited is None:
            visited = set()

        key = (row, col)
        if key in visited:
            return []

        color = self.board[row][col]
        if color is None:
            return []

        visited.add(key)
        group = [key]

        for r, c in self.get_neighbors(row, col):
            if self.board[r][c] == color:
                group.extend(self.get_group(r, c, visited))

        return group

    def has_liberties(self, row, col):
        color = self.board[row][col]
        if color is None:
            return True

        group = self.get_group(row, col)

        for r, c in group:
            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr][nc] is None:
                    return True

        return False

    def count_liberties(self, row, col):
        """Count the number of liberties for a stone group"""
        color = self.board[row][col]
        if color is None:
            return 0

        group = self.get_group(row, col)
        liberties = set()

        for r, c in group:
            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr][nc] is None:
                    liberties.add((nr, nc))

        return len(liberties)

    def remove_captures(self, color):
        captured_count = 0
        captured_groups = []

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row][col] == color and not self.has_liberties(row, col):
                    group = self.get_group(row, col)
                    captured_groups.append(group)
                    for r, c in group:
                        self.board[r][c] = None
                        captured_count += 1

        if captured_groups:
            logging.debug(f"Captured {len(captured_groups)} group(s) containing {captured_count} stones")
            for i, group in enumerate(captured_groups):
                logging.debug(f"  Captured group {i + 1}: {group}")

        return captured_count

    def update_captures_display(self):
        self.black_captures_label.config(text=f"Captures: {self.captures['black']}")
        self.white_captures_label.config(text=f"Captures: {self.captures['white']}")

    def reset_game(self):
        logging.info("=" * 60)
        logging.info("GAME RESET: Starting new game")

        if self.move_count > 0:
            logging.info(f"Previous game statistics:")
            logging.info(f"  Total moves: {self.move_count}")
            logging.info(f"  Final captures - Black: {self.captures['black']}, White: {self.captures['white']}")
            logging.info(
                f"  Final time - Black: {int(self.time_remaining['black'])}s, White: {int(self.time_remaining['white'])}s")

        self.board = [[None for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = 'black'
        self.captures = {'black': 0, 'white': 0}
        self.last_move = None
        self.game_over = False
        self.move_count = 0

        # Reset timer
        minutes = int(self.time_var.get())
        seconds = minutes * 60
        self.time_remaining = {'black': seconds, 'white': seconds}
        self.timer_running = True
        self.last_time = time.time()

        self.turn_label.config(text="Current Turn: Black")
        self.update_captures_display()
        self.update_timer_display()
        self.draw_board()

        logging.info(f"New game initialized with {minutes} minutes per player")
        logging.info("=" * 60)


if __name__ == '__main__':
    root = tk.Tk()
    game = GoGame(root)
    root.mainloop()