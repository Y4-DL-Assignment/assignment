import tkinter as tk
from tkinter import messagebox, ttk
import time
import logging
from dqn_agent import DQNAgent


class GoGame:
    """Go Game with Deep Q-Learning AI opponent"""

    def __init__(self, root):
        self.root = root
        self.root.title("Go Game (9x9) - Deep Q-Learning")
        self.root.configure(bg='#f0e6d2')

        self.BOARD_SIZE = 9
        self.CELL_SIZE = 50
        self.PADDING = 30
        self.STONE_RADIUS = 20

        # Game state
        self.board = [[None for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = 'black'
        self.captures = {'black': 0, 'white': 0}
        self.last_move = None
        self.game_over = False
        self.move_count = 0

        # AI settings
        self.ai_enabled = False
        self.ai_player = 'white'  # AI plays as white by default
        self.agent = DQNAgent(board_size=self.BOARD_SIZE)
        self.previous_state = None
        self.previous_action = None

        # Try to load existing trained agent
        if self.agent.load():
            logging.info("Loaded pre-trained agent")
        else:
            logging.info("Starting with new untrained agent")

        # Timer settings
        self.time_limit = {'black': 600, 'white': 600}
        self.time_remaining = {'black': 600, 'white': 600}
        self.last_time = None
        self.timer_running = False

        self.create_widgets()
        logging.info("=" * 60)
        logging.info("Game initialized - Deep Q-Learning Go Game (9x9)")
        logging.info("=" * 60)

    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title_label = tk.Label(self.root, text="Deep Q-Learning Go Game",
                               font=('Arial', 20, 'bold'), bg='#f0e6d2', fg='black')
        title_label.pack(pady=(15, 5))

        # AI Control Frame
        ai_frame = tk.Frame(self.root, bg='#f0e6d2', relief=tk.RAISED, borderwidth=2)
        ai_frame.pack(pady=10)

        tk.Label(ai_frame, text="AI Settings:", font=('Arial', 12, 'bold'),
                 bg='#f0e6d2', fg='black').pack(side=tk.LEFT, padx=10)

        self.ai_button = tk.Button(ai_frame, text="Enable AI (White)",
                                   command=self.toggle_ai,
                                   font=('Arial', 11, 'bold'), bg='#9C27B0', fg='black',
                                   padx=15, pady=5)
        self.ai_button.pack(side=tk.LEFT, padx=5)

        tk.Button(ai_frame, text="Save Agent", command=self.save_agent,
                  font=('Arial', 10), bg='#2196F3', fg='black',
                  padx=10, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(ai_frame, text="Load Agent", command=self.load_agent,
                  font=('Arial', 10), bg='#FF9800', fg='black',
                  padx=10, pady=5).pack(side=tk.LEFT, padx=5)

        self.epsilon_label = tk.Label(ai_frame,
                                      text=f"Epsilon: {self.agent.epsilon:.3f}",
                                      font=('Arial', 10), bg='#f0e6d2', fg='black')
        self.epsilon_label.pack(side=tk.LEFT, padx=10)

        self.memory_label = tk.Label(ai_frame,
                                     text=f"Memory: {len(self.agent.memory)}",
                                     font=('Arial', 10), bg='#f0e6d2', fg='black')
        self.memory_label.pack(side=tk.LEFT, padx=5)

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
        tk.Label(black_frame, text="⚫ Black Player (Human)", font=('Arial', 14, 'bold'),
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
        self.white_label = tk.Label(white_frame, text="⚪ White Player (Human)",
                                    font=('Arial', 14, 'bold'),
                                    bg='#f0e6d2', fg='black')
        self.white_label.pack(pady=5)
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
                                text="Click 'Enable AI' to play against Deep Q-Learning agent.\nThe AI learns from each game! Save the agent to preserve training.\n\n©2025 SLIIT - DL Assignment Group",
                                font=('Arial', 10), bg='#f0e6d2', fg='black', wraplength=600)
        instructions.pack(pady=10)

        self.start_timer()

    def toggle_ai(self):
        """Toggle AI opponent on/off"""
        self.ai_enabled = not self.ai_enabled

        if self.ai_enabled:
            self.ai_button.config(text="Disable AI", bg='#f44336')
            self.white_label.config(text="⚪ White Player (AI)")
            logging.info("AI ENABLED - Agent will play as White")

            # If it's AI's turn, make a move
            if self.current_player == self.ai_player and not self.game_over:
                self.root.after(500, self.ai_make_move)
        else:
            self.ai_button.config(text="Enable AI (White)", bg='#9C27B0')
            self.white_label.config(text="⚪ White Player (Human)")
            logging.info("AI DISABLED")

    def save_agent(self):
        """Save the trained agent to file"""
        self.agent.save()
        messagebox.showinfo("Agent Saved",
                            f"DQN Agent saved successfully!\nMemory size: {len(self.agent.memory)}\nEpsilon: {self.agent.epsilon:.4f}")

    def load_agent(self):
        """Load a trained agent from file"""
        if self.agent.load():
            self.epsilon_label.config(text=f"Epsilon: {self.agent.epsilon:.3f}")
            self.memory_label.config(text=f"Memory: {len(self.agent.memory)}")
            messagebox.showinfo("Agent Loaded",
                                f"DQN Agent loaded successfully!\nMemory size: {len(self.agent.memory)}\nEpsilon: {self.agent.epsilon:.4f}")
        else:
            messagebox.showerror("Load Failed", "Agent file not found!")

    def ai_make_move(self):
        """AI makes a move using the DQN agent"""
        if self.game_over or not self.ai_enabled or self.current_player != self.ai_player:
            return

        logging.info("AI is thinking...")

        # Get current state
        state = self.agent.get_state(self.board, self.ai_player)
        valid_actions = self.agent.get_valid_actions(self.board)

        if not valid_actions:
            logging.warning("No valid moves for AI - passing turn")
            self.pass_turn()
            return

        # Choose action using epsilon-greedy policy
        action = self.agent.act(state, valid_actions)

        if action is None:
            self.pass_turn()
            return

        row = action // self.BOARD_SIZE
        col = action % self.BOARD_SIZE

        logging.info(f"AI chose action {action} -> position ({row}, {col})")

        # Store state and action for learning
        self.previous_state = state
        self.previous_action = action

        # Make the move
        self.place_stone(row, col, is_ai=True)

        # Update UI displays
        self.epsilon_label.config(text=f"Epsilon: {self.agent.epsilon:.3f}")
        self.memory_label.config(text=f"Memory: {len(self.agent.memory)}")

    def calculate_reward(self, captured_count, is_ai_move):
        """
        Calculate reward for reinforcement learning
        Rewards:
        - +10 per captured stone
        - +1 for valid move
        - -10 per stone lost to opponent
        """
        reward = 0

        if is_ai_move:
            reward += captured_count * 10  # Reward for captures
            reward += 1  # Small reward for valid move
        else:
            reward -= captured_count * 10  # Penalty when opponent captures

        return reward

    def set_time(self):
        """Set game timer"""
        try:
            minutes = int(self.time_var.get())
            seconds = minutes * 60
            self.time_limit = {'black': seconds, 'white': seconds}
            self.time_remaining = {'black': seconds, 'white': seconds}
            self.update_timer_display()
            messagebox.showinfo("Time Set", f"Timer set to {minutes} minutes per player")
            logging.info(f"Timer set to {minutes} minutes per player")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid time")
            logging.error("Invalid time input")

    def start_timer(self):
        """Start the game timer"""
        self.timer_running = True
        self.last_time = time.time()
        self.update_timer()

    def update_timer(self):
        """Update timer display and check for timeout"""
        if not self.timer_running or self.game_over:
            return

        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time

        self.time_remaining[self.current_player] -= elapsed

        if self.time_remaining[self.current_player] <= 0:
            self.time_remaining[self.current_player] = 0
            self.game_over = True
            self.timer_running = False
            opponent = 'white' if self.current_player == 'black' else 'black'

            logging.warning(f"TIME OUT: {self.current_player.capitalize()}")
            logging.info(f"GAME OVER: {opponent.capitalize()} wins by timeout")

            # Learn from timeout
            if self.ai_enabled and self.previous_state is not None:
                reward = -50 if self.current_player == self.ai_player else 50
                next_state = self.agent.get_state(self.board, self.ai_player)
                self.agent.remember(self.previous_state, self.previous_action,
                                    reward, next_state, True)
                self.agent.replay()

            messagebox.showinfo("Time Out!",
                                f"{self.current_player.capitalize()} ran out of time!\n{opponent.capitalize()} wins!")
            return

        self.update_timer_display()
        self.root.after(100, self.update_timer)

    def update_timer_display(self):
        """Update timer labels"""
        for player in ['black', 'white']:
            minutes = int(self.time_remaining[player] // 60)
            seconds = int(self.time_remaining[player] % 60)
            time_str = f"Time: {minutes:02d}:{seconds:02d}"

            if player == 'black':
                self.black_timer_label.config(text=time_str)
            else:
                self.white_timer_label.config(text=time_str)

    def resign(self, player):
        """Handle player resignation"""
        if self.game_over:
            return

        result = messagebox.askyesno("Resign",
                                     f"Are you sure {player.capitalize()} wants to resign?")
        if result:
            self.game_over = True
            self.timer_running = False
            opponent = 'white' if player == 'black' else 'black'

            logging.warning(f"RESIGNATION: {player.capitalize()}")
            logging.info(f"GAME OVER: {opponent.capitalize()} wins")
            logging.info(f"Final - Black: {self.captures['black']}, White: {self.captures['white']}")

            # Learn from resignation
            if self.ai_enabled and self.previous_state is not None:
                reward = -100 if player == self.ai_player else 100
                next_state = self.agent.get_state(self.board, self.ai_player)
                self.agent.remember(self.previous_state, self.previous_action,
                                    reward, next_state, True)
                self.agent.replay()
                logging.info("AI learned from game outcome")

            messagebox.showinfo("Game Over",
                                f"{player.capitalize()} resigned.\n{opponent.capitalize()} wins!")

    def pass_turn(self):
        """Pass turn to opponent"""
        if self.game_over:
            return

        opponent = 'white' if self.current_player == 'black' else 'black'
        logging.info(f"Move {self.move_count + 1}: {self.current_player.capitalize()} PASSED")

        messagebox.showinfo("Pass", f"{self.current_player.capitalize()} passed their turn.")

        self.current_player = opponent
        self.turn_label.config(text=f"Current Turn: {self.current_player.capitalize()}")
        self.last_move = None
        self.move_count += 1
        self.draw_board()

        # Trigger AI move if it's AI's turn
        if self.ai_enabled and self.current_player == self.ai_player:
            self.root.after(500, self.ai_make_move)

    def draw_board(self):
        """Draw the Go board"""
        self.canvas.delete('all')

        # Draw grid lines
        for i in range(self.BOARD_SIZE):
            # Horizontal
            x1 = self.PADDING
            y = self.PADDING + i * self.CELL_SIZE
            x2 = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE
            self.canvas.create_line(x1, y, x2, y, fill='#8B4513', width=2)

            # Vertical
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
        """Draw a stone on the board"""
        x = self.PADDING + col * self.CELL_SIZE
        y = self.PADDING + row * self.CELL_SIZE

        fill_color = 'black' if color == 'black' else 'white'
        outline_color = '#333' if color == 'black' else '#666'

        self.canvas.create_oval(x - self.STONE_RADIUS, y - self.STONE_RADIUS,
                                x + self.STONE_RADIUS, y + self.STONE_RADIUS,
                                fill=fill_color, outline=outline_color, width=2)

    def handle_click(self, event):
        """Handle mouse click on board"""
        if self.game_over:
            return

        # Don't allow human moves during AI turn
        if self.ai_enabled and self.current_player == self.ai_player:
            return

        col = round((event.x - self.PADDING) / self.CELL_SIZE)
        row = round((event.y - self.PADDING) / self.CELL_SIZE)

        if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
            self.place_stone(row, col, is_ai=False)

    def place_stone(self, row, col, is_ai=False):
        """Place a stone on the board"""
        if self.board[row][col] is not None:
            logging.warning(f"ILLEGAL: Position ({row}, {col}) occupied")
            return

        # Place stone
        self.board[row][col] = self.current_player
        self.move_count += 1

        player_type = "AI" if is_ai else "Human"
        logging.info(f"Move {self.move_count}: {self.current_player.capitalize()} ({player_type}) -> ({row}, {col})")

        opponent = 'white' if self.current_player == 'black' else 'black'
        captured_count = self.remove_captures(opponent)

        # Check for suicide
        if not self.has_liberties(row, col) and captured_count == 0:
            self.board[row][col] = None
            self.move_count -= 1
            logging.warning(f"ILLEGAL: Suicide move at ({row}, {col})")

            if not is_ai:
                messagebox.showwarning("Illegal Move", "Suicide is not allowed!")
            else:
                # AI made illegal move - strong negative reward
                if self.previous_state is not None:
                    next_state = self.agent.get_state(self.board, self.ai_player)
                    self.agent.remember(self.previous_state, self.previous_action,
                                        -50, next_state, False)
                    self.agent.replay()
                    logging.info("AI penalized for illegal move")
            return

        # Update captures
        if captured_count > 0:
            self.captures[self.current_player] += captured_count
            self.update_captures_display()
            logging.info(f"CAPTURE: {captured_count} {opponent} stone(s)")

        # Learn from the move (for AI)
        if self.ai_enabled and self.previous_state is not None:
            reward = self.calculate_reward(captured_count, is_ai)
            next_state = self.agent.get_state(self.board, self.ai_player)
            self.agent.remember(self.previous_state, self.previous_action,
                                reward, next_state, False)
            self.agent.replay()
            logging.debug(f"AI learning: reward={reward}")

        self.last_move = (row, col)
        self.current_player = opponent
        self.turn_label.config(text=f"Current Turn: {self.current_player.capitalize()}")
        self.draw_board()

        # Trigger AI move if it's AI's turn
        if self.ai_enabled and self.current_player == self.ai_player and not self.game_over:
            self.root.after(500, self.ai_make_move)

    def get_neighbors(self, row, col):
        """Get valid neighboring positions"""
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
        """Get all stones in the same connected group"""
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
        """Check if a stone/group has any liberties (empty adjacent points)"""
        color = self.board[row][col]
        if color is None:
            return True

        group = self.get_group(row, col)

        for r, c in group:
            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr][nc] is None:
                    return True

        return False

    def remove_captures(self, color):
        """Remove captured stones (stones with no liberties)"""
        captured_count = 0

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row][col] == color and not self.has_liberties(row, col):
                    group = self.get_group(row, col)
                    for r, c in group:
                        self.board[r][c] = None
                        captured_count += 1

        return captured_count

    def update_captures_display(self):
        """Update the captures display labels"""
        self.black_captures_label.config(text=f"Captures: {self.captures['black']}")
        self.white_captures_label.config(text=f"Captures: {self.captures['white']}")

    def reset_game(self):
        """Reset the game to initial state"""
        logging.info("=" * 60)
        logging.info("GAME RESET")

        if self.move_count > 0:
            logging.info(f"Previous game: {self.move_count} moves")
            logging.info(f"Captures - Black: {self.captures['black']}, White: {self.captures['white']}")

        self.board = [[None for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = 'black'
        self.captures = {'black': 0, 'white': 0}
        self.last_move = None
        self.game_over = False
        self.move_count = 0
        self.previous_state = None
        self.previous_action = None

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

        logging.info(f"New game started")
        logging.info("=" * 60)
