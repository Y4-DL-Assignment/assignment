import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import messagebox, filedialog
import os


# ========================
# Deep SARSA Network Definition (matches snake_deep_sarsa.py)
# ========================
class D_SARSA(nn.Module):
    def __init__(self, input_size=16, hidden1=512, hidden2=512, hidden3=256, output_size=3):
        super(D_SARSA, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_size)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ========================
# Academic Network Visualizer
# ========================
class AcademicNetworkVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Deep SARSA Network Architecture Analysis")
        self.geometry("1400x900")
        self.configure(bg="#f5f5f5")

        # Initialize model (Deep SARSA architecture)
        self.model = D_SARSA(input_size=16, hidden1=512, hidden2=512, hidden3=256, output_size=3)
        self.model.eval()
        # Try to automatically load pretrained model if present
        self.model_loaded = False
        self.model_path = None
        # Common candidate paths (script dir / deep_ql folder)
        try_paths = [
            os.path.join(os.path.dirname(__file__), 'deep_sarsa_snake_optimized_20000_10x10.pth'),
            os.path.join(os.path.dirname(__file__), 'deep_sarsa_snake_optimized_20000_10x10.pt'),
            'deep_sarsa_snake_optimized_20000_10x10.pth',
            os.path.join('deep_ql', 'deep_sarsa_snake_optimized_20000_10x10.pth'),
            os.path.join('.', 'deep_ql', 'deep_sarsa_snake_optimized_20000_10x10.pth')
        ]
        for p in try_paths:
            try:
                if p and os.path.exists(p):
                    data = torch.load(p, map_location='cpu')
                    # if it's a state_dict
                    if isinstance(data, dict) and not any(k.startswith('__') for k in data.keys()):
                        # try to detect if it contains keys for model or full checkpoint
                        if 'state_dict' in data:
                            state = data['state_dict']
                        else:
                            state = data
                        # load state dict (handle wrapped keys)
                        try:
                            self.model.load_state_dict(state)
                        except Exception:
                            # try to remap keys if they are prefixed
                            new_state = {k.replace('model.', ''): v for k, v in state.items()}
                            self.model.load_state_dict(new_state)
                    else:
                        # might be a full model saved with torch.save(model)
                        try:
                            # attempt load into model weights if possible
                            self.model.load_state_dict(data.state_dict())
                        except Exception:
                            # as a last resort, try to replace model (not ideal)
                            try:
                                self.model = data
                            except Exception:
                                pass
                    self.model.eval()
                    self.model_loaded = True
                    self.model_path = p
                    break
            except Exception:
                continue

        # Feature and action definitions
        self.input_features = [
            'Danger↑', 'Danger↓', 'Danger←', 'Danger→',
            'Danger↖', 'Danger↗', 'Danger↙', 'Danger↘',
            'Dir↑', 'Dir↓', 'Dir←', 'Dir→',
            'Food↑', 'Food↓', 'Food←', 'Food→'
        ]
        self.action_names = ['Straight', 'Turn Right', 'Turn Left']

        # Setup UI
        self._setup_header()
        self._setup_content()
        self._generate_sample_state()

    def _setup_header(self):
        """Create academic header section"""
        header = tk.Frame(self, bg="#1a1a1a", height=80)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="Deep SARSA Network Architecture Analysis",
            font=("Helvetica", 18, "bold"),
            bg="#1a1a1a",
            fg="#ffffff"
        )
        title.pack(pady=10)

        subtitle = tk.Label(
            header,
            text="State-Action Value Function | Snake Game Learning Agent",
            font=("Helvetica", 11),
            bg="#1a1a1a",
            fg="#aaaaaa"
        )
        subtitle.pack()

        # Model status label and load button
        status_frame = tk.Frame(header, bg="#1a1a1a")
        status_frame.pack(side=tk.RIGHT, padx=12)
        self.model_status_lbl = tk.Label(status_frame, text="Model: (none)", bg="#1a1a1a", fg="#bfeec9", font=("Helvetica", 9))
        self.model_status_lbl.pack(side=tk.LEFT, padx=(0,8))
        if self.model_loaded and self.model_path:
            self.model_status_lbl.config(text=f"Model: {os.path.basename(self.model_path)}")

    def _setup_content(self):
        """Create main content areas"""
        main_container = tk.Frame(self, bg="#f5f5f5")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left column - Input and Output
        left_column = tk.Frame(main_container, bg="#f5f5f5")
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._create_input_panel(left_column)
        self._create_output_panel(left_column)

        # Right column - Network visualization
        right_column = tk.Frame(main_container, bg="#f5f5f5")
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._create_network_panel(right_column)

        # Bottom - Controls
        self._create_controls()

    def _create_input_panel(self, parent):
        """Input state features panel"""
        panel = tk.LabelFrame(
            parent,
            text="Input State (16 features)",
            font=("Helvetica", 11, "bold"),
            bg="#ffffff",
            fg="#000000",
            padx=10,
            pady=10
        )
        panel.pack(fill=tk.X, pady=(0, 10))

        # Input grid (4x4)
        self.input_labels = []
        for i in range(16):
            row = i // 4
            col = i % 4

            frame = tk.Frame(panel, bg="#f0f0f0", relief=tk.SOLID, bd=1)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            feature_name = tk.Label(
                frame,
                text=self.input_features[i],
                font=("Helvetica", 9),
                bg="#f0f0f0",
                fg="#333333"
            )
            feature_name.pack(pady=(3, 0))

            value_label = tk.Label(
                frame,
                text="0",
                font=("Courier", 12, "bold"),
                bg="#f0f0f0",
                fg="#0066cc"
            )
            value_label.pack(pady=(0, 3))

            self.input_labels.append((frame, value_label))

        # Configure grid weights
        for i in range(4):
            panel.grid_columnconfigure(i, weight=1)

    def _create_output_panel(self, parent):
        """Output Q-values panel"""
        panel = tk.LabelFrame(
            parent,
            text="Output Q-Values (3 actions)",
            font=("Helvetica", 11, "bold"),
            bg="#ffffff",
            fg="#000000",
            padx=10,
            pady=10
        )
        panel.pack(fill=tk.X)

        self.output_labels = []
        self.output_bars = []
        colors = ["#3b82f6", "#10b981", "#f59e0b"]

        for i, (action, color) in enumerate(zip(self.action_names, colors)):
            action_frame = tk.Frame(panel, bg="#ffffff")
            action_frame.pack(fill=tk.X, pady=8)

            # Label
            label = tk.Label(
                action_frame,
                text=f"{action}:",
                font=("Helvetica", 10, "bold"),
                bg="#ffffff",
                fg="#000000",
                width=12,
                anchor="w"
            )
            label.pack(side=tk.LEFT, padx=(0, 10))

            # Bar container
            bar_container = tk.Frame(action_frame, bg="#e5e5e5", height=30)
            bar_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            bar_container.pack_propagate(False)

            # Value bar
            bar = tk.Frame(bar_container, bg=color, height=30)
            bar.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

            # Value text
            value_label = tk.Label(
                action_frame,
                text="0.00",
                font=("Courier", 10, "bold"),
                bg="#ffffff",
                fg="#000000",
                width=7,
                anchor="e"
            )
            value_label.pack(side=tk.LEFT, padx=(10, 0))

            self.output_labels.append(value_label)
            self.output_bars.append((bar_container, bar, color))

    def _create_network_panel(self, parent):
        """Network architecture visualization"""
        panel = tk.LabelFrame(
            parent,
            text="Network Architecture & Activations",
            font=("Helvetica", 11, "bold"),
            bg="#ffffff",
            fg="#000000",
            padx=5,
            pady=5
        )
        panel.pack(fill=tk.BOTH, expand=True)

        # Canvas for drawing network
        self.network_canvas = tk.Canvas(
            panel,
            bg="#fafafa",
            highlightthickness=1,
            highlightbackground="#cccccc",
            relief=tk.SOLID,
            bd=1
        )
        self.network_canvas.pack(fill=tk.BOTH, expand=True)
        self.network_canvas.bind("<Configure>", self._on_canvas_resize)

    def _create_controls(self):
        """Control panel"""
        controls = tk.Frame(self, bg="#f5f5f5", height=50)
        controls.pack(fill=tk.X, padx=15, pady=(10, 15))
        controls.pack_propagate(False)

        btn_frame = tk.Frame(controls, bg="#f5f5f5")
        btn_frame.pack(side=tk.LEFT)

        random_btn = tk.Button(
            btn_frame,
            text="Random Input",
            command=self._generate_random_state,
            font=("Helvetica", 10),
            bg="#666666",
            fg="black",
            padx=15,
            pady=8,
            relief=tk.RAISED,
            bd=2
        )
        random_btn.pack(side=tk.LEFT, padx=5)

        info_label = tk.Label(
            controls,
            text="Observe how input features activate neurons and produce Q-values for each action",
            font=("Helvetica", 9, "italic"),
            bg="#f5f5f5",
            fg="#666666"
        )
        info_label.pack(side=tk.RIGHT, padx=10)

    def _generate_sample_state(self):
        """Generate a typical danger scenario"""
        state = np.array([
            1, 0, 1, 0,  # Dangers
            0, 0, 0, 0,
            0, 0, 1, 0,  # Direction (left)
            0, 1, 0, 0  # Food (down)
        ], dtype=np.float32)
        self._compute_and_visualize(state)

    def _generate_random_state(self):
        """Generate random binary state"""
        state = np.random.randint(0, 2, 16).astype(np.float32)
        self._compute_and_visualize(state)

    def _compute_and_visualize(self, state):
        """Forward pass and visualization"""
        # Forward pass with layer activations
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            # Manual forward pass to capture activations
            x = state_tensor
            h1 = torch.relu(self.model.fc1(x))
            h1_drop = self.model.dropout(h1)
            h2 = torch.relu(self.model.fc2(h1_drop))
            h2_drop = self.model.dropout(h2)
            h3 = torch.relu(self.model.fc3(h2_drop))
            output = self.model.fc4(h3)

        # Store activations
        self.activations = {
            'input': state,
            'h1': h1.squeeze().numpy(),
            'h2': h2.squeeze().numpy(),
            'h3': h3.squeeze().numpy(),
            'output': output.squeeze().numpy()
        }

        # Update visualizations
        self._update_input_display(state)
        self._update_output_display(output.squeeze().numpy())
        self._draw_network_architecture()

    def _update_input_display(self, state):
        """Update input feature display"""
        for i, (frame, label) in enumerate(self.input_labels):
            val = int(state[i])
            label.config(text=str(val))

            if val == 1:
                frame.config(bg="#cce5ff")
                label.config(bg="#cce5ff", fg="#0033aa")
            else:
                frame.config(bg="#f0f0f0")
                label.config(bg="#f0f0f0", fg="#999999")

    def _update_output_display(self, q_values):
        """Update output Q-values display"""
        max_q = np.max(q_values)
        min_q = np.min(q_values)
        range_q = max_q - min_q if max_q > min_q else 1.0

        for i, (label, (container, bar, color)) in enumerate(zip(self.output_labels, self.output_bars)):
            q_val = q_values[i]
            label.config(text=f"{q_val:.3f}")

            # Normalize for bar width
            normalized = (q_val - min_q) / range_q if range_q > 0 else 0
            normalized = max(0, min(1, normalized))

            # Update bar width
            container_width = container.winfo_width()
            if container_width > 1:
                bar.config(width=int(container_width * normalized))

    def _draw_network_architecture(self):
        """Draw the network graph with neuron activations"""
        canvas = self.network_canvas
        canvas.delete("all")

        w = canvas.winfo_width()
        h = canvas.winfo_height()

        if w < 2 or h < 2:
            return

        # Layer info: (name, size, max_display)
        layers = [
            ("Input\n(16)", 16, 16),
            ("Hidden 1\n(512)", 512, 12),
            ("Hidden 2\n(512)", 512, 12),
            ("Hidden 3\n(256)", 256, 10),
            ("Output\n(3)", 3, 3)
        ]

        activation_data = [
            self.activations['input'],
            self.activations['h1'],
            self.activations['h2'],
            self.activations['h3'],
            self.activations['output']
        ]

        # Calculate positions
        x_spacing = w / (len(layers) + 1)
        layer_positions = []

        for layer_idx, ((name, total_size, display_size), activations) in enumerate(zip(layers, activation_data)):
            x = int((layer_idx + 1) * x_spacing)
            layer_pos = []

            # Select which neurons to display
            if display_size >= total_size:
                indices = list(range(total_size))
            else:
                indices = np.linspace(0, total_size - 1, display_size, dtype=int)

            # Normalize activations
            if len(activations) > 0:
                act_values = activations[indices]
                min_act = np.min(act_values)
                max_act = np.max(act_values)
                norm_act = (act_values - min_act) / (max_act - min_act + 1e-6) if max_act > min_act else act_values
            else:
                norm_act = np.zeros(display_size)

            # Vertical spacing
            y_spacing = h / (display_size + 2)

            for neuron_idx, (act_idx, norm_val) in enumerate(zip(indices, norm_act)):
                y = int((neuron_idx + 1) * y_spacing)
                layer_pos.append((x, y, float(norm_val), int(act_idx)))

            layer_positions.append((layer_pos, name))

        # Draw connections
        for layer_idx in range(len(layer_positions) - 1):
            curr_layer, _ = layer_positions[layer_idx]
            next_layer, _ = layer_positions[layer_idx + 1]

            for x1, y1, act1, _ in curr_layer:
                for x2, y2, act2, _ in next_layer:
                    # Connection color based on activation
                    strength = (act1 + act2) / 2
                    color = self._get_color_gradient(strength)
                    canvas.create_line(x1, y1, x2, y2, fill=color, width=0.5, dash=(2, 2))

        # Draw neurons
        for layer_idx, (layer_pos, name) in enumerate(layer_positions):
            # Layer name
            x_pos = layer_pos[0][0] if layer_pos else 0
            canvas.create_text(
                x_pos, 15,
                text=name,
                font=("Helvetica", 9, "bold"),
                fill="#333333"
            )

            # Neurons
            for x, y, norm_val, neuron_idx in layer_pos:
                radius = 4 + int(6 * norm_val)

                # Color by layer
                if layer_idx == 0:
                    color = "#3b82f6"
                elif layer_idx == len(layer_positions) - 1:
                    color = "#ef4444"
                else:
                    color = "#10b981"

                # Alpha blend by activation
                canvas.create_oval(
                    x - radius, y - radius,
                    x + radius, y + radius,
                    fill=color,
                    outline="#333333",
                    width=1
                )

    def _get_color_gradient(self, value):
        """Get color from gradient based on activation value"""
        # Blue (low) -> Green (medium) -> Red (high)
        if value < 0.5:
            r = int(59 + (16 - 59) * (value * 2))
            g = int(130 + (176 - 130) * (value * 2))
            b = int(246 - (246 - 52) * (value * 2))
        else:
            r = int(16 + (239 - 16) * ((value - 0.5) * 2))
            g = int(176 + (78 - 176) * ((value - 0.5) * 2))
            b = int(52 + (78 - 52) * ((value - 0.5) * 2))

        return f'#{r:02x}{g:02x}{b:02x}'

    def _on_canvas_resize(self, event):
        """Redraw when canvas is resized"""
        if hasattr(self, 'activations'):
            self._draw_network_architecture()

    def _on_load_model(self):
        """Load a different model file"""
        file_path = tk.filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Files", "*.pth *.pt"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        # Update status label
        self.model_status_lbl.config(text="Model: (loading...)", fg="#ffcc00")

        # Load the selected model file
        try:
            data = torch.load(file_path, map_location='cpu')
            # if it's a state_dict
            if isinstance(data, dict) and not any(k.startswith('__') for k in data.keys()):
                # try to detect if it contains keys for model or full checkpoint
                if 'state_dict' in data:
                    state = data['state_dict']
                else:
                    state = data
                # load state dict (handle wrapped keys)
                try:
                    self.model.load_state_dict(state)
                except Exception:
                    # try to remap keys if they are prefixed
                    new_state = {k.replace('model.', ''): v for k, v in state.items()}
                    self.model.load_state_dict(new_state)
            else:
                # might be a full model saved with torch.save(model)
                try:
                    # attempt load into model weights if possible
                    self.model.load_state_dict(data.state_dict())
                except Exception:
                    # as a last resort, try to replace model (not ideal)
                    try:
                        self.model = data
                    except Exception:
                        pass
            self.model.eval()
            self.model_loaded = True
            self.model_path = file_path
            self.model_status_lbl.config(text=f"Model: {os.path.basename(file_path)}", fg="#bfeec9")
        except Exception as e:
            self.model_status_lbl.config(text="Model: (error)", fg="#ff4444")
            messagebox.showerror("Model Load Error", f"Failed to load model file:\n{e}")

        # Redraw network architecture if activations are available
        if hasattr(self, 'activations'):
            self._draw_network_architecture()


if __name__ == "__main__":
    app = AcademicNetworkVisualizer()
    app.mainloop()