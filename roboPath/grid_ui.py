import pygame
import sys
import os
import threading
import torch
import numpy as np
from gridworld_env import GridWorldEnv
from dqn_agent import DQNAgent

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (230, 230, 230)
ORANGE = (255, 165, 0)

# Grid settings
CELL_SIZE = 65
MARGIN = 4
GRID_SIZE = 6
GRID_PIXELS = GRID_SIZE * (CELL_SIZE + MARGIN)
BOTTOM_PANEL = 150

# Training graph settings
GRAPH_WIDTH = 800
GRAPH_HEIGHT = 400
GRAPH_MARGIN = 50

WINDOW_SIZE = (GRID_PIXELS + 100, GRID_PIXELS + BOTTOM_PANEL)
TRAINING_WINDOW_SIZE = (GRAPH_WIDTH + 2 * GRAPH_MARGIN, GRAPH_HEIGHT + 2 * GRAPH_MARGIN + BOTTOM_PANEL)

# Global state
stop_requested = False
training_running = False
training_data = {
    "episodes": [],
    "rewards": [],
    "success_rates": [],
    "epsilons": [],
    "losses": []
}


def draw_grid(screen, env):
    """Draw grid with agent, goal, bomb, and obstacles."""
    for i in range(env.n):
        for j in range(env.n):
            color = GRAY
            if (i, j) == env.agent:
                color = BLUE
            elif (i, j) == env.goal:
                color = GREEN
            elif (i, j) == env.bomb:
                color = RED
            elif (i, j) in env.obstacles:
                color = BLACK

            pygame.draw.rect(
                screen, color,
                [(MARGIN + CELL_SIZE) * j + MARGIN,
                 (MARGIN + CELL_SIZE) * i + MARGIN,
                 CELL_SIZE, CELL_SIZE]
            )


def draw_learning_curves(screen, font, title_font):
    """Draw real-time learning curves during training."""
    screen.fill(WHITE)
    
    # Title
    title = title_font.render("DQN Training Progress", True, BLACK)
    screen.blit(title, (GRAPH_MARGIN, 20))
    
    if len(training_data["episodes"]) < 2:
        msg = font.render("Collecting training data...", True, GRAY)
        screen.blit(msg, (GRAPH_MARGIN, GRAPH_MARGIN + 100))
        return
    
    episodes = training_data["episodes"]
    rewards = training_data["rewards"]
    success_rates = training_data["success_rates"]
    
    # Graph background
    graph_rect = pygame.Rect(GRAPH_MARGIN, GRAPH_MARGIN + 30, GRAPH_WIDTH, GRAPH_HEIGHT)
    pygame.draw.rect(screen, LIGHT_GRAY, graph_rect)
    pygame.draw.rect(screen, BLACK, graph_rect, 2)
    
    # Calculate scales
    max_episode = max(episodes) if episodes else 1
    max_reward = max(rewards) if rewards else 1
    min_reward = min(rewards) if rewards else 0
    reward_range = max_reward - min_reward if max_reward != min_reward else 1
    
    # Draw grid lines
    for i in range(5):
        y = GRAPH_MARGIN + 30 + (GRAPH_HEIGHT * i // 4)
        pygame.draw.line(screen, GRAY, (GRAPH_MARGIN, y), (GRAPH_MARGIN + GRAPH_WIDTH, y), 1)
    
    # Draw reward curve
    if len(rewards) > 1:
        points = []
        for ep, rew in zip(episodes, rewards):
            x = GRAPH_MARGIN + int((ep / max_episode) * GRAPH_WIDTH)
            y = GRAPH_MARGIN + 30 + GRAPH_HEIGHT - int(((rew - min_reward) / reward_range) * GRAPH_HEIGHT)
            y = max(GRAPH_MARGIN + 30, min(GRAPH_MARGIN + 30 + GRAPH_HEIGHT, y))
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(screen, BLUE, False, points, 3)
    
    # Draw success rate curve
    if len(success_rates) > 1:
        points = []
        for ep, sr in zip(episodes, success_rates):
            x = GRAPH_MARGIN + int((ep / max_episode) * GRAPH_WIDTH)
            y = GRAPH_MARGIN + 30 + GRAPH_HEIGHT - int((sr / 100) * GRAPH_HEIGHT)
            y = max(GRAPH_MARGIN + 30, min(GRAPH_MARGIN + 30 + GRAPH_HEIGHT, y))
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(screen, GREEN, False, points, 2)

    # Draw epsilon curve
    if "epsilons" in training_data and len(training_data["epsilons"]) > 1:
        epsilons = training_data["epsilons"]
        points = []
        for ep, eps in zip(episodes, epsilons):
            x = GRAPH_MARGIN + int((ep / max_episode) * GRAPH_WIDTH)
            y = GRAPH_MARGIN + 30 + GRAPH_HEIGHT - int(eps * GRAPH_HEIGHT)
            points.append((x, y))
        pygame.draw.lines(screen, ORANGE, False, points, 2)

    # Draw loss curve
    if "losses" in training_data and len(training_data["losses"]) > 1:
        losses = training_data["losses"]
        valid_losses = [l for l in losses if l > 0]
        if valid_losses:
            max_loss = max(valid_losses)
            points = []
            for ep, l in zip(episodes, losses):
                if l > 0:
                    x = GRAPH_MARGIN + int((ep / max_episode) * GRAPH_WIDTH)
                    scaled_loss = np.log1p(l) / np.log1p(max_loss)
                    y = GRAPH_MARGIN + 30 + GRAPH_HEIGHT - int(scaled_loss * GRAPH_HEIGHT)
                    points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(screen, RED, False, points, 2)
    
    # Y-axis labels
    for i in range(5):
        value = min_reward + (reward_range * (4 - i) / 4)
        label = font.render(f"{value:.1f}", True, BLACK)
        y = GRAPH_MARGIN + 30 + (GRAPH_HEIGHT * i // 4)
        screen.blit(label, (5, y - 10))
    
    # X-axis labels
    for i in range(5):
        value = int((max_episode * i / 4))
        label = font.render(f"{value}", True, BLACK)
        x = GRAPH_MARGIN + (GRAPH_WIDTH * i // 4)
        screen.blit(label, (x, GRAPH_MARGIN + 30 + GRAPH_HEIGHT + 5))
    
    # Legend
    legend_x = GRAPH_MARGIN + 10
    legend_y = GRAPH_MARGIN + 50
    
    pygame.draw.line(screen, BLUE, (legend_x, legend_y), (legend_x + 30, legend_y), 3)
    screen.blit(font.render("Avg Reward", True, BLACK), (legend_x + 40, legend_y - 8))
    
    pygame.draw.line(screen, GREEN, (legend_x, legend_y + 25), (legend_x + 30, legend_y + 25), 3)
    screen.blit(font.render("Success Rate", True, BLACK), (legend_x + 40, legend_y + 17))
    
    pygame.draw.line(screen, ORANGE, (legend_x, legend_y + 50), (legend_x + 30, legend_y + 50), 3)
    screen.blit(font.render("Epsilon", True, BLACK), (legend_x + 40, legend_y + 42))
    
    pygame.draw.line(screen, RED, (legend_x, legend_y + 75), (legend_x + 30, legend_y + 75), 3)
    screen.blit(font.render("Loss (log)", True, BLACK), (legend_x + 40, legend_y + 67))


def draw_panel(screen, font, current_mode, message="", training_view=False):
    """Draw bottom panel with buttons and status message."""
    panel_y = (GRAPH_HEIGHT + 2 * GRAPH_MARGIN + 10) if training_view else (GRID_PIXELS + 10)
    panel_height = BOTTOM_PANEL
    window_width = GRAPH_WIDTH + 2 * GRAPH_MARGIN if training_view else GRID_PIXELS + 100
    
    pygame.draw.rect(screen, LIGHT_GRAY, (0, panel_y, window_width, panel_height))
    
    # Mode and message
    text = font.render(f"Mode: {current_mode}", True, BLACK)
    screen.blit(text, (20, panel_y + 10))

    if message:
        msg_text = font.render(message, True, BLUE)
        screen.blit(msg_text, (20, panel_y + 40))

    # Buttons
    button_y = panel_y + 80
    button_w, button_h = 110, 35
    spacing = 10
    start_x = 20

    train_rect = pygame.Rect(start_x, button_y, button_w, button_h)
    eval_rect = pygame.Rect(start_x + button_w + spacing, button_y, button_w, button_h)
    stop_rect = pygame.Rect(start_x + 2 * (button_w + spacing), button_y, button_w, button_h)

    pygame.draw.rect(screen, YELLOW, train_rect)
    pygame.draw.rect(screen, GREEN, eval_rect)
    pygame.draw.rect(screen, RED, stop_rect)

    screen.blit(font.render("Train", True, BLACK), (train_rect.x + 25, train_rect.y + 8))
    screen.blit(font.render("Evaluate", True, BLACK), (eval_rect.x + 12, eval_rect.y + 8))
    screen.blit(font.render("Stop", True, WHITE), (stop_rect.x + 30, stop_rect.y + 8))

    return train_rect, eval_rect, stop_rect


def train_with_visualization(env, agent, episodes=5000):
    """Training function with progress callbacks."""
    global stop_requested, training_data
    
    epsilon = 1.0
    end_epsilon = 0.01
    epsilon_decay = 0.9995
    
    all_rewards = []
    success_window = []
    goal_count = 0

    for ep in range(episodes):
        if stop_requested:
            print("\nTraining stopped by user")
            break
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if stop_requested:
                break
            
            action = agent.select_action(state, epsilon)
            result = env.step(action)

            agent.store_experience(state, action, result.reward, result.next_state, result.done)
            agent.train_step()
            
            state = result.next_state
            total_reward += result.reward
            done = result.done

            if result.info.get("reason") == "goal":
                goal_count += 1
                success_window.append(1)
            elif done:
                success_window.append(0)

        if stop_requested:
            break

        if len(success_window) > 100:
            success_window.pop(0)
        
        epsilon = max(end_epsilon, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        # Update visualization data every 10 episodes
        if ep % 10 == 0:
            avg_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            success_rate = np.mean(success_window) * 100 if success_window else 0
            
            training_data["episodes"].append(ep + 1)
            training_data["rewards"].append(avg_reward)
            training_data["success_rates"].append(success_rate)
            training_data["epsilons"].append(epsilon)
            training_data["losses"].append(agent.last_loss)

        if (ep + 1) % 500 == 0:
            success_rate = np.mean(success_window) * 100 if success_window else 0
            avg_reward = np.mean(all_rewards[-50:])
            print(f"\n[Episode {ep+1:>5}] Avg Reward: {avg_reward:>7.2f} | "
                  f"Success: {success_rate:>5.1f}% | Goals: {goal_count:>5} | Eps: {epsilon:.4f}")

    if not stop_requested:
        torch.save(agent.q_net.state_dict(), "trained_dqn_model.pth")
        print("\nModel saved to trained_dqn_model.pth")


def run_training_thread(episodes):
    """Run training in background thread."""
    global training_running, stop_requested
    
    try:
        training_data["episodes"].clear()
        training_data["rewards"].clear()
        training_data["success_rates"].clear()
        training_data["epsilons"].clear()
        training_data["losses"].clear()
        
        env = GridWorldEnv(grid_size=GRID_SIZE, max_steps=200, use_obstacles=True, fixed_start=True)
        agent = DQNAgent(env, lr=0.001, gamma=0.95, batch_size=128, tau=0.005)
        
        print("\nStarting DQN Training with visualization...")
        train_with_visualization(env, agent, episodes)
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        training_running = False
        print("Training session ended.")


def run_evaluation(screen, font):
    """Run evaluation with manual obstacle placement."""
    global stop_requested
    stop_requested = False

    if not os.path.exists("trained_dqn_model.pth"):
        print("No trained model found. Please train first!")
        return
    
    env = GridWorldEnv(grid_size=GRID_SIZE, max_steps=200, use_obstacles=True, fixed_start=True)
    agent = DQNAgent(env)
    agent.q_net.load_state_dict(torch.load("trained_dqn_model.pth", weights_only=True))
    agent.q_net.eval()

    # Edit mode
    mode = "Edit"
    message = "G=goal, B=bomb, O=obstacle, R=remove | Click to place"
    selected_type = "obstacle"
    editing = True

    while editing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    selected_type = "goal"
                    message = "Goal mode: Click to place"
                elif event.key == pygame.K_b:
                    selected_type = "bomb"
                    message = "Bomb mode: Click to place"
                elif event.key == pygame.K_o:
                    selected_type = "obstacle"
                    message = "Obstacle mode: Click to place"
                elif event.key == pygame.K_r:
                    selected_type = "remove"
                    message = "Remove mode: Click to remove"
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                _, eval_rect, stop_rect = draw_panel(screen, font, mode, message)

                if y < GRID_PIXELS:
                    j = x // (CELL_SIZE + MARGIN)
                    i = y // (CELL_SIZE + MARGIN)
                    
                    if 0 <= i < env.n and 0 <= j < env.n:
                        pos = (i, j)
                        if selected_type == "goal":
                            env.goal = pos
                            message = "Goal placed"
                        elif selected_type == "bomb":
                            env.bomb = pos
                            message = "Bomb placed"
                        elif selected_type == "obstacle":
                            if env.add_obstacle(pos):
                                message = "Obstacle added"
                            else:
                                message = "Cannot add obstacle"
                        elif selected_type == "remove":
                            env.remove_obstacle(pos)
                            message = "Obstacle removed"

                elif eval_rect.collidepoint(x, y):
                    editing = False
                elif stop_rect.collidepoint(x, y):
                    stop_requested = True
                    return

        screen.fill(WHITE)
        draw_grid(screen, env)
        draw_panel(screen, font, mode, f"{selected_type.upper()} | {message}")
        pygame.display.flip()

    # Evaluation mode
    print("\nRunning evaluation...")
    env.agent = (0, 0)
    state = env._get_obs()
    total_reward = 0
    step = 0
    running = True
    
    while running and step < env.max_steps:
        if stop_requested:
            break

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
        action = torch.argmax(q_values).item()

        result = env.step(action)
        state = result.next_state
        total_reward += result.reward
        step += 1

        screen.fill(WHITE)
        draw_grid(screen, env)
        draw_panel(screen, font, "Evaluating", f"Step {step} | Reward {total_reward:.1f}")
        pygame.display.flip()
        pygame.time.wait(100)

        if result.done:
            running = False
            break

    # Show result
    reason = result.info.get("reason", "timeout")
    if reason == "goal":
        result_message = f"SUCCESS in {step} steps"
    elif reason == "obstacle":
        result_message = "FAILED: Hit obstacle"
    elif reason == "bomb":
        result_message = "FAILED: Hit bomb"
    elif reason == "wall":
        result_message = "FAILED: Hit wall"
    else:
        result_message = "TIMEOUT"

    screen.fill(WHITE)
    draw_grid(screen, env)
    draw_panel(screen, font, "Result", result_message)
    pygame.display.flip()
    print(f"\n{result_message}")
    pygame.time.wait(3000)


def run_grid_ui():
    """Main UI loop."""
    global training_running, stop_requested
    
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("GridWorld DQN Visualizer")
    font = pygame.font.SysFont("Arial", 18)
    title_font = pygame.font.SysFont("Arial", 24, bold=True)

    env = GridWorldEnv(grid_size=GRID_SIZE, use_obstacles=True)
    mode = "Ready"
    training_thread = None
    training_view = False

    clock = pygame.time.Clock()

    print("\n" + "="*60)
    print("GridWorld DQN Visualizer")
    print("="*60)
    print("Controls:")
    print("  TRAIN: Start training with learning curves")
    print("  EVALUATE: Test trained agent")
    print("  STOP: Interrupt training/evaluation")
    print("="*60 + "\n")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if training_running:
                    stop_requested = True
                    if training_thread:
                        training_thread.join(timeout=3)
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                train_rect, eval_rect, stop_rect = draw_panel(screen, font, mode, training_view=training_view)

                if train_rect.collidepoint(x, y):
                    if not training_running:
                        mode = "Training"
                        training_running = True
                        training_view = True
                        stop_requested = False
                        
                        screen = pygame.display.set_mode(TRAINING_WINDOW_SIZE)
                        
                        training_thread = threading.Thread(
                            target=run_training_thread,
                            args=(5000,),
                            daemon=True
                        )
                        training_thread.start()
                        print("Training started...")

                elif eval_rect.collidepoint(x, y):
                    if not training_running:
                        stop_requested = False
                        run_evaluation(screen, font)
                        mode = "Ready"
                        screen = pygame.display.set_mode(WINDOW_SIZE)
                        training_view = False

                elif stop_rect.collidepoint(x, y):
                    if training_running:
                        stop_requested = True
                        mode = "Stopping"
                        print("Stop requested...")

        # Check training completion
        if training_thread and not training_thread.is_alive() and training_running:
            training_running = False
            training_view = False
            mode = "Ready"
            stop_requested = False
            screen = pygame.display.set_mode(WINDOW_SIZE)
            print("Ready for next task")

        # Render
        screen.fill(WHITE)
        
        if training_view and training_running:
            draw_learning_curves(screen, font, title_font)
            ep = training_data["episodes"][-1] if training_data["episodes"] else 0
            rew = training_data["rewards"][-1] if training_data["rewards"] else 0
            suc = training_data["success_rates"][-1] if training_data["success_rates"] else 0
            draw_panel(screen, font, f"Training Ep: {ep}", 
                      message=f"Reward: {rew:.2f} | Success: {suc:.1f}%",
                      training_view=True)
        else:
            draw_grid(screen, env)
            draw_panel(screen, font, mode, training_view=False)
        
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    run_grid_ui()