# import numpy as np
# from tqdm import trange


# def train_dqn(
#     env,
#     agent,
#     episodes=20000,
#     start_epsilon=1.0,
#     end_epsilon=0.01,
#     epsilon_decay=0.9995,
#     print_freq=500,
#     save_path='trained_gridworld_dqn.pth',
#     progress_callback=None,
#     stop_check=None
# ):
#     """
#     Train DQN agent on GridWorld environment.
    
#     Args:
#         env: GridWorld environment
#         agent: DQNAgent instance
#         episodes: Number of training episodes
#         start_epsilon: Initial exploration rate
#         end_epsilon: Final exploration rate
#         epsilon_decay: Epsilon decay factor per episode
#         print_freq: Print statistics every N episodes
#         save_path: Path to save trained model
#         progress_callback: Optional callback(episode, progress_percent) for progress updates
#         stop_check: Optional callable that returns True to stop training early
        
#     Returns:
#         training_data: Dictionary with training metrics
#     """
#     epsilon = start_epsilon
#     all_rewards = []
#     all_scores = []
#     success_window = []
#     goal_count = 0
#     epsilons = []
#     losses = []

#     print(f"\n{'='*70}")
#     print(f"Training DQN on GridWorld (Snake-Style)")
#     print(f"{'='*70}")
#     print(f"Episodes: {episodes} | Grid: {env.n}×{env.n}")
#     print(f"State dim: 20 | Actions: 4 | Max steps: {env.max_steps}")
#     print(f"Epsilon: {start_epsilon} → {end_epsilon} (decay: {epsilon_decay})")
#     print(f"{'='*70}\n")

#     for ep in trange(episodes, desc="Training", ncols=80):
#         # Check for early stopping
#         if stop_check and stop_check():
#             print("\nTraining stopped by user")
#             break

#         state = env.reset()
#         total_reward = 0
#         done = False
#         episode_goals = 0

#         while not done:
#             # Check for early stopping during episode
#             if stop_check and stop_check():
#                 break

#             # Select and execute action
#             action = agent.select_action(state, epsilon)
#             result = env.step(action)

#             # Store experience and train
#             agent.store_experience(state, action, result.reward, result.next_state, result.done)
#             agent.train_step()
            
#             state = result.next_state
#             total_reward += result.reward
#             done = result.done

#             # Track goal achievements
#             if result.info.get("reason") == "goal":
#                 goal_count += 1
#                 episode_goals += 1

#         # Early stop check after episode
#         if stop_check and stop_check():
#             break

#         # Track episode success
#         if episode_goals > 0:
#             success_window.append(1)
#         else:
#             success_window.append(0)

#         # Maintain window size
#         if len(success_window) > 100:
#             success_window.pop(0)
        
#         # Decay epsilon
#         epsilon = max(end_epsilon, epsilon * epsilon_decay)
        
#         # Track metrics
#         all_rewards.append(total_reward)
#         all_scores.append(episode_goals)
#         epsilons.append(epsilon)
#         losses.append(agent.last_loss)

#         # Progress callback
#         if progress_callback:
#             progress_percent = (ep + 1) / episodes * 100
#             progress_callback(ep + 1, progress_percent)

#         # Progress reporting
#         if (ep + 1) % print_freq == 0:
#             success_rate = np.mean(success_window) * 100 if success_window else 0
#             avg_reward = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
#             avg_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
#             avg_loss = np.mean([l for l in losses[-100:] if l > 0]) if losses else 0
            
#             print(f"\n[Episode {ep+1:>5}] "
#                   f"Avg Reward: {avg_reward:>7.2f} | "
#                   f"Avg Score: {avg_score:>5.2f} | "
#                   f"Success Rate: {success_rate:>5.1f}% | "
#                   f"Goals: {goal_count:>6} | "
#                   f"ε: {epsilon:.4f} | "
#                   f"Loss: {avg_loss:.4f}")

#     # Save model (only if not stopped early or if stopped gracefully)
#     if not (stop_check and stop_check()):
#         agent.save(save_path)
#     else:
#         print("\nTraining stopped - model not saved")

#     # Final statistics
#     final_success_rate = np.mean(success_window) * 100 if success_window else 0
#     final_avg_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
    
#     print(f"\n{'='*70}")
#     print(f"Training Complete!")
#     print(f"{'='*70}")
#     if all_rewards:
#         print(f"Final Avg Reward: {np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards):.2f}")
#     print(f"Final Avg Score: {final_avg_score:.2f} goals per episode")
#     print(f"Final Success Rate: {final_success_rate:.1f}%")
#     print(f"Total Goals Reached: {goal_count}")
#     print(f"{'='*70}\n")

#     return {
#         "episodes": list(range(1, len(all_rewards) + 1)),
#         "rewards": all_rewards,
#         "scores": all_scores,
#         "epsilons": epsilons,
#         "losses": losses,
#         "success_rates": [np.mean(success_window[max(0, i-99):i+1]) * 100 
#                           for i in range(len(success_window))]
#     }




"""
Training function for GridWorld DQN
Now with progress callbacks for GUI integration
"""

import numpy as np
from tqdm import trange


def train_dqn(
    env,
    agent,
    episodes=25000,
    start_epsilon=1.0,
    end_epsilon=0.01,
    epsilon_decay=0.9997,
    print_freq=500,
    save_path='trained_gridworld_dqn.pth',
    progress_callback=None,
    stop_check=None
):
    """
    Train DQN agent on GridWorld environment.
    
    Args:
        env: GridWorld environment
        agent: DQNAgent instance
        episodes: Number of training episodes
        start_epsilon: Initial exploration rate
        end_epsilon: Final exploration rate
        epsilon_decay: Epsilon decay factor per episode
        print_freq: Print statistics every N episodes
        save_path: Path to save trained model
        progress_callback: Optional function(episode, progress_percent) for GUI updates
        stop_check: Optional function() -> bool to check if training should stop
        
    Returns:
        training_data: Dictionary with training metrics
    """
    epsilon = start_epsilon
    all_rewards = []
    all_scores = []
    success_window = []
    goal_count = 0
    epsilons = []
    losses = []

    print(f"\n{'='*70}")
    print(f"Training DQN on GridWorld (Snake-Style with Moving Obstacles)")
    print(f"{'='*70}")
    print(f"Episodes: {episodes} | Grid: {env.n}×{env.n}")
    print(f"State dim: 20 | Actions: 4 | Max steps: {env.max_steps}")
    print(f"Obstacle move frequency: Every {env.obstacle_move_freq} steps")
    print(f"Epsilon: {start_epsilon} → {end_epsilon} (decay: {epsilon_decay})")
    print(f"{'='*70}\n")

    for ep in trange(episodes, desc="Training", ncols=80):
        # Check if training should stop (from GUI stop button)
        if stop_check and stop_check():
            print("\n⏸️ Training stopped by user")
            break
        
        state = env.reset()
        total_reward = 0
        done = False
        episode_goals = 0

        while not done:
            # Check stop again during episode
            if stop_check and stop_check():
                break
            
            # Select and execute action
            action = agent.select_action(state, epsilon)
            result = env.step(action)

            # Store experience and train
            agent.store_experience(state, action, result.reward, result.next_state, result.done)
            agent.train_step()
            
            state = result.next_state
            total_reward += result.reward
            done = result.done

            # Track goal achievements
            if result.info.get("reason") == "goal":
                goal_count += 1
                episode_goals += 1

        # Track episode success
        if episode_goals > 0:
            success_window.append(1)
        else:
            success_window.append(0)

        # Maintain window size
        if len(success_window) > 100:
            success_window.pop(0)
        
        # Decay epsilon
        epsilon = max(end_epsilon, epsilon * epsilon_decay)
        
        # Track metrics
        all_rewards.append(total_reward)
        all_scores.append(episode_goals)
        epsilons.append(epsilon)
        losses.append(agent.last_loss)

        # Update progress callback for GUI
        if progress_callback:
            progress_percent = (ep + 1) / episodes * 100
            progress_callback(ep + 1, progress_percent)

        # Progress reporting
        if (ep + 1) % print_freq == 0:
            success_rate = np.mean(success_window) * 100 if success_window else 0
            avg_reward = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            avg_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
            avg_loss = np.mean([l for l in losses[-100:] if l > 0]) if losses else 0
            
            print(f"\n[Episode {ep+1:>5}] "
                  f"Avg Reward: {avg_reward:>7.2f} | "
                  f"Avg Score: {avg_score:>5.2f} | "
                  f"Success Rate: {success_rate:>5.1f}% | "
                  f"Goals: {goal_count:>6} | "
                  f"ε: {epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f}")

    # Save model only if training completed
    if not (stop_check and stop_check()):
        agent.save(save_path)

    # Final statistics
    final_success_rate = np.mean(success_window) * 100 if success_window else 0
    final_avg_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Final Avg Reward: {np.mean(all_rewards[-100:]):.2f}")
    print(f"Final Avg Score: {final_avg_score:.2f} goals per episode")
    print(f"Final Success Rate: {final_success_rate:.1f}%")
    print(f"Total Goals Reached: {goal_count}")
    print(f"{'='*70}\n")

    return {
        "episodes": list(range(1, len(all_rewards) + 1)),
        "rewards": all_rewards,
        "scores": all_scores,
        "epsilons": epsilons,
        "losses": losses,
        "success_rates": [np.mean(success_window[max(0, i-99):i+1]) * 100 
                          for i in range(len(success_window))]
    }