import numpy as np
import torch
from tqdm import trange
from agents.dqn_agent import DQNAgent
from env.gridworld_env import GridWorldEnv


def train_dqn(
    env,
    agent,
    episodes=10000,
    start_epsilon=1.0,
    end_epsilon=0.01,
    epsilon_decay=0.9995,
    print_freq=500
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
        
    Returns:
        all_rewards: List of episode rewards
        success_rate: Final success rate
    """
    epsilon = start_epsilon
    all_rewards = []
    success_window = []
    goal_count = 0

    print(f"\nStarting DQN Training")
    print(f"Episodes: {episodes} | Grid: {env.n}x{env.n}")
    print(f"Epsilon: {start_epsilon} -> {end_epsilon} (decay: {epsilon_decay})\n")

    for ep in trange(episodes, desc="Training", ncols=80):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
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
                success_window.append(1)
            elif done:
                success_window.append(0)

        # Maintain window size
        if len(success_window) > 100:
            success_window.pop(0)
        
        # Decay epsilon
        epsilon = max(end_epsilon, epsilon * epsilon_decay)
        
        # Track rewards
        all_rewards.append(total_reward)

        # Progress reporting
        if (ep + 1) % print_freq == 0:
            success_rate = np.mean(success_window) * 100 if success_window else 0
            avg_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            print(f"\n[Episode {ep+1:>5}] Avg Reward: {avg_reward:>7.2f} | "
                  f"Success Rate: {success_rate:>5.1f}% | Goals: {goal_count:>5} | "
                  f"Epsilon: {epsilon:.4f}")

    # Final statistics
    final_success_rate = np.mean(success_window) * 100 if success_window else 0
    print(f"\nTraining Complete!")
    print(f"Final Avg Reward: {np.mean(all_rewards[-50:]):.2f}")
    print(f"Final Success Rate: {final_success_rate:.1f}%")
    print(f"Total Goals Reached: {goal_count}\n")

    return all_rewards, final_success_rate