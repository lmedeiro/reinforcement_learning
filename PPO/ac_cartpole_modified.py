import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from ac_agent import Agent, Policy
import AlmaNegritaToolsCartpole as alma
from cp_cont import CartPoleEnv
import pandas as pd

from pdb import set_trace as bp

print_interval = 200
batch_size = 32
buffer_size = 128
# It seems to be crucial for PPO to not look too far back for learning.
render = False
tf_monitoring = False


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_episodes=5000):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.n
    observation_space_dim = env.observation_space.shape[-1]
    # bp()
    # Instantiate agent and its policy
    # policy = Policy(observation_space_dim, action_space_dim, train_device='cuda')
    # agent = Agent(policy)
    policy = alma.ProximalPolicyOpt(observation_space_dim, action_space_dim, tf_monitoring=tf_monitoring)
    agent = alma.Agent(policy, buffer_size=buffer_size, batch_size=batch_size)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()
        # bp()
        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities, state_value = agent.get_action(observation)
            # Perform the action on the environment, get new state and reward
            next_observation, reward, done, info = env.step(int(action.cpu().detach().float()))
            agent.store_outcome(observation, action_probabilities, action, reward, state_value)
            observation = next_observation

            # if done:
            #     bp()
            if render:
                env.render()
            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if print_things and episode_number % print_interval == 0:
            print("Episode {} finished. Total avg reward: {:.3g} ({} timesteps)"
                  .format(episode_number, np.mean(reward_history[len(reward_history) - print_interval:]), timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Let the agent do its magic (update the policy)
        if len(agent.states) >= batch_size:
            # bp()
            agent.episode_update(episode_number)

    # Training is finished - plot rewards
    if print_things:
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history")
        plt.savefig('alma_negrita_test.png')
        plt.show()
        print("Training finished.")
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["PG"]*len(reward_history),
                         "reward": reward_history})
    data.to_pickle('alma_negrita_test.pckl')
    # torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % (env_name, train_run_id))
    return data


# Function to test a trained policy
def test(env_name, episodes, params, render):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(params)
    agent = Agent(policy)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=5000, help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(args.env, train_episodes=args.train_episodes)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, args.render_test)

