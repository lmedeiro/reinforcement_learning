"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import time
import gym
import numpy as np
import argparse
import wimblepong
from wimblepong import AlmaNegritaTools as agent_tools
import pandas as pd
from pdb import set_trace as bp
import torch
import torchvision
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--buffer_size_multiplier", type=int, help="Buffer size multiplier", default=4)
parser.add_argument("--batch_size", type=int, help="batch size", default=32)
parser.add_argument("--model_type", type=str, help="Model Type", default='NN')
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongSimpleAI-v0", visual=True)
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 100000
training_session_time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
print_interval = 200
# Define the player
player_id = 1
# bp()
observation = env.reset()
# bp()
action_space_dim = env.action_space.n
cnn_in_channels = 1
# We used different values for resulting image, however, 150x150 seem to produce the best results.
desired_size = [cnn_in_channels, 150, 150]
observation_space_dim = np.array(desired_size)
batch_size = args.batch_size
model_type = args.model_type

buffer_size = batch_size * args.buffer_size_multiplier
tf_monitoring = False
training_settings = "buffer_size: {}, batch_size: {}, model_type: {}".format(buffer_size, batch_size, model_type)
print(training_settings)
training_settings_file_name = './training_logs/training_session_settings_' + training_session_time_stamp + '.txt'
with open(training_settings_file_name, 'w') as f:
    data = training_settings
    f.write(data)

policy = agent_tools.ProximalPolicyOpt(observation_space_dim, action_space_dim, train_device='cuda',
                                       tf_monitoring=tf_monitoring, cnn_in_channels=cnn_in_channels,
                                       model_type=model_type)
player = agent_tools.Agent(policy, desired_observation_size=desired_size[1:], batch_size=batch_size,
                           buffer_size=buffer_size, in_channels=cnn_in_channels)

env.set_names(p1="Alma_Negrita")

# Arrays to keep track of rewards
reward_history, timestep_history = [], []
average_reward_history = pd.DataFrame(columns=['episode_number','avg_reward_100'])


# Housekeeping
states = []
win1 = 0
# The main training loop was borrowed from the previous exercises.
for episode_number in range(0, episodes):
    # observation = env.reset()
    done = False
    reward_sum, timesteps = 0, 0
    # bp()
    while not done:
        action, action_probabilities, state_value = player.get_action(observation)
        next_observation, rew1, done, info = env.step(action)
        if args.housekeeping:
            states.append(observation)
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        reward_sum += rew1
        player.store_outcome(observation, action_probabilities, action, rew1, state_value)
        timesteps += 1
        observation = next_observation

        if done:
            observation = env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            if episode_number % 5 == 4:
                env.switch_sides()
    reward_history.append(reward_sum)
    timestep_history.append(timesteps)
    # Below we choose to update after the end of each episode, if we have enough batches.
    if len(player.states) >= batch_size:
        player.episode_update(episode_number=episode_number, global_timestep=timesteps)
    if episode_number > 100:
        avg = np.mean(reward_history[-100:])
    else:
        avg = np.mean(reward_history)
    average_reward_history.loc[average_reward_history.index.size] = [episode_number, avg]
    if episode_number % print_interval == 0:
        print("episode {} over. Broken WR: {:.3f}. Avg reward: {}".format(episode_number,
                                                                          win1 / (episode_number + 1),
                                                                          avg))
        if tf_monitoring:
            policy.writer.add_scalar('loss/avg_rewards',
                                     avg,
                                     global_step=episode_number / print_interval)
        average_reward_history.to_pickle('training_logs/avg_reward_history_' + training_session_time_stamp + '.pckl')
        player.save_policy_parameters(unique_name=training_session_time_stamp)



