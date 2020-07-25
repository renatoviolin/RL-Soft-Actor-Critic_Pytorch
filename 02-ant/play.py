from sac2019 import SACAgent as SAC
import numpy as np
import os
import torch
import gym
import pybullet_envs
from gym import wrappers

monitor_path = './monitor/'
if not os.path.exists(monitor_path):
    os.makedirs(monitor_path)

env_name = "AntBulletEnv-v0"
env = gym.make(env_name)
max_episode_steps = env._max_episode_steps
env = wrappers.Monitor(env, monitor_path, force=True)
start_timesteps = 10_000
eval_freq = 5_000
max_timesteps = 500_000
batch_size = 100


total_timesteps = 0
episode_reward = 0
episode_timesteps = 0
episode_num = 0
done = False
obs = env.reset()

gamma = 0.99
tau = 0.005
alpha = 0.2
a_lr = 1e-3
q_lr = 1e-3
p_lr = 1e-3
buffer_maxlen = 1_000_000

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)


policy = SAC(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)
policy.load_checkpoint('models/actor', 'models/critic')


def evaluate_policy_deterministic(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.get_action_deterministic(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("\n------------------------------------------")
    print(f"DETERMINISTIC: Evaluation Step: {avg_reward}")
    print("------------------------------------------\n")


evaluate_policy_deterministic(policy)
