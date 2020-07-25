from sac2019 import SACAgent as SAC
import numpy as np
import os
import torch
import gym
import pybullet_envs
from gym import wrappers

if not os.path.exists("./models"):
    os.makedirs("./models")


def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("\n------------------------------------------")
    print(f"SAMPLE: Evaluation Step: {avg_reward}")
    print("------------------------------------------\n")


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


env_name = "AntBulletEnv-v0"
env = gym.make(env_name)
start_timesteps = 10_000
eval_freq = 5_000
max_timesteps = 500_000
batch_size = 100
max_episode_steps = env._max_episode_steps


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

while total_timesteps < max_timesteps:

    if total_timesteps < start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.get_action(obs)

    new_obs, reward, done, _ = env.step(action)
    episode_reward += reward
    done_bool = 0.0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
    policy.replay_buffer.add(obs, action, reward, new_obs, done_bool)
    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1

    if done:
        if total_timesteps >= start_timesteps:
            policy.train(episode_timesteps, batch_size)
        print("Total Timesteps: {} Episode Timesteps {} Episode Num: {} Reward: {}".format(total_timesteps, episode_timesteps, episode_num, episode_reward))
        obs = env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if total_timesteps % eval_freq == 0:
        evaluate_policy(policy)
        evaluate_policy_deterministic(policy)
        policy.save_checkpoint('models/actor', 'models/critic')
