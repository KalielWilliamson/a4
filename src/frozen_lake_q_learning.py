#!/usr/bin/env python
# coding: utf-8
import json
import time
import gym
import tqdm
import itertools
from pylab import *

env = gym.make('FrozenLake8x8-v1')


def low_pass(data, alpha=0.99):
    low_pass = [data[0]]
    for i in range(1, len(data)):
        low_pass.append(alpha * low_pass[-1] + (1.0 - alpha) * data[i])
    return low_pass


def q_learn(max_iteration, max_steps, learning_rate, gamma):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    reward_list = []
    for i in range(max_iteration):
        state = env.reset()
        reward_sum = 0.0
        for _ in range(max_steps):
            action_list = Q[state, :]
            fuzzy_actions = action_list + np.random.randn(1, env.action_space.n) * (1. / (i + 1))
            selected_action = np.argmax(fuzzy_actions)
            new_state, reward, done, _ = env.step(selected_action)
            Q[state, selected_action] = Q[state, selected_action] + learning_rate * (
                    reward + gamma * np.max(Q[new_state, :]) - Q[state, selected_action])
            state = new_state
            reward_sum += reward
            if done:
                break
        reward_list.append(reward_sum)
    return reward_list, Q


def test_policy(env, Q, gamma, n=100):
    def run_episode(env, Q, gamma):
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while step_idx < 10000:
            action = np.argmax(Q[obs, :])
            obs, reward, done, _ = env.step(action)
            total_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward

    scores = [run_episode(env, Q, gamma) for _ in range(n)]
    return np.mean(scores)


max_steps = [1e5]
max_iters = [1e5]
gammas = [0.9, 0.999999]
learning_rates = [0.99999]
hyperparameters = list(itertools.product(max_steps, max_iters, gammas, learning_rates))


def run():
    for args in tqdm.tqdm(hyperparameters, desc="Frozen Lake Q Learning Experiment Iterations"):
        env.reset()
        max_steps, max_iteration, gamma, learning_rate = args
        start_time = time.process_time()
        reward_list, Q = q_learn(int(max_steps), int(max_iteration), learning_rate, gamma)
        wall_time = time.process_time() - start_time

        # test the learned policy
        mean_reward = test_policy(env, Q, gamma)

        results = {
            'max_iteration': max_iteration,
            'max_steps': max_steps,
            'gamma': gamma,
            'average_reward': mean_reward,
            'learning_rate': learning_rate,
            'wall_time': wall_time,
            'training_loss': reward_list,
            'Q': Q.tolist(),
            'name': 'Frozen Lake Q Learning'
        }
        with open(f'artifacts/frozen_lake_q_learning/{time.time_ns()}.json', 'w') as f:
            json.dump(results, f)
