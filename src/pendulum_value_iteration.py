#!/usr/bin/env python
# coding: utf-8

import itertools
import json
import math
import time

import gym
import numpy as np
import tqdm

env = gym.make('Pendulum-v1')
env.reset()
states = set()
possible_rewards = []
num_possible_states = 1000

while len(states) < num_possible_states:
    state = env.reset()
    states.add(tuple(state))

for state in states:
    x, y = state[0], state[1]
    angle = math.atan2(y, x)
    expected_reward = -(angle**2 + 0.1*state[2]**2)
    expected_reward += 16.2736044 # making the reward positive for every step taken
    possible_rewards.append(expected_reward)


possible_states = np.asarray(list(states))
possible_actions = np.arange(-2.0, 2.0, 0.01)
possible_rewards = np.asarray(possible_rewards)


def descretize_state(state, possible_states):
    index = np.argmin(np.linalg.norm(possible_states - state, axis=1))
    return index, possible_states[index]


def value_iteration(env, possible_actions, possible_states, gamma, epsilon, max_iteration):
    v = np.zeros(len(possible_states))
    deltas = []
    for i in range(max_iteration):
        prev_v = np.copy(v)
        for i, state in enumerate(possible_states):
            q_sa = []
            for j, action in enumerate(possible_actions):
                env.reset()
                env.state = state
                next_state, reward, done, _ = env.step([action])

                # making the reward positive for every step taken
                # -16.27 is the worst reward for the pendulum environment
                reward += 16.2736044

                idx, next_state = descretize_state(next_state, possible_states)
                q_sa.append(reward + gamma * prev_v[idx])
            v[i] = max(q_sa)
        delta = np.sum(np.fabs(prev_v - v))
        deltas.append(delta)
        if delta <= epsilon:
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return v, deltas


def policy_iteration(V, env, possible_actions, possible_states, gamma):
    new_policy = np.zeros(len(possible_states))
    for i, state in enumerate(possible_states):
        new_v = np.zeros(len(possible_actions))
        for j, action in enumerate(possible_actions):
            env.reset()
            env.state = state
            next_state, reward, done, info = env.step([action])

            # making the reward positive for every step taken
            # -16.27 is the worst reward for the pendulum environment
            reward += 16.2736044

            idx, next_state = descretize_state(next_state, possible_states)
            new_v[j] += reward + gamma * V[idx]
        new_policy[i] = int(np.argmax(new_v))
    return new_policy.astype(np.int32)


def test_policy(env, policy, gamma, n=100):
    def run_episode(env, policy, gamma):
        obs = env.reset()
        state_index, state = descretize_state(obs, possible_states)
        total_reward = 0
        step_idx = 0
        while step_idx < 10000:
            action = policy[state_index]
            obs, reward, done, _ = env.step([action])
            state_index, state = descretize_state(obs, possible_states)
            total_reward += reward * (gamma ** step_idx)
            step_idx += 1
            if done:
                break
        return total_reward

    scores = [run_episode(env, policy, gamma) for _ in range(n)]
    return np.mean(scores)


max_iterations = [1e1, 1e2, 1e4, 1e6]
gammas = [0.1, 0.6, 0.9]
epsilons = [1e-1, 1e-3, 1e-5]
hyperparameters = list(itertools.product(max_iterations, gammas, epsilons))


def run():
    for params in tqdm.tqdm(hyperparameters, 'Pendulum Value Iteration Experiment Iterations'):
        max_iter, gamma, epsilon = params

        start_time = time.process_time()
        V, deltas = value_iteration(env, possible_actions, possible_states, gamma, epsilon, int(max_iter))
        value_iteration_time = time.process_time() - start_time

        start_time = time.process_time()
        policy = policy_iteration(V, env, possible_actions, possible_states, gamma)
        policy_time = time.process_time() - start_time

        env.reset()
        start_time = time.process_time()
        average_policy_score = test_policy(env, policy, gamma, n=1000)
        policy_score_time = start_time - time.process_time()

        result = {
            'average_policy_score': average_policy_score,
            'policy': policy.tolist(),
            'V': V.tolist(),
            'deltas': deltas,
            'max_iterations': max_iter,
            'gamma': gamma,
            'epsilon': epsilon,
            'value_iteration_wall_time': value_iteration_time,
            'policy_time': policy_time,
            'policy_score_time': policy_score_time,
            'name': 'Pendulum Value Iteration Experiment b'
        }

        with open(f'artifacts/pendulum_value_iteration/{time.time_ns()}.json', 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    run()
