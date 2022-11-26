#!/usr/bin/env python
# coding: utf-8
import itertools
import json
import time
import gym
import numpy as np
import tqdm

env = gym.make('FrozenLake8x8-v1')


def test_policy(env, policy, gamma, n=100):
    def run_episode(env, policy, gamma):
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while step_idx < 10000:
            obs, reward, done, _ = env.step(int(policy[obs]))
            total_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward

    scores = [run_episode(env, policy, gamma) for _ in range(n)]
    return np.mean(scores)


def extract_policy(v, gamma):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma, epsilon, max_iterations):
    value_func_old = np.random.rand(env.nS)
    value_func_new = np.zeros(env.nS)
    done_states = []
    differences = []
    for iteration in range(max_iterations):
        delta = 0
        for s in range(env.nS):
            maxvsa = -1
            for a in range(env.nA):
                vsa = 0
                for possible_next_state in env.P[s][a]:
                    prob_action = possible_next_state[0]
                    cur_reward = possible_next_state[2]
                    if possible_next_state[3]:
                        future_reward = 0
                        done_states.append(possible_next_state[1])
                    else:
                        future_reward = gamma * value_func_old[possible_next_state[1]]
                    vsa += prob_action * (cur_reward + future_reward)
                if vsa > maxvsa:
                    maxvsa = vsa
            # diff=math.pow((value_func_old[s]-maxvsa),2)
            diff = abs(value_func_old[s] - maxvsa)
            delta = max(delta, diff)
            value_func_new[s] = maxvsa
        # delta=math.sqrt(delta)
        differences.append(delta)
        if delta <= epsilon: break
        value_func_old = value_func_new

    return differences, value_func_new, done_states


max_iterations = [1e1, 1e4, 1e10, 1e15]
gammas = [0.1, 0.9, 0.99999]
epsilons = [1e-1, 1e5, 1e-11]
hyperparameters = list(itertools.product(max_iterations, gammas, epsilons))


def run():
    for args in tqdm.tqdm(hyperparameters, desc="Frozen Lake Value Iteration Experiment Iterations"):
        env.reset()
        max_iteration, gamma, epsilon = args

        start_time = time.process_time()
        differences, V, done_states = value_iteration(env, gamma, epsilon=epsilon, max_iterations=int(max_iteration))
        value_iteration_wall_time = time.process_time() - start_time

        start_time = time.process_time()
        policy = extract_policy(V, gamma)
        extract_policy_wall_time = time.process_time() - start_time

        average_policy_score = test_policy(env, policy, gamma)

        result = {
            'average_policy_score': average_policy_score,
            'policy': policy.tolist(),
            'done_states': done_states,
            'V': V.tolist(),
            'differences': differences,
            'value_iteration_wall_time': value_iteration_wall_time,
            'extract_policy_wall_time': extract_policy_wall_time,
            'gamma': gamma,
            'epsilon': epsilon,
            'max_iterations': max_iteration,
            'name': 'Frozen Lake Value Iteration Experiment'
        }

        with open(f'artifacts/frozenlake_vi2/{time.time_ns()}.json', 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    run()
