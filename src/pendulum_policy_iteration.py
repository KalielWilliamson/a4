import itertools
import json
import math
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import gym
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
    expected_reward = -(angle ** 2 + 0.1 * state[2] ** 2)
    expected_reward += 16.2736044  # making the reward positive for every step taken
    possible_rewards.append(expected_reward)

possible_states = np.asarray(list(states))
possible_actions = np.arange(-2.0, 2.0, 0.2)
possible_rewards = np.asarray(possible_rewards)


def descretize_state(state, possible_states):
    # find the closest state in possible_states
    state = np.array(state)
    state = state.reshape(1, -1)
    distances = np.linalg.norm(possible_states - state, axis=1)
    closest_state_index = np.argmin(distances)
    closest_state = possible_states[closest_state_index]
    return closest_state_index, closest_state


def init_policy(possible_states, possible_actions):
    policy = defaultdict(lambda: {})
    for state_index, state in enumerate(possible_states):
        for action_index, action in enumerate(possible_actions):
            policy[state_index][action] = 1.0 / len(possible_actions)
    return policy


def one_step_lookahead(env, state, V, possible_actions, possible_states, gamma):
    action_values = np.zeros(len(possible_actions))
    for action_index, action in enumerate(possible_actions):
        env.reset()
        env.state = state
        next_state, reward, done, info = env.step([action])
        next_state_index, next_state = descretize_state(next_state, possible_states)

        action_values[action_index] += reward + (gamma * V[next_state_index])
    return np.argmax(action_values)


def policy_evaluation(env, possible_states, possible_actions, max_policy_iteration, gamma,
                      epsilon):
    policy = init_policy(possible_states, possible_actions)
    V = np.zeros(len(possible_states))
    new_V = np.zeros(len(possible_states))
    differences = []

    # because it's deterministic, no need to loop more than once
    for i in range(max_policy_iteration):
        delta = 0
        new_V = np.zeros(len(possible_states))
        for state_index, state in enumerate(possible_states):
            values = []
            for action in possible_actions:
                env.reset()
                env.state = state

                action_probability = policy[state_index][action]

                next_state, reward, done, info = env.step([action])

                # making the reward positive for every step taken
                # -16.27 is the worst reward for the pendulum environment
                reward += 16.2736044

                next_state_index, next_state = descretize_state(next_state, possible_states)
                possible_rewards[next_state_index] = reward

                values.append(action_probability * V[next_state_index])

            new_V[state_index] = possible_rewards[state_index] + gamma * sum(values)

            delta = max(delta, abs(new_V[state_index] - V[state_index]))
            V[state_index] = new_V[state_index]
        differences.append(delta)
        if delta < epsilon:
            break

        # policy improvement
        optimal_policy_found = True
        for state_index, state in enumerate(possible_states):
            # Compute state value
            env.reset()
            env.state = state

            policy_action = dict_argmax(policy[state_index])
            lookahead_action = one_step_lookahead(env, state, V, possible_actions, possible_states, gamma)

            if policy_action != possible_actions[lookahead_action]:
                optimal_policy_found = False
            for action in policy[state_index]:
                if action == possible_actions[lookahead_action]:
                    policy[state_index][action] = 1.0
                else:
                    policy[state_index][action] = 0.0

            if optimal_policy_found:
                break

    # for all dict in new_policy, set the value to the max value
    for state_index, state in enumerate(possible_states):
        if isinstance(policy[state_index], dict):
            # get key of max value
            policy[state_index] = dict_argmax(policy[state_index])

        discounted_reward_sum = V[state_index]
        action = policy[state_index]
        next_state, reward, done, _ = env.step([action])
        next_state_index, next_state_value = descretize_state(next_state, possible_states)
        # discounted downstream values
        discounted_reward_sum += reward + gamma * V[next_state_index]

        new_V[state_index] = discounted_reward_sum

    V = new_V.copy()

    return differences, V, policy


def dict_argmax(dictionary: Dict):
    max_value = max(dictionary.values())  # TODO handle multiple keys with the same max value
    for key, value in dictionary.items():
        if value == max_value:
            return key


def policy_improvement(V, env, possible_actions, possible_states, gamma):
    new_policy = dict()
    for state_index, state in enumerate(possible_states):
        action_values = np.zeros(len(possible_actions))
        for action_index, action in enumerate(possible_actions):
            env.reset()
            env.state = state
            next_state, reward, done, info = env.step([action])
            next_state_index, next_state = descretize_state(next_state, possible_states)

            action_values[action_index] += reward + (gamma * V[next_state_index])
        new_policy[state_index] = np.max(action_values)

    return new_policy


def l2(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i]) ** 2
    return math.sqrt(s)


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
            total_reward += reward * math.pow(gamma, step_idx)
            step_idx += 1
            if done:
                break
        return total_reward

    scores = [run_episode(env, policy, gamma) for _ in range(n)]
    return np.mean(scores)


max_policy_iterations = [50, 100, 1000]
gammas = [0.1, 0.9]
epsilons = [0.1, 0.001]
hyperparameters = list(itertools.product(max_policy_iterations, gammas, epsilons))


def run():
    env = gym.make('Pendulum-v1', g=9.81)
    for max_policy_iteration, gamma, epsilon in tqdm.tqdm(hyperparameters,
                                                          desc="Pendulum Policy Iteration Experiment"):
        iteration_differences = []
        env.reset()

        start_time = time.process_time()
        differences, V, policy = policy_evaluation(env, possible_states, max_policy_iteration=int(max_policy_iteration),
                                                   possible_actions=possible_actions,
                                                   gamma=gamma, epsilon=epsilon)

        wall_time = time.process_time() - start_time

        average_test_score = test_policy(env, policy, gamma)

        result = {
            'iteration_differences': iteration_differences,
            'policy_evaluation_differences': differences,
            'V': V.tolist(),
            'policy': policy,
            'gamma': gamma,
            'max_policy_iteration': max_policy_iteration,
            'epsilon': epsilon,
            'wall_time': wall_time,
            'average_test_score': average_test_score,
            'name': 'Pendulum Policy Iteration Experiment'
        }

        with open(f'artifacts/pendulum_policy_iteration/{time.time_ns()}.json', 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    run()
