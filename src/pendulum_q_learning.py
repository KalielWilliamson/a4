import itertools
import json
import math
import sys
import time

import numpy as np
import gym
import tqdm
from matplotlib.pyplot import subplot, plot, legend, show

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
possible_actions = np.arange(-2.0, 2.0, 0.01)
possible_rewards = np.asarray(possible_rewards)


def descretize_state(state, possible_states):
    index = np.argmin(np.linalg.norm(possible_states - state, axis=1))
    return index, possible_states[index]


def low_pass(data, alpha=0.99):
    low_pass = [data[0]]
    for i in range(1, len(data)):
        low_pass.append(alpha * low_pass[-1] + (1.0 - alpha) * data[i])
    return low_pass


def q_learn(max_iteration, max_steps, learning_rate, gamma, possible_states, possible_actions):
    Q = np.zeros([len(possible_states), len(possible_actions)])
    training_loss = []
    for i in range(max_iteration):
        state = env.reset()
        s_idx, state = descretize_state(state, possible_states)
        reward_sum = 0.0

        for _ in range(max_steps):
            # exploit vs explore to find action
            if np.random.random() < 1.0 / (i + 1):
                # get index of randomly selected action
                action_idx = np.random.randint(0, len(possible_actions))
            else:
                action_idx = np.argmax(Q[s_idx, :])

            action = possible_actions[action_idx]

            # take action, observe new state and reward
            env.state = state
            new_state, reward, done, _ = env.step([action])

            # making the reward positive for every step taken
            # -16.27 is the worst reward for the pendulum environment
            reward += 16.2736044

            next_sidx, next_state = descretize_state(new_state, possible_states)
            Q[s_idx, action_idx] = Q[s_idx, action_idx] + learning_rate * (
                    reward + gamma * np.max(Q[next_sidx, :]) - Q[s_idx, action_idx])
            s_idx = next_sidx
            reward_sum += reward
            if done:
                break
            training_loss.append(reward_sum)
        return Q, training_loss


def test_policy(env, Q, gamma, n=100):
    def run_episode(env, Q, gamma):
        obs = env.reset()
        state_index, state = descretize_state(obs, possible_states)
        total_reward = 0
        step_idx = 0
        while step_idx < 10000:
            action = np.argmax(Q[state_index, :])
            env.state = state
            new_state, reward, done, _ = env.step(possible_states[action])
            state_index, state = descretize_state(new_state, possible_states)
            total_reward += reward * math.pow(gamma, step_idx)
            step_idx += 1
            if done:
                break
        return total_reward

    scores = [run_episode(env, Q, gamma) for _ in range(n)]
    return np.mean(scores)


def get_policy(Q):
    policy = []
    for s in range(len(possible_states)):
        policy.append(np.argmax(Q[s, :]))
    return policy


max_steps = [1e3, 1e5, 1e10]
max_iters = [1e3, 1e5, 1e10]
gammas = [0.1, 0.6, 0.9]
learning_rates = [0.1, 0.5, 0.9]
hyperparameters = list(itertools.product(max_steps, max_iters, gammas, learning_rates))


def run():
    for args in tqdm.tqdm(hyperparameters, desc="Pendulum Q Learning Experiment Iterations"):
        max_steps, max_iteration, gamma, learning_rate = args
        start_time = time.process_time()
        Q, training_loss = q_learn(max_steps=int(max_steps), max_iteration=int(max_iteration),
                                   learning_rate=learning_rate, gamma=gamma, possible_states=possible_states,
                                   possible_actions=possible_actions)
        wall_time = time.process_time() - start_time
        mean_reward = test_policy(env, Q, gamma)
        results = {
            'max_iteration': max_iteration,
            'max_steps': max_steps,
            'gamma': gamma,
            'mean_reward': mean_reward,
            'learning_rate': learning_rate,
            'wall_time': wall_time,
            'training_loss': training_loss,
            'Q': Q.tolist(),
            'name': 'Pendulum Q Learning Experiment b'
        }
        with open(f'artifacts/pendulum_q_learning/{time.time_ns()}.json', 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    run()
