import itertools
import json
import time

import gym
import numpy as np
import tqdm


def avg_reward(env, s, a):
    avg_reward = 0
    done_states = []
    for prob, next_s, reward, done in env.P[s][a]:
        avg_reward += prob * reward
        if done:
            done_states.append(next_s)
        # if not done:
        #     avg_reward += prob * STEP_REWARD
        # elif reward == 0.0:
        #     done_states.append(next_s)
        #     avg_reward += 0.0
        # else:
        #     #             avg_reward += prob * 10
        #     avg_reward += prob * WIN_REWARD
    return avg_reward, done_states


def random_policy(env):
    return np.random.randint(0, 4, size=env.nS)


def one_step_lookahead(env, s, value_function, discount):
    action_values = np.zeros(env.nA)
    done_states = []
    for a in range(env.nA):
        value, look_ds = avg_reward(env, s, a)
        done_states.extend(look_ds)
        for p, next_s, _, _ in env.P[s][a]:
            value += discount * p * value_function[next_s]
        action_values[a] = value
    return action_values, done_states


def evaluate_policy(env, policy, max_backups=1000, epsilon=1e-6, discount=0.9):
    old_value = np.zeros(env.nS)
    done_states = []
    deltas = []
    for i in range(max_backups):
        new_value = np.zeros(env.nS)
        for s in range(env.nS):
            action_values, ols_ds = one_step_lookahead(env, s, old_value, discount)
            done_states.extend(ols_ds)
            new_value[s] = action_values[policy[s]]
        delta = np.max(np.abs(new_value - old_value))
        deltas.append(delta)
        if delta < epsilon:
            break
        old_value = new_value
    return new_value, done_states, deltas


def greedy_policy(env, value_function, discount=0.9):
    policy = np.zeros(env.nS, dtype=np.int32)
    for s in range(env.nS):
        action_values, ds = one_step_lookahead(env, s, value_function, discount)
        policy[s] = np.argmax(action_values)
    return policy


def policy_iteration(env, max_steps=100, discount=0.9, epsilon=1e-6):
    old_policy = random_policy(env)
    done_states = []
    deltas = []
    test_scores = []
    for i in range(max_steps):
        value_function, ep_ds, episode_deltas = evaluate_policy(env, old_policy, discount=discount, epsilon=epsilon)
        deltas.extend(episode_deltas)
        done_states.extend(ep_ds)
        new_policy = greedy_policy(env, value_function, discount=discount)

        # test policy
        test_score = test_policy(env, new_policy, gamma=1.0)
        test_scores.append(test_score)

        if np.array_equal(new_policy, old_policy):
            break
        old_policy = new_policy
    # dedup done states
    done_states = list(set(done_states))
    return old_policy, value_function, done_states, deltas, test_scores


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


STEP_REWARD = 0.0
LOSE_REWARD = 0.0
WIN_REWARD = 1.0

max_policy_iterations = [1e2, 1e10]
gammas = [0.1, 0.999]
epsilons = [0.1, 0.001]
hyperparameters = list(itertools.product(max_policy_iterations, gammas, epsilons))


def main():
    env = gym.make('FrozenLake8x8-v1')
    env.reset()

    for max_iterations, gamma, epsilon in tqdm.tqdm(hyperparameters, desc="Frozen Lake Policy Iteration Experiment"):
        start_time = time.process_time()
        policy, opt_value_func, done_states, deltas, test_scores = policy_iteration(env, discount=gamma, epsilon=epsilon, max_steps=int(max_iterations))
        wall_time = time.process_time() - start_time

        average_test_score = test_policy(env, policy, gamma=1.0)

        result = {
            'V': opt_value_func.tolist(),
            'policy': policy.tolist(),
            'policy_eval_differences': deltas,
            'done_states': done_states,
            'gamma': gamma,
            'max_iterations': max_iterations,
            'epsilon': epsilon,
            'wall_time': wall_time,
            'test_scores': test_scores,
            'average_test_score': average_test_score,
            'name': 'Frozen Lake Policy Iteration Experiment'
        }

        with open(f'artifacts/frozenlakev2/{time.time_ns()}.json', 'w') as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
