import numpy as np
from simple_envs_utils import q_table_init, e_greedy_policy


'''
Algoritmo n-step sarsa, basado en pseudo codigo de Sutton y Barto
'''
def n_step_sarsa(env, n, num_episodes, alpha, gamma, epsilon, max_steps_per_episode):
    q_table = q_table_init(env)
    episode_rewards = []
    possible_actions = env.action_space

    for episode in range(num_episodes):
        states = [env.reset()]
        actions = [e_greedy_policy(q_table, states[0], possible_actions, epsilon)]
        rewards = [0]

        T = float('inf')
        t = 0
        episode_reward = 0
        finishing_episode = True
        while finishing_episode:
            if t < T:
                next_state, reward, s_terminal = env.step(actions[t])
                episode_reward += reward
                states.append(next_state)
                rewards.append(reward)
                if s_terminal == True:
                    T = t + 1
                else:
                    actions.append(e_greedy_policy(q_table, states[t + 1], possible_actions, epsilon))

            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma ** n) * q_table[states[tau + n]][actions[tau + n]]
                q_table[states[tau]][actions[tau]] += alpha * (G - q_table[states[tau]][actions[tau]])

            if tau == T - 1:
                finishing_episode = False
            t += 1
            if t >= max_steps_per_episode and T == float('inf'):
                finishing_episode = False

        episode_rewards.append(episode_reward)
    return q_table, episode_rewards