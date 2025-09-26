import numpy as np
import random
from simple_envs_utils import q_table_init, e_greedy_policy


'''
Algoritmo tabular dyna q, basado en pseudo codigo de Sutton y Barto
'''
def tabular_dyna_q(env, n, num_episodes, alpha, gamma, epsilon, max_steps_per_episode):
    q_table = q_table_init(env)
    model = {}
    episode_rewards = []
    possible_actions = env.action_space

    for episode in range(num_episodes):
        current_state = env.reset()
        s_terminal = False
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = e_greedy_policy(q_table, current_state, possible_actions, epsilon)
            next_state, reward, s_terminal = env.step(action)
            episode_reward += reward
            q_table[current_state][action] += alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[current_state][action])
            model[(current_state, action)] = (reward, next_state)

            for i in range(n):
                visited_key_action_pairs = list(model.keys())
                prev_visited_state_and_action = visited_key_action_pairs[random.randint(0, len(visited_key_action_pairs) - 1)]

                prev_visited_state = prev_visited_state_and_action[0]
                prev_visited_action = prev_visited_state_and_action[1]

                reward_and_next_state = model[prev_visited_state_and_action]
                prev_visited_reward = reward_and_next_state[0]
                prev_visited_next_state = reward_and_next_state[1]

                q_table[prev_visited_state][prev_visited_action] += alpha * (prev_visited_reward + gamma * max(q_table[prev_visited_next_state].values()) - q_table[prev_visited_state][prev_visited_action])

            current_state = next_state
            if s_terminal == True:
                break
        episode_rewards.append(episode_reward)
    return q_table, episode_rewards