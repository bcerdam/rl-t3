import numpy as np
from simple_envs_utils import q_table_init, value_iteration, action_from_v

'''
Algoritmo Rmax, basado en pseudo codigo de capsula subida por el profe.
'''
def rmax(env, k, Rmax, num_episodes, gamma, max_steps_per_episode, theta):
    N_t = {}
    N_p = {}
    episode_rewards = []

    for episode in range(num_episodes):
        S = env.reset()
        s_terminal = False
        episode_reward = 0

        for step in range(max_steps_per_episode):
            V = value_iteration(env, gamma, theta, Rmax, k, N_t, N_p, list(q_table_init(env).keys()))
            action = action_from_v(env, V, S, gamma, Rmax, k, N_t, N_p)
            s_p, r, s_terminal = env.step(action)
            episode_reward += r
            state_action = (S, action)

            if state_action not in N_t:
                N_t[state_action] = 0
            N_t[state_action] += 1

            if state_action not in N_p:
                N_p[state_action] = {}

            s_p_r = (s_p, r)
            if s_p_r not in N_p[state_action]:
                N_p[state_action][s_p_r] = 0
            N_p[state_action][s_p_r] += 1

            S = s_p
            if s_terminal == True:
                break
        episode_rewards.append(episode_reward)
    return episode_rewards