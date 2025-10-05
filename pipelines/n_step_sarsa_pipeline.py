from collections import defaultdict
from .simple_envs_utils import q_table_init, e_greedy_policy


'''
Algoritmo n-step sarsa, basado en pseudo codigo de Sutton y Barto
'''
def n_step_sarsa(env, n, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, init_q_val=0, memory=False):
    q_table = q_table_init(env)
    episode_rewards = []
    episode_lengths = []
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
        episode_lengths.append(t)

    if memory == True:
        return episode_lengths
    else:
        return q_table, episode_rewards, episode_lengths


def multi_goal_n_step_sarsa(env, n, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, initial_q_value=0):
    q_table = q_table_init(env, initial_q_value)
    episode_rewards = []
    episode_lengths = []
    possible_actions = env.action_space
    all_goals = env.goals

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
                    T = t+1
                else:
                    actions.append(e_greedy_policy(q_table, states[t+1], possible_actions, epsilon))

            tau = t-n+1
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+n, T)+1):
                    G += (gamma**(i-tau-1)) * rewards[i]
                if tau+n < T:
                    G += (gamma**n) * q_table[states[tau+n]][actions[tau+n]]
                q_table[states[tau]][actions[tau]] += alpha * (G - q_table[states[tau]][actions[tau]])

                for g in all_goals:
                    other_G = 0
                    for i in range(tau+1, min(tau+n, T)+1):
                        if states[i][0] == g:
                            sub_g_reward = 1.0
                        else:
                            sub_g_reward = 0.0
                        other_G += (gamma**(i-tau-1)) * sub_g_reward

                    if tau+n < T:
                        other_G += (gamma**n) * q_table[(states[tau+n][0], g)][actions[tau+n]]
                    q_table[(states[tau][0], g)][actions[tau]] += alpha * (other_G - q_table[(states[tau][0], g)][actions[tau]])
            if tau == T-1:
                finishing_episode = False
            t += 1
            if t >= max_steps_per_episode and T == float('inf'):
                finishing_episode = False
        episode_rewards.append(episode_reward)
        episode_lengths.append(t)
    return q_table, episode_rewards, episode_lengths


def memory_efficient_n_step_sarsa(env, n, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, init_q_val=0, memory=False):
    q_table = defaultdict(lambda: defaultdict(lambda: init_q_val))
    episode_rewards = []
    episode_lengths = []
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
        episode_lengths.append(t)

    if memory == True:
        return episode_lengths
    else:
        return q_table, episode_rewards, episode_lengths