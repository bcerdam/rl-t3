from simple_envs_utils import q_table_init, e_greedy_policy


'''
Algoritmo q-learning, basado en pseudo codigo de Sutton y Barto
'''
def q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, init_q_value=0):
    q_table = q_table_init(env, init_q_value)
    episode_rewards = []
    episode_lengths = []
    possible_actions = env.action_space

    for episode in range(num_episodes):
        init_s = env.reset()
        current_state = init_s
        s_terminal = False
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = e_greedy_policy(q_table, current_state, possible_actions, epsilon)

            next_state, reward, s_terminal = env.step(action)
            episode_reward += reward

            q_table[current_state][action] += alpha * \
                                              (reward + gamma * max(q_table[next_state].values()) -
                                               q_table[current_state][action])

            current_state = next_state

            if s_terminal:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step+1)

    return q_table, episode_rewards, episode_lengths


def multi_goal_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, init_q_value=0):
    q_table = q_table_init(env, init_q_value)
    episode_rewards = []
    episode_lengths = []
    possible_actions = env.action_space
    all_goals = env.goals

    for episode in range(num_episodes):
        init_s = env.reset()
        current_state = init_s
        s_terminal = False
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = e_greedy_policy(q_table, current_state, possible_actions, epsilon)
            next_state, reward, s_terminal = env.step(action)
            episode_reward += reward
            q_table[current_state][action] += alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[current_state][action])

            for goal in all_goals:
                if next_state[0] == goal:
                    sub_g_reward = 1.0
                else:
                    sub_g_reward = 0.0
                q_table[(current_state[0], goal)][action] += alpha * (sub_g_reward + gamma * max(q_table[(next_state[0], goal)].values()) - q_table[(current_state[0], goal)][action])
            current_state = next_state
            if s_terminal:
                break
        episode_rewards.append(episode_reward)
        episode_lengths.append(step+1)
    return q_table, episode_rewards, episode_lengths