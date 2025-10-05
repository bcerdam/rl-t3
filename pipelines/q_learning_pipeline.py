from .simple_envs_utils import q_table_init, e_greedy_policy
from collections import defaultdict


'''
Algoritmo q-learning, basado en pseudo codigo de Sutton y Barto
'''
def q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, init_q_value=0, memory=False):
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

    if memory == True:
        return episode_lengths
    else:
        return q_table, episode_rewards, episode_lengths


'''
Algoritmo q-learning, basado en pseudo codigo de Sutton y Barto, modificado para funcionar en enviroment multi-goal.
'''
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


'''
Algoritmo q-learning, basado en pseudo codigo de Sutton y Barto. Es casi igual que la primera funcion pero
para este tipo de problemas, tuve que modificar la inicialiacion de la q_table, y solamente hacerlo para
estados que se han visitado.
'''
def memory_efficient_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, init_q_value, memory=False):
    q_table = defaultdict(lambda: defaultdict(lambda: init_q_value))
    episode_lengths = []
    possible_actions = env.action_space

    for episode in range(num_episodes):
        current_state = env.reset()
        s_terminal = False

        for step in range(max_steps_per_episode):
            action = e_greedy_policy(q_table, current_state, possible_actions, epsilon)
            next_state, reward, s_terminal = env.step(action)

            if q_table[next_state]:
                next_max_q = max(q_table[next_state].values())
            else:
                next_max_q = init_q_value

            q_table[current_state][action] += alpha * (reward + gamma * next_max_q - q_table[current_state][action])
            current_state = next_state
            if s_terminal == True:
                break
        episode_lengths.append(step + 1)

    if memory == True:
        return episode_lengths
    else:
        return q_table, episode_lengths


'''
Algoritmo q-learning, basado en pseudo codigo de Sutton y Barto, modificado para que funcione en HunterEnv.
'''
def decentralized_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, init_q_value):
    q_table_hunter1 = defaultdict(lambda: defaultdict(lambda: init_q_value))
    q_table_hunter2 = defaultdict(lambda: defaultdict(lambda: init_q_value))
    episode_lengths = []
    possible_single_agent_actions = env.single_agent_action_space

    for episode in range(num_episodes):
        current_state = env.reset()
        s_terminal = False

        for step in range(max_steps_per_episode):
            action_hunter1 = e_greedy_policy(q_table_hunter1, current_state, possible_single_agent_actions, epsilon)
            action_hunter2 = e_greedy_policy(q_table_hunter2, current_state, possible_single_agent_actions, epsilon)
            next_state, rewards, s_terminal = env.step((action_hunter1, action_hunter2))

            if q_table_hunter1[next_state]:
                next_max_q_hunter1 = max(q_table_hunter1[next_state].values())
            else:
                next_max_q_hunter1 = init_q_value
            q_table_hunter1[current_state][action_hunter1] += alpha * (
                        rewards[0] + gamma * next_max_q_hunter1 - q_table_hunter1[current_state][action_hunter1])

            if q_table_hunter2[next_state]:
                next_max_q_hunter2 = max(q_table_hunter2[next_state].values())
            else:
                next_max_q_hunter2 = init_q_value
            q_table_hunter2[current_state][action_hunter2] += alpha * (
                        rewards[1] + gamma * next_max_q_hunter2 - q_table_hunter2[current_state][action_hunter2])

            current_state = next_state
            if s_terminal == True:
                break
        episode_lengths.append(step + 1)
    return episode_lengths


def decentralized_competitive_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps_per_episode, init_q_value):
    q_table_hunter1 = defaultdict(lambda: defaultdict(lambda: init_q_value))
    q_table_hunter2 = defaultdict(lambda: defaultdict(lambda: init_q_value))
    q_table_prey = defaultdict(lambda: defaultdict(lambda: init_q_value))
    episode_lengths = []
    possible_single_agent_actions = env.single_agent_action_space

    for episode in range(num_episodes):
        current_state = env.reset()
        s_terminal = False

        for step in range(max_steps_per_episode):
            action_hunter1 = e_greedy_policy(q_table_hunter1, current_state, possible_single_agent_actions, epsilon)
            action_hunter2 = e_greedy_policy(q_table_hunter2, current_state, possible_single_agent_actions, epsilon)
            action_prey = e_greedy_policy(q_table_prey, current_state, possible_single_agent_actions, epsilon)
            next_state, rewards, s_terminal = env.step((action_hunter1, action_hunter2, action_prey))

            if q_table_hunter1[next_state]:
                next_max_q_hunter1 = max(q_table_hunter1[next_state].values())
            else:
                next_max_q_hunter1 = init_q_value
            q_table_hunter1[current_state][action_hunter1] += alpha * (rewards[0] + gamma * next_max_q_hunter1 - q_table_hunter1[current_state][action_hunter1])

            if q_table_hunter2[next_state]:
                next_max_q_hunter2 = max(q_table_hunter2[next_state].values())
            else:
                next_max_q_hunter2 = init_q_value
            q_table_hunter2[current_state][action_hunter2] += alpha * (rewards[1] + gamma * next_max_q_hunter2 - q_table_hunter2[current_state][action_hunter2])

            if q_table_prey[next_state]:
                next_max_q_prey = max(q_table_prey[next_state].values())
            else:
                next_max_q_prey = init_q_value
            q_table_prey[current_state][action_prey] += alpha * (rewards[2] + gamma * next_max_q_prey - q_table_prey[current_state][action_prey])

            current_state = next_state
            if s_terminal == True:
                break
        episode_lengths.append(step + 1)
    return episode_lengths