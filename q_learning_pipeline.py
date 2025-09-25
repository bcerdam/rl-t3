import numpy as np
import time

'''
inicializa tabla Q(s, a) = 0 para todos los estados y toas las acciones
'''
def q_table_init(env):
    q_table = {}
    possible_actions = env.action_space
    for row in range(env._height):
        for col in range(env._width):
            state = (row, col)
            q_table[state] = {action: 0 for action in possible_actions}
    return q_table


'''
Elige accion de manera epsilon greedy
'''
def e_greedy_policy(q_table, state, possible_actions, epsilon):
    random_decimal = np.random.random()
    if random_decimal < epsilon:
        return np.random.choice(possible_actions)
    else:
        return possible_actions[np.argmax([q_table[state][action] for action in possible_actions])]

'''
Algoritmo q-learning, basado en pseudo codigo de Sutton y Barto
'''
def q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps_per_episode):
    q_table = q_table_init(env)
    episode_rewards = []
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

    return q_table, episode_rewards

'''
Funcion que permite visualizar si politica es buena, basicamente muestra el agente resolviendo el problema
'''
def aprox_optimal_policy_visualization(env, q_table):
    current_state = env.reset()
    s_terminal = False

    while not s_terminal:
        env.show()

        possible_actions = env.action_space
        action = possible_actions[np.argmax([q_table[current_state][act] for act in possible_actions])]

        next_state, reward, s_terminal = env.step(action)
        current_state = next_state
        time.sleep(0.5)

    env.show()