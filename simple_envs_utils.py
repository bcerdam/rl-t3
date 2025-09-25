import numpy as np
import matplotlib.pyplot as plt
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


'''
funcion que plottea promedio de retornos, para la pregunta c.
'''
def plot_mean_returns(mean_return_q_learning, mean_return_sarsa, mean_return_4_step_sarsa):
    plt.figure(figsize=(14, 10))
    plt.plot(mean_return_q_learning, label="Q-learning")
    plt.plot(mean_return_sarsa, label="SARSA (N-step SARSA con n=1)")
    plt.plot(mean_return_4_step_sarsa, label="4-Step SARSA (N-step SARSA con n=4)")
    plt.xlabel("Episodios")
    plt.ylabel("Retorno Promedio")
    plt.title("Q-learning vs SARSA vs 4-step SARSA en CliffEnv")
    plt.ylim(bottom=-200)
    plt.legend()
    plt.grid(True)
    plt.tight_layout
    # plt.savefig('figuras/pregunta_c.jpeg', dpi=500)
    plt.show()