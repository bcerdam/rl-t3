import numpy as np
import matplotlib.pyplot as plt
import time

'''
inicializa tabla Q(s, a) = 0 para todos los estados y toas las acciones
'''
def q_table_init(env, init_q_val=0):
    q_table = {}
    possible_actions = env.action_space

    if "RoomEnv" in str(type(env)):
        all_positions = []
        for r in range(env._height):
            for c in range(env._width):
                all_positions.append((r, c))

        all_goals = env.goals
        for agent_pos in all_positions:
            for goal_pos in all_goals:
                state = (agent_pos, goal_pos)
                q_table[state] = {action: init_q_val for action in possible_actions}

    elif "EscapeRoomEnv" in str(type(env)):
        for row in range(env._height):
            for col in range(env._width):
                for has_key in [True, False]:
                    state = (row, col, has_key)
                    q_table[state] = {action: init_q_val for action in possible_actions}
    else:
        for row in range(env._height):
            for col in range(env._width):
                state = (row, col)
                q_table[state] = {action: init_q_val for action in possible_actions}
    return q_table


'''
Elige accion de manera epsilon greedy
'''
def e_greedy_policy(q_table, state, possible_actions, epsilon):
    random_decimal = np.random.random()
    if random_decimal < epsilon:
        return possible_actions[np.random.randint(len(possible_actions))]
    else:
        if state not in q_table.keys():
            return possible_actions[np.random.randint(len(possible_actions))]

        q_values = []
        for action in possible_actions:
            q_values.append(q_table[state][action])
        return possible_actions[np.argmax(q_values)]


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


def plot_multiple_mean_lengths(results_dict):
    plt.figure(figsize=(14, 10))
    for label, data in results_dict.items():
        plt.plot(data, label=label)
    plt.xlabel("Episodios")
    plt.ylabel("Largo Promedio del Episodio")
    plt.title("Comparación Centralized, Decentralized Cooperative y Decentralized Competitive Q-learning.")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figuras/pregunta_g_4.jpeg', dpi=500)
    plt.show()




'''
implementacion value_iteration basado en pseudo coddigo de libro Sutton y Barto, tiene modificaciones para que funcione con pseudo codigo de Rmax
visto en la capsula que subio el profe.
'''
def value_iteration(env, gamma, theta, Rmax, k, N_t, N_p, S):
    V = {s: 0 for s in S}
    possible_actions = env.action_space
    not_converged = True

    while not_converged == True:
        delta = 0
        for s in S:
            v = V[s]
            action_values = []

            for action in possible_actions:
                if (s, action) not in N_t or N_t[(s, action)] < k:
                    v_terminal = 0
                    current_action_value = Rmax + gamma * v_terminal
                    action_values.append(current_action_value)
                else:
                    current_action_value = 0
                    if (s, action) in N_p:
                        total_transitions = N_t[(s, action)]

                        for transition in list(N_p[(s, action)].keys()):
                            s_p = transition[0]
                            r = transition[1]
                            amount = N_p[(s, action)][transition]
                            current_action_value += (amount / total_transitions) * (r + gamma * V[s_p])
                    action_values.append(current_action_value)

            max_v = 0
            if action_values != []:
                max_v = np.max(np.array(action_values))
            V[s] = max_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            not_converged = False
    return V


'''
Funcion que obtiene A de S usando politica greedy a partir de V, necesaria para parte de pseudo codigo de Rmax.
'''
def action_from_v(env, V, s, gamma, Rmax, k, N_t, N_p):
    possible_actions = env.action_space
    action_values = []

    for action in possible_actions:
        if (s, action) not in N_t or N_t[(s, action)] < k:
            v_terminal = 0
            current_action_value = Rmax + gamma * v_terminal
            action_values.append(current_action_value)
        else:
            current_action_value = 0
            if (s, action) in N_p:
                total_transitions = N_t[(s, action)]
                for transition in list(N_p[(s, action)].keys()):
                    s_p = transition[0]
                    r = transition[1]
                    amount = N_p[(s, action)][transition]
                    current_action_value += (amount / total_transitions) * (r + gamma * V[s_p])
            action_values.append(current_action_value)

    return possible_actions[np.argmax(action_values)]