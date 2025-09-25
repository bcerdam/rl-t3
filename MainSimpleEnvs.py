import numpy as np
import matplotlib.pyplot as plt
from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
from q_learning_pipeline import q_learning, aprox_optimal_policy_visualization

def show(env, current_state, reward=None):
    env.show()
    print(f"Raw state: {current_state}")
    if reward:
        print(f"Reward: {reward}")


def get_action_from_user(valid_actions):
    key = input()
    while key not in valid_actions:
        key = input()
    return valid_actions[key]


def play_simple_env(simple_env):
    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    s = simple_env.reset()
    show(simple_env, s)
    done = False
    while not done:
        print("Action: ", end="")
        action = get_action_from_user(key2action)
        s, r, done = simple_env.step(action)
        show(simple_env, s, r)


if __name__ == "__main__":
    env = CliffEnv()


    '''
        Inicio pregunta c
    '''

    ALPHA = 0.1
    GAMMA = 1.0
    EPSILON = 0.1
    N_EPISODES = 500
    MAX_STEPS_PER_EPISODE = 10**6
    N_RUNS = 100

    run_episode_rewards = np.zeros((N_RUNS, N_EPISODES))
    for run in range(N_RUNS):
        print(f"run {run + 1}/{N_RUNS}")
        q_table, episode_rewards = q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE)
        run_episode_rewards[run, :] = episode_rewards
    mean_return_q_learning = np.mean(run_episode_rewards, axis=0)

    # Plotteo de retorno por episodio promediado a lo largo de 100 ejecuciones
    plt.plot(mean_return_q_learning)
    plt.show()

    # Valor de retorno promedio en general, para todas las ejecuciones, para todos los episodios
    print(f"Retorno promedio 500 episodios: {np.mean(mean_return_q_learning)}")

    # Permite visualizar como agente resuelve el problema con la politica encontrada
    # Descomentar para poder observar.
    # aprox_optimal_policy_visualization(env, q_table)


    '''
        Fin pregunta c
    '''