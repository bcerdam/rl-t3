import numpy as np
from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
from pipelines.q_learning_pipeline import q_learning
from pipelines.n_step_sarsa_pipeline import n_step_sarsa
from pipelines.simple_envs_utils import aprox_optimal_policy_visualization, plot_mean_returns
from pipelines.dyna_pipeline import tabular_dyna_q
from pipelines.rmax_pipeline import rmax


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


    '''
        Inicio pregunta c
    '''
    # env = CliffEnv()
    #
    # ALPHA = 0.1
    # GAMMA = 1.0
    # EPSILON = 0.1
    # N_EPISODES = 500
    # MAX_STEPS_PER_EPISODE = 10**6
    # N_RUNS = 100
    #
    # ### Q-LEARNING ###
    # run_episode_rewards = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Q-learning run {run + 1}/{N_RUNS}")
    #     q_table, episode_rewards, episode_lengths = q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE)
    #     run_episode_rewards[run, :] = episode_rewards
    # mean_return_q_learning = np.mean(run_episode_rewards, axis=0)
    # ### Q-LEARNING ###
    #
    # ## SARSA ###
    # run_episode_rewards = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"SARSA run {run + 1}/{N_RUNS}")
    #     q_table_sarsa, rewards_sarsa, episode_lengths = n_step_sarsa(env, 1, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE)
    #     run_episode_rewards[run, :] = rewards_sarsa
    # mean_return_sarsa = np.mean(run_episode_rewards, axis=0)
    # ## SARSA ###
    #
    # ### N-STEP SARSA ###
    # run_episode_rewards = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"4-step SARSA run {run + 1}/{N_RUNS}")
    #     q_table_4_step_sarsa, rewards_4_step_sarsa, episode_lengths = n_step_sarsa(env, 4, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE)
    #     run_episode_rewards[run, :] = rewards_4_step_sarsa
    # mean_return_4_step_sarsa = np.mean(run_episode_rewards, axis=0)
    # ### N-STEP SARSA ###
    #
    # # Ejecuciones se promedian -> Episodios se promedian -> Unico valor es printeado. i.e retorno promedio sobre todas las ejecuciones y todos los episodios.
    # print(f"Retorno promedio 500 episodios Q-learning: {np.mean(mean_return_q_learning)}")
    # print(f"Retorno promedio 500 episodios SARSA: {np.mean(mean_return_sarsa)}")
    # print(f"Retorno promedio 500 episodios 4-step SARSA: {np.mean(mean_return_4_step_sarsa)}")
    #
    # # Permite visualizar como agente resuelve el problema con la politica encontrada, descomentar para poder observar.
    # aprox_optimal_policy_visualization(env, q_table)
    # aprox_optimal_policy_visualization(env, q_table_sarsa)
    # aprox_optimal_policy_visualization(env, q_table_4_step_sarsa)
    #
    # # Plotteo primeros 500 episodios promediados sobre 100 ejecuciones.
    # plot_mean_returns(mean_return_q_learning, mean_return_sarsa, mean_return_4_step_sarsa)

    '''
        Fin pregunta c
    '''

    '''
        Inicio pregunta d
    '''

    # env = EscapeRoomEnv()
    # ALPHA = 0.5
    # GAMMA = 1.0
    # EPSILON = 0.1
    # N_EPISODES = 20
    # MAX_STEPS_PER_EPISODE = 10**3
    # N_RUNS = 5
    # planning_steps = [0, 1, 10, 100, 1000, 10000]
    #
    # mean_returns_dyna = []
    # full_run_episodes_rewards = []
    # for steps in planning_steps:
    #     print(f"Tabular Dyna-Q, n={steps}")
    #     rewards = []
    #     run_episodes_rewards = []
    #     for run in range(N_RUNS):
    #         q_table_dyna, rewards_dyna = tabular_dyna_q(env, steps, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE)
    #         run_episodes_rewards.append(rewards_dyna)
    #         rewards.append(np.mean(rewards_dyna))
    #     full_run_episodes_rewards.append(run_episodes_rewards)
    #     mean_return_dyna = np.mean(rewards)
    #     mean_returns_dyna.append(mean_return_dyna)
    #
    # # Retorno medio por episodio para todos los planning steps.
    # mean_across_runs = np.mean(np.array(full_run_episodes_rewards), axis=1).T
    # print(mean_across_runs)
    #
    # # Permite visualizar como agente resuelve el problema con la politica encontrada, descomentar para poder observar.
    # # aprox_optimal_policy_visualization(env, q_table_dyna)

    # env = EscapeRoomEnv()
    # GAMMA = 1.0
    # N_EPISODES = 20
    # MAX_STEPS_PER_EPISODE = 1000
    # N_RUNS = 5
    # K = 2
    # R_MAX = 10
    # THETA = 0.00001
    #
    # rewards = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Rmax ejecucion: {run + 1}/{N_RUNS}")
    #     reward_by_episode = rmax(env, K, R_MAX, N_EPISODES, GAMMA, MAX_STEPS_PER_EPISODE, THETA)
    #     rewards[run, :] = reward_by_episode
    #
    # mean_return_per_episode = np.mean(rewards, axis=0)
    # print(mean_return_per_episode)

    '''
        Fin pregunta d
    '''