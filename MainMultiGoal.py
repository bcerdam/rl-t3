import numpy as np
from q_learning_pipeline import q_learning, multi_goal_q_learning
from n_step_sarsa_pipeline import n_step_sarsa, multi_goal_n_step_sarsa
from simple_envs_utils import aprox_optimal_policy_visualization, plot_mean_returns, plot_multiple_mean_lengths
from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
from MainSimpleEnvs import play_simple_env


def play_room_env():
    n_episodes = 10
    for _ in range(n_episodes):
        env = RoomEnv()
        play_simple_env(env)


if __name__ == '__main__':

    '''
    Inicio pregunta e)
    '''

    env = RoomEnv()

    ALPHA = 0.1
    EPSILON = 0.1
    GAMMA = 0.99
    N_EPISODES = 500
    N_RUNS = 100
    INITIAL_Q_VALUE = 1.0
    MAX_STEPS_PER_EPISODE = 10 ** 3


    # ### Q-LEARNING ###
    # all_episode_lengths = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Q-learning, ejecucion {run + 1}/{N_RUNS}")
    #     q_learning_q_table, q_learning_episode_rewards, q_learning_episode_lengths = q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths[run, :] = q_learning_episode_lengths
    # q_learning_mean_episode_lengths = np.mean(all_episode_lengths, axis=0)
    # ### Q-LEARNING ###
    #
    # ### SARSA ###
    # all_episode_lengths_sarsa = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Sarsa, ejecucion {run + 1}/{N_RUNS}")
    #     q_table, ep_rewards, ep_lengths = n_step_sarsa(env, 1, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths_sarsa[run, :] = ep_lengths
    # sarsa_mean_episode_lengths = np.mean(all_episode_lengths_sarsa, axis=0)
    # ### SARSA ###
    #
    # ### 8-STEP SARSA ###
    # all_episode_lengths_8_step_sarsa = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"8-Step Sarsa, ejecucion {run + 1}/{N_RUNS}")
    #     q_table, ep_rewards, ep_lengths = n_step_sarsa(env, 8, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths_8_step_sarsa[run, :] = ep_lengths
    # sarsa_8_step_mean_episode_lengths = np.mean(all_episode_lengths_8_step_sarsa, axis=0)
    # ### 8-STEP SARSA ###
    #
    # ### MULTI-GOAL Q-LEARNING ###
    # all_episode_lengths_multi_goal_q_learning = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Multi-Goal Q-learning, ejecucion {run + 1}/{N_RUNS}")
    #     multi_goal_q_table, multi_goal_q_episode_rewards, multi_goal_q_episode_lengths = multi_goal_q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths_multi_goal_q_learning[run, :] = multi_goal_q_episode_lengths
    # mean_lengths_multi_goal_q = np.mean(all_episode_lengths_multi_goal_q_learning, axis=0)
    # ### MULTI-GOAL Q-LEARNING ###

    # ### MULTI-GOAL 1-STEP SARSA ###
    # all_episode_lengths_multi_goal_1_step_sarsa = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Multi-Goal 1-Step Sarsa, ejecucion {run + 1}/{N_RUNS}")
    #     multi_goal_1_step_sarsa_q_table, multi_goal_1_step_sarsa_episode_rewards, multi_goal_1_step_sarsa_episode_lengths = multi_goal_n_step_sarsa(
    #         env, 1, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths_multi_goal_1_step_sarsa[run, :] = multi_goal_1_step_sarsa_episode_lengths
    # multi_goal_1_step_sarsa_mean_episode_lengths = np.mean(all_episode_lengths_multi_goal_1_step_sarsa, axis=0)
    # ### MULTI-GOAL 1-STEP SARSA ###
    #
    # ### MULTI-GOAL 8-STEP SARSA ###
    # all_episode_lengths_multi_goal_8_step_sarsa = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Multi-Goal 8-Step Sarsa, ejecucion {run + 1}/{N_RUNS}")
    #     multi_goal_8_step_sarsa_q_table, multi_goal_8_step_sarsa_episode_rewards, multi_goal_8_step_sarsa_episode_lengths = multi_goal_n_step_sarsa(
    #         env, 8, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths_multi_goal_8_step_sarsa[run, :] = multi_goal_8_step_sarsa_episode_lengths
    # multi_goal_8_step_sarsa_mean_episode_lengths = np.mean(all_episode_lengths_multi_goal_8_step_sarsa, axis=0)
    # ### MULTI-GOAL 8-STEP SARSA ###

    # # ### RESULTADOS ###
    # final_results = {
    #     "Q-learning": mean_lengths_q,
    #     "Sarsa": sarsa_mean_episode_lengths,
    #     "8-Step Sarsa": sarsa_8_step_mean_episode_lengths,
    #     "Multi-Goal Q-learning": mean_lengths_multi_goal_q,
    #     "Multi-Goal Sarsa": multi_goal_1_step_sarsa_mean_episode_lengths,
    #     "Multi-Goal 8-Step Sarsa": multi_goal_8_step_sarsa_mean_episode_lengths
    # }
    # plot_multiple_mean_lengths(final_results)
    #
    # # ### RESULTADOS ###
    '''
    Fin pregunta e)
    '''
