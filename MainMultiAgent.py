import numpy as np
from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from MainSimpleEnvs import show, get_action_from_user
from pipelines.q_learning_pipeline import memory_efficient_q_learning, decentralized_q_learning, decentralized_competitive_q_learning
from pipelines.simple_envs_utils import plot_multiple_mean_lengths

def play_hunter_env():
    hunter_env = HunterAndPreyEnv()

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down", '': "None"}
    num_of_agents = hunter_env.num_of_agents
    s = hunter_env.reset()
    show(hunter_env, s)
    done = False
    while not done:
        print("Hunter A: ", end="")
        hunter1 = get_action_from_user(key2action)
        print("Hunter B: ", end="")
        hunter2 = get_action_from_user(key2action)
        action = hunter1, hunter2
        if num_of_agents == 3:
            print("Prey: ", end="")
            prey = get_action_from_user(key2action)
            action = hunter1, hunter2, prey
        s, r, done = hunter_env.step(action)
        show(hunter_env, s, r)


if __name__ == '__main__':

    '''
    Inicio pregunta f)
    '''

    # env = CentralizedHunterEnv()
    #
    # ALPHA = 0.1
    # GAMMA = 0.95
    # EPSILON = 0.1
    # INITIAL_Q_VALUE = 1.0
    # N_EPISODES = 50000
    # N_RUNS = 30
    # MAX_STEPS_PER_EPISODE = 3*(10**2)
    #
    # ### CENTRALIZED Q-LEARNING ###
    # all_episode_lengths = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Centralized Q_learning: {run + 1}/{N_RUNS}...")
    #     centralized_q_table, centralized_episode_lengths = memory_efficient_q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths[run, :] = centralized_episode_lengths
    #
    # centralized_mean_episode_lengths = np.mean(all_episode_lengths, axis=0)
    # centralized_mean_episode_lengths = np.mean(centralized_mean_episode_lengths.reshape(-1, 100), axis=1)
    # ### CENTRALIZED Q-LEARNING ###
    #
    # ### DECENTRALIZED Q-LEARNING ###
    # env_decentralized = HunterEnv()
    # all_episode_lengths_decentralized = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Decentralized Q_learning: {run + 1}/{N_RUNS}...")
    #     decentralized_episode_lengths = decentralized_q_learning(env_decentralized, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths_decentralized[run, :] = decentralized_episode_lengths
    # decentralized_mean_episode_lengths = np.mean(all_episode_lengths_decentralized, axis=0)
    # decentralized_mean_episode_lengths = np.mean(decentralized_mean_episode_lengths.reshape(-1, 100), axis=1)
    # ### dECENTRALIZED Q-LEARNING ###
    #
    # ### DECENTRALIZED COMPETITIVE Q-LEARNING ###
    # env_decentralized_competitive = HunterAndPreyEnv()
    #
    # all_episode_lengths_decentralized_competitive = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Decentralized Competitive Q_learning: {run + 1}/{N_RUNS}...")
    #     decentralized_competitive_episode_lengths = decentralized_competitive_q_learning(env_decentralized_competitive, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE)
    #     all_episode_lengths_decentralized_competitive[run, :] = decentralized_competitive_episode_lengths
    #
    # decentralized_competitive_mean_episode_lengths = np.mean(all_episode_lengths_decentralized_competitive, axis=0)
    # decentralized_competitive_mean_episode_lengths = np.mean(decentralized_competitive_mean_episode_lengths.reshape(-1, 100), axis=1)
    # ### DECENTRALIZED COMPETITIVE Q-LEARNING ###
    #
    # ### RESULTADOS ###
    #
    # final_results = {"Centralized Q-learning": centralized_mean_episode_lengths,
    #                  "Decentralized Cooperative Q-learning": decentralized_mean_episode_lengths,
    #                  "Decentralized Competitive Q-learning": decentralized_competitive_mean_episode_lengths}
    # plot_multiple_mean_lengths(final_results)
    #
    # ### RESULTADOS ###


    '''
    Fin pregunta f)
    '''