import numpy as np
from pipelines.simple_envs_utils import plot_multiple_mean_lengths
from pipelines.q_learning_pipeline import q_learning, memory_efficient_q_learning
from pipelines.n_step_sarsa_pipeline import n_step_sarsa, memory_efficient_n_step_sarsa

from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MainSimpleEnvs import show, get_action_from_user, play_simple_env
from MemoryWrappers.BinaryMemory import BinaryMemory
from MemoryWrappers.KOrderMemory import KOrderMemory
from MemoryWrappers.KOrderMemoryBufferVariant import KOrderMemoryBufferVariant


def play_env_with_binary_memory():
    num_of_bits = 1
    env = InvisibleDoorEnv()
    env = BinaryMemory(env, num_of_bits)

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    key2memory = {str(i): i for i in range(2**num_of_bits)}
    s = env.reset()
    show(env, s)
    done = False
    while not done:
        print("Environment action: ", end="")
        env_action = get_action_from_user(key2action)
        print(f"Memory action ({', '.join(key2memory.keys())}): ", end="")
        mem_action = get_action_from_user(key2memory)
        action = env_action, mem_action
        s, r, done = env.step(action)
        show(env, s, r)


def play_env_with_k_order_memory():
    memory_size = 2
    env = InvisibleDoorEnv()
    env = KOrderMemory(env, memory_size)
    play_simple_env(env)


def play_env_without_extra_memory():
    env = InvisibleDoorEnv()
    play_simple_env(env)


if __name__ == '__main__':

    '''
    Inicio pregunta g)
    '''

    # env = InvisibleDoorEnv()
    #
    # ALPHA = 0.1
    # GAMMA = 0.99
    # EPSILON = 0.01
    # INITIAL_Q_VALUE = 1.0
    # N_EPISODES = 1000
    # N_RUNS = 30
    # MAX_STEPS_PER_EPISODE = 5*(10**3)
    #
    # ### Q-LEARNING ###
    # all_lengths_q = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Q-learning: {run + 1}/{N_RUNS}...")
    #     q_learning_episode_lengths = q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, True)
    #     all_lengths_q[run, :] = q_learning_episode_lengths
    # q_learning_mean_lengths = np.mean(all_lengths_q, axis=0)
    # ### Q-LEARNING ###
    #
    # ### 1-step SARSA ###
    # all_lengths_sarsa1 = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Sarsa: {run + 1}/{N_RUNS}...")
    #     sarsa_episode_lengths = n_step_sarsa(env, 1, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, True)
    #     all_lengths_sarsa1[run, :] = sarsa_episode_lengths
    # sarsa_mean_lengths = np.mean(all_lengths_sarsa1, axis=0)
    # ### 1-step SARSA ###
    #
    # ### 16-step SARSA ###
    # all_lengths_sarsa16 = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"16-Step Sarsa: {run + 1}/{N_RUNS}...")
    #     sarsa_16_step_episode_lengths = n_step_sarsa(env, 16, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, True)
    #     all_lengths_sarsa16[run, :] = sarsa_16_step_episode_lengths
    # sarsa_16_step_mean_lengths = np.mean(all_lengths_sarsa16, axis=0)
    # ### 16-step SARSA ###
    #
    # ### Resultados Primera Parte ###
    # final_results = {
    #     "Q-learning": q_learning_mean_lengths,
    #     "1-step SARSA": sarsa_mean_lengths,
    #     "16-Step SARSA": sarsa_16_step_mean_lengths
    # }
    # plot_multiple_mean_lengths(final_results)
    # ### Resultados Primera Parte ###
    #
    # MEMORY_SIZE = 2
    # env = KOrderMemory(InvisibleDoorEnv(), memory_size=MEMORY_SIZE)
    #
    # ### Q-LEARNING K=2 ORDER MEMORY ###
    # all_lengths_q_learning_size_2_memory = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Q-learning size 2 Memory: {run + 1}/{N_RUNS}...")
    #     ep_lengths = memory_efficient_q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_q_learning_size_2_memory[run, :] = ep_lengths
    # q_learning_size_2_memory_mean_lengths = np.mean(all_lengths_q_learning_size_2_memory, axis=0)
    # ### Q-LEARNING K=2 ORDER MEMORY ###
    #
    # ### 1-step SARSA K=2 ORDER MEMORY ###
    # all_lengths_sarsa_size_2_memory = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Sarsa size 2 Memory: {run + 1}/{N_RUNS}...")
    #     ep_lengths = memory_efficient_n_step_sarsa(env, 1, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_sarsa_size_2_memory[run, :] = ep_lengths
    # sarsa_size_2_memory_mean_lengths = np.mean(all_lengths_sarsa_size_2_memory, axis=0)
    # ### 1-step SARSA K=2 ORDER MEMORY ###
    #
    # ### 16-step SARSA K=2 ORDER MEMORY ###
    # all_lengths_sarsa_16_step_size_2_memory = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"16-Step Sarsa size 2 Memory: {run + 1}/{N_RUNS}...")
    #     ep_lengths = memory_efficient_n_step_sarsa(env, 16, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_sarsa_16_step_size_2_memory[run, :] = ep_lengths
    # sarsa_16_step_size_2_memory_mean_lengths = np.mean(all_lengths_sarsa_16_step_size_2_memory, axis=0)
    # ### 16-step SARSA K=2 ORDER MEMORY ###
    #
    # ### Resultados Segunda Parte ###
    # final_results = {
    #     "Q-learning 2-order Memory": q_learning_size_2_memory_mean_lengths,
    #     "1-Step Sarsa 2-order  Memory": sarsa_size_2_memory_mean_lengths,
    #     "16-Step Sarsa 2-order Memory": sarsa_16_step_size_2_memory_mean_lengths
    # }
    # plot_multiple_mean_lengths(final_results)
    # ### Resultados Segunda Parte ###
    #
    # NUM_OF_BITS = 1
    # env = BinaryMemory(InvisibleDoorEnv(), num_of_bits=NUM_OF_BITS)
    #
    # ### BINARY MEMORY Q-LEARNING ###
    # all_lengths_q_learning_binary = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Q-learning with Binary Memory: {run + 1}/{N_RUNS}...")
    #     q_learning_binary_episode_lengths = memory_efficient_q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_q_learning_binary[run, :] = q_learning_binary_episode_lengths
    # q_learning_binary_mean_lengths = np.mean(all_lengths_q_learning_binary, axis=0)
    # ### BINARY MEMORY Q-LEARNING ###
    #
    # ### BINARY MEMORY 1-step SARSA ###
    # all_lengths_sarsa_binary = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Sarsa with Binary Memory: {run + 1}/{N_RUNS}...")
    #     sarsa_binary_episode_lengths = memory_efficient_n_step_sarsa(env, 1, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_sarsa_binary[run, :] = sarsa_binary_episode_lengths
    # sarsa_binary_mean_lengths = np.mean(all_lengths_sarsa_binary, axis=0)
    # ### BINARY MEMORY 1-step SARSA ###
    #
    # ### BINARY MEMORY 16-step SARSA ###
    # all_lengths_sarsa_16_step_binary = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"16-Step Sarsa with Binary Memory: {run + 1}/{N_RUNS}...")
    #     sarsa_16_step_binary_episode_lengths = memory_efficient_n_step_sarsa(env, 16, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_sarsa_16_step_binary[run, :] = sarsa_16_step_binary_episode_lengths
    # sarsa_16_step_binary_mean_lengths = np.mean(all_lengths_sarsa_16_step_binary, axis=0)
    # ### BINARY MEMORY 16-step SARSA ###
    #
    # ### Resultados Tercera Parte ###
    # final_results = {
    #     "Q-learning with Binary Memory": q_learning_binary_mean_lengths,
    #     "Sarsa with Binary Memory": sarsa_binary_mean_lengths,
    #     "16-Step Sarsa with Binary Memory": sarsa_16_step_binary_mean_lengths
    # }
    # plot_multiple_mean_lengths(final_results)
    # ### Resultados Tercera Parte ###
    #
    # BUFFER_SIZE = 1
    # env = KOrderMemoryBufferVariant(InvisibleDoorEnv(), k_order_buffer=BUFFER_SIZE)
    #
    # ### CUSTOM MEMORY Q-LEARNING ###
    # all_lengths_q_learning_custom_memory = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Q-learning with Custom Memory: {run + 1}/{N_RUNS}...")
    #     q_learning_custom_memory_episode_lengths = memory_efficient_q_learning(env, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_q_learning_custom_memory[run, :] = q_learning_custom_memory_episode_lengths
    # q_learning_custom_memory_mean_lengths = np.mean(all_lengths_q_learning_custom_memory, axis=0)
    # ### CUSTOM MEMORY Q-LEARNING ###
    #
    # ### CUSTOM MEMORY 1-Step SARSA ###
    # all_lengths_sarsa_custom_memory = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"Sarsa with Custom Memory: {run + 1}/{N_RUNS}...")
    #     sarsa_custom_memory_episode_lengths = memory_efficient_n_step_sarsa(env, 1, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_sarsa_custom_memory[run, :] = sarsa_custom_memory_episode_lengths
    # sarsa_custom_memory_mean_lengths = np.mean(all_lengths_sarsa_custom_memory, axis=0)
    # ### CUSTOM MEMORY 1-Step SARSA ###
    #
    # ### CUSTOM MEMORY 16-Step SARSA ###
    # all_lengths_sarsa_16_step_custom_memory = np.zeros((N_RUNS, N_EPISODES))
    # for run in range(N_RUNS):
    #     print(f"16-Step Sarsa with Custom Memory: {run + 1}/{N_RUNS}...")
    #     sarsa_16_step_custom_memory_episode_lengths = memory_efficient_n_step_sarsa(env, 16, N_EPISODES, ALPHA, GAMMA, EPSILON, MAX_STEPS_PER_EPISODE, INITIAL_Q_VALUE, memory=True)
    #     all_lengths_sarsa_16_step_custom_memory[run, :] = sarsa_16_step_custom_memory_episode_lengths
    # sarsa_16_step_custom_memory_mean_lengths = np.mean(all_lengths_sarsa_16_step_custom_memory, axis=0)
    # ### CUSTOM MEMORY 16-Step SARSA ###
    #
    # ### Resultados Cuarda Parte ###
    # final_results = {
    #     "Q-learning with Custom Memory": q_learning_custom_memory_mean_lengths,
    #     "Sarsa with Custom Memory": sarsa_custom_memory_mean_lengths,
    #     "16-Step Sarsa with Custom Memory": sarsa_16_step_custom_memory_mean_lengths
    # }
    # plot_multiple_mean_lengths(final_results)
    # ### Resultados Cuarda Parte ###

    '''
    Fin pregunta g)
    '''


