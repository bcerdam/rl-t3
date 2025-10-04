from Environments.AbstractEnv import AbstractEnv


class KOrderMemoryBufferVariant(AbstractEnv):

    def __init__(self, env, k_order_buffer):
        self.__env = env
        self.__k_order_buffer = k_order_buffer
        self.__memory_buffer = []
        self.__buffer_co = None

        self.__action_space = []
        possible_actions = self.__env.action_space
        for enviroment_action in possible_actions:
            for action in ["save", "ignore"]:
                self.__action_space.append((enviroment_action, action))

    @property
    def action_space(self):
        return self.__action_space

    def reset(self):
        self.__buffer_co = self.__env.reset()
        self.__memory_buffer = []
        return self.__buffer_co, tuple(self.__memory_buffer)

    def step(self, action):
        env_action, memory_action = action

        if memory_action == "save":
            self.__memory_buffer.append(self.__buffer_co)
            if len(self.__memory_buffer) > self.__k_order_buffer:
                self.__memory_buffer.pop(0)

        next_observation, reward, done = self.__env.step(env_action)
        self.__buffer_co = next_observation

        new_state = (self.__buffer_co, tuple(self.__memory_buffer))
        return new_state, reward, done

    def show(self):
        self.__env.show()
        print(f"Memory: {self.__memory_buffer}")