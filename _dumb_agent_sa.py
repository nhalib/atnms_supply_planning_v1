from req_py_libraries import *
from _environment import _environment

class dumb_agent():


    def __init__(self,fixed_action):

        self.action_count = 3
        self.input_size = 21

        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Epsilon greedy parameter
        self.epsilon_min = 0.1  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)

        self.fixed_action = fixed_action

        self.max_time_steps = 10000

        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        # Using huber loss for stability
        self.loss_function = keras.losses.Huber()

        self.step_count = 0

        self.environment = _environment()

        # Number of frames to take random action and observe output
        self.epsilon_random_frames = 2000
        # Number of frames for exploration
        self.epsilon_greedy_frames = 6000

        # after how many steps should we update the weights
        self.update_qn_weights_epoch = 4

        self._init_buffers()

        self.episode_reward = []
        self.viz_df = []

        self.reward_plot_df = []



    def _init_buffers(self):
        # buffers to hold experiences
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []


    def _simulator(self):


        state = np.array(self.environment._environment_reset()[0])
        for timestep in range(self.max_time_steps):

            viz_dict = {} # this records all the data

            #viz_dict["state"] = state
            viz_dict["step_count"] = self.step_count

            action = self.fixed_action # uniformly random action [0,1]

            viz_dict["action"] = action

            # Apply the sampled action in our environment
            [reward,state_next] = self.environment._environment_step(suggested_action=action)

            viz_dict["state"] = state_next
            viz_dict["reward"] = reward
            self.episode_reward.append(reward)

            state_next = np.array(state_next)

            state = state_next

            self.step_count += 1
            self.viz_df.append(viz_dict)

            if self.step_count % 500 == 0:

                print(self.step_count)

                # dataframe that stores the environment interactions
                mean_reward = np.mean(self.episode_reward)
                self.episode_reward = []

                dict = {}
                dict["mean_reward"] = mean_reward
                dict["step_count"] = self.step_count

                self.reward_plot_df.append(dict)

        pd.DataFrame(self.reward_plot_df).to_excel("_data/mean_reward_da_sa_{0}.xlsx".format(self.fixed_action))

        pd.DataFrame(self.viz_df).to_excel("_data/viz_df_da_sa_{0}.xlsx".format(self.fixed_action))

        [lt_performance_df] = self.environment._environment_lt_perf_data()
        pd.DataFrame(lt_performance_df).to_excel("_data/lt_perf_da_sa_{0}.xlsx".format(self.fixed_action))


if __name__ == "__main__":

    r1 = dumb_agent(fixed_action=0)
    r1._simulator()