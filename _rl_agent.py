from req_py_libraries import *
from _environment import _environment

class rl_agent():


    def __init__(self):

        self.action_count = 3
        self.input_size = 21

        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Epsilon greedy parameter
        self.epsilon_min = 0.1  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)

        self.max_time_steps = 15000

        self.optimizer = keras.optimizers.Adam(learning_rate=0.0025, clipnorm=1.0)

        # Using huber loss for stability
        self.loss_function = keras.losses.Huber()

        self.step_count = 0

        self.environment = _environment()

        # Number of frames to take random action and observe output
        self.epsilon_random_frames = 4000
        # Number of frames for exploration
        self.epsilon_greedy_frames = 7000

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


    # a simple feed forward neural network to suggest best action, based on the state.
    def q_network_definition(self):

        inputs = layers.Input(shape=(self.input_size,))

        layer1 = layers.Dense(self.input_size,activation="relu",
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros())(inputs)
        layer2 = layers.Dense(self.input_size,activation="relu",
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros())(layer1)
        # since the output is the state-action value for the multiple actions for the said state.
        layer3 = layers.Dense(self.action_count,
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros(), activation='linear')(layer2)

        return keras.Model(inputs=inputs,outputs=layer3)

    def _simulator(self):

        q_network = self.q_network_definition()
        q_network_target = self.q_network_definition()

        state = np.array(self.environment._environment_reset()[0])
        for timestep in range(self.max_time_steps):

            viz_dict = {} # this records all the data

            #viz_dict["state"] = state
            viz_dict["step_count"] = self.step_count

            if self.step_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                action = np.random.choice(self.action_count)
            else:
                # Predict action Q-values; From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)

                qsa_values = q_network(state_tensor, training=False)

                # The above isn't actually action probs, but they're actually the Q(s,a) values
                # for the different actions for the same state. We end up selecting the action with the highest Q(s,a) value
                action = tf.argmax(qsa_values[0]).numpy()

            viz_dict["action"] = action

            # Decay probability of taking random action
            self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # Apply the sampled action in our environment
            [reward,state_next] = self.environment._environment_step(suggested_action=action)

            viz_dict["state"] = state_next
            viz_dict["reward"] = reward

            state_next = np.array(state_next)
            self.action_history.append(action)
            self.state_history.append(state)
            self.state_next_history.append(state_next)
            self.rewards_history.append(reward)
            self.episode_reward.append(reward)

            state = state_next

            # update weights with every 4 samples
            if self.step_count % self.update_qn_weights_epoch == 0 and self.step_count > 0:

                state_sample = np.array([self.state_history[i] for i in range(self.update_qn_weights_epoch)])
                state_next_sample = np.array([self.state_next_history[i] for i in range(self.update_qn_weights_epoch)])
                rewards_sample = [self.rewards_history[i] for i in range(self.update_qn_weights_epoch)]
                action_sample = [self.action_history[i] for i in range(self.update_qn_weights_epoch)]

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = q_network_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, self.action_count)

                with tf.GradientTape() as tape:

                    # Train the model on the states and updated Q-values
                    q_values = q_network(state_sample)
                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = self.loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, q_network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

                # reset experience buffers after every %self.update_after% steps
                self._init_buffers()

            if self.step_count % (self.update_qn_weights_epoch) == 0:
                # update the target network with new weights
                q_network_target.set_weights(q_network.get_weights())

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


        pd.DataFrame(self.reward_plot_df).to_excel("_data/mean_reward.xlsx")
        pd.DataFrame(self.viz_df).to_excel("_data/viz_df_rl_agent.xlsx")

        [lt_performance_df] = self.environment._environment_lt_perf_data()
        pd.DataFrame(lt_performance_df).to_excel("_data/lt_perf_rl_agent.xlsx")

if __name__ == "__main__":

    r1 = rl_agent()
    r1._simulator()


