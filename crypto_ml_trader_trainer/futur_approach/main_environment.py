#
# #%%
# from numpy import single
# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.environments import tf_py_environment
# from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
# from tf_agents.keras_layers import inner_reshape
# from tf_agents.networks.encoding_network import EncodingNetwork
# from tf_agents.policies import random_py_policy, random_tf_policy
# import tensorflow as tf
# import pandas as pd
# import tensorflow.keras as keras
# import tensorflow.keras.layers as kl
# import tensorflow.keras.models as km
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.specs import tensor_spec
# from tf_agents.trajectories import trajectory
# from tf_agents.utils import common
# from tf_agents.networks import Network, nest_map, sequential
# import tf_agents as tfa
#
# from market_indicator_environment import MarketIndicatorEnvironment
#
# # Data and env creation
# from utilities.DateUtility import dateparse
#
# data_file_path = "./input/btc_usd_1min.csv"
# train_ratio = 0.7
#
# data = pd.read_csv(data_file_path,
#                         delimiter=',',
#                         names=["time", "open", "high", "low", "close", "volume", "trades"],
#                         parse_dates=True,
#                         date_parser=dateparse,
#                         index_col='time')
# data_size = len(data)
# data_train_size = round(data_size * train_ratio)
#
# data_train = data.iloc[:data_train_size]
# data_val = data.iloc[data_train_size:]
#
# environment_train = MarketIndicatorEnvironment(data_train)
# environment_val = MarketIndicatorEnvironment(data_val)
#
# # utils.validate_py_environment(environment, episodes=2)
#
# #%%
# # Random Policy
#
# # noinspection PyTypeChecker
# # my_random_py_policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
# #     action_spec=environment_train.action_spec())
# #
# #
# # def compute_avg_return(environment, policy, num_episodes=1000):
# #
# #     total_return = 0.0
# #     for _ in range(num_episodes):
# #
# #         time_step = environment.reset()
# #         episode_return = 0.0
# #
# #         while not time_step.is_last():
# #             action_step = policy.action(time_step)
# #             time_step = environment.step(action_step.action)
# #             episode_return += time_step.reward
# #         total_return += episode_return
# #
# #     avg_return = total_return / num_episodes
# #     return avg_return
# #
# #
# # avg_return = compute_avg_return(environment_train, my_random_py_policy, 10)
# # print(f"Average return {avg_return}")
#
#
#
# #%%
# # DQN Policy
# # ========================================
#
# num_iterations = 20000 # @param {type:"integer"}
#
# initial_collect_steps = 100  # @param {type:"integer"}
# collect_steps_per_iteration = 1  # @param {type:"integer"}
# replay_buffer_max_length = 100000  # @param {type:"integer"}
#
# batch_size = 64  # @param {type:"integer"}
# learning_rate = 1e-3  # @param {type:"number"}
# log_interval = 200  # @param {type:"integer"}
#
# num_eval_episodes = 10  # @param {type:"integer"}
# eval_interval = 1000  # @param {type:"integer"}
#
# train_env = tf_py_environment.TFPyEnvironment(environment_train, check_dims=True)
# eval_env = tf_py_environment.TFPyEnvironment(environment_val)
#
# batch_size = train_env.batch_size
#
# fc_layer_params = (100, 50)
# action_tensor_spec = train_env.action_spec()
# num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
#
# # Define a helper function to create Dense layers configured with the right
# # activation and kernel initializer.
# def dense_layer(num_units):
#   return tf.keras.layers.Dense(
#       num_units,
#       activation=tf.keras.activations.relu,
#       kernel_initializer=tf.keras.initializers.VarianceScaling(
#           scale=2.0, mode='fan_in', distribution='truncated_normal'))
#
# # def create_fc_network(layer_units):
# #   return sequential.Sequential([dense_layer(num_units) for num_units in layer_units])
#
#
#
# #%%
# # DQN part 2
# #====================
#
# # QNetwork consists of a sequence of Dense layers followed by a dense layer
# # with `num_actions` units to generate one q_value per available action as
# # it's output.
#
# def build_q_network(num_actions: int):
#     input1 = keras.layers.Input(shape=((15,15,1,)), name="input_market")
#     input2 = keras.layers.Input(shape=((4,)), name="input_wallet")
#
#     m = kl.Reshape((15,15,1), input_shape=(225,))(input1)
#     m = kl.Conv2D(filters=64, kernel_size=(5,5), strides=3)(m)
#     m = kl.Flatten()(m)
#
#     m2 = kl.Dense(128)(input2)
#
#     end = kl.Concatenate()([m, m2])
#     end = kl.Dense(100)(end)
#     end = kl.Dense(50)(end)
#
#     output = kl.Dense(num_actions)(end)
#
#     model = keras.models.Model(inputs=[input1, input2], outputs=output)
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss=keras.losses.mean_squared_error)
#
#     return model
#
#
# network_main = build_q_network(num_actions)
# network_target = build_q_network(num_actions)
#
# keras.utils.plot_model(network_main, "a.png", show_shapes=True, show_layer_names=True)
#
# #%%
# train_step_counter = tf.Variable(0)
#
#
# # noinspection PyTypeChecker
# my_random_tf_policy = random_tf_policy.RandomTFPolicy(time_step_spec=train_env.time_step_spec(),
#                                                       action_spec=train_env.action_spec())
#
# #%%
# ## ReplayBuffer class
#
# class ReplayBufferEntry:
#
#     def __init__(self, state, reward, ):
#
# class UniformReplayBuffer:
#
#
#
#     def __init__(self, size: int, shape: tuple, batch_size: int = 32):
#
#
#
# replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#     data_spec=tensor_spec.from_spec(agent.collect_data_spec),
#     batch_size=train_env.batch_size,
#     max_length=replay_buffer_max_length)
#
# def compute_avg_return(environment, policy, num_episodes=10):
#     return 0
#
#     total_return = 0.0
#     for _ in range(num_episodes):
#
#         time_step = environment.reset()
#         episode_return = 0.0
#         policy_state = policy.get_initial_state(batch_size=batch_size)
#
#         while not time_step.is_last():
#             action_step = policy.action(time_step, policy_state)
#             time_step = environment.step(action_step.action)
#             episode_return += time_step.reward
#         total_return += episode_return
#
#     avg_return = total_return / num_episodes
#     return avg_return.numpy()[0]
#
# def collect_step(environment, policy, buffer):
#     time_step = environment.current_time_step()
#     policy_state = policy.get_initial_state(batch_size=batch_size)
#     action_step = policy.action(time_step, policy_state)
#     next_time_step = environment.step(action_step.action)
#     traj = trajectory.from_transition(time_step, action_step, next_time_step)
#
#     # Add trajectory to the replay buffer
#     buffer.add_batch(traj)
#
# def collect_data(env, policy, buffer, steps):
#     for _ in range(steps):
#         collect_step(env, policy, buffer)
#
#
#
# collect_data(train_env, my_random_tf_policy, replay_buffer, initial_collect_steps)
#
# dataset = replay_buffer.as_dataset(
#     sample_batch_size=batch_size,
#     num_steps=2, single_deterministic_pass=False).prefetch(3)
# iterator = iter(dataset)
#
#
#
#
#
#
# # (Optional) Optimize by wrapping some of the code in a graph using TF function.
# agent.train = common.function(agent.train)
#
# # Reset the train step
# agent.train_step_counter.assign(0)
#
# # Evaluate the agent's policy once before training.
# avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
# returns = [avg_return]
#
#
#
# for _ in range(num_iterations):
#
#   # Collect a few steps using collect_policy and save to the replay buffer.
#   collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
#
#   # Sample a batch of data from the buffer and update the agent's market_net.
#   experience, unused_info = next(iterator)
#   train_loss = agent.train(experience).loss
#
#   step = agent.train_step_counter.numpy()
#
#   if step % log_interval == 0:
#     print('step = {0}: loss = {1}'.format(step, train_loss))
#
#   if step % eval_interval == 0:
#     avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
#     print('step = {0}: Average Return = {1}'.format(step, avg_return))
#     returns.append(avg_return)