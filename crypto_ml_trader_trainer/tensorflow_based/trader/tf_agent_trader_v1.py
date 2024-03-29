
#%%
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
import tensorflow as tf
import pandas as pd
import tensorflow.keras.layers as kl
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.networks import sequential

from crypto_ml_trader_trainer.tensorflow_based.trader.market_indicator_tf_environment import MarketIndicatorTfEnvironment

# Data and env creation
from crypto_ml_trader_trainer.utilities.DateUtility import dateparse

data_file_path = "./input/ohlc_btc-usd_1min_2021.csv"
train_ratio = 0.7

data = pd.read_csv(data_file_path,
                        delimiter=',',
                        names=["time", "open", "high", "low", "close", "volume", "trades"],
                        parse_dates=True,
                        date_parser=dateparse,
                        index_col='time')
data_size = len(data)
data_train_size = round(data_size * train_ratio)

data_train = data.iloc[:data_train_size]
data_val = data.iloc[data_train_size:]

environment_train = MarketIndicatorTfEnvironment(data_train)
environment_val = MarketIndicatorTfEnvironment(data_val)

# utils.validate_py_environment(environment, episodes=2)

#%%
# Random Policy

# noinspection PyTypeChecker
# my_random_py_policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
#     action_spec=environment_train.action_spec())
#
#
# def compute_avg_return(environment, policy, num_episodes=1000):
#
#     total_return = 0.0
#     for _ in range(num_episodes):
#
#         time_step = environment.reset()
#         episode_return = 0.0
#
#         while not time_step.is_last():
#             action_step = policy.action(time_step)
#             time_step = environment.step(action_step.action)
#             episode_return += time_step.reward
#         total_return += episode_return
#
#     avg_return = total_return / num_episodes
#     return avg_return
#
#
# avg_return = compute_avg_return(environment_train, my_random_py_policy, 10)
# print(f"Average return {avg_return}")



#%%
# DQN Policy
# ========================================

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 2000  # @param {type:"integer"}
collect_steps_per_iteration = 30  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 32  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}



train_env = tf_py_environment.TFPyEnvironment(environment_train, check_dims=True)
eval_env = tf_py_environment.TFPyEnvironment(environment_val)

batch_size = train_env.batch_size

fc_layer_params = (100, 50)
action_tensor_spec = train_env.action_spec()
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# def create_fc_network(layer_units):
#   return sequential.Sequential([dense_layer(num_units) for num_units in layer_units])



#%%
# DQN part 2
#====================

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# it's output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential([kl.Reshape((15,15,1)), kl.Conv2D(filters=64, kernel_size=(5,5), strides=3), kl.Flatten()])
q_net2 = sequential.Sequential([kl.Dense(59), kl.Reshape((59,))])


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

net = sequential.Sequential([
    dense_layer(100),
    dense_layer(50),
    q_values_layer
])


agent = dqn_agent.DqnAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    q_network=net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()
# net.summary()


eval_policy = agent.policy
collect_policy = agent.collect_policy

# noinspection PyTypeChecker
my_random_tf_policy = random_tf_policy.RandomTFPolicy(time_step_spec=train_env.time_step_spec(),
                                                      action_spec=train_env.action_spec())

#%%
## DQN PART 3

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tensor_spec.from_spec(agent.collect_data_spec),
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        policy_state = policy.get_initial_state(batch_size=batch_size)

        while not time_step.is_last():
            action_step = policy.action(time_step, policy_state)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    policy_state = policy.get_initial_state(batch_size=batch_size)
    action_step = policy.action(time_step, policy_state)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)



collect_data(train_env, my_random_tf_policy, replay_buffer, initial_collect_steps)

dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2, single_deterministic_pass=False).prefetch(3)
iterator = iter(dataset)



# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]



for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

  # Sample a batch of data from the buffer and update the agent's market_net.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)