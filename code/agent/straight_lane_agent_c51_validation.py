import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('/')
    sys.path.append('../simulation2')
    sys.path.append('../environment2')
    sys.path.append('../../carla/')
except IndexError:
    print('Modules not found.')
    exit(-1)

import tensorflow as tf
from tensorflow_addons.layers.gelu import GELU
from tensorflow_addons.optimizers.yogi import Yogi
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.categorical_q_network import CategoricalQNetwork
from tf_agents.agents.categorical_dqn.categorical_dqn_agent import CategoricalDqnAgent
from tf_agents.utils.common import Checkpointer
from environment.carla_environment import CarlaEnvironment

# Allowing tensorflow to expand in physical memory, if it fails to allocate memory in GPU.
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# 1. Creating the tf-environment for validation.
carla_environment = CarlaEnvironment(log_dir='validation_log4/')
eval_env = TFPyEnvironment(environment=carla_environment)

# Allowing tensorflow to expand in physical memory, if it fails to allocate memory in GPU.
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# 2. Constructing the Categorical QNetworks: Online & Target.
# Default Activation Function: "Gelu".
# Default Weight Initialization: "He (Xavier) Initialization".
fc_layer_units = [128, 128]
conv_layer_units = [ (4, 3, 1) ]
num_atoms = 51

online_q_net = CategoricalQNetwork(
    input_tensor_spec=eval_env.observation_spec(),
    action_spec=eval_env.action_spec(),
    num_atoms=num_atoms,
    conv_layer_params=conv_layer_units,
    fc_layer_params=fc_layer_units,
    activation_fn=GELU()
)
target_q_net = CategoricalQNetwork(
    input_tensor_spec=eval_env.observation_spec(),
    action_spec=eval_env.action_spec(),
    num_atoms=num_atoms,
    conv_layer_params=conv_layer_units,
    fc_layer_params=fc_layer_units,
    activation_fn=GELU()
)

# Defining train_step, which will be used to store the current step.
train_step = tf.Variable(initial_value=0)
total_steps = 150000

# Defining decay epsilon-greedy strategy.
decay_epsilon_greedy = PolynomialDecay(
    initial_learning_rate=0.9,
    decay_steps=total_steps,
    end_learning_rate=0.001,
)

# 3. Constructing the DQN Agent.
optimizer = Yogi(learning_rate=0.00025)
loss = Huber()
n_steps = 3
tau = 0.001
gamma = 0.99
min_q = -200
max_q = 200

agent = CategoricalDqnAgent(
    time_step_spec=eval_env.time_step_spec(),
    action_spec=eval_env.action_spec(),
    categorical_q_network=online_q_net,
    optimizer=optimizer,
    min_q_value=min_q,
    max_q_value=max_q,
    epsilon_greedy=lambda: decay_epsilon_greedy(train_step),
    n_step_update=n_steps,
    target_categorical_q_network=target_q_net,
    target_update_tau=tau,
    target_update_period=1,
    td_errors_loss_fn=loss,
    gamma=gamma,
    train_step_counter=train_step
)
agent.initialize()

# 3. Restoring agent's training progress (Checkpoint)...
checkpoint_dir = 'checkpoint/'
train_checkpointer = Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy
)
train_checkpointer.initialize_or_restore()


# 8. Evaluating the agent.
def evaluate(env, policy, num_episodes):
    total_return = 0.0

    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return

    average_return = total_return / num_episodes
    return average_return.numpy()[0]


# Resetting the train step.
agent.train_step_counter.assign(0)

# Resetting eval environment.
eval_env.reset()

# Evaluate the agent's policy once before training.
num_of_episodes = 1
avg_return = evaluate(eval_env, agent.policy, num_of_episodes)

print('\nAverage return in', num_of_episodes, 'episodes =', avg_return)

carla_environment.close()