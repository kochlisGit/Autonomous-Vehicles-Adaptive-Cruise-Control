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
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics.tf_metrics import AverageReturnMetric, AverageEpisodeLengthMetric
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import random_tf_policy
from tf_agents.utils.common import Checkpointer
from environment.carla_environment import CarlaEnvironment
import matplotlib.pyplot as plt

# Allowing tensorflow to expand in physical memory, if it fails to allocate memory in GPU.
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# 1. Creating the tf-environment for training.
carla_environment = CarlaEnvironment()
train_env = TFPyEnvironment(environment=carla_environment)

# 2. Constructing the Categorical QNetworks: Online & Target.
# Default Activation Function: "Gelu".
# Default Weight Initialization: "He (Xavier) Initialization".
fc_layer_units = [128, 128]
conv_layer_units = [ (4, 3, 1) ]
num_atoms = 51

online_q_net = CategoricalQNetwork(
    input_tensor_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
    num_atoms=num_atoms,
    conv_layer_params=conv_layer_units,
    fc_layer_params=fc_layer_units,
    activation_fn=GELU()
)
target_q_net = CategoricalQNetwork(
    input_tensor_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
    num_atoms=num_atoms,
    conv_layer_params=conv_layer_units,
    fc_layer_params=fc_layer_units,
    activation_fn=GELU()
)

# Defining train_step, which will be used to store the current step.
train_step = tf.Variable(initial_value=0)
total_steps = 30000

# Defining decay epsilon-greedy strategy.
decay_epsilon_greedy = PolynomialDecay(
    initial_learning_rate=0.7,
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
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
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

# 4. Constructing the Replay Memory.
memory_size = 20000
batch_size = 64

replay_buffer = TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=memory_size
)

# Initializing Observer of replay buffer to store experiences (trajectories) to memory.
replay_buffer_observer = replay_buffer.add_batch

# Defining Metrics for measuring training progress.
train_metrics = [ AverageReturnMetric(), AverageEpisodeLengthMetric() ]

# 5. Defining initial policy as random to collect enough examples to fill the memory buffer (Training delay).
initial_collect_policy = random_tf_policy.RandomTFPolicy( train_env.time_step_spec(), train_env.action_spec() )
initial_collect_steps = 2000


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 1000 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


init_driver = DynamicStepDriver(
    env=train_env,
    policy=initial_collect_policy,
    observers=[ replay_buffer.add_batch, ShowProgress(initial_collect_steps) ],
    num_steps=initial_collect_steps
)

# Collecting experiences.
print('Collecting random initial experiences...')
init_driver.run()

# 6. Training the agent.
dataset = replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=n_steps+1, num_parallel_calls=3).prefetch(3)

all_train_loss = []
all_metrics = []

collect_driver = DynamicStepDriver(
    env=train_env,
    policy=agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=n_steps+1
)

# 7. Building Policy Saver & Checkpointer (Training Saver).
checkpoint_dir = 'checkpoint/'
train_checkpointer = Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy
)
train_checkpointer.initialize_or_restore()


def train_agent(num_steps, checkpointer):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    dataset_iter = iter(dataset)

    for step in range(num_steps+1):
        current_metrics = []

        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(dataset_iter)

        train_loss = agent.train(trajectories)
        all_train_loss.append( train_loss.loss.numpy() )

        for i in range( len(train_metrics) ):
            current_metrics.append(train_metrics[i].result().numpy())

        all_metrics.append(current_metrics)

        if step % 1000 == 0:
            print( "\nIteration: {}, loss:{:.2f}".format( step, train_loss.loss.numpy() ) )

            for i in range( len(train_metrics) ):
                print( '{}: {}'.format( train_metrics[i].name, train_metrics[i].result().numpy() ) )

        if step % 10000 == 0:
            checkpointer.save(step)
            print('Training has been saved.')


train_agent(total_steps, train_checkpointer)

# 8. Plotting metrics and results.
average_return = [ metric[0] for metric in all_metrics ]
plt.plot(average_return)
plt.show()

episode_length = [ metric[1] for metric in all_metrics ]
plt.plot(episode_length)
plt.show()

plt.plot( carla_environment.get_num_of_episode_steps() )
plt.show()

plt.plot( carla_environment.get_episode_mean_target_speed() )
plt.show()

# Terminating client's connection and closing the training environment.
train_env.close()
carla_environment.close()
