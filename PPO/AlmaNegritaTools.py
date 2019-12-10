import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from .utils import discount_rewards
from pdb import set_trace as bp
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


class ActorNN(torch.nn.Module):
    def __init__(self, state_space, action_space, train_device='cpu'):
        """
        This is the NN model, which takes in the state space, and action space parameters. This model
        contains 100 neurons hidden layer.
        :param state_space:
        :param action_space:
        :param train_device:
        """
        super().__init__()
        self.train_device = train_device
        self.state_space = state_space
        self.name = 'ActorNN'
        # bp()
        self.flat_state_space = self.state_space[0] * self.state_space[1] * self.state_space[2]
        self.action_space = action_space
        self.hidden = 100
        self.state_value = 0
        self.fc_0 = torch.nn.Linear(in_features=self.flat_state_space, out_features=self.hidden)
        self.fc_1 = torch.nn.Linear(in_features=self.hidden, out_features=self.action_space)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 0.5)
                torch.nn.init.uniform_(m.bias, -0.3, 0.3)

    def forward(self, x):
        # bp()
        if len(x.shape) > 3:
            x = x.flatten(start_dim=1)
        else:
            x = x.flatten()
        x = self.fc_0(x)
        x = F.relu(x) # The idea was to produce a simple non-linearity
        x = self.fc_1(x)
        # bp()
        if len(x.shape) < 2:
            new_shape = tuple(torch.cat((torch.tensor([1]), torch.tensor(x.shape))))
            x = torch.reshape(x,
                              shape=new_shape)
        x = self.softmax(x)
        x = torch.distributions.Categorical(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                m.weight += torch.distributions.Normal(torch.zeros_like(m.weight), torch.ones_like(m.weight))
                m.bias += torch.distributions.Uniform(-0.3, 0.3)


class ActorCNN(torch.nn.Module):
    def __init__(self, state_space, action_space, train_device='cpu',
                 in_channels=1):
        """
        This is the CNN actor model.
        :param state_space:
        :param action_space:
        :param train_device:
        :param in_channels:
        """
        super().__init__()
        self.train_device = train_device
        self.state_space = state_space
        # bp()
        self.name = 'ActorCNN'
        self.flat_state_space = self.state_space[0] * self.state_space[1] * self.state_space[2]
        self.action_space = action_space
        self.hidden = 100
        self.state_value = 0
        self.cnn_0 = torch.nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=3)
        new_size = self.state_space[1:] - self.cnn_0.kernel_size + 1
        # size: kernel_size + 1
        self.cnn_1 = torch.nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3)
        new_size = new_size - self.cnn_1.kernel_size + 1
        # size: kernel_size + 1
        # 28 x 28
        flat_size = new_size[0] * new_size[1] * self.cnn_1.out_channels
        self.fc_0 = torch.nn.Linear(in_features=flat_size, out_features=self.hidden)
        self.fc_1 = torch.nn.Linear(in_features=self.hidden, out_features=self.action_space)
        self.softmax = torch.nn.Softmax(dim=1)
    #     self.init_weights()
    #
    # def init_weights(self):
    #     for m in self.modules():
    #         if type(m) is torch.nn.Linear:
    #             torch.nn.init.normal_(m.weight)
    #             torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # bp()
        x = self.cnn_0(x)
        x = F.relu(x)
        x = self.cnn_1(x)
        x = F.relu(x)
        # bp()
        if len(x.shape) > 3:
            x = x.flatten(start_dim=1)
        else:
            x = x.flatten()
        x = self.fc_0(x)
        x = F.relu(x)
        x = self.fc_1(x)
        # bp()
        if len(x.shape) < 2:
            new_shape = tuple(torch.cat((torch.tensor([1]), torch.tensor(x.shape))))
            x = torch.reshape(x,
                              shape=new_shape)
        x = self.softmax(x)
        x = torch.distributions.Categorical(x)
        return x
        
class CriticCNN(torch.nn.Module):
    def __init__(self, state_space, action_space=None, train_device='cpu',
                 in_channels=1):
        super().__init__()
        self.train_device = train_device
        self.state_space = state_space
        # bp()
        self.name = 'CriticCNN'
        self.flat_state_space = self.state_space[0] * self.state_space[1] * self.state_space[2]
        self.action_space = action_space
        self.hidden = 100
        self.state_value = 0
        self.cnn_0 = torch.nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=3)
        new_size = self.state_space[1:] - self.cnn_0.kernel_size + 1
        # size: kernel_size + 1
        self.cnn_1 = torch.nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3)
        new_size = new_size - self.cnn_1.kernel_size + 1
        # size: kernel_size + 1p
        # 28 x 28
        flat_size = new_size[0] * new_size[1] * self.cnn_1.out_channels
        self.fc_0 = torch.nn.Linear(in_features=flat_size, out_features=self.hidden)
        self.fc_1 = torch.nn.Linear(in_features=self.hidden, out_features=1)
        # self.softmax = torch.nn.Softmax(dim=1)
    #     self.init_weights()
    #
    # def init_weights(self):
    #     for m in self.modules():
    #         if type(m) is torch.nn.Linear:
    #             torch.nn.init.normal_(m.weight)
    #             torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # bp()
        x = self.cnn_0(x)
        x = F.relu(x)
        x = self.cnn_1(x)
        x = F.relu(x)
        # bp()
        if len(x.shape) > 3:
            x = x.flatten(start_dim=1)
        else:
            x = x.flatten()
        x = self.fc_0(x)
        x = F.relu(x)
        x = self.fc_1(x)
        # bp()
        if len(x.shape) < 2:
            new_shape = tuple(torch.cat((torch.tensor([1]), torch.tensor(x.shape))))
            x = torch.reshape(x,
                              shape=new_shape)
        # x = F.relu(x)
        return x


class Critic(torch.nn.Module):
    def __init__(self, state_space, train_device='cpu'):
        super().__init__()
        self.train_device = train_device
        self.state_space = state_space
        self.flat_state_space = self.state_space[0] * self.state_space[1] * self.state_space[2]
        self.hidden = 100
        self.state_value = 0
        # bp()
        self.fc_0 = torch.nn.Linear(in_features=self.flat_state_space, out_features=self.hidden)
        self.fc_1 = torch.nn.Linear(in_features=self.hidden, out_features=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 0.5)
                torch.nn.init.uniform_(m.bias, -0.3, 0.3)

    def forward(self, x):
        # bp()
        # print(x.shape)
        if len(x.shape) > 3:
            x = x.flatten(start_dim=1)
        else:
            x = x.flatten()
        x = self.fc_0(x)
        x = F.relu(x)
        x = self.fc_1(x)
        # x = F.relu(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                m.weight += torch.distributions.Normal(torch.zeros_like(m.weight), torch.ones_like(m.weight))
                m.bias += torch.distributions.Uniform(-0.3, 0.3)


class ProximalPolicyOpt(object):
    def __init__(self, state_space, action_space, train_device='cuda', tf_monitoring=False,
                 cnn_in_channels=1, model_type='NN'):
        self.train_device = train_device
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.state_value = 0
        self.actor_learning_rate = 1e-3 # This was the learning rate that performed the best.
        # Larger learning rates caused the agent smaller learning averages.
        self.critic_learning_rate = 1e-3
        self.in_channels = cnn_in_channels
        if model_type == 'NN':
            self.actor = ActorNN(self.state_space, self.action_space, train_device=train_device).to(train_device)
            self.critic = Critic(self.state_space, train_device=train_device).to(train_device)
        else:
            self.actor = ActorCNN(self.state_space, self.action_space, train_device=train_device,
                                  in_channels=self.in_channels).to(train_device)
            self.critic = CriticCNN(self.state_space, self.action_space, train_device=train_device,
                                    in_channels=self.in_channels).to(train_device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_learning_rate)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_learning_rate)
        self.clip_parameter = 0.2 # parameter suggested by PPO paper
        self.update_time = 10 # parameter initially suggested by reference implementation. Different
        # values were attempted, but 10 ended up performining the best.
        self.max_grad_norm = 0.5 # Parameter suggested by reference implementation
        self.training_step = 0
        self.writer = SummaryWriter('./training_logs')
        self.tensorboard_monitoring = tf_monitoring

    def get_action(self, observation):
        """
        get action based on the policy. This functions expects observation to be processed before it is given
        to the policy. It should be processed by the agent using the policy.
        :param observation:
        :return:
        """
        observation = observation.to(self.train_device)
        with torch.no_grad():
            action_probabilities = self.actor.forward(observation)
        # bp()
        action = action_probabilities.sample()
        # act_log_prob = action_probabilities.log_prob(action)
        act_log_prob = action_probabilities.probs[0][action]
        return action, act_log_prob

    def get_value(self, observation):
        observation = observation.to(self.train_device)
        with torch.no_grad():
            value_estimate = self.critic.forward(observation)
        return value_estimate

    def load_model(self, actor, critic):
        self.actor.load_state_dict(actor, strict=False)
        self.critic.load_state_dict(critic, critic=False)

    def save_parameters(self, storage_directory='models/', unique_identifier=None):
        if unique_identifier is None:
            unique_identifier = time.strftime('%Y-%m-%d-%H-%M-%S')
        actor_file_name = storage_directory + 'actor_' + unique_identifier + '.pckl'
        critic_file_name = storage_directory + 'critic_' + unique_identifier + '.pckl'
        torch.save(self.actor.state_dict(), actor_file_name)
        torch.save(self.critic.state_dict(), critic_file_name)

    # TODO: Verify that update function works
    def update(self, states, actions, action_probs,
               discounted_rewards, episode_number="not_set",
               batch_size=32, global_timestep=None):
        """
            The idea is that data is passed for this function in a format
            that is already to be processed. All data related processing is 
            done by the agent. 
        """
        # bp()
        for i in range(self.update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(states))),
                                      batch_size=batch_size,
                                      drop_last=False):
                current_values = self.critic(states[index]).squeeze(-1)
                current_advantage = discounted_rewards[index] - current_values
                current_advantage = current_advantage.detach()
                current_action_probs = self.actor(states[index])
                current_actions = actions[index]
                current_action_probs = current_action_probs.probs[range(len(current_actions)), current_actions]
                # idea is to see how the actions that were taken affect the current state of the network.
                # Below is where we calculate the ratio and surrogates according to PPO
                ratio = current_action_probs / action_probs[index]
                if self.tensorboard_monitoring:
                    for item in ratio:
                        self.writer.add_scalar('loss/ratio',
                                               item,
                                               global_step=self.training_step)
                surrogate_1 = ratio * current_advantage
                surrogate_2 = torch.clamp(ratio, 1 - self.clip_parameter, 1 + self.clip_parameter) * current_advantage
                # Below is where we compute the losses according to PPO
                action_loss = - torch.min(surrogate_1, surrogate_2).mean()
                if self.tensorboard_monitoring:
                    self.writer.add_scalar('loss/action_loss',
                                            action_loss.detach(),
                                            global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                # We also tried without clipping the gradients, however, performance did not improve.
                self.actor_optimizer.step()

                # Below is where we update the critic
                value_loss = F.mse_loss(discounted_rewards[index], current_values)
                if self.tensorboard_monitoring:
                    self.writer.add_scalar('loss/value_loss',
                                            value_loss.detach(),
                                            global_step=self.training_step)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.training_step += 1
        return 0


class Agent(object):
    def __init__(self, policy=None, desired_observation_size=[150, 150], train_device='cuda', batch_size=64,
                 buffer_size=1024, in_channels=1, model_type='NN'):
        self.train_device = train_device
        self.name = 'Alma_Negrita'
        # Load default model with parameters
        if policy is None:
            cnn_in_channels = in_channels
            desired_size = [cnn_in_channels, desired_observation_size[0], desired_observation_size[1]]
            observation_space_dim = np.array(desired_size)
            action_space_dim = 3
            self.policy = ProximalPolicyOpt(observation_space_dim, action_space_dim, train_device='cuda',
                                            tf_monitoring=False, cnn_in_channels=cnn_in_channels,
                                            model_type=model_type)

        else:
            self.policy = policy
        self.gamma = 0.99 # Tried with different parameters, but this performed the best. In these larger games,
        # gamma = 0.99 seems to be the benchmark, as we don't want to consider rewards too far back.
        self.states = []
        self.action_probs = []
        self.in_channels = in_channels
        self.rewards = []
        self.actions = []
        self.state_values = []
        self.buffer_size = buffer_size
        self.action_space = [0, 1, 2]
        self.previous_observation = None
        self.desired_observation_size = np.array(desired_observation_size)
        self.preprocess_image = torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(num_output_channels=1),
             torchvision.transforms.Resize(size=self.desired_observation_size),
             torchvision.transforms.ToTensor(),
             # torchvision.transforms.Normalize(mean=[0.0], std=[1.0]),

        ])
        self.batch_size = batch_size

    @staticmethod
    def discount_rewards(r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def episode_update(self, episode_number=0, global_timestep=None):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        actions = torch.stack(self.actions, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0) \
            .to(self.train_device).squeeze(-1)
        # bp()
        if self.in_channels == 2:
            states = self.get_two_layer_observation(states, batch_update=True)
        self.states, self.action_probs, self.rewards, self.state_values, self.actions = [], [], [], [], []
        discounted_rewards = self.discount_rewards(rewards, self.gamma)
        self.policy.update(states, actions, action_probs,
                           discounted_rewards, episode_number=episode_number,
                           batch_size=self.batch_size, global_timestep=global_timestep)

    def reset(self):
        self.states, self.action_probs, self.rewards, self.state_values, self.actions = [], [], [], [], []
        self.previous_observation = None

    def get_name(self):
        return self.name

    def load_model(self):
        actor_weights = torch.load('./actor.mdl')
        critic_weights = torch.load('./critic.mdl')
        self.policy.load_model(actor=actor_weights, critic=critic_weights)

    def get_action(self, observation):
        observation = self.process_observation(observation)
        x = self.get_basic_observation(observation)
        # In case we have chosen to work with 2 input channels, then we process the input accordingly.
        # This is where we work with the alternative, 1st layer observation, 2nd layer difference between current
        # and previous observation.
        if self.in_channels == 2:
            x = self.get_two_layer_observation(x)
        action, action_probabilities = self.policy.get_action(x)
        # bp()
        state_value = self.policy.get_value(x)
        self.previous_observation = observation.to(self.train_device)
        return action, action_probabilities, state_value

    def get_difference_observation(self, observation):
        if self.previous_observation is not None:
            # x = torch.abs(observation - self.previous_observation) # / self.previous_observation
            # bp()
            x = observation.to(self.train_device) - self.previous_observation
        else:
            x = observation
        return x

    def get_basic_observation(self, observation):
        # bp()

        if len(observation.shape) < 4:
            new_shape = tuple(torch.cat((torch.tensor([1]), torch.tensor(observation.shape))))
            observation = torch.reshape(observation,
                                        shape=new_shape)
        observation = observation.to(self.train_device)
        return observation

    def get_two_layer_observation(self, observation, batch_update=False):
        if not batch_update:
            diff_observation = self.get_difference_observation(observation)
            # bp()
            x = torch.cat((observation, diff_observation), dim=1)
            return x
        else:
            # bp()
            diff_observation = observation[1:] - observation[:observation.shape[0] - 1]
            x = torch.cat((observation[1:], diff_observation), dim=1)
            return x

    def process_observation(self, observation):
        observation[observation == 43] = 0
        observation[observation == 48] = 0
        observation[observation == 58] = 0
        # Above, we are basically zeroing the background, so there is a clear
        # distinction between ball and agents.
        observation = self.preprocess_image(Image.fromarray(observation))
        # Simple preprocessing of image into grey scale.
        # observation = self.get_difference_observation(observation)
        # observation[observation != 0] = 1
        return observation

    def save_policy_parameters(self, unique_name=None):
        self.policy.save_parameters(unique_identifier=unique_name)

    def store_outcome(self, observation, action_prob,
                      action_taken, reward,
                      state_value,
                      ):
        if self.buffer_size > len(self.rewards):
            # As long as we are within the buffer_size limit, we keep appending values.
            observation = self.process_observation(observation)
            self.states.append(observation)
            self.action_probs.append(torch.tensor(action_prob))
            self.rewards.append(torch.Tensor([reward]))
            self.actions.append(action_taken)
            self.state_values.append(state_value)



