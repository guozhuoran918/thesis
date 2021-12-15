import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from . import configuration as config
from collections import namedtuple, OrderedDict,defaultdict

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class MLP(nn.Module):
    def __init__(self, n_states,n_actions,hidden_dim=1024):
        """ 初始化q网络，为全连接网络
            n_states: 输入的特征数即环境的状态数
            n_actions: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, 512) # 输出层
        self.fc4 = nn.Linear(512,n_actions)
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, layer_config=None, non_linearity='relu'):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_config = self.form_layer_config(layer_config)
        self.non_linearity = non_linearity
        self.model = self.compose_model()

    def get_default_config(self):
        return [
            [self.input_dim, 48],
            [48, 32],
            [32, 32],
            [32, self.output_dim]
        ]

    def form_layer_config(self, layer_config):
        if layer_config is None:
            return self.get_default_config()

        if len(layer_config) < 2:
            raise ValueError("Layer config must have at least two layers")

        if layer_config[0][0] != self.input_dim:
            raise ValueError("Input dimension of first layer config must be the same as input to the model")

        if layer_config[-1][1] != self.output_dim:
            raise ValueError("output dimension of last layer config must be the same as expected model output")

        for idx in range(len(layer_config) - 1):
            assert layer_config[idx][1] == layer_config[idx+1][0], "Dimension mismatch between layers %d and %d" % (idx, idx + 1)

        return layer_config

    def get_non_linear_class(self):
        if self.non_linearity == 'tanh':
            return nn.Tanh
        else:
            return nn.ReLU

    def compose_model(self):
        non_linear = self.get_non_linear_class()
        layers = OrderedDict()
        for idx in range(len(self.layer_config)):
            input_dim, output_dim = self.layer_config[idx]
            layers['linear-%d' % idx] = nn.Linear(input_dim, output_dim)
            if idx != len(self.layer_config) - 1:
                layers['nonlinear-%d' % idx] = non_linear()

        return nn.Sequential(layers)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x)


class MedAgent:
    def __init__(self, env, **kwargs):
        self.env = env
        # 3 takes care of the age, gender and race and 3*num_symptoms represents the symptoms flattened out
        self.input_dim = kwargs.get('input_dim',8*env.num_symptoms+2)
        # self.output_dim = env.num_symptoms + env.num_conditions
        self.output_dim = kwargs.get('output_dim',env.num_symptoms+33)
        self.n_actions = self.output_dim
        self.layer_config = kwargs.get('layer_config', None)
        self.learning_start  = kwargs.get('learning_start', 1)
        self.batch_size = kwargs.get('batch_size', 1)
        self.gamma = kwargs.get('gamma', 0.999)
        self.eps_start = kwargs.get('eps_start', 0.9)
        self.epsilon = self.eps_start
        self.frame_idx = 0
        self.eps_end = kwargs.get('eps_end', 0.05)
        self.eps_decay = kwargs.get('eps_decay', 200)
        self.target_update = kwargs.get('target_update', 10)
        self.replay_capacity = kwargs.get('replay_capacity', 10)
        self.non_linearity = kwargs.get('non_linearity', 'relu')
        self.optimiser_name = kwargs.get('optimiser_name', 'adam')
        self.optimiser_params = kwargs.get('optimiser_params', {})
        self.debug = kwargs.get('debug', False)
        self.train = kwargs.get('train',True)
        self.context = kwargs.get('context',False)

        self.memory = ReplayMemory(self.replay_capacity)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0

        self.policy_network = MLP(self.input_dim, self.output_dim, 1024).to(
            self.device)
        self.target_network = MLP(self.input_dim, self.output_dim,1024).to(
            self.device)
        for target_param, param in zip(self.target_network.parameters(),self.policy_network.parameters()): # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        # self.target_network = DQN(self.input_dim, self.output_dim, self.layer_config, self.non_linearity).to(
        #     self.device)

        # we aren't interested in tracking gradients for the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        # self.target_network.eval()

        optimiser_cls = self.get_optimiser()
        self.optimiser = optimiser_cls(self.policy_network.parameters(), **self.optimiser_params)

        self.state = None
        self.reset_env()

    def reset_env(self):
        self.env.reset()
        self.state = self.state_to_tensor(self.env.state)

    def state_to_tensor(self, state):
        if state is None:
            return None

        tensor = np.zeros(self.input_dim)

        # tensor[0] = state.gender
        # tensor[1] = state.race
        # tensor[2] = state.age
        if self.context:
            tensor[0] = state.age
            tensor[1] = state.gender
            tensor[2:]=state.symptoms.reshape(-1)
        else:
            tensor[0:] = state.symptoms.reshape(-1)

        return torch.tensor(tensor, device=self.device, dtype=torch.float).reshape(-1, self.input_dim)

    def get_optimiser(self):
        if self.optimiser_name == 'sgd':
            optimiser = optim.RMSprop
        elif self.optimiser_name == 'adam':
            optimiser = optim.Adam
        else:
            optimiser = optim.RMSprop

        return optimiser

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        if self.steps_done < self.learning_start:
            return None

        transitions = self.memory.sample(self.batch_size)

        # see https://stackoverflow.com/a/19343/3343043 for detailed explanation
        batch = Transition(*zip(*transitions))

        loss = self.compute_loss(batch)

        self.optimiser.zero_grad()
        loss.backward()

        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimiser.step()

        return loss

    def double_q_update(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        # for target_param, param in zip(self.target_network.parameters(),self.policy_network.parameters()): # 复制参数到目标网路targe_net
        #     target_param.data.copy_(param.data)

    def compute_loss(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_network.forward(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target_network.forward(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        return loss

    def select_action(self, state):
        sample = random.random()
        self.frame_idx += 1
        if sample < self.epsilon(self.frame_idx):
            if self.debug:
                print("Taking Random Action")
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            if self.debug:
                print("Taking Best Action")
            with torch.no_grad():
                return self.policy_network.forward(state).max(1)[1].view(1, 1)
            
    def top_results(self,state):
        return self.policy_network.forward(state)

    def decay_epsilon(self):
        # if self.steps_done > self.learning_start:
            # self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
            #                np.exp(-1 * (self.steps_done-self.learning_start) / self.eps_decay)
        self.epsilon = lambda frame_idx: self.eps_end + \
            (self.eps_start - self.eps_end) * \
            math.exp(-1. * frame_idx / self.eps_decay)

    def step(self):
        self.decay_epsilon()
        state = self.state
        action = self.select_action(state)
        values,indices = self.top_results(state).topk(5)

        if self.debug:
            # print("State: ", state)
            print(self.env.state.symptoms)
            print(np.where(self.env.state.symptoms == 1)[0])
            print("Took action: ", action)
            print("top 5 action",indices)

        self.steps_done += 1

        # if action.item()>=61:
        #     # print(indices.tolist())
        #     next_state, _reward, done = self.env.top_5(indices.tolist()[0])
        # else:
        next_state, _reward, done = self.env.take_action(action.item())
        if self.debug:
            print("Next state: ", next_state)
            print("Reward: ", _reward)
            print("Done: ", done)
        next_state = self.state_to_tensor(next_state)
        reward = torch.tensor([_reward], device=self.device)
        if self.train:
            self.memory.push(state, action, next_state, reward)
        self.state = next_state
        # if self.train:
        #     if self.steps_done % self.target_update == 0:
        #         self.double_q_update()
            # print("update")

        return _reward, done

    def save(self, path):
        torch.save(self.target_network.state_dict(), path+'dqn_checkpoint.pth')
        torch.save(self.policy_network.state_dict(),path+'dqn_checkpoint_policy.pth')

    def load(self, path):
        self.target_network.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            param.data.copy_(target_param.data)

    def __del__(self):
        del self.env

class AgentQlearning(object):
    def __init__(self, env, config,debug):
        self.env = env
        self.debug = debug
        self.state_dim = 3 + 3*env.num_symptoms
        # self.action_dim = env.num_symptoms + env.num_conditions
        self.action_dim = env.num_symptoms+1
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = 0
        self.sample_count = 0
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.Q_table = defaultdict(lambda:np.zeros(self.action_dim))
    
    def select_action(self,state):
        self.sample_count+=1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) # 选择Q(s,a)最大对应的动作
        else:
            action = np.random.choice(self.action_dim) # 随机选择动作
        return action
    
    def predict(self,state):
        action = np.argmax(self.Q_table[str(state)])
        return action
    
    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action] 
        if done == config.DIALOGUE_STATUS_FAILED or config.DIALOGUE_STATUS_SUCCESS: # end 
            Q_target = reward  
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)]) 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
    
    def save(self,path):
        import dill
        torch.save(
            obj=self.Q_table,
            f=path+"Qleaning_model.pkl",
            pickle_module=dill
        )
        print("save model successfully")
    
    def load(self, path):
        import dill
        self.Q_table =torch.load(f=path+'Qleaning_model.pkl',pickle_module=dill)
        print("load model successfully")
    
    def reset_env(self):
        self.env.reset()
        self.state = self.env.state

    def step(self):
        state = self.state
        action = self.select_action(state)
        if self.debug:
            print("State: ", state)
            print("Took action: ", action)
        next_state, _reward, done = self.env.take_action(action)
        self.update(state,action,_reward,next_state,done)
        self.state = next_state
        return next_state,_reward,done

    def __del__(self):
        del self.env



class AgentQlearning_v1(object):
    '''
    action space: inquiry,diagnose
    '''
    def __init__(self, env, config,debug):
        self.env = env
        self.debug = debug
        self.state_dim = 3 + 3*env.num_symptoms
        # self.action_dim = env.num_symptoms + env.num_conditions
        self.action_dim = 2
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = 0
        self.sample_count = 0
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.Q_table = defaultdict(lambda:np.zeros(self.action_dim))
    
    def select_action(self,state):
        self.sample_count+=1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) # 选择Q(s,a)最大对应的动作
        else:
            action = np.random.choice(self.action_dim) # 随机选择动作
        return action
    
    def predict(self,state):
        action = np.argmax(self.Q_table[str(state)])
        return action
    
    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action] 
        if done == config.DIALOGUE_STATUS_FAILED or config.DIALOGUE_STATUS_SUCCESS: # end 
            Q_target = reward  
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)]) 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
    
    def save(self,path):
        import dill
        torch.save(
            obj=self.Q_table,
            f=path+"Qleaning_model.pkl",
            pickle_module=dill
        )
        print("save model successfully")
    
    def load(self, path):
        import dill
        self.Q_table =torch.load(f=path+'Qleaning_model.pkl',pickle_module=dill)
        print("load model successfully")
    
    def reset_env(self):
        self.env.reset()
        self.state = self.env.state

    def step(self):
        state = self.state
        action = self.select_action(state)
        if self.debug:
            print("State: ", state)
            print("Took action: ", action)
        next_state, _reward, done = self.env.take_action(action)
        self.update(state,action,_reward,next_state,done)
        self.state = next_state
        return next_state,_reward,done
    
    def get_candidate_symptoms(self):
        pass


    def get_top3_diseases(self,symtom):
        pass

    def get_top5_symtoms(self,disease):
        pass


    def __del__(self):
        del self.env