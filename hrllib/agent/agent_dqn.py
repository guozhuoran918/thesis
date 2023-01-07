from pytest import param
import torch
import random
import copy
from collections import deque
import pickle
import math
import numpy as np
import json
import sys, os
from hrllib.agent.utils import state2rep
from hrllib.agent import Agent
from hrllib.policy_learning.dqn_torch import DQN

class AgentDQN(Agent):
    def __init__(self,symptoms,conditions,parameter):
        super(AgentDQN,self).__init__(symptoms = symptoms,conditions = conditions,parameter=parameter)
        self.symptom_set  = symptoms
        self.all_symptoms = parameter.get("symptoms")
        self.all_symptoms_db = self.load_db(self.all_symptoms)
        self.input_dim = 9*len(self.all_symptoms_db.keys()) if parameter.get("nlice") else 3*len(self.all_symptoms_db.keys())
        self.action_space = self._build_action_space(self.symptom_set)
        # self.output_dim = env.num_symptoms + env.num_conditions
        self.output_dim = len(self.action_space)
        self.hidden_size = parameter.get('hidden_size_dqn')
        self.parameter = parameter
        self.state = None
        self.master_experience_replay_size = 10000
        self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        self.dqn = DQN(input_size=self.input_dim,hidden_size=self.hidden_size,output_size=self.output_dim,parameter=parameter)
        self.action_space = self._build_action_space(self.symptom_set)

    def next(self,state,turn,greedy_strategy,**kwargs):
        self.agent_action["turn"] = turn 
        state_rep = state2rep(state,self.all_symptoms_db,self.parameter)
        if greedy_strategy is True:
            greedy = random.random()
            if greedy < self.parameter.get('epsilon'):
                action_index = random.randint(0,len(self.action_space)-1)
            else:
                action_index = self.dqn.predict(Xs=[state_rep])[1]

        else:
            action_index = self.dqn.predict(Xs=[state_rep])[1]

      
        agent_action = copy.deepcopy(self.action_space[action_index])
        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"
        agent_action["action_index"] = action_index
        return agent_action,action_index

    def train(self,batch):
        loss = self.dqn.singleBatch(batch=batch,params=self.parameter)
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()

    def save_model(self,model_performance,episodes_index,checkpoint_path = None):
        self.dqn.save_model(model_performance,episodes_index,checkpoint_path)

    def train_dqn(self,**kwargs):
        """
        Train dqn.
        :return:
        """
        lower_rewards = []
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size", 16)
        group_id = kwargs.get("label")
        for iter in range(math.ceil(len(self.experience_replay_pool) / batch_size)):
                batch = random.sample(self.experience_replay_pool, min(batch_size, len(self.experience_replay_pool)))
                #print(batch)
                loss = self.train(batch=batch)
                cur_bellman_err += loss["loss"]
          
                temp = [x[2] for x in batch]
                lower_rewards.extend(temp)
  
        ave_lower_reward = np.mean(lower_rewards)
        print('*'+str(group_id)+' '+"cur bellman err %.4f, experience replay pool %s, ave lower reward %.4f" % (
                float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool),float(ave_lower_reward)))
        # print('*'+str(group_id)+' '+"cur bellman err %.4f, experience replay pool %s, ave lower reward %.4f" % (
        # float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool),float(ave_lower_reward)))


    def get_q_values(self,state,**kwargs):
        state = state2rep(state)
        Q_values, max_index = self.dqn.predict(Xs=[state])
        return Q_values.cpu().detach().numpy()

    # /*todo*/
    def reward_shaping(self, state, next_state):
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item
        # slot number in state.
        slot_dict = copy.deepcopy(state["current_slots"]["inform_slots"])
        slot_dict.update(state["current_slots"]["explicit_inform_slots"])
        slot_dict.update(state["current_slots"]["implicit_inform_slots"])
        slot_dict.update(state["current_slots"]["proposed_slots"])
        slot_dict.update(state["current_slots"]["agent_request_slots"])
        slot_dict = delete_item_from_dict(slot_dict, False)

        next_slot_dict = copy.deepcopy(next_state["current_slots"]["inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["explicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["implicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["proposed_slots"])
        next_slot_dict.update(next_state["current_slots"]["agent_request_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, False)
        gamma = self.parameter.get("gamma")
        return gamma * len(next_slot_dict) - len(slot_dict)

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
        shaping = self.reward_shaping(state, next_state) 
        alpha = 0.0
        if episode_over is True:
            pass
        else:
            reward = reward + alpha * shaping
        state_rep = state2rep(state=state, slot_set=self.all_symptoms_db, parameter=self.parameter) # sequence representation.
        next_state_rep =state2rep(state=next_state, slot_set=self.all_symptoms_db, parameter=self.parameter)
        self.experience_replay_pool.append((state_rep, agent_action, reward, next_state_rep, episode_over))
        #print(reward)

    
    def train_mode(self):
        self.dqn.current_net.train()

    def eval_mode(self):
        self.dqn.current_net.eval()

    def update_target_network(self):
        self.dqn.update_target_network()

    def load_db(self,file):
        with open(file) as fp:
            dbs = json.load(fp)
        return dbs
    
