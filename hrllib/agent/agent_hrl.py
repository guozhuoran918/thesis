import json
import numpy as np
import copy
import sys, os
from pytest import TempPathFactory
import torch
import random
import math
from collections import deque
from hrllib import configuration
from hrllib.agent.agent_dqn import AgentDQN as LowerAgent
from hrllib.policy_learning.dqn_torch import DQN,DQN2
from hrllib.agent.utils import state2rep
from hrllib.classifier import dl_classifier

class AgentHRL(object):
    def __init__(self,symptom_set,disease_set,parameter):
        # self.action_set = self.load_db(action_set)
        self.symptom_set = self.load_db(symptom_set)
        self.parameter = parameter
        self.diseases =  self.load_db(disease_set)
        self.master_experience_replay_size = 10000
        self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        self.id2lowerAgent={}
        self.count = 0
        self.disease_replay = deque(maxlen=10000)
        self.master_action_space = []
        self.group_id = [1,2,3,4,5]
        self.all_data = parameter.get("file_all")
        self.past_lower_agent_pool = {key: 0 for key in self.id2lowerAgent.keys()}
        # build lower agents
        temp_parameter={}
        for id in self.group_id:
            label = str(id)
            self.master_action_space.append(label)
            group_path = os.path.join(self.all_data,'group'+str(label))
            all_symptom_file = group_path+"/"+"symptom"+str(id)+".json"
            conditon_file = group_path+"/"+"condition"+str(id)+".json"
            group_symptom_db = self.load_db(all_symptom_file)
            group_condition_db = self.load_db(conditon_file)
            temp_parameter[label] = copy.deepcopy(parameter)
            self.id2lowerAgent[label] = LowerAgent(group_symptom_db,group_condition_db,parameter=temp_parameter[label])
            path_list = parameter["saved_model"].split('/')
            path_list.insert(-1,"lower")
            path_list.insert(-1,str(label))
            temp_parameter[label]['saved_model'] = '/'.join(path_list)
            temp_parameter[label]['gamma'] = temp_parameter[label]['gamma_worker']

        # build master agent
        self.input_size = len(self.symptom_set.keys())*8
        self.hidden_size = parameter.get("hidden_size_dqn", 100)
        self.output_size = len(self.id2lowerAgent)+1
        self.master = DQN2(input_size=self.input_size,
        hidden_size=self.hidden_size,output_size=self.output_size,parameter = parameter,
        named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        self.current_lower_agent_id = -1
        self.action = None
        print("master:", self.master_action_space)
        self.behave_prob = 1
        self.count = 0
        self.subtask_terminal = True
        self.subtask_turn = 0
        self.subtask_max_turn = 5
        self.past_lower_agent_pool = {key: 0 for key in self.id2lowerAgent.keys()}
        if parameter.get("train_mode") is False:
            self.master.restore_model(self.parameter.get("saved_model"))
            self.master.current_net.eval()
            self.master.target_net.eval()
            for label, agent in self.id2lowerAgent.items():
                self.id2lowerAgent[label].dqn.restore_model(temp_parameter[label]["saved_model"])
                self.id2lowerAgent[label].dqn.current_net.eval()
                self.id2lowerAgent[label].dqn.target_net.eval()
        
        self.agent_action = {
            "turn": 1,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},
            "implicit_inform_slots": {},
            "speaker": "agent"
        }


    def initialize(self):
        self.agent_action = {     
            "turn": None,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},
            "implicit_inform_slots": {},
            "speaker": "agent"

        }
        self.subtask_turn = 0
        self.master_reward = 0
        self.subtask_terminal = True
    
    def train(self, batch):
        """
        Training the agent.
        Args:
            batch: the sample used to training.
        Return:
             dict with a key `loss` whose value it a float.
        """
        loss = self.master.singleBatch(batch=batch,params=self.parameter,weight_correction=self.parameter.get("weight_correction"))
        return loss

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        # for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
        #     batch = random.sample(self.experience_replay_pool, batch_size)

        #     loss = self.train(batch=batch)
        #     cur_bellman_err += loss["loss"]
        # print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))

        # Training of lower agents.
        
        for iter in range(math.ceil(len(self.experience_replay_pool) / batch_size)):
                batch = random.sample(self.experience_replay_pool, min(batch_size, len(self.experience_replay_pool)))
                loss = self.train(batch=batch)
                cur_bellman_err += loss["loss"]
                print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (
                float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))
        if self.count % 10 == 9:
                #print(len(self.id2lowerAgent))
                    for group_id, lower_agent in self.id2lowerAgent.items():
                    # if len(lower_agent.experience_replay_pool) ==10000 or (len(lower_agent.experience_replay_pool)-self.past_lower_agent_pool[group_id])>100:
                        if len(lower_agent.experience_replay_pool) > 150:
                            lower_agent.train_dqn(label=group_id)
                            self.past_lower_agent_pool[group_id] = len(lower_agent.experience_replay_pool)

        self.count += 1
        # Training of lower agents.
        # for disease_id, lower_agent in self.id2lowerAgent.items():
        #    lower_agent.train_dqn()

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.master.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=checkpoint_path)
        # Saving lower agent
        for key, lower_agent in self.id2lowerAgent.items():
            temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/' + str(key))
            lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)


    def update_target_network(self):
        self.master.update_target_network()
        for key in self.id2lowerAgent.keys():
            self.id2lowerAgent[key].update_target_network()
    

    def next(self,state,turn,greedy_strategy,**kwargs):
        epsilon = self.parameter.get("epsilon")
        state_rep = state2rep(state,self.symptom_set,self.parameter)

        if self.subtask_terminal ==True:
            self.master_state = copy.deepcopy(state)
            self.__master_next(state_rep,greedy_strategy)
            self.subtask_terminal = False
            self.subtask_turn = 0

        if self.master_action_index>(len(self.id2lowerAgent) -1):
                agent_action = {'action': 'inform', 'inform_slots': {"disease": 'UNK'}, 'request_slots': {},
                                "explicit_inform_slots": {}, "implicit_inform_slots": {}}
                agent_action["turn"] = turn
                agent_action["inform_slots"] = {"disease": None}
                agent_action["speaker"] = 'agent'
                agent_action["action_index"] = None
                lower_action_index = -1
                self.subtask_terminal = True
        else:
            self.subtask_turn +=1
            self.current_lower_agent_id = self.master_action_space[self.master_action_index]
            agent_action, lower_action_index = self.id2lowerAgent[str(self.current_lower_agent_id)].next(state, self.subtask_turn, greedy_strategy=greedy_strategy)
            if self.subtask_turn>=self.subtask_max_turn:
                self.subtask_terminal = True
                self.subtask_turn = 0
            else:
                assert len(list(agent_action["request_slots"].keys())) == 1
        return agent_action,self.master_action_index,lower_action_index


    def __master_next(self,state_rep,greedy_strategy):
                # Master agent takes an action.
        epsilon = self.parameter.get("epsilon")
        #print(greedy_strategy)
        if greedy_strategy == True:
            greedy = random.random()
            if greedy < epsilon:
                self.master_action_index = random.randint(0, self.output_size - 1)
            else:
                self.master_action_index = self.master.predict(Xs=[state_rep])[1]
 
        else:
            self.master_action_index = self.master.predict(Xs=[state_rep])[1]
 


    def load_db(self,file):
        with open(file) as fp:
            dbs = json.load(fp)
        return dbs


    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, lower_reward, master_action_index):
        # samples of master agent.
        # print(state)
        #print(reward)

        shaping = self.reward_shaping(state, next_state)
        alpha = self.parameter.get("weight_for_reward_shaping")
        '''
        if reward == self.parameter.get("reward_for_repeated_action"):
            lower_reward = reward
            # reward = reward * 2
        else:
            lower_reward = max(0, shaping * alpha)
            # lower_reward = shaping * alpha
        '''
        if episode_over is True:
            pass
        else:
            reward = reward + alpha * shaping
        if int(agent_action) >= 0:
            self.id2lowerAgent[self.current_lower_agent_id].record_training_sample(state, agent_action, lower_reward,
                                                                                   next_state, episode_over)


        state_rep = state2rep(state=state,slot_set=self.symptom_set, parameter=self.parameter)  # sequence representation.
        next_state_rep = state2rep(state=next_state, slot_set=self.symptom_set, parameter=self.parameter)
        master_state_rep = state2rep(state=self.master_state, slot_set=self.symptom_set, parameter=self.parameter)
        self.master_reward += reward

        if self.subtask_terminal or int(agent_action) == -1 or episode_over==True:
            if self.master_reward >-60 and self.master_reward <=0:
                self.master_reward = self.master_reward /4
            if self.master_action_index > (len(self.id2lowerAgent) - 1):
                subtask_turn = 1
            else:
                if self.subtask_turn == 0:
                    subtask_turn = 5
                else:
                    subtask_turn = self.subtask_turn
            #print(subtask_turn)
            self.experience_replay_pool.append((master_state_rep, master_action_index, self.master_reward, next_state_rep, episode_over,subtask_turn))
            self.master_reward = 0
    
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


    def train_mode(self):
        self.master.current_net.train()

    def eval_mode(self):
        self.master.current_net.eval()