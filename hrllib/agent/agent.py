# basic agent class
from collections import deque
import json
import numpy as np
import copy
import sys,os
from hrllib.agent.utils import state2rep
from hrllib import configuration
from hrllib.agent import relaybuffer

class Agent(object):

    def __init__(self,symptoms,conditions,parameter,**kwargs):
        self.state = None
        self.parameter = parameter
        self.symptoms_db = symptoms
        self.conditions_db = conditions
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        self.agent_action = {
            "turn":1,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }
        self._build_action_space(self.symptoms_db)
    def next(self,*args,**kwargs):
        raise NotImplementedError('The `next` function of agent has not been implemented.')

    def train(self,batch):
        raise NotImplementedError('The `train` function of agent has not been implemented.')

    def initialize(self):
        self.agent_action = {
            "turn":None,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
  

        state = state2rep(state=state, slot_set=self.symptoms_db,parameter=self.parameter) # sequence representation.
        next_state = state2rep(state=next_state, slot_set=self.symptoms_db,parameter=self.parameter)

        self.experience_replay_pool.append((state, agent_action, reward, next_state, episode_over))

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))

    def train_mode(self):
        """
        Set the agent as the train mode, i.e., the parameters will be updated and dropout will be activated.
        """
        raise NotImplementedError("The `train_mode` function of agent has not been implemented")

    def eval_mode(self):
        """
        Set the agent as the train mode, i.e., the parameters will be unchanged and dropout will be deactivated.
        """
        raise NotImplementedError("The `train_mode` function of agent has not been implemented")

    def _build_action_space(self, symptoms):
        """
        Building the Action Space for the RL-based Agent.
        All diseases are treated as actions.
        :return: Action Space, a list of feasible actions.
        """
        feasible_actions = []
        # Adding the request actions. And the slots are extracted from the links between disease and symptom,
        # i.e., disease_symptom
        for slot in sorted(symptoms.keys()):
        
            feasible_actions.append({'action': 'request', 'inform_slots': {}, 'request_slots': {slot:configuration.VALUE_UNKNOWN},"explicit_inform_slots":{}, "implicit_inform_slots":{}})

        return feasible_actions

    def flush_pool(self):
         self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
 