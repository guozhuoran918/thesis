import sys
import os
import time
import json
from collections import deque
import copy
import numpy as np
from hrllib.agent import AgentDQN,AgentHRL
from hrllib import configuration
from hrllib.dialogue.dialogue_hrl import HRL
from hrllib.user import User
import random

class RunningSteward(object):
    def __init__(self,parameter):
        self.epoch_size = parameter.get("simulation_size",1000)
        self.checkpoint_path = parameter.get("checkpoint_path")
        if os.path.isdir(self.checkpoint_path) is False:
            os.mkdir(self.checkpoint_path)
        self.parameter = parameter
        self.symptoms_db = parameter["symptoms"]
        self.conditions_db = parameter["conditions"]
        self.dataset = parameter["dataset"]
        self.test_dataset = parameter["test"]
        self.learning_curve = {}
        self.user = User(self.dataset,parameter)
        self.test_user = User(self.test_dataset,parameter)
        self.agent = AgentHRL(self.symptoms_db,self.conditions_db,parameter)
        self.dialogue_manager = HRL(self.user,self.agent,self.symptoms_db,self.conditions_db,parameter)
        self.dialogue_manager.build_deep_learning_classifier()
        self.best_result =     {
            "success_rate":0.0,
            "top3":0.0,
            "top5":0.0,
            "average_reward": 0.0,
            "average_turn": 0.0,
            "average_match": 0.0,
        }
    
    def simulate(self, epoch_number, train_mode=True):

        for index in range(0,epoch_number,1):
            if train_mode is True:
                self.dialogue_manager.train()
                self.simulation_epoch(epoch_size=1000,index = index)
            result = self.evaluation_model(index)
            
            print(result)
            if result["success_rate"] > self.best_result["success_rate"]:
                self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index = index, checkpoint_path=self.checkpoint_path)
                self.dialogue_manager.save_dl_model(model_performance=result, episodes_index=index,
                                                            checkpoint_path=self.checkpoint_path)
                self.best_result = copy.deepcopy(result)
            else:
                pass
        self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index=index, checkpoint_path=self.checkpoint_path)
        self.dialogue_manager.save_dl_model(model_performance=result, episodes_index=index, checkpoint_path=self.checkpoint_path)
        self.__dump_performance__(epoch_index=index)


    def simulation_epoch(self,epoch_size,index):
        success_count = 0
        total_reward = 0
        total_turns = 0
        # dataset_len = len(self.user.user)
        dataset_len = epoch_size
        match =np.zeros(dataset_len,dtype=int)
        self.dialogue_manager.state_tracker.agent.eval_mode()
        dataset = self.user.user
        for n in range(0,epoch_size):
            self.dialogue_manager.initialize()
            over = False
            while over is False:
                reward, over, dialogue_status= self.dialogue_manager.next(save_record=True,greedy_strategy=True)
                total_reward +=reward
                if dialogue_status == configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM:
                    match[n] =1
            total_turns += self.dialogue_manager.state_tracker.turn
            if dialogue_status == configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
        success_rate = float("%.3f" % (float(success_count) / dataset_len))
        average_reward = float("%.3f" % (float(total_reward) / dataset_len))
        average_turn = float("%.3f" % (float(total_turns) / dataset_len))
        average_match = float("%.3f" % (float(np.mean(match))))
        self.dialogue_manager.state_tracker.agent.train_mode()
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn,
               "average_match":average_match}
        return res

    def evaluation_model(self):
        self.dialogue_manager.state_tracker.agent.eval_mode()
        reach_max_turn = 0
        success_count = 0
        total_reward = 0
        total_top3 = 0
        total_top5 = 0
        slot_record = {}
        dataset_len = len(self.test_user.user)
        total_turns = np.zeros(dataset_len,dtype=int) 
        dataset = self.test_user.user
        self.dialogue_manager.set_user(self.test_user)
        # self.__dump_diaglogue__(self.dialogue_manager.user.user)
        match =np.zeros(dataset_len,dtype=int)
        for index in dataset.keys():
            self.dialogue_manager.initialize(index)
            over = False
            while over is False:
                reward,over,dialogue_status = self.dialogue_manager.next(
                    save_record=False,greedy_strategy=False)
                if dialogue_status == configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM:
                    match[int(index)] = 1
                total_reward+=reward
                total_turns[int(index)] +=1

            slot_record[index] = self.dialogue_manager.current_slots 
            if dialogue_status == configuration.DIALOGUE_STATUS_SUCCESS:
                success_count +=1
                total_top3+=1
                total_top5+=1
            if dialogue_status == configuration.DIALOGUE_STATUS_TOP3:
                total_top3+=1
                total_top5+=1
            if dialogue_status == configuration.DIALOGUE_STATUS_TOP5:
                total_top5+=1     
            if dialogue_status == configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
                reach_max_turn+=1
        

        success_rate = float("%.3f" % (float(success_count) / dataset_len))
        average_reward = float("%.3f" % (float(total_reward) / dataset_len))
        average_turn = float("%.3f" % (float(np.mean(total_turns))))
        average_top3 = float("%.3f" % (float(total_top3) / dataset_len))
        average_top5 = float("%.3f" % (float(total_top5) / dataset_len))
        average_match = float("%.3f" % (float(np.mean(match))))
        
        if(self.parameter.get("train_mode") == True):
            self.dialogue_manager.train_deep_learning_classifier(epochs=20)
        res = {
            "success_rate":success_rate,
            "top3":average_top3,
            "top5":average_top5,
            "average_reward": average_reward,
            "average_turn": average_turn,
            "average_match": average_match,
        }

        self.__dump_diaglogue__(slot_record)
        return res
    def __dump_performance__(self, epoch_index):
        filename = "best"+str(epoch_index)+".json"
        json_str = json.dumps(self.best_result)
        path = os.path.join(self.checkpoint_path,filename)
        with open(path,'w') as json_file:
            json_file.write(json_str)

    def __dump_diaglogue__(self,records):
        filename = "diaglogue.json"
        json_str = json.dumps(records)
        path = os.path.join(self.checkpoint_path,filename)
        with open(path,'w') as json_file:
            json_file.write(json_str)


