import copy
from inspect import Parameter
import random
from collections import deque
import sys, os
import json
from gevent import config
from torch import alpha_dropout
from hrllib.agent.utils import state2rep
from hrllib.user.state_tracker import StateTracker
from hrllib.user.user import get_pro_age,get_pro_incidence,get_pro_sex
from hrllib import configuration
import numpy as np
from hrllib.classifier import dl_classifier

class HRL(object):

    def __init__ (self,user,agent,symptom_set,disease_set,parameter):
        self.state_tracker = StateTracker(user = user,agent=agent, parameter= parameter)
        self.parameter = parameter
        self.context = parameter.get("context")
        self.user = user
        self.agent = agent
        self.hint = 0
        self.repeated_action_count = 0
        self.symptom_db = self.load_db(symptom_set)
        self.disease_db = self.load_db(disease_set)
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.action_history = []
        self.master_action_history = []
        self.lower_action_history = []
        self.inform_wrong_disease_count = 0
        self.disease_replay = deque(maxlen = 10000)
        self.id2disease = dict(map(reversed,self.disease_db.items()))
        self.worker_right_inform_num = 0
        self.current_slots = {}

        if self.parameter.get("train_mode")==False:
            self.test_by_group = {x:[0,0,0] for x in ['1', '2', '3', '4', '5', '6', '7', '8', '9','10']}
            #这里的三维向量分别表示成功次数、group匹配正确的个数、隶属于某个group的个数
            self.disease_record = []
            self.lower_reward_by_group = {x: [] for x in ['1', '2', '3', '4', '5', '6', '7', '8', '9','10']}
            self.master_index_by_group = []
            self.symptom_by_group = {x: [0,0] for x in ['1', '2', '3', '4', '5', '6', '7', '8', '9','10']}
    def set_user(self,new_user):
        self.user = new_user
        self.state_tracker.user = new_user

    def next(self,save_record,greedy_strategy):

            lower_reward = 0
            state = self.state_tracker.get_state()
            # group_id = self.state_tracker.user["group_id"]
            self.master_action_space = self.agent.master_action_space
            agent_action, master_action_index, lower_action_index = self.agent.next(state=state, turn=self.state_tracker.turn,
                                                                                              greedy_strategy=greedy_strategy)
            action_type = "disease" if lower_action_index == -1 else "symptom"
            over = True if lower_action_index == -1 else False
            if len(agent_action["request_slots"])>0:
                lower_action = list(agent_action["request_slots"].keys())[0]
                action_type = "symptom"
            elif len(agent_action["inform_slots"])>0:
                lower_action = list(agent_action["inform_slots"].keys())[0]
                action_type ="disease"


            # if self.parameter.get("train_mode") == True:
            #     condition = False
            # else:
            #     condition = state["turn"] == self.parameter.get("max_turn")+16

            if action_type == "disease":
                disease = self.state_tracker.user.goal["disease"]
                labels = list(range(0,54))
                state_rep = state2rep(state,self.symptom_db,self.parameter)
                Ys,pre_disease = self.model.predict([state_rep])
                self.disease_replay.append((state_rep,self.disease_db[disease]))
                lower_action_index = -1
                sorted_ = np.argsort(Ys.detach().cpu().numpy())[0]
                top5 = []
                for i in range(5):
                    top5.append(self.id2disease[sorted_[-(i+1)]])
            
                master_action_index = len(self.master_action_space)
                agent_action = {'action': 'inform', 'inform_slots': {"disease":top5}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}}
     
                if self.parameter.get("train_mode") == False:
                    self.disease_record.append([self.disease_db[disease],self.id2disease[pre_disease[0]]])

            self.state_tracker.state_updater(agent_action=agent_action)
            user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
            self.state_tracker.state_updater(user_action = user_action)
            if master_action_index < len(self.master_action_space):
                self.master_action_history.append(self.master_action_space[master_action_index])
            if action_type == "symptom":
                reward, lower_reward = self.next_by_hrl_joint2(dialogue_status, lower_action, state, master_action_index, episode_over, reward)
  
            if dialogue_status ==  configuration.DIALOGUE_STATUS_FAILED:
                self.inform_wrong_disease_count +=1

            if dialogue_status == configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
                disease = self.state_tracker.user.goal["disease"]
                labels = list(range(0,54))
                state_rep = state2rep(self.state_tracker.get_state(),self.symptom_db,self.parameter)
                Ys,pre_disease = self.model.predict([state_rep])
                self.disease_replay.append((state_rep,self.disease_db[disease]))
                lower_action_index = -1
                
                top5 = []
                if self.context is True and greedy_strategy is False:
                    #todo 
                    Q_context = Ys.detach().cpu().numpy()[0]
                    new_context = self.policy_transformation(Q_context)
                    sorted_ = np.argsort(new_context)
                else:
                    sorted_ = np.argsort(Ys.detach().cpu().numpy())[0]

                for i in range(5):
                        top5.append(self.id2disease[sorted_[-(i+1)]])
            
                if disease == top5[0]:
                    dialogue_status = configuration.DIALOGUE_STATUS_SUCCESS
                elif disease in top5[0:3]:
                    dialogue_status = configuration.DIALOGUE_STATUS_TOP3
                elif disease in top5:
                    dialogue_status = configuration.DIALOGUE_STATUS_TOP5
                else:
                    dialogue_status = configuration.DIALOGUE_STATUS_FAILED
                if self.parameter.get("train_mode") == False:
                    self.disease_record.append([self.disease_db[disease],self.id2disease[pre_disease[0]]])

            assert dialogue_status is not configuration.DIALOGUE_STATUS_REACH_MAX_TURN

            if episode_over is True:
                 state = self.state_tracker.get_state()
                 self.action_history = []
                 self.lower_action_history = []
                 self.current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
                 if len(top5) >0:
                     self.current_slots["disease"] = top5
                 self.current_slots["goal"] = self.state_tracker.user.goal["disease"]
                #  self.current_slots["diagnosis"] = copy.deepcopy(state["current_slots"]["disease"])
            if save_record or self.state_tracker.get_state()["turn"]==2:
                    self.record_training_sample(
                        state=state,
                        agent_action=lower_action_index,
                        next_state=self.state_tracker.get_state(),
                        reward=reward,
                        episode_over=episode_over,
                        lower_reward = lower_reward,
                        master_action_index = master_action_index
                        )

            return reward, episode_over, dialogue_status
        
    def initialize(self,goal_index = None):
        self.state_tracker.initialize()
        self.inform_wrong_disease_count = 0
        self.lower_action_history = []
        self.current_slots = {}
        user_action = self.state_tracker.user.initialize(goal_index)
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()

    def train(self):
        self.state_tracker.agent.train_dqn()
        self.state_tracker.agent.update_target_network()
        
    def lower_reward_function(self, state, next_state):
        """
        The reward for lower agent
        :param state:
        :param next_state:
        :return:
        """
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item

        # slot number in state.
        slot_dict = copy.deepcopy(state["current_slots"]["inform_slots"])
        slot_dict = delete_item_from_dict(slot_dict, False)
        next_slot_dict = copy.deepcopy(next_state["current_slots"]["inform_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, False)
        gamma = self.parameter.get("gamma")
        return gamma * len(next_slot_dict) - len(slot_dict)

    def set_agent(self,agent):
        self.state_tracker.set_agent(agent=agent)

    
    def load_db(self,file):
        with open(file) as fp:
            dbs = json.load(fp)
        return dbs

    def build_deep_learning_classifier(self):
        input_size = len(self.symptom_db)*9 if self.parameter['nlice'] else len(self.symptom_db)*3
        self.model = dl_classifier(input_size=input_size, hidden_size=256,
                                   output_size=len(self.disease_db),
                                   parameter=self.parameter)
        if self.parameter.get("train_mode") == False:
            temp_path = self.parameter.get("saved_model")
            path_list = temp_path.split('/')
            path_list.insert(-1, 'classifier')
            saved_model = '/'.join(path_list)
            self.model.restore_model(saved_model)
            self.model.eval_mode()

    def train_deep_learning_classifier(self, epochs):
        #self.model.train_dl_classifier(epochs=5000)
        #print("############   the deep learning model is training over  ###########")
        for iter in range(epochs):
            batch = random.sample(self.disease_replay, min(self.parameter.get("batch_size"),len(self.disease_replay)))
            loss = self.model.train(batch=batch)

        test_batch = random.sample(self.disease_replay, min(1000,len(self.disease_replay)))
        test_acc = self.model.test(test_batch=test_batch)
        print('disease_replay:{},loss:{:.4f}, test_acc:{:.4f}'.format(len(self.disease_replay), loss["loss"], test_acc))
        return loss["loss"], test_acc
        #self.model.test_dl_classifier()

    def save_dl_model(self, model_performance, episodes_index, checkpoint_path=None):
        temp_checkpoint_path = os.path.join(checkpoint_path, 'classifier/')
        self.model.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
      
        lower_reward = kwargs.get("lower_reward")
        master_action_index = kwargs.get("master_action_index")
        self.state_tracker.agent.record_training_sample(state, agent_action, reward, next_state, episode_over, lower_reward, master_action_index)


    
    def next_by_hrl_joint2(self, dialogue_status, lower_action, state, master_action_index, episode_over, reward):
        '''
                    self.acc_by_group[group_id][2] += 1
                    if self.master_action_space[master_action_index] == group_id:
                        self.acc_by_group[group_id][1] += 1
                        if self.lower_reward_function(state=state, next_state=self.state_tracker.get_state()) > 0:
                            self.acc_by_group[group_id][0] += 1
                    '''
        alpha = self.parameter.get("weight_for_reward_shaping")
        if dialogue_status == configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
            self.repeated_action_count += 1

   

        # if self.parameter.get("train_mode") == False:
        #     if lower_action not in self.lower_action_history:
        #         # self.lower_reward_by_group[self.master_action_space[master_action_index]].append(lower_reward)
        #         self.symptom_by_group[self.master_action_space[master_action_index]][1] += 1
        #         if self.lower_reward_function(state=state, next_state=self.state_tracker.get_state()) > 0:
        #             self.symptom_by_group[self.master_action_space[master_action_index]][0] += 1

        lower_reward = alpha * self.lower_reward_function(state=state, next_state=self.state_tracker.get_state())
        if lower_action in self.lower_action_history:  # repeated action
            lower_reward = -alpha
            self.state_tracker.agent.subtask_terminal = True
            # episode_over = True
            # self.repeated_action_count += 1
            reward = self.parameter.get("reward_for_repeated_action")
        # elif self.state_tracker.agent.subtask_terminal is False or lower_reward>0:
        else:
            lower_reward = max(0, lower_reward)
            if lower_reward > 0:
                self.state_tracker.agent.subtask_terminal = True
                self.worker_right_inform_num += 1
                lower_reward = alpha
            self.lower_action_history.append(lower_action)

        # if self.parameter.get("train_mode") == False:
        #     self.lower_reward_by_group[self.master_action_space[master_action_index]].append(lower_reward)

        if episode_over == True:
            self.state_tracker.agent.subtask_terminal = True
            self.state_tracker.agent.subtask_turn = 0

        elif self.state_tracker.agent.subtask_terminal:
            # reward = alpha / 2 * self.worker_right_inform_num - 2
            '''
            if self.master_action_space[master_action_index] == group_id:
                reward = 10
            else:
                reward = -1
            '''
            self.worker_right_inform_num = 0
            # if self.parameter.get("train_mode") == False:
            #     self.master_index_by_group.append([group_id, master_action_index])
            # self.lower_action_history = []
        return reward, lower_reward


    def policy_transformation(self,conditions):

        for i in range(len(conditions)):
            co = self.id2disease.get(i)
            pro_age = get_pro_age(co,self.state_tracker.user.goal["age"],self.state_tracker.user.piors_map)/100
            pro_gender=get_pro_sex(co,self.state_tracker.user.goal["gender"],self.state_tracker.user.piors_map)/100
            conditions[i] = conditions[i]*pro_age*pro_gender
        return conditions
            
