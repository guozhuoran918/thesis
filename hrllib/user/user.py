from collections import namedtuple
from hrllib.agent.utils import load_db
import numpy as np
import json
import os
import random
from psutil import users
from sympy import solve_undetermined_coeffs
AiMedPatient = namedtuple('AiMedPatient', ('age', 'race', 'gender', 'symptoms', 'condition'))
AiMedState = namedtuple('AiMedState', ('age', 'race', 'gender', 'symptoms'))
RLPatient = namedtuple('AiMedPatient', ('symptoms', 'condition'))
RLState = namedtuple('AiMedState', ('symptoms'))
RLPatient_Context= namedtuple('AiMedPatient', ('age', 'gender', 'symptoms', 'condition'))
RLState_Context = namedtuple('AiMedState', ('age','gender', 'symptoms'))
default_age={
"age-0-1-years": 1.536246615603,
"age-1-4-years" : 4.6087398468089,
"age-5-14-years" : 11.4611269862763,
"age-15-29-years" : 21.3235812629491,
"age-30-44-years" :20.2551962493181,
"age-45-59-years" : 20.4205176044076,
"age-60-74-years" : 13.987739661922,
"age-g-75-years" : 6.4068517727151
}
default_sex={
    "sex-male":50,
    "sex-female":50,
}

def get_pro_age(condition,age,all_piors):
    value = all_piors.get(condition)
    age_keys = value.get("age")
    age_keys = default_age if age_keys=={} else age_keys
    for a in age_keys.keys():
        ages = a.split("-")
        age_min,age_max = ages[1],ages[2]
        if age_min =="g":
            age_min=age_max
            age_max=140
        age_min,age_max = int(age_min),int(age_max)
        if (age>=age_min and age<=age_max):
            return age_keys[a]
    return 0



def get_pro_sex(condition,sex,all_piors):
    value = all_piors.get(condition)
    sex_keys = value.get("sex")
    sex_keys = default_sex if sex_keys=={} else sex_keys
    gender = "sex-male" if sex == '0' else "sex-female"
    return sex_keys[gender]

def get_pro_incidence(condition,all_piors):
    v = all_piors.get(condition)
    value = v.get("incidence")
    value = value.split('/')
    result = float(value[0])/float(value[1])
    return result

import random 
import copy
from hrllib import configuration
from hrllib.agent.agent import Agent

class User(object):
    def __init__(self,dataset,parameter):
        self.max_turn = configuration.MAX_TURN
        self.parameter = parameter
        self.max_turn = parameter["max_turn"]
        self.line_number = 0
        self.data = open(dataset).readlines()[1:]
        self.piors_map = load_db(parameter["piors"])
        self.user = {}
        self.goal = {}
        self.state = None
        self.episode_over = False
        self.dialogue_status = self.dialogue_status = configuration.DIALOGUE_STATUS_NOT_COME_YET
        self.pre_goal()


    def pre_goal(self):
        for line in self.data:
            temp = {}
            line = "" if line is None else line.strip()
            parts = line.split(",")
            user_id = str(parts[0])
            _gender = parts[1]
            gender = 0 if _gender == 'M' else 1
            race = parts[2]
            # age = int(parts[3])
            disease = parts[6]
            age = int(self.get_age(disease))
            temp = {}
            temp["gender"] = gender
            temp["race"] = race
            temp["age"] = age
            temp["disease"] = disease
            temp["request_slots"] = {"disease": "UNK"}
            temp["explicit_inform_slots"] = {}
            temp["implicit_inform_slots"] = {}
            # temp["symptoms"] = {}
            symptom_list = parts[8]
            symptoms = symptom_list.split(";")
            nums_main = self.parameter["nums_main_complaint"]
            replacement = True if nums_main>len(symptoms) else False
            temp_initial_symptom = np.random.choice(symptoms,nums_main,replace=replacement)
        
            # initial_symptom = temp_initial_symptom[0].split(":")[0]
            for item in symptom_list.split(";"):                          
                    sym_list = item.split(":")
                    _symptom,_nature, _location_main,_location, _intensity, _duration, _onset, _exciation, _frequency, _ = sym_list
                    complaint = "explicit_inform_slots" if item in temp_initial_symptom else "implicit_inform_slots" 
                    if(_symptom == "Alterred_stool"):
                        _symptom ="Altered_stool"
                    if(_symptom == "Nausea_"):
                        _symptom="Nausea"
                    if(_symptom == "Pain_relief_"):
                        _symptom= "Pain_relief"
                    if(_symptom == "Vomitting"):
                        _symptom== "Vomiting"
                    if(_symptom == "Incontinence_"):
                        _symptom="Incontinence"
                    temp[complaint][_symptom] = {}
                    temp[complaint][_symptom]["nature"] = _nature
                    temp[complaint][_symptom]["location_main"] = _location_main
                    temp[complaint][_symptom]["location"] = _location
                    temp[complaint][_symptom]["intensity"] = _intensity
                    temp[complaint][_symptom]["duration"] = _duration
                    temp[complaint][_symptom]["onset"] = _onset
                    temp[complaint][_symptom]["exciation"] = _exciation
                    temp[complaint][_symptom]["frequency"] = _frequency

            self.user[user_id] = temp


    def _init(self,user_index = None):
   
        self.state = {
                "turn":0,
                "action":None,
                "history":{}, # For slots that have been informed.
                "request_slots":{}, # For slots that user requested in this turn.
                "inform_slots":{}, # For slots that belong to goal["request_slots"] or other slots not in explicit/implicit_inform_slots.
                "explicit_inform_slots":{}, # For slots that belong to goal["explicit_inform_slots"]
                "implicit_inform_slots":{}, # For slots that belong to goal["implicit_inform_slots"]
                "rest_slots":{} # For slots that have not been informed.
            }
        if user_index is None:
            id,temp = random.choice(list(self.user.items()))
            self.goal = temp
        else:
            self.goal = self.user[user_index]
        self.episode_over = False
        self.dialogue_status = configuration.DIALOGUE_STATUS_NOT_COME_YET
   


    def initialize(self,user_index):
        self._init(user_index)
    
        self.state["action"] = "request"
        self.state["request_slots"]["disease"] = configuration.VALUE_UNKNOWN
        inform_slots = list(self.goal["explicit_inform_slots"].keys())
        for slot in inform_slots:
            self.state["inform_slots"][slot] = self.goal["explicit_inform_slots"][slot]
 
        for slot in self.goal["implicit_inform_slots"].keys():
            if slot not in self.state["request_slots"].keys():
                self.state["rest_slots"][slot] = "implicit_inform_slots" # Remember where the rest slot comes from.
        for slot in self.goal["explicit_inform_slots"].keys():
            if slot not in self.state["request_slots"].keys():
                self.state["rest_slots"][slot] = "explicit_inform_slots"
        if "disease" not in self.state["request_slots"].keys():
                self.state["rest_slots"]["disease"] = "request_slots" 

        user_action = self._assemble_user_action()
        return user_action

    def _assemble_user_action(self):
        """
        Assembling the user action according to the current status.
        Returns:
            A dict, containing the information of this turn and the user's current state.
        """
        user_action = {
            "turn":self.state["turn"],
            "action":self.state["action"],
            "speaker":"user",
            "request_slots":self.state["request_slots"],
            "inform_slots":self.state["inform_slots"],
            "explicit_inform_slots":self.state["explicit_inform_slots"],
            "implicit_inform_slots":self.state["implicit_inform_slots"]
        }
        return user_action      

    def next(self,agent_action,turn):
        agent_act_type = agent_action["action"]
        self.state["turn"] = turn
        if self.state["turn"]>= self.max_turn:
            self.episode_over = True
            self.dialogue_status = configuration.DIALOGUE_STATUS_REACH_MAX_TURN
            self.state["action"] = configuration.CLOSE_DIALOGUE

        else:
            pass

        if self.episode_over is not True:
            self.state["history"].update(self.state["inform_slots"])
            self.state["history"].update(self.state["explicit_inform_slots"])
            self.state["history"].update(self.state["implicit_inform_slots"])
            self.state["inform_slots"].clear()
            self.state["explicit_inform_slots"].clear()
            self.state["implicit_inform_slots"].clear()
            if agent_act_type == configuration.CLOSE_DIALOGUE:
                self._response_closing()
            elif agent_act_type in ["inform","explicit_inform","implicit_inform"]:
                self._response_inform(agent_action)
            elif agent_act_type == "request":
                self._response_request(agent_action=agent_action)
        else:
            pass
        user_action = self._assemble_user_action()
        reward = self._reward_function()
        return user_action,reward,self.episode_over,self.dialogue_status
   
    def _response_closing(self):
       self.state["action"] = configuration.THANKS
       self.episode_over = True

    def _response_inform(self,agent_action):
        agent_all_inform_slots = copy.deepcopy(agent_action["inform_slots"])
        agent_all_inform_slots.update(agent_action["explicit_inform_slots"])
        agent_all_inform_slots.update(agent_action["implicit_inform_slots"])

        user_all_inform_slots = copy.deepcopy(self.goal["explicit_inform_slots"])
        user_all_inform_slots.update(self.goal["implicit_inform_slots"])

        if "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"][0] == self.goal["disease"]:
            self.state["action"] = configuration.CLOSE_DIALOGUE
            self.dialogue_status = configuration.DIALOGUE_STATUS_SUCCESS
            self.state["history"]["disease"] = agent_action["inform_slots"]["disease"]
            self.episode_over = True
            self.state["inform_slots"]["disease"] = agent_action["inform_slots"]["disease"]
            self.state["explicit_inform_slots"].clear()
            self.state["implicit_inform_slots"].clear()
            self.state["request_slots"].pop("disease")
            if "disease" in self.state["rest_slots"]: self.state["rest_slots"].pop("disease")

        elif "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"][0] != self.goal["disease"]:
            top3 = agent_action["inform_slots"]["disease"][0:3]
            top5 = agent_action["inform_slots"]["disease"]
            if self.goal["disease"] in top3:
                self.dialogue_status = configuration.DIALOGUE_STATUS_TOP3
            elif self.goal["disease"] in top5:
                self.dialogue_status = configuration.DIALOGUE_STATUS_TOP5
            else:
                self.dialogue_status = configuration.DIALOGUE_STATUS_FAILED
            self.state["action"] = configuration.CLOSE_DIALOGUE
            self.episode_over = True
            self.state["inform_slots"]["disease"] = agent_action["inform_slots"]["disease"]
            self.state["explicit_inform_slots"].clear()
            self.state["implicit_inform_slots"].clear()
   
        else:
            pass


    def _response_request(self, agent_action):
        """
        The user informs slot must be one of implicit_inform_slots, because the explicit_inform_slots are all informed
        at beginning.
        # It would be easy at first whose job is to answer the implicit slot requested by agent.
        :param agent_action:
        :return:
        """
        # TODO (Qianlong): response to request action.
        if len(agent_action["request_slots"].keys()) > 0:
            for slot in agent_action["request_slots"].keys():
                # The requested slots are come from explicit_inform_slots.
                if slot in self.goal["explicit_inform_slots"].keys():
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["explicit_inform_slots"][slot]
                    # For requesting right symptoms of the user goal.
                    self.dialogue_status = configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                elif slot in self.goal["implicit_inform_slots"].keys():
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["implicit_inform_slots"][slot]
                    # For requesting right symptoms of the user goal.
                    self.dialogue_status = configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                # The requested slots not in the user goals.
                else:
                    self.state["action"] = "not"
                    self.state["inform_slots"][slot] = False
                    self.dialogue_status == configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE
        self.episode_over = False


    def _reward_function(self):
        if self.dialogue_status == configuration.DIALOGUE_STATUS_NOT_COME_YET:
            reward = 0
        elif self.dialogue_status == configuration.DIALOGUE_STATUS_SUCCESS:
            reward = self.parameter.get("reward_success")
        elif self.dialogue_status == configuration.DIALOGUE_STATUS_FAILED or self.dialogue_status == configuration.DIALOGUE_STATUS_TOP3 or self.dialogue_status == configuration.DIALOGUE_STATUS_TOP5:
            reward = self.parameter.get("reward_fail")
        elif self.dialogue_status == configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM:
            reward = self.parameter.get("reward_right")
        elif self.dialogue_status == configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE:
            reward = self.parameter.get("reward_wrong")
        elif self.dialogue_status == configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
            reward = self.parameter.get("reach_max_turn")

        return reward


    def get_age(self,condition):
        import random
        #synthea generated dataset is not based on the age distribution, so we re-set age
        value = self.piors_map.get(condition)
        ages = value.get("age")
        age_keys = default_age if ages=={} else ages
        age_position = self.randomS(age_keys)
        age = age_position.split("-")
        age_min,age_max = age[1],age[2]
        if age_min =="g":
            age_min=age_max
            age_max=140
        age_min,age_max = int(age_min),int(age_max)
        return random.randint(age_min,age_max)

    def randomS(self,age_keys):
        import random
        target = random.randint(0,int(sum(age_keys.values())))
        sum_=0
        for k,v in age_keys.items():
            sum_+=v
            if sum_>=target:
                return k
    
            

# class User:
#     def __init__(
#             self,
#             data_file,
#             symptom_map_file,
#             condition_map_file,
#             body_parts,
#             excitation_enc,
#             frequency_enc,
#             nature_enc,
#             vas_enc,
#             onset_enc,
#             duration_enc,
#             piors,
#             classifer,
#             max_turn,
#             epoch,
#             policy,
#             **kwargs
#     ):
#         """
#         data_file: A file of generated patient, symptoms, condition data
#         symptom_map_file: the encoding file for symptoms
#         condition_map_file: the encoding file for conditions
#         initial_symptom_file: a map of conditions
#         clf: a classifier which can output a probabilistic description of possible conditions based on
#         symptoms and patient demography.
#         """
#         self.data_file = data_file
#         self.symptom_map_file = symptom_map_file
#         self.condition_map_file = condition_map_file
#         self.body_file = body_parts
#         self.excitation_file=excitation_enc
#         self.frequency_file=frequency_enc
#         self.nature_file=nature_enc
#         self.vas_file=vas_enc
#         self.onset_file=onset_enc
#         self.duration_file=duration_enc
#         self.piors =piors
#         self.epoch = epoch
#         self.classifier = classifer
#         self.line_number = 0
#         self.state = None
#         self.patient = None
#         self.data = None
#         self.symptom_map = None
#         self.condition_map = None
#         self.body_map=None
#         self.excitation_map=None
#         self.frequency_map=None
#         self.nature_map=None
#         self.vas_map=None
#         self.onset_map=None
#         self.duration_map=None
#         self.initial_symptom_map = None
#         self.piors_map = None
#         self.num_symptoms = None
#         self.num_conditions = None
#         self.current_Q = None
#         self.max_turn = max_turn
#         self.status = None
#         self.conditions_reverse = None
#         self.check_file_exists()
#         self.policy = policy
#         self.load_data_file()
#         self.load_symptom_map()
#         self.load_condition_map()
#         self.turn = 0
#         self.is_inquiry = 1
#         self.is_diagnose = 2

#         self.inquiry_list = set([])

#         self.RACE_CODE = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}
#         self.reward_correct_inquiry = kwargs.get('reward_inquiry', self.max_turn)
#         self.reward_wrong_inquiry = kwargs.get('reward_inquiry', 0)
#         self.reward_repeated = kwargs.get('reward_repeated', -1)
#         self.reward_diagnose_correct = kwargs.get('reward_diagnose_correct', self.max_turn)
#         self.reward_diagnose_incorrect = kwargs.get('reward_diagnose_incorrect', 0)

#     def check_file_exists(self):
#         files = [self.data_file, self.symptom_map_file, self.condition_map_file,self.body_file,self.frequency_file,self.nature_file,self.excitation_file,self.vas_file]
#         for file in files:
#             if not os.path.exists(file):
#                 raise ValueError("File: %s does not exist" % file)

#     def load_data_file(self):
#         self.data = open(self.data_file)

#     def close_data_file(self):
#         if self.data is not None:
#             self.data.close()

#     def load_symptom_map(self):
#         with open(self.symptom_map_file) as fp:
#             symptoms = json.load(fp)
#             sorted_symptoms = sorted(symptoms.keys())
#             self.symptom_map = {code: idx for idx, code in enumerate(sorted_symptoms)}
#             self.num_symptoms = len(self.symptom_map)
#         with open(self.body_file) as fp:
#             self.body_map = json.load(fp)
#         with open(self.frequency_file) as fp:
#             self.frequency_map = json.load(fp)
#         with open(self.excitation_file) as fp:
#             self.excitation_map = json.load(fp)
#         with open(self.nature_file) as fp:
#             self.nature_map = json.load(fp)
#         with open(self.vas_file) as fp:
#             self.vas_map = json.load(fp)
#         with open(self.onset_file) as fp:
#             self.onset_map = json.load(fp)
#         with open(self.duration_file) as fp:
#             self.duration_map = json.load(fp)
#         with open(self.piors) as fp:
#             self.piors_map = json.load(fp)
#     def load_condition_map(self):
#         with open(self.condition_map_file) as fp:
#             conditions = json.load(fp)
#             sorted_conditions = sorted(conditions.keys())
#             self.condition_map = {code: idx for idx, code in enumerate(sorted_conditions)}
#             self.num_conditions = len(self.condition_map)
#         self.conditions_reverse = dict(map(reversed, self.condition_map.items()))
     
#     def readline(self):
#         line = self.data.readline()
#         line = "" if line is None else line.strip()
#         return line

#     def get_line(self):
#         if self.line_number == 0:
#             self.readline() # header line

#         line = self.readline()
#         if not line or len(line) == 0:
#             # EOF
#             self.data.seek(0)
#             self.readline() # header line
#             line = self.readline()

#         self.line_number += 1
#         return line

#     def parse_line(self, line):
#         parts = line.split(",")
#         # print(parts)
#         _gender = parts[1]
#         _race = parts[2]      
#         age = int(parts[3])
#         condition = parts[4]
#         age = self.get_age(condition)
#         symptom_list = parts[6]
#         gender = 0 if _gender == 'M' else 1
#         race = self.RACE_CODE.get(_race)
#         condition = self.condition_map.get(condition)
#         symptoms_nlice = np.zeros(8 *self.num_symptoms, dtype=np.uint16)
#         # symptoms = np.zeros(len(self.num_symptoms), dtype=np.uint16)
#         for item in symptom_list.split(";"):
#             sym_list = item.split(":")
#             # if(len(sym_list)==9):
#             _symptom, _nature, _location, _intensity, _duration, _onset, _exciation, _frequency, _ = sym_list
#             #todo add nlice in the future
#             _symptom.replace("Alterred_stool", "Altered_stool")
#             _symptom.replace("Nausea_", "Nausea")
#             _symptom.replace("Pain_relief_", "Pain_relief")
#             _symptom.replace("Vomitting", "Vomiting")
#             _symptom_idx = self.symptom_map.get(_symptom)*8
#             _nature_idx = _symptom_idx + 1
#             _location_idx = _symptom_idx + 2
#             _intensity_idx = _symptom_idx + 3
#             _duration_idx = _symptom_idx + 4
#             _onset_idx = _symptom_idx + 5
#             _excitation_idx = _symptom_idx + 6
#             _frequency_idx = _symptom_idx + 7
#             _nature_val = 1 if _nature == "" or _nature == "other" else self.nature_map.get(_nature)
#             _location_val = 1 if _location == "" or _location == "other" else self.body_map.get(_location)
#             _intensity_val = 1 if _intensity == "" else self.vas_map.get(_intensity)
#             _duration_val = 1 if _duration == "" else self.duration_map.get(_duration)
#             _onset_val = 1 if _onset == "" else self.onset_map.get(_onset)
#             _excitation_val = 1 if _exciation == "" else self.excitation_map.get(_exciation)
#             _frequency_val = 1 if _frequency == "" else self.frequency_map.get(_frequency)
#             symptoms_nlice[_symptom_idx] = 1
#             symptoms_nlice[_nature_idx] = _nature_val
#             symptoms_nlice[_location_idx] = _location_val
#             symptoms_nlice[_intensity_idx] = _intensity_val
#             symptoms_nlice[_duration_idx] = _duration_val
#             symptoms_nlice[_onset_idx] = _onset_val
#             symptoms_nlice[_excitation_idx] = _excitation_val
#             symptoms_nlice[_frequency_idx] = _frequency_val

#         # ('age', 'race', 'gender', 'symptoms', 'condition')
#         symptoms = np.array(symptoms_nlice)
#         patient = RLPatient_Context(age,gender,symptoms, condition)
#         return patient

#     def get_age(self,condition):
#         import random
#         #synthea generated dataset is not based on the age distribution, so we re-set age
#         value = self.piors_map.get(condition)
#         ages = value.get("age")
#         age_keys = default_age if ages=={} else ages
#         age_position = self.randomS(age_keys)
#         age = age_position.split("-")
#         age_min,age_max = age[1],age[2]
#         if age_min =="g":
#             age_min=age_max
#             age_max=140
#         age_min,age_max = int(age_min),int(age_max)
#         return random.randint(age_min,age_max)
    
#     def randomS(self,age_keys):
#         import random
#         target = random.randint(0,int(sum(age_keys.values())))
#         sum_=0
#         for k,v in age_keys.items():
#             sum_+=v
#             if sum_>=target:
#                 return k
#     def reset(self):
#         if self.line_number == self.epoch:
#             self.line_number = 0
#             self.load_data_file()
#         line = self.get_line()
#         self.patient = self.parse_line(line)
#         self.state = self.generate_state()
#         self.inquiry_list = set([])
#         self.turn = 0
#         self.pick_initial_symptom()

#     def pick_initial_symptom(self):
#         _existing_symptoms = np.where(self.patient.symptoms == 1)[0]
#         symptoms = self.find_symptom(_existing_symptoms)
#         # if(len(symptoms)<=2):
#         initial_symptom = np.random.choice(symptoms)
#             # self.state.symptoms[initial_symptom] = np.array([0, 1, 0])
#             # self.state.symptoms[initial_symptom] = 1
#         # print(initial_symptom)
#         symptom_nlice = []
#         for i in range(8):
#             symptom_nlice.append(self.patient.symptoms[int(initial_symptom)*8+i])
#         self.state.symptoms[int(initial_symptom)] = np.array(symptom_nlice)
#         self.inquiry_list.add(int(initial_symptom))
#         # else:
#         #     initial_symptom = np.random.choice(_existing_symptoms,2)
#         #     # print(initial_symptom)
#         #     for i in range(len(initial_symptom)):
#         #         # self.state.symptoms[initial_symptom[i]] = np.array([0, 1, 0])
#         #         self.state.symptoms[initial_symptom[i]] = 1
#         #         self.inquiry_list.add(initial_symptom[i])
#         self.status = configuration.DIALOGUE_STATUS_COMMING
#     def find_symptom(self,index):
#         symptoms=[]
#         for s in index:
#             if(s%8==0):
#                 symptoms.append(s/8)
#         return symptoms

#     def generate_state(self):
#         _symptoms = np.zeros((self.num_symptoms,8), dtype=np.uint8)  # all symptoms start as unknown
#         # _symptoms[:, 2] = 0

#         return RLState_Context(self.patient.age,self.patient.gender,_symptoms)

#     def is_valid_action(self, action):
#         if action < self.num_symptoms:
#             return True, self.is_inquiry, action  # it's an inquiry action
#         else:
#             action = action % (self.num_symptoms-1)
#             # if action == config.ACTION_DIAGNOSE:
#             return True, self.is_diagnose, action  # it's a diagnose action

#         return False, None, None

#     def take_action(self, action):
#         self.turn = self.turn+1
#         self.current_Q = action
#         top1 = action.max(1)[1].view(1, 1)
#         is_valid, action_type, action_value = self.is_valid_action(top1.item())
#         if not is_valid:
#             raise ValueError("Invalid action: %s" % action)
#         if action_type == self.is_inquiry:
#             return self.inquire(action_value)
#         else:
#             return self.diagnose(action_value)

#     def patient_has_symptom(self, symptom_idx):
#         return self.patient.symptoms[symptom_idx*8] == 1

#     def inquire(self, action_value):
#         """
#         returns state, reward, done
#         """
#         if action_value in self.inquiry_list:
#             # repeated inquiry
#             # return self.state, -1, True # reward is -3 if inquiry is repeated
#             return self.state, self.reward_repeated, configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE

#         # does the patient have the symptom
#         if self.patient_has_symptom(action_value):
#             # value = np.array([0, 1, 0])
#             value = 1
#             # self.state.symptoms[action_value] = value
#             self.update_nlice(action_value)
#             self.inquiry_list.add(action_value)
#             return self.state, self.reward_correct_inquiry,configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM
#         else:
#             value = [2,0,0,0,0,0,0,0]
#             self.state.symptoms[action_value] = np.array(value)
#             self.inquiry_list.add(action_value)
#             return self.state, self.reward_wrong_inquiry,configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE

#     def diagnose(self, action):
#         # we'll need to make a prediction
#         # if len(self.inquiry_list) < 2:
#         #     return None,self.reward_repeated,config.DIALOGUE_STATUS_FAILED
#         classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
#         if not self.policy:
#             action = classlabel[action-1]
#         else:
#             conditions = self.policy_transformation()
#             sorted_ = np.argsort(conditions)  
#             action = classlabel[sorted_[-1]]
#         is_correct = self.iscorrect(self.patient.condition,action)
#         # is_correct = self.patient.condition==action
#         reward = self.reward_diagnose_correct if is_correct else self.reward_diagnose_incorrect
#         if is_correct:
#             return None,reward,configuration.DIALOGUE_STATUS_SUCCESS
#         else:
#             return None,reward,configuration.DIALOGUE_STATUS_FAILED
    
    
#     def top_5(self):
#         classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
#         # index =self.current_Q.keys().item()
#         # value = self.current_Q.values().item()
#         result = self.current_Q.flatten().tolist()
#         if not self.policy:
#             conditions =[]
#             for v in result:
#                 inx = result.index(v)
#                 if inx>=(self.num_symptoms):
#                     conditions.append(v)
#         else:
#             conditions = self.policy_transformation()
#         sorted_ = np.argsort(conditions)
#         top_5 = []
#         debug = []
#         for i in range(5):
#              top_5.append(classlabel[sorted_[-(i+1)]])
#              debug.append(sorted_[-(i+1)])
    
#         is_correct = False
#         for a in top_5:
#             # is_correct = self.iscorrect(self.patient.condition,classlabel[a-1])
#             is_correct = self.iscorrect(self.patient.condition,a)
#             if(is_correct == True):
#                 return None,self.reward_diagnose_correct,configuration.DIALOGUE_STATUS_SUCCESS
#         return None,self.reward_diagnose_incorrect,configuration.DIALOGUE_STATUS_FAILED

#     def top_3(self):
#         classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
#         # index =self.current_Q.keys().item()
#         # value = self.current_Q.values().item()
#         result = self.current_Q.flatten().tolist()
#         if not self.policy:
#             conditions =[]
#             for v in result:
#                 inx = result.index(v)
#                 if inx>=(self.num_symptoms):
#                     conditions.append(v)
#         else:
#             conditions = self.policy_transformation()
#         sorted_ = np.argsort(conditions)    
#         top_3 = []
#         debug = []
#         for i in range(3):
#              top_3.append(classlabel[sorted_[-(i+1)]])
#              debug.append(sorted_[-(i+1)])
    
#         is_correct = False
#         for a in top_3:
#             # is_correct = self.iscorrect(self.patient.condition,classlabel[a-1])
#             is_correct = self.iscorrect(self.patient.condition,a)
#             if(is_correct == True):
#                 return None,self.reward_diagnose_correct,configuration.DIALOGUE_STATUS_SUCCESS
#         return None,self.reward_diagnose_incorrect,configuration.DIALOGUE_STATUS_FAILED

#     def policy_transformation(self):
#         result = self.current_Q.flatten().tolist()
#         conditions = []
#         for v in result:
#             inx = result.index(v)
#             if inx>=(self.num_symptoms):
#                 conditions.append(v)
#         sorted_ = np.argsort(conditions)
#         for i in sorted_:
#             co = self.conditions_reverse.get(classlabel[i])
#             pro_age = get_pro_age(co,self.patient.age,self.piors_map)/100
#             pro_gender=get_pro_sex(co,self.patient.gender,self.piors_map)/100
#             incidence = get_pro_incidence(co,self.piors_map)
#             # conditions[i] = conditions[i]*pro_age*pro_gender+incidence
#             conditions[i] = conditions[i]*pro_age*pro_gender
#         return conditions

#     def iscorrect(self,condition,action):
#         if condition in [48,49] and action in [48,49]:
#                 return True
#         elif condition in [37,38,39] and action in [37,38,39]:
#                 return True
#         elif condition in [40,41] and action in [40,41]:
#                 return True
#         elif condition in [28,44,45] and action in [28,44,45]:
#                 return True
#         elif condition == action:
#             return True
#         else:
#             return False

#     def update_nlice(self,action_value):
#         symptom_nlice=[]
#         for i in range(8):
#             symptom_nlice = self.patient.symptoms[action_value*8+i]
#         self.state.symptoms[action_value] = np.array(symptom_nlice)

#     def __del__(self):
#         self.close_data_file()