import json
import torch
import os
from collections import namedtuple
from itertools import repeat
from scipy.sparse import  csc_matrix
from fractions import Fraction
# from scipy import sparse
import numpy as np
from . import configuration as config
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
classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]

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

class AiBasicMedEnv:
    def __init__(
            self,
            data_file,
            symptom_map_file,
            condition_map_file,
            clf,
            **kwargs
    ):
        """
        data_file: A file of generated patient, symptoms, condition data
        symptom_map_file: the encoding file for symptoms
        condition_map_file: the encoding file for conditions
        initial_symptom_file: a map of conditions
        clf: a classifier which can output a probabilistic description of possible conditions based on
        symptoms and patient demography.
        """
        self.data_file = data_file
        self.symptom_map_file = symptom_map_file
        self.condition_map_file = condition_map_file
        self.clf = clf
        self.line_number = 0
        self.state = None
        self.patient = None
        self.data = None
        self.symptom_map = None
        self.condition_map = None
        self.initial_symptom_map = None
        self.num_symptoms = None
        self.num_conditions = None

        self.check_file_exists()

        self.load_data_file()
        self.load_symptom_map()
        self.load_condition_map()

        self.is_inquiry = 1
        self.is_diagnose = 2

        self.inquiry_list = set([])

        self.RACE_CODE = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}

        self.reward_inquiry = kwargs.get('reward_inquiry', -1)
        self.reward_repeated = kwargs.get('reward_repeated', -2)
        self.reward_diagnose_correct = kwargs.get('reward_diagnose_correct', 5)
        self.reward_diagnose_incorrect = kwargs.get('reward_diagnose_incorrect', 0)

    def check_file_exists(self):
        files = [self.data_file, self.symptom_map_file, self.condition_map_file]
        for file in files:
            if not os.path.exists(file):
                raise ValueError("File: %s does not exist" % file)

    def load_data_file(self):
        self.data = open(self.data_file)

    def close_data_file(self):
        if self.data is not None:
            self.data.close()

    def load_symptom_map(self):
        with open(self.symptom_map_file) as fp:
            symptoms = json.load(fp)
            sorted_symptoms = sorted(symptoms.keys())
            self.symptom_map = {code: idx for idx, code in enumerate(sorted_symptoms)}
            self.num_symptoms = len(self.symptom_map)

    def load_condition_map(self):
        with open(self.condition_map_file) as fp:
            conditions = json.load(fp)
            sorted_conditions = sorted(conditions.keys())
            self.condition_map = {code: idx for idx, code in enumerate(sorted_conditions)}
            self.num_conditions = len(self.condition_map)

    def readline(self):
        line = self.data.readline()
        line = "" if line is None else line.strip()
        return line

    def get_line(self):
        if self.line_number == 0:
            self.readline() # header line

        line = self.readline()
        if not line or len(line) == 0:
            # EOF
            self.data.seek(0)
            self.readline() # header line
            line = self.readline()

        self.line_number += 1
        return line

    def parse_line(self, line):
        parts = line.split(",")
        # print(parts)
        _gender = parts[1]
        _race = parts[2]      
        age = int(parts[3])
        condition = parts[4]
        symptom_list = parts[6]
        gender = 0 if _gender == 'M' else 1
        race = self.RACE_CODE.get(_race)
        condition = self.condition_map.get(condition)
        symptoms = list(repeat(0, self.num_symptoms))
        for item in symptom_list.split(";"):
            sym_list = item.split(":")
            # if(len(sym_list)==9):
            _symptom, _nature, _location, _intensity, _duration, _onset, _exciation, _frequency, _ = sym_list
            #todo add nlice in the future
            idx = self.symptom_map.get(_symptom)
            symptoms[idx] = 1
        # ('age', 'race', 'gender', 'symptoms', 'condition')
        symptoms = np.array(symptoms)
        patient = AiMedPatient(age, race, gender, symptoms, condition)
        return patient
    

    def reset(self):
        line = self.get_line()
        self.patient = self.parse_line(line)
        self.state = self.generate_state(
            self.patient.age,
            self.patient.race,
            self.patient.gender
        )
        self.inquiry_list = set([])

        self.pick_initial_symptom()

    def pick_initial_symptom(self):
        _existing_symptoms = np.where(self.patient.symptoms == 1)[0]

        initial_symptom = np.random.choice(_existing_symptoms)

        self.state.symptoms[initial_symptom] = np.array([0, 1, 0])
        self.inquiry_list.add(initial_symptom)

    def generate_state(self, age, race, gender):
        _symptoms = np.zeros((self.num_symptoms, 3), dtype=np.uint8)  # all symptoms start as unknown
        _symptoms[:, 2] = 1

        return AiMedState(age, race, gender, _symptoms)

    def is_valid_action(self, action):
        if action < self.num_symptoms:
            return True, self.is_inquiry, action  # it's an inquiry action
        else:
            action = action % self.num_symptoms

            if action == config.ACTION_DIAGNOSE:
                return True, self.is_diagnose, action  # it's a diagnose action

        return False, None, None

    def take_action(self, action):
        is_valid, action_type, action_value = self.is_valid_action(action)
        if not is_valid:
            raise ValueError("Invalid action: %s" % action)
        if action_type == self.is_inquiry:
            return self.inquire(action_value)
        else:
            return self.diagnose(action_value)

    def patient_has_symptom(self, symptom_idx):
        return self.patient.symptoms[symptom_idx] == 1

    def inquire(self, action_value):
        """
        returns state, reward, done
        """
        if action_value in self.inquiry_list:
            # repeated inquiry
            # return self.state, -1, True # reward is -3 if inquiry is repeated
            return self.state, self.reward_repeated, False

        # does the patient have the symptom
        if self.patient_has_symptom(action_value):
            value = np.array([0, 1, 0])
        else:
            value = np.array([1, 0, 0])

        self.state.symptoms[action_value] = value
        self.inquiry_list.add(action_value)

        return self.state, self.reward_inquiry, False # reward is -1 for non-repeated inquiry

    def get_patient_vector(self):
        patient_vector = np.zeros(3 + self.num_symptoms, dtype=np.uint8)
        patient_vector[0] = self.state.gender
        patient_vector[1] = self.state.race
        patient_vector[2] = self.state.age

        has_symptom = np.where(self.state.symptoms[:, 1] == 1)[0] + 3
        patient_vector[has_symptom] = 1

        return patient_vector.reshape(1, -1)

    def predict_condition(self):
        patient_vector = self.get_patient_vector()
        patient_vector = csc_matrix(patient_vector)

        prediction = self.clf.predict(patient_vector)

        return prediction

    def diagnose(self, action_value):
        # we'll need to make a prediction
        prediction = self.predict_condition()[0]

        is_correct = action_value == prediction
        reward = self.reward_diagnose_correct if is_correct else self.reward_diagnose_incorrect

        return None, reward, True

    def __del__(self):
        self.close_data_file()


class AiBasicMedEnvSample:
    def __init__(
            self,
            definition_file,
            symptom_map_file,
            condition_map_file,
            clf
    ):
        """
        data_file: A file of generated patient, symptoms, condition data
        symptom_map_file: the encoding file for symptoms
        condition_map_file: the encoding file for conditions
        initial_symptom_file: a map of conditions
        clf: a classifier which can output a probabilistic description of possible conditions based on
        symptoms and patient demography.
        """
        self.symptom_map_file = symptom_map_file
        self.condition_map_file = condition_map_file
        self.definition_file = definition_file
        self.clf = clf

        self.state = None
        self.patient = None
        self.symptom_map = None
        self.condition_map = None
        self.definition = None
        self.initial_symptom_map = None
        self.num_symptoms = None
        self.num_conditions = None

        self.load_definition_file()
        self.load_symptom_map()
        self.load_condition_map()

        self.is_inquiry = 1
        self.is_diagnose = 2

        self.inquiry_list = set([])

        self.RACE_CODE = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}

    def load_symptom_map(self):
        with open(self.symptom_map_file) as fp:
            symptoms = json.load(fp)
            sorted_symptoms = sorted(symptoms.keys())
            self.symptom_map = {code: idx for idx, code in enumerate(sorted_symptoms)}
            self.num_symptoms = len(self.symptom_map)

    def load_condition_map(self):
        with open(self.condition_map_file) as fp:
            conditions = json.load(fp)
            sorted_conditions = sorted(conditions.keys())
            self.condition_map = {code: idx for idx, code in enumerate(sorted_conditions)}
            self.num_conditions = len(self.condition_map)

    def load_definition_file(self):
        with open(self.definition_file) as fp:
            self.definition = json.load(fp)

    def generate_patient(self):
        return AiMedPatient(-1, -1, -1, -1, -1)

    def reset(self):
        self.patient = self.generate_patient()
        self.state = self.generate_state(
            self.patient.age,
            self.patient.race,
            self.patient.gender
        )
        self.inquiry_list = set([])

        self.pick_initial_symptom()

    def pick_initial_symptom(self):
        _existing_symptoms = np.where(self.patient.symptoms == 1)[0]

        initial_symptom = np.random.choice(_existing_symptoms)

        self.state.symptoms[initial_symptom] = np.array([0, 1, 0])
        self.inquiry_list.add(initial_symptom)

    def generate_state(self, age, race, gender):
        _symptoms = np.zeros((self.num_symptoms, 3), dtype=np.uint8)  # all symptoms start as unknown
        _symptoms[:, 2] = 1

        return AiMedState(age, race, gender, _symptoms)

    def is_valid_action(self, action):
        if action < self.num_symptoms:
            return True, self.is_inquiry, action  # it's an inquiry action
        else:
            action = action % self.num_symptoms

            if action < self.num_conditions:
                return True, self.is_diagnose, action  # it's a diagnose action

        return False, None, None

    def take_action(self, action):
        is_valid, action_type, action_value = self.is_valid_action(action)
        if not is_valid:
            raise ValueError("Invalid action: %s" % action)
        if action_type == self.is_inquiry:
            return self.inquire(action_value)
        else:
            return self.diagnose(action_value)

    def patient_has_symptom(self, symptom_idx):
        return self.patient.symptoms[symptom_idx] == 1

    def inquire(self, action_value):
        """
        returns state, reward, done
        """
        if action_value in self.inquiry_list:
            # repeated inquiry
            # return self.state, -1, True  # we terminate on a repeated inquiry
            return None, -1, True

        # does the patient have the symptom
        if self.patient_has_symptom(action_value):
            value = np.array([0, 1, 0])
        else:
            value = np.array([1, 0, 0])

        self.state.symptoms[action_value] = value
        self.inquiry_list.add(action_value)

        return self.state, 0, False

    def get_patient_vector(self):
        patient_vector = np.zeros(3 + self.num_symptoms, dtype=np.uint8)
        patient_vector[0] = self.state.gender
        patient_vector[1] = self.state.race
        patient_vector[2] = self.state.age

        has_symptom = np.where(self.state.symptoms[:, 1] == 1)[0] + 3
        patient_vector[has_symptom] = 1

        return patient_vector.reshape(1, -1)

    def predict_condition(self):
        patient_vector = self.get_patient_vector()
        patient_vector = csc_matrix(patient_vector)

        prediction = self.clf.predict(patient_vector)

        return prediction

    def diagnose(self, action_value):
        # enforce that there should be at least one inquiry in addition to the initial symptom
        if len(self.inquiry_list) < 2:
            # return self.state, -1, True  # we always terminate on a repeated enquiry
            return None, -1, True

        # we'll need to make a prediction
        prediction = self.predict_condition()[0]

        is_correct = action_value == prediction
        reward = 1 if is_correct else 0

        return None, reward, True



class RLBasicMedEnv:
    def __init__(
            self,
            data_file,
            symptom_map_file,
            condition_map_file,
            clf,
            classifer,
            max_turn,
            epoch,
            **kwargs
    ):
        """
        data_file: A file of generated patient, symptoms, condition data
        symptom_map_file: the encoding file for symptoms
        condition_map_file: the encoding file for conditions
        initial_symptom_file: a map of conditions
        clf: a classifier which can output a probabilistic description of possible conditions based on
        symptoms and patient demography.
        """
        self.data_file = data_file
        self.symptom_map_file = symptom_map_file
        self.condition_map_file = condition_map_file
        self.clf = clf
        self.epoch = epoch
        self.classifier = classifer
        self.line_number = 0
        self.state = None
        self.patient = None
        self.data = None
        self.symptom_map = None
        self.condition_map = None
        self.initial_symptom_map = None
        self.num_symptoms = None
        self.num_conditions = None
        self.max_turn = max_turn
        self.status = None
        self.check_file_exists()

        self.load_data_file()
        self.load_symptom_map()
        self.load_condition_map()

        self.is_inquiry = 1
        self.is_diagnose = 2

        self.inquiry_list = set([])

        self.RACE_CODE = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}
        self.reward_correct_inquiry = kwargs.get('reward_inquiry', self.max_turn)
        self.reward_wrong_inquiry = kwargs.get('reward_inquiry', 0)
        self.reward_repeated = kwargs.get('reward_repeated', -self.max_turn)
        self.reward_diagnose_correct = kwargs.get('reward_diagnose_correct', 2*self.max_turn)
        self.reward_diagnose_incorrect = kwargs.get('reward_diagnose_incorrect', -self.max_turn)

    def check_file_exists(self):
        files = [self.data_file, self.symptom_map_file, self.condition_map_file]
        for file in files:
            if not os.path.exists(file):
                raise ValueError("File: %s does not exist" % file)

    def load_data_file(self):
        self.data = open(self.data_file)

    def close_data_file(self):
        if self.data is not None:
            self.data.close()

    def load_symptom_map(self):
        with open(self.symptom_map_file) as fp:
            symptoms = json.load(fp)
            sorted_symptoms = sorted(symptoms.keys())
            self.symptom_map = {code: idx for idx, code in enumerate(sorted_symptoms)}
            self.num_symptoms = len(self.symptom_map)

    def load_condition_map(self):
        with open(self.condition_map_file) as fp:
            conditions = json.load(fp)
            sorted_conditions = sorted(conditions.keys())
            self.condition_map = {code: idx for idx, code in enumerate(sorted_conditions)}
            self.num_conditions = len(self.condition_map)

    def readline(self):
        line = self.data.readline()
        line = "" if line is None else line.strip()
        return line

    def get_line(self):
        if self.line_number == 0:
            self.readline() # header line

        line = self.readline()
        if not line or len(line) == 0:
            # EOF
            self.data.seek(0)
            self.readline() # header line
            line = self.readline()

        self.line_number += 1
        return line

    def parse_line(self, line):
        parts = line.split(",")
        # print(parts)
        _gender = parts[1]
        _race = parts[2]      
        age = int(parts[3])
        condition = parts[4]
        symptom_list = parts[6]
        gender = 0 if _gender == 'M' else 1
        race = self.RACE_CODE.get(_race)
        condition = self.condition_map.get(condition)
        symptoms = list(repeat(0, self.num_symptoms))
        for item in symptom_list.split(";"):
            sym_list = item.split(":")
            # if(len(sym_list)==9):
            _symptom, _nature, _location, _intensity, _duration, _onset, _exciation, _frequency, _ = sym_list
            #todo add nlice in the future
            idx = self.symptom_map.get(_symptom)
            symptoms[idx] = 1
        # ('age', 'race', 'gender', 'symptoms', 'condition')
        symptoms = np.array(symptoms)
        patient = AiMedPatient(age, race, gender, symptoms, condition)
        return patient

    def reset(self):
        if self.line_number == self.epoch:
            self.line_number = 0
            self.load_data_file()
        line = self.get_line()
        self.patient = self.parse_line(line)
        self.state = self.generate_state(
            self.patient.age,
            self.patient.race,
            self.patient.gender
        )
        self.inquiry_list = set([])

        self.pick_initial_symptom()

    def pick_initial_symptom(self):
        _existing_symptoms = np.where(self.patient.symptoms == 1)[0]
        if(len(_existing_symptoms)<=2):
            initial_symptom = np.random.choice(_existing_symptoms)
            self.state.symptoms[initial_symptom] = np.array([0, 1, 0])
            self.inquiry_list.add(initial_symptom)
        else:
            initial_symptom = np.random.choice(_existing_symptoms,2)
            # print(initial_symptom)
            for i in range(len(initial_symptom)):
                self.state.symptoms[initial_symptom[i]] = np.array([0, 1, 0])
                self.inquiry_list.add(initial_symptom[i])
        self.status = config.DIALOGUE_STATUS_COMMING

    def generate_state(self, age, race, gender):
        _symptoms = np.zeros((self.num_symptoms, 3), dtype=np.uint8)  # all symptoms start as unknown
        _symptoms[:, 2] = 1

        return AiMedState(age, race, gender, _symptoms)

    def is_valid_action(self, action):
        if action < self.num_symptoms:
            return True, self.is_inquiry, action  # it's an inquiry action
        else:
            action = action % (self.num_symptoms-1)
            # if action == config.ACTION_DIAGNOSE:
            return True, self.is_diagnose, action  # it's a diagnose action

        return False, None, None

    def take_action(self, action):
        is_valid, action_type, action_value = self.is_valid_action(action)
        if not is_valid:
            raise ValueError("Invalid action: %s" % action)
        if action_type == self.is_inquiry:
            return self.inquire(action_value)
        else:
            return self.diagnose(action_value)

    def patient_has_symptom(self, symptom_idx):
        return self.patient.symptoms[symptom_idx] == 1

    def inquire(self, action_value):
        """
        returns state, reward, done
        """
        if action_value in self.inquiry_list:
            # repeated inquiry
            # return self.state, -1, True # reward is -3 if inquiry is repeated
            return self.state, self.reward_repeated, config.DIALOGUE_STATUS_FAILED

        # does the patient have the symptom
        if self.patient_has_symptom(action_value):
            value = np.array([0, 1, 0])
            self.state.symptoms[action_value] = value
            self.inquiry_list.add(action_value)
            return self.state, self.reward_correct_inquiry,config.DIALOGUE_STATUS_COMMING
        else:
            value = np.array([1, 0, 0])
            self.state.symptoms[action_value] = value
            self.inquiry_list.add(action_value)
            return self.state, self.reward_wrong_inquiry,config.DIALOGUE_STATUS_COMMING
        
    

        # reward is -1 for non-repeated inquiry

    def get_patient_vector_nb(self):

        patient_vector = np.zeros(3 + 8*self.num_symptoms, dtype=np.uint16)
        patient_vector[0] = self.state.gender
        patient_vector[1] = self.state.race
        patient_vector[2] = self.state.age
        has_symptom = (np.where(self.state.symptoms[:, 1] == 1)[0])*8 + 3
        for _symptom_idx in has_symptom:
            _nature_idx =_symptom_idx+1
            _location_idx = _symptom_idx + 2
            _vas_idx = _symptom_idx + 3
            _duration_idx = _symptom_idx + 4
            _onset_idx = _symptom_idx + 5
            _excitation_idx = _symptom_idx + 6
            _frequency_idx = _symptom_idx + 7

            _symptom_val = 1
            _nature_val =  1
            _location_val =  1
            _vas_val =  1
            _excitation_val =  1
            _frequency_val =  1
            _duration_val = 0
            _onset_val = 0
            patient_vector[_symptom_idx] = 1
            patient_vector[_nature_idx] = _nature_val
            patient_vector[_location_idx] = _location_val
            patient_vector[_vas_idx] = _vas_val
            patient_vector[_duration_idx] = _duration_val
            patient_vector[_onset_idx] = _onset_val
            patient_vector[_frequency_idx] = _frequency_val
            patient_vector[_excitation_idx] = _excitation_val
            patient_vector[has_symptom] = 1
        patient_vector = csc_matrix(patient_vector.reshape(1, -1))
        num_features = self.num_symptoms*8 +3
        reg_indices = np.array([0, 1, 2])
        symptom_indices = np.arange(3, num_features, 8, dtype=np.uint16)
        nature_indices = symptom_indices + 1
        location_indices = symptom_indices + 2
        intensity_indices = symptom_indices + 3
        duration_indices = symptom_indices + 4
        onset_indices = symptom_indices + 5
        excitation_indices = symptom_indices + 6
        frequency_indices = symptom_indices + 7
        _nb_indices = np.hstack([
        reg_indices,
        symptom_indices,
        nature_indices, location_indices, intensity_indices, excitation_indices, frequency_indices,
        duration_indices, onset_indices
        ])
        nb_vector = patient_vector[:,_nb_indices]
        return nb_vector

    def get_patient_vector_rf(self):
        patient_vector = np.zeros(7 + 8*self.num_symptoms, dtype=np.uint16)
        patient_vector[0] = self.state.age
        patient_vector[1] = self.state.gender
        race_idx = self.state.race +2
        patient_vector[race_idx] = 1
        has_symptom = (np.where(self.state.symptoms[:, 1] == 1)[0])*8 + 5
        for _symptom_idx in has_symptom:
            _nature_idx =_symptom_idx+1
            _location_idx = _symptom_idx + 2
            _vas_idx = _symptom_idx + 3
            _duration_idx = _symptom_idx + 4
            _onset_idx = _symptom_idx + 5
            _excitation_idx = _symptom_idx + 6
            _frequency_idx = _symptom_idx + 7

            _symptom_val = 1
            _nature_val =  1
            _location_val =  1
            _vas_val =  1
            _excitation_val =  1
            _frequency_val =  1
            _duration_val = 0
            _onset_val = 0
            patient_vector[_symptom_idx] = 1
            patient_vector[_nature_idx] = _nature_val
            patient_vector[_location_idx] = _location_val
            patient_vector[_vas_idx] = _vas_val
            patient_vector[_duration_idx] = _duration_val
            patient_vector[_onset_idx] = _onset_val
            patient_vector[_frequency_idx] = _frequency_val
            patient_vector[_excitation_idx] = _excitation_val
            patient_vector[has_symptom] = 1
        patient_vector = csc_matrix(patient_vector.reshape(1, -1))
        return patient_vector

    def predict_condition(self):
        if self.classifier =="nb":
            patient_vector = self.get_patient_vector_nb()
        elif self.classifier == "rf":
            patient_vector = self.get_patient_vector_rf()
        # patient_vector = csc_matrix(patient_vector)
        prediction = self.clf.predict_proba(patient_vector)

        return prediction

    def diagnose(self, action):
        # we'll need to make a prediction
        # if len(self.inquiry_list) < 2:
        #     return None,self.reward_repeated,config.DIALOGUE_STATUS_FAILED
        classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
        prediction = self.predict_condition()
        sorted_number = sorted(enumerate(prediction[0]),key = lambda x :x[1],reverse = True)
        idx = [i[0] for i in sorted_number]
        nums = [i[1] for i in sorted_number]
        sorted_ = np.argsort(prediction)
        top_5 = []
        for i in range(5):
             top_5.append(classlabel[sorted_[0][-(i+1)]])
        # print(nb_top_5)
        # classification = {}
        # for idx in nb_top_5:
        #     classification[self.condition_map[idx]] = float(prediction[0, idx])
        # rf_classification = {
        #     "prediction": self.condition_map[nb_top_5[0]],
        #     "top_5": classification
        #     }
        is_correct = self.patient.condition in top_5
        reward = self.reward_diagnose_correct if is_correct else self.reward_diagnose_incorrect
        if is_correct:
            return None,reward,config.DIALOGUE_STATUS_SUCCESS
        else:
            return None,reward,config.DIALOGUE_STATUS_FAILED

    # def iscorrect(self,top_5):
    #     if self.patient.condition == 48:
    #         top_5.append(49) 
    #     if self.patient.condition == 49:
    #         top_5.append(48)
    #     if self.patient.condition == 40:
    #         top_5.append(41)
    #     if self.patient.condition == 41:
    #         top_5.append(40)

    #     if self.patient.condition == 37:
    #         top_5.append(38) 
    #         top_5.append(39)
    #     if self.patient.condition == 38:
    #         top_5.append(37) 
    #         top_5.append(39)
    #     if self.patient.condition == 39:
    #         top_5.append(38) 
    #         top_5.append(37)
        
    #     if self.patient
      
    def __del__(self):
        self.close_data_file()



'''
NO RACE GENDER SEX
There are also three types of symptom states, positive, negative, and not mentioned, represented by 1, - 1, 0 in symptom
vectors respectively
'''
class RLBasicMedEnvNOPATIENT:
    def __init__(
            self,
            data_file,
            symptom_map_file,
            condition_map_file,
            body_parts,
            excitation_enc,
            frequency_enc,
            nature_enc,
            vas_enc,
            onset_enc,
            duration_enc,
            clf,
            classifer,
            max_turn,
            epoch,
            **kwargs
    ):
        """
        data_file: A file of generated patient, symptoms, condition data
        symptom_map_file: the encoding file for symptoms
        condition_map_file: the encoding file for conditions
        initial_symptom_file: a map of conditions
        clf: a classifier which can output a probabilistic description of possible conditions based on
        symptoms and patient demography.
        """
        self.data_file = data_file
        self.symptom_map_file = symptom_map_file
        self.condition_map_file = condition_map_file
        self.body_file = body_parts
        self.excitation_file=excitation_enc
        self.frequency_file=frequency_enc
        self.nature_file=nature_enc
        self.vas_file=vas_enc
        self.onset_file=onset_enc
        self.duration_file=duration_enc
        self.clf = clf
        self.epoch = epoch
        self.classifier = classifer
        self.line_number = 0
        self.state = None
        self.patient = None
        self.data = None
        self.symptom_map = None
        self.condition_map = None
        self.body_map=None
        self.excitation_map=None
        self.frequency_map=None
        self.nature_map=None
        self.vas_map=None
        self.onset_map=None
        self.duration_map=None
        self.initial_symptom_map = None
        self.num_symptoms = None
        self.num_conditions = None
        self.max_turn = max_turn
        self.status = None
        self.check_file_exists()

        self.load_data_file()
        self.load_symptom_map()
        self.load_condition_map()

        self.is_inquiry = 1
        self.is_diagnose = 2

        self.inquiry_list = set([])

        self.RACE_CODE = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}
        self.reward_correct_inquiry = kwargs.get('reward_inquiry', self.max_turn)
        self.reward_wrong_inquiry = kwargs.get('reward_inquiry', 0)
        self.reward_repeated = kwargs.get('reward_repeated', -1)
        self.reward_diagnose_correct = kwargs.get('reward_diagnose_correct', 2*self.max_turn)
        self.reward_diagnose_incorrect = kwargs.get('reward_diagnose_incorrect', 0)

    def check_file_exists(self):
        files = [self.data_file, self.symptom_map_file, self.condition_map_file,self.body_file,self.frequency_file,self.nature_file,self.excitation_file,self.vas_file]
        for file in files:
            if not os.path.exists(file):
                raise ValueError("File: %s does not exist" % file)

    def load_data_file(self):
        self.data = open(self.data_file)

    def close_data_file(self):
        if self.data is not None:
            self.data.close()

    def load_symptom_map(self):
        with open(self.symptom_map_file) as fp:
            symptoms = json.load(fp)
            sorted_symptoms = sorted(symptoms.keys())
            self.symptom_map = {code: idx for idx, code in enumerate(sorted_symptoms)}
            self.num_symptoms = len(self.symptom_map)
        with open(self.body_file) as fp:
            self.body_map = json.load(fp)
        with open(self.frequency_file) as fp:
            self.frequency_map = json.load(fp)
        with open(self.excitation_file) as fp:
            self.excitation_map = json.load(fp)
        with open(self.nature_file) as fp:
            self.nature_map = json.load(fp)
        with open(self.vas_file) as fp:
            self.vas_map = json.load(fp)
        with open(self.onset_file) as fp:
            self.onset_map = json.load(fp)
        with open(self.duration_file) as fp:
            self.duration_map = json.load(fp)
    def load_condition_map(self):
        with open(self.condition_map_file) as fp:
            conditions = json.load(fp)
            sorted_conditions = sorted(conditions.keys())
            self.condition_map = {code: idx for idx, code in enumerate(sorted_conditions)}
            self.num_conditions = len(self.condition_map)
     
    def readline(self):
        line = self.data.readline()
        line = "" if line is None else line.strip()
        return line

    def get_line(self):
        if self.line_number == 0:
            self.readline() # header line

        line = self.readline()
        if not line or len(line) == 0:
            # EOF
            self.data.seek(0)
            self.readline() # header line
            line = self.readline()

        self.line_number += 1
        return line

    def parse_line(self, line):
        parts = line.split(",")
        # print(parts)
        _gender = parts[1]
        _race = parts[2]      
        age = int(parts[3])
        condition = parts[4]
        symptom_list = parts[6]
        gender = 0 if _gender == 'M' else 1
        race = self.RACE_CODE.get(_race)
        condition = self.condition_map.get(condition)
        symptoms_nlice = np.zeros(8 *self.num_symptoms, dtype=np.uint16)
        # symptoms = np.zeros(len(self.num_symptoms), dtype=np.uint16)
        for item in symptom_list.split(";"):
            sym_list = item.split(":")
            # if(len(sym_list)==9):
            _symptom, _nature, _location, _intensity, _duration, _onset, _exciation, _frequency, _ = sym_list
            #todo add nlice in the future
            _symptom.replace("Alterred_stool", "Altered_stool")
            _symptom.replace("Nausea_", "Nausea")
            _symptom.replace("Pain_relief_", "Pain_relief")
            _symptom.replace("Vomitting", "Vomiting")
            _symptom_idx = self.symptom_map.get(_symptom)*8
            _nature_idx = _symptom_idx + 1
            _location_idx = _symptom_idx + 2
            _intensity_idx = _symptom_idx + 3
            _duration_idx = _symptom_idx + 4
            _onset_idx = _symptom_idx + 5
            _excitation_idx = _symptom_idx + 6
            _frequency_idx = _symptom_idx + 7
            _nature_val = 1 if _nature == "" or _nature == "other" else self.nature_map.get(_nature)
            _location_val = 1 if _location == "" or _location == "other" else self.body_map.get(_location)
            _intensity_val = 1 if _intensity == "" else self.vas_map.get(_intensity)
            _duration_val = 1 if _duration == "" else self.duration_map.get(_duration)
            _onset_val = 1 if _onset == "" else self.onset_map.get(_onset)
            _excitation_val = 1 if _exciation == "" else self.excitation_map.get(_exciation)
            _frequency_val = 1 if _frequency == "" else self.frequency_map.get(_frequency)
            symptoms_nlice[_symptom_idx] = 1
            symptoms_nlice[_nature_idx] = _nature_val
            symptoms_nlice[_location_idx] = _location_val
            symptoms_nlice[_intensity_idx] = _intensity_val
            symptoms_nlice[_duration_idx] = _duration_val
            symptoms_nlice[_onset_idx] = _onset_val
            symptoms_nlice[_excitation_idx] = _excitation_val
            symptoms_nlice[_frequency_idx] = _frequency_val

        # ('age', 'race', 'gender', 'symptoms', 'condition')
        symptoms = np.array(symptoms_nlice)
        patient = RLPatient(symptoms, condition)
        return patient

    def reset(self):
        if self.line_number == self.epoch:
            self.line_number = 0
            self.load_data_file()
        line = self.get_line()
        self.patient = self.parse_line(line)
        self.state = self.generate_state()
        self.inquiry_list = set([])

        self.pick_initial_symptom()

    def pick_initial_symptom(self):
        _existing_symptoms = np.where(self.patient.symptoms == 1)[0]
        symptoms = self.find_symptom(_existing_symptoms)

        # if(len(symptoms)<=2):
        initial_symptom = np.random.choice(symptoms)
            # self.state.symptoms[initial_symptom] = np.array([0, 1, 0])
            # self.state.symptoms[initial_symptom] = 1
        symptom_nlice = []
        for i in range(8):
            symptom_nlice.append(self.patient.symptoms[initial_symptom*8+i])
        self.state.symptoms[initial_symptom] = np.array(symptom_nlice)
        self.inquiry_list.add(initial_symptom)
        # else:
        #     initial_symptom = np.random.choice(_existing_symptoms,2)
        #     # print(initial_symptom)
        #     for i in range(len(initial_symptom)):
        #         # self.state.symptoms[initial_symptom[i]] = np.array([0, 1, 0])
        #         self.state.symptoms[initial_symptom[i]] = 1
        #         self.inquiry_list.add(initial_symptom[i])
        self.status = config.DIALOGUE_STATUS_COMMING
    def find_symptom(self,index):
        symptoms=[]
        for s in index:
            if(s%8==0):
                symptoms.append(s%8)
        return symptoms

    def generate_state(self):
        _symptoms = np.zeros((self.num_symptoms,8), dtype=np.uint8)  # all symptoms start as unknown
        # _symptoms[:, 2] = 0

        return RLState(_symptoms)

    def is_valid_action(self, action):
        if action < self.num_symptoms:
            return True, self.is_inquiry, action  # it's an inquiry action
        else:
            action = action % (self.num_symptoms-1)
            # if action == config.ACTION_DIAGNOSE:
            return True, self.is_diagnose, action  # it's a diagnose action

        return False, None, None

    def take_action(self, action):
        is_valid, action_type, action_value = self.is_valid_action(action)
        if not is_valid:
            raise ValueError("Invalid action: %s" % action)
        if action_type == self.is_inquiry:
            return self.inquire(action_value)
        else:
            return self.diagnose(action_value)

    def patient_has_symptom(self, symptom_idx):
        return self.patient.symptoms[symptom_idx*8] == 1

    def inquire(self, action_value):
        """
        returns state, reward, done
        """
        if action_value in self.inquiry_list:
            # repeated inquiry
            # return self.state, -1, True # reward is -3 if inquiry is repeated
            return self.state, self.reward_repeated, config.DIALOGUE_STATUS_INFORM_WRONG_DISEASE

        # does the patient have the symptom
        if self.patient_has_symptom(action_value):
            # value = np.array([0, 1, 0])
            value = 1
            # self.state.symptoms[action_value] = value
            self.update_nlice(action_value)
            self.inquiry_list.add(action_value)
            return self.state, self.reward_correct_inquiry,config.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM
        else:
            value = [2,0,0,0,0,0,0,0]
            self.state.symptoms[action_value] = np.array(value)
            self.inquiry_list.add(action_value)
            return self.state, self.reward_wrong_inquiry,config.DIALOGUE_STATUS_INFORM_WRONG_DISEASE

    def diagnose(self, action):
        # we'll need to make a prediction
        # if len(self.inquiry_list) < 2:
        #     return None,self.reward_repeated,config.DIALOGUE_STATUS_FAILED
        classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
        action = classlabel[action-1]
        is_correct = self.iscorrect(self.patient.condition,action)
        reward = self.reward_diagnose_correct if is_correct else self.reward_diagnose_incorrect
        if is_correct:
            return None,reward,config.DIALOGUE_STATUS_SUCCESS
        else:
            return None,reward,config.DIALOGUE_STATUS_FAILED
    def top_5(self,action):
        classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
        new_a= []
        is_correct = False
        for a in action:
            a = a % (self.num_symptoms-1)
            is_correct = self.iscorrect(self.patient.condition,classlabel[a-1])
            if(is_correct == True):
                return None,self.reward_diagnose_correct,config.DIALOGUE_STATUS_SUCCESS
        return None,self.reward_diagnose_incorrect,config.DIALOGUE_STATUS_FAILED



    def iscorrect(self,condition,action):
        if condition in [48,49] and action in [48,49]:
                return True
        elif condition in [37,38,39] and action in [37,38,39]:
                return True
        elif condition in [40,41] and action in [40,41]:
                return True
        elif condition in [28,44,45] and action in [28,44,45]:
                return True
        elif condition == action:
            return True
        else:
            return False

    def update_nlice(self,action_value):
        symptom_nlice=[]
        for i in range(8):
            symptom_nlice = self.patient.symptoms[action_value*8+i]
        self.state.symptoms[action_value] = np.array(symptom_nlice)

    def __del__(self):
        self.close_data_file()



'''
with RACE GENDER SEX
There are also three types of symptom states, positive, negative, and not mentioned, represented by 1, - 1, 0 in symptom
vectors respectively
'''
class RLBasicMedEnvContext:
    def __init__(
            self,
            data_file,
            symptom_map_file,
            condition_map_file,
            body_parts,
            excitation_enc,
            frequency_enc,
            nature_enc,
            vas_enc,
            onset_enc,
            duration_enc,
            piors,
            classifer,
            max_turn,
            epoch,
            policy,
            **kwargs
    ):
        """
        data_file: A file of generated patient, symptoms, condition data
        symptom_map_file: the encoding file for symptoms
        condition_map_file: the encoding file for conditions
        initial_symptom_file: a map of conditions
        clf: a classifier which can output a probabilistic description of possible conditions based on
        symptoms and patient demography.
        """
        self.data_file = data_file
        self.symptom_map_file = symptom_map_file
        self.condition_map_file = condition_map_file
        self.body_file = body_parts
        self.excitation_file=excitation_enc
        self.frequency_file=frequency_enc
        self.nature_file=nature_enc
        self.vas_file=vas_enc
        self.onset_file=onset_enc
        self.duration_file=duration_enc
        self.piors =piors
        self.epoch = epoch
        self.classifier = classifer
        self.line_number = 0
        self.state = None
        self.patient = None
        self.data = None
        self.symptom_map = None
        self.condition_map = None
        self.body_map=None
        self.excitation_map=None
        self.frequency_map=None
        self.nature_map=None
        self.vas_map=None
        self.onset_map=None
        self.duration_map=None
        self.initial_symptom_map = None
        self.piors_map = None
        self.num_symptoms = None
        self.num_conditions = None
        self.current_Q = None
        self.max_turn = max_turn
        self.status = None
        self.conditions_reverse = None
        self.check_file_exists()
        self.policy = policy
        self.load_data_file()
        self.load_symptom_map()
        self.load_condition_map()

        self.is_inquiry = 1
        self.is_diagnose = 2

        self.inquiry_list = set([])

        self.RACE_CODE = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}
        self.reward_correct_inquiry = kwargs.get('reward_inquiry', self.max_turn)
        self.reward_wrong_inquiry = kwargs.get('reward_inquiry', 0)
        self.reward_repeated = kwargs.get('reward_repeated', -1)
        self.reward_diagnose_correct = kwargs.get('reward_diagnose_correct', self.max_turn)
        self.reward_diagnose_incorrect = kwargs.get('reward_diagnose_incorrect', 0)

    def check_file_exists(self):
        files = [self.data_file, self.symptom_map_file, self.condition_map_file,self.body_file,self.frequency_file,self.nature_file,self.excitation_file,self.vas_file]
        for file in files:
            if not os.path.exists(file):
                raise ValueError("File: %s does not exist" % file)

    def load_data_file(self):
        self.data = open(self.data_file)

    def close_data_file(self):
        if self.data is not None:
            self.data.close()

    def load_symptom_map(self):
        with open(self.symptom_map_file) as fp:
            symptoms = json.load(fp)
            sorted_symptoms = sorted(symptoms.keys())
            self.symptom_map = {code: idx for idx, code in enumerate(sorted_symptoms)}
            self.num_symptoms = len(self.symptom_map)
        with open(self.body_file) as fp:
            self.body_map = json.load(fp)
        with open(self.frequency_file) as fp:
            self.frequency_map = json.load(fp)
        with open(self.excitation_file) as fp:
            self.excitation_map = json.load(fp)
        with open(self.nature_file) as fp:
            self.nature_map = json.load(fp)
        with open(self.vas_file) as fp:
            self.vas_map = json.load(fp)
        with open(self.onset_file) as fp:
            self.onset_map = json.load(fp)
        with open(self.duration_file) as fp:
            self.duration_map = json.load(fp)
        with open(self.piors) as fp:
            self.piors_map = json.load(fp)
    def load_condition_map(self):
        with open(self.condition_map_file) as fp:
            conditions = json.load(fp)
            sorted_conditions = sorted(conditions.keys())
            self.condition_map = {code: idx for idx, code in enumerate(sorted_conditions)}
            self.num_conditions = len(self.condition_map)
        self.conditions_reverse = dict(map(reversed, self.condition_map.items()))
     
    def readline(self):
        line = self.data.readline()
        line = "" if line is None else line.strip()
        return line

    def get_line(self):
        if self.line_number == 0:
            self.readline() # header line

        line = self.readline()
        if not line or len(line) == 0:
            # EOF
            self.data.seek(0)
            self.readline() # header line
            line = self.readline()

        self.line_number += 1
        return line

    def parse_line(self, line):
        parts = line.split(",")
        # print(parts)
        _gender = parts[1]
        _race = parts[2]      
        age = int(parts[3])
        condition = parts[4]
        age = self.get_age(condition)
        symptom_list = parts[6]
        gender = 0 if _gender == 'M' else 1
        race = self.RACE_CODE.get(_race)
        condition = self.condition_map.get(condition)
        symptoms_nlice = np.zeros(8 *self.num_symptoms, dtype=np.uint16)
        # symptoms = np.zeros(len(self.num_symptoms), dtype=np.uint16)
        for item in symptom_list.split(";"):
            sym_list = item.split(":")
            # if(len(sym_list)==9):
            _symptom, _nature, _location, _intensity, _duration, _onset, _exciation, _frequency, _ = sym_list
            #todo add nlice in the future
            _symptom.replace("Alterred_stool", "Altered_stool")
            _symptom.replace("Nausea_", "Nausea")
            _symptom.replace("Pain_relief_", "Pain_relief")
            _symptom.replace("Vomitting", "Vomiting")
            _symptom_idx = self.symptom_map.get(_symptom)*8
            _nature_idx = _symptom_idx + 1
            _location_idx = _symptom_idx + 2
            _intensity_idx = _symptom_idx + 3
            _duration_idx = _symptom_idx + 4
            _onset_idx = _symptom_idx + 5
            _excitation_idx = _symptom_idx + 6
            _frequency_idx = _symptom_idx + 7
            _nature_val = 1 if _nature == "" or _nature == "other" else self.nature_map.get(_nature)
            _location_val = 1 if _location == "" or _location == "other" else self.body_map.get(_location)
            _intensity_val = 1 if _intensity == "" else self.vas_map.get(_intensity)
            _duration_val = 1 if _duration == "" else self.duration_map.get(_duration)
            _onset_val = 1 if _onset == "" else self.onset_map.get(_onset)
            _excitation_val = 1 if _exciation == "" else self.excitation_map.get(_exciation)
            _frequency_val = 1 if _frequency == "" else self.frequency_map.get(_frequency)
            symptoms_nlice[_symptom_idx] = 1
            symptoms_nlice[_nature_idx] = _nature_val
            symptoms_nlice[_location_idx] = _location_val
            symptoms_nlice[_intensity_idx] = _intensity_val
            symptoms_nlice[_duration_idx] = _duration_val
            symptoms_nlice[_onset_idx] = _onset_val
            symptoms_nlice[_excitation_idx] = _excitation_val
            symptoms_nlice[_frequency_idx] = _frequency_val

        # ('age', 'race', 'gender', 'symptoms', 'condition')
        symptoms = np.array(symptoms_nlice)
        patient = RLPatient_Context(age,gender,symptoms, condition)
        return patient

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
    def reset(self):
        if self.line_number == self.epoch:
            self.line_number = 0
            self.load_data_file()
        line = self.get_line()
        self.patient = self.parse_line(line)
        self.state = self.generate_state()
        self.inquiry_list = set([])

        self.pick_initial_symptom()

    def pick_initial_symptom(self):
        _existing_symptoms = np.where(self.patient.symptoms == 1)[0]
        symptoms = self.find_symptom(_existing_symptoms)
        # if(len(symptoms)<=2):
        initial_symptom = np.random.choice(symptoms)
            # self.state.symptoms[initial_symptom] = np.array([0, 1, 0])
            # self.state.symptoms[initial_symptom] = 1
        # print(initial_symptom)
        symptom_nlice = []
        for i in range(8):
            symptom_nlice.append(self.patient.symptoms[int(initial_symptom)*8+i])
        self.state.symptoms[int(initial_symptom)] = np.array(symptom_nlice)
        self.inquiry_list.add(int(initial_symptom))
        # else:
        #     initial_symptom = np.random.choice(_existing_symptoms,2)
        #     # print(initial_symptom)
        #     for i in range(len(initial_symptom)):
        #         # self.state.symptoms[initial_symptom[i]] = np.array([0, 1, 0])
        #         self.state.symptoms[initial_symptom[i]] = 1
        #         self.inquiry_list.add(initial_symptom[i])
        self.status = config.DIALOGUE_STATUS_COMMING
    def find_symptom(self,index):
        symptoms=[]
        for s in index:
            if(s%8==0):
                symptoms.append(s/8)
        return symptoms

    def generate_state(self):
        _symptoms = np.zeros((self.num_symptoms,8), dtype=np.uint8)  # all symptoms start as unknown
        # _symptoms[:, 2] = 0

        return RLState_Context(self.patient.age,self.patient.gender,_symptoms)

    def is_valid_action(self, action):
        if action < self.num_symptoms:
            return True, self.is_inquiry, action  # it's an inquiry action
        else:
            action = action % (self.num_symptoms-1)
            # if action == config.ACTION_DIAGNOSE:
            return True, self.is_diagnose, action  # it's a diagnose action

        return False, None, None

    def take_action(self, action):
        self.current_Q = action
        top1 = action.max(1)[1].view(1, 1)
        is_valid, action_type, action_value = self.is_valid_action(top1.item())
        if not is_valid:
            raise ValueError("Invalid action: %s" % action)
        if action_type == self.is_inquiry:
            return self.inquire(action_value)
        else:
            return self.diagnose(action_value)

    def patient_has_symptom(self, symptom_idx):
        return self.patient.symptoms[symptom_idx*8] == 1

    def inquire(self, action_value):
        """
        returns state, reward, done
        """
        if action_value in self.inquiry_list:
            # repeated inquiry
            # return self.state, -1, True # reward is -3 if inquiry is repeated
            return self.state, self.reward_repeated, config.DIALOGUE_STATUS_INFORM_WRONG_DISEASE

        # does the patient have the symptom
        if self.patient_has_symptom(action_value):
            # value = np.array([0, 1, 0])
            value = 1
            # self.state.symptoms[action_value] = value
            self.update_nlice(action_value)
            self.inquiry_list.add(action_value)
            return self.state, self.reward_correct_inquiry,config.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM
        else:
            value = [2,0,0,0,0,0,0,0]
            self.state.symptoms[action_value] = np.array(value)
            self.inquiry_list.add(action_value)
            return self.state, self.reward_wrong_inquiry,config.DIALOGUE_STATUS_INFORM_WRONG_DISEASE

    def diagnose(self, action):
        # we'll need to make a prediction
        # if len(self.inquiry_list) < 2:
        #     return None,self.reward_repeated,config.DIALOGUE_STATUS_FAILED
        classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
        if not self.policy:
            action = classlabel[action-1]
        else:
            conditions = self.policy_transformation()
            sorted_ = np.argsort(conditions)  
            action = classlabel[sorted_[-1]]
        is_correct = self.iscorrect(self.patient.condition,action)
        # is_correct = self.patient.condition==action
        reward = self.reward_diagnose_correct if is_correct else self.reward_diagnose_incorrect
        if is_correct:
            return None,reward,config.DIALOGUE_STATUS_SUCCESS
        else:
            return None,reward,config.DIALOGUE_STATUS_FAILED
    
    
    def top_5(self):
        classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
        # index =self.current_Q.keys().item()
        # value = self.current_Q.values().item()
        result = self.current_Q.flatten().tolist()
        if not self.policy:
            conditions =[]
            for v in result:
                inx = result.index(v)
                if inx>=(self.num_symptoms):
                    conditions.append(v)
        else:
            conditions = self.policy_transformation()
        sorted_ = np.argsort(conditions)
        top_5 = []
        debug = []
        for i in range(5):
             top_5.append(classlabel[sorted_[-(i+1)]])
             debug.append(sorted_[-(i+1)])
    
        is_correct = False
        for a in top_5:
            # is_correct = self.iscorrect(self.patient.condition,classlabel[a-1])
            is_correct = self.iscorrect(self.patient.condition,a)
            if(is_correct == True):
                return None,self.reward_diagnose_correct,config.DIALOGUE_STATUS_SUCCESS
        return None,self.reward_diagnose_incorrect,config.DIALOGUE_STATUS_FAILED

    def top_3(self):
        classlabel = [24, 29, 49, 38, 5, 3, 43, 48, 41, 37, 12, 31, 9, 42, 35, 28, 8, 14, 36, 4, 40, 19, 20, 21, 39, 53, 10, 25, 0, 44, 2, 17, 45]
        # index =self.current_Q.keys().item()
        # value = self.current_Q.values().item()
        result = self.current_Q.flatten().tolist()
        if not self.policy:
            conditions =[]
            for v in result:
                inx = result.index(v)
                if inx>=(self.num_symptoms):
                    conditions.append(v)
        else:
            conditions = self.policy_transformation()
        sorted_ = np.argsort(conditions)    
        top_3 = []
        debug = []
        for i in range(3):
             top_3.append(classlabel[sorted_[-(i+1)]])
             debug.append(sorted_[-(i+1)])
    
        is_correct = False
        for a in top_3:
            # is_correct = self.iscorrect(self.patient.condition,classlabel[a-1])
            is_correct = self.iscorrect(self.patient.condition,a)
            if(is_correct == True):
                return None,self.reward_diagnose_correct,config.DIALOGUE_STATUS_SUCCESS
        return None,self.reward_diagnose_incorrect,config.DIALOGUE_STATUS_FAILED

    def policy_transformation(self):
        result = self.current_Q.flatten().tolist()
        conditions = []
        for v in result:
            inx = result.index(v)
            if inx>=(self.num_symptoms):
                conditions.append(v)
        sorted_ = np.argsort(conditions)
        for i in sorted_:
            co = self.conditions_reverse.get(classlabel[i])
            pro_age = get_pro_age(co,self.patient.age,self.piors_map)/100
            pro_gender=get_pro_sex(co,self.patient.gender,self.piors_map)/100
            incidence = get_pro_incidence(co,self.piors_map)
            # conditions[i] = conditions[i]*pro_age*pro_gender+incidence
            conditions[i] = conditions[i]*pro_age*pro_gender
        return conditions

    def iscorrect(self,condition,action):
        if condition in [48,49] and action in [48,49]:
                return True
        elif condition in [37,38,39] and action in [37,38,39]:
                return True
        elif condition in [40,41] and action in [40,41]:
                return True
        # elif condition in [28,44,45] and action in [28,44,45]:
        #         return True
        elif condition == action:
            return True
        else:
            return False

    def update_nlice(self,action_value):
        symptom_nlice=[]
        for i in range(8):
            symptom_nlice = self.patient.symptoms[action_value*8+i]
        self.state.symptoms[action_value] = np.array(symptom_nlice)

    def __del__(self):
        self.close_data_file()
