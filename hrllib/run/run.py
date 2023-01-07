import sys, os
sys.path.append(os.getcwd().replace("hrllib/run",""))
from hrllib.agent import AgentHRL,AgentDQN
from hrllib.run.runner import RunningSteward
import time
import argparse
import pickle
import sys, os
import random
import json
import torch
import datetime

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    if s.lower() == 'true':
        return True
    else:
        return False
parser = argparse.ArgumentParser()

#simulation configuration
parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default = 1, help="The number of simulate epoch.")
parser.add_argument("--simulation_size", dest="simulation_size", type=int, default=5500, help="The number of simulated sessions in each simulated epoch.")
parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=10000, help="the size of experience replay.")
parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=512, help="the hidden_size of DQN.")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=16, help="the batch size when training.")
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1, help="The greedy probability of DQN")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.95, help="The discount factor of immediate reward in RL.")
parser.add_argument("--gamma_worker", dest="gamma_worker", type=float, default=0.9, help="The discount factor of immediate reward of the lower agent in HRL.")
parser.add_argument("--train_mode", dest="train_mode", type=boolean_string, default=False, help="Runing this code in training mode? [True, False]")
# parser.add_argument("--case_study",dest="case_study", type=boolean_string, default=False, help="Runing the trained model for case study [True, False]")

parser.add_argument("--weight_correction", dest="weight_correction", type=boolean_string, default=False, help="weight corrention for the master agent in HRL? {True, False}")
#dataset
file0 = "./data_v2/group/"
parser.add_argument('--file_all', dest="file_all", type=str, default=file0, help='the path for ten groups of diseases')

parser.add_argument('--nlice',dest="nlice",type = boolean_string,default=True)
parser.add_argument('--body',dest = "body",type = str,default ="./data_v2/body-parts-enc.json")
parser.add_argument('--body-main',dest = "body-main",type = str,default ="./data_v2/body_main_parts-enc.json")
parser.add_argument('--excitation',dest = "excitation",type = str,default ="./data_v2/excitation_encoding.json")
parser.add_argument('--frequency',dest = "frequency",type = str,default ="./data_v2/frequency_encoding.json")
parser.add_argument('--nature',dest = "nature",type = str,default ="./data_v2/nature_encoding.json")
parser.add_argument('--vas',dest = "vas",type = str,default ="./data_v2/vas_encoding.json")
parser.add_argument('--onset',dest = "onset",type = str,default ="./data_v2/onset_encoding.json")
parser.add_argument('--duration',dest = "duration",type = str,default ="./data_v2/duration_encoding.json")
parser.add_argument('--conditions',dest="conditions",type=str,default="./data_v2/condition.json")
parser.add_argument('--symptoms',dest="symptoms",type=str,default="./data_v2/symptom.json")

parser.add_argument('--test',dest="test",type=str,default="./data_v2/testcases.csv")
parser.add_argument('--dataset',dest="dataset",type=str,default="./data_v2/total_1k.csv")
max_turn =22
parser.add_argument('--piors',dest="piors",type=str,default="./data_v2/all_piors.json")
parser.add_argument('--context',dest="context",type=boolean_string, default=False)
#save model

nowTime=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
output = "./hrloutput/"+nowTime
if not os.path.exists(output):
    os.makedirs(output)
saved_model1 = './hrloutput/model1/'+'20220418214231'
parser.add_argument('--checkpoint_path',dest="checkpoint_path",type=str,default = output+"/")
parser.add_argument('--saved_model',dest="saved_model",type=str,default = saved_model1+"/")
# reward design
parser.add_argument("--reward_for_not_come_yet", dest="reward_for_not_come_yet", type=float,default=0.0)
parser.add_argument("--reward_success", dest="reward_success", type=float,default= 1.0)
parser.add_argument("--reward_fail", dest="reward_fail", type=float,default=-1.0)
parser.add_argument("--reward_right", dest="reward_right", type=float,default=1.0)
parser.add_argument("--reward_wrong", dest="reward_wrong", type=float,default=0.0)
parser.add_argument("--minus_left_slots", dest="minus_left_slots", type=boolean_string, default=False,help="Success reward minus the number of left slots as the final reward for a successful session.{True, False}")
parser.add_argument("--reach_max_turn", dest="reach_max_turn", type=float, default=-1.0)
parser.add_argument("--reward_for_repeated_action", dest='reward_for_repeated_action', type=float, default= -1.0, help='the reward for repeated action')
parser.add_argument("--weight_for_reward_shaping", dest='weight_for_reward_shaping', type=float, default=0.0, help="weight for reward shaping. 0 means no reward shaping.")
parser.add_argument("--max_turn",dest="max_turn",type=int,default=max_turn)
parser.add_argument("--disease_number",dest="disease_number",type=int,default=55)
parser.add_argument("--exp_str",dest="exp_str",type=str, default="55conditons+nlice+rewared+1mainconplaints")
parser.add_argument("--nums_main_complaint",dest="nums_main_complaint",type=int,default=1)
args = parser.parse_args()
parameter = vars(args)


def run(parameter):
    print(json.dumps(parameter, indent=2))
    time.sleep(2)
    dataset = parameter.get("dataset")
    steward = RunningSteward(parameter=parameter)
    train_mode = parameter.get("train_mode")
    simulate_epoch_number = parameter.get("simulate_epoch_number")
    steward.dialogue_manager.state_tracker.set_agent(steward.agent)
    if train_mode is True:
        steward.simulate(epoch_number=simulate_epoch_number,train_mode=train_mode)
    else:
        for index in range(simulate_epoch_number):
            steward.evaluation_model()

if __name__ == "__main__":
    torch.cuda.manual_seed(12345)
    torch.manual_seed(12345)
    run(parameter=parameter)
    
