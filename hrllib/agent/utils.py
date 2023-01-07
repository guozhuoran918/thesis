
import numpy as np
import copy
import sys, os
import json
import traceback
from hrllib import configuration
# todo nlice-state
def load_db(file):
        with open(file) as fp:
            dbs = json.load(fp)
        return dbs
def state2rep(state, slot_set, parameter):
    """
    Mapping dialogue state, which contains the history utterances and informed/requested slots up to this turn, into
    vector so that it can be fed into the model.
    This mapping function uses informed/requested slots that user has informed and requested up to this turn .
    :param state: Dialogue state
    :return: Dialogue state representation with 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-rank, which is a vector representing dialogue state.
    """
    ######################
    # Current_slots rep.
    #####################
    # try:
    #     slot_set.pop("disease")
    # except:
    #     pass

    current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
    current_slots.update(state["current_slots"]["explicit_inform_slots"])
    current_slots.update(state["current_slots"]["implicit_inform_slots"])
    current_slots.update(state["current_slots"]["proposed_slots"])
    current_slots.update(state["current_slots"]["agent_request_slots"])  # request slot is represented in the following part.


    excitation = load_db(parameter.get("excitation"))
    frequency = load_db(parameter.get("frequency"))
    nature = load_db(parameter.get("nature"))
    vas = load_db(parameter.get("vas"))
    onset = load_db(parameter.get("onset"))
    duration = load_db(parameter.get("duration"))
    body = load_db(parameter.get("body"))
    main_body = load_db(parameter.get("body-main"))
    # one hot
  
    if(parameter.get("nlice")):
        current_slots_rep = np.zeros((len(slot_set.keys()),9))
        for slot in slot_set:
                if slot in current_slots.keys():
                    # if current_slots[slot] == False:
                    #     temp_slot = [-1,-1,-1,-1,-1,-1,-1,-1]
                    if isinstance(current_slots[slot],dict):
                        nlice = current_slots[slot]
                        _nature_val = 1 if nlice["nature"] == "" or nlice["nature"] =="other" else nature.get(nlice["nature"])
                        _location_main_val = 1 if nlice["location_main"] or nlice["location_main"] == "other" else main_body.get(nlice["location_main"])
                        _location_val = 1 if nlice["location"] == "" or nlice["location"] == "other" else body.get(nlice["location"])
                        _intensity_val = 1 if nlice["intensity"] == "" else vas.get(nlice["intensity"])
                        _duration_val = 1 if nlice["duration"] == "" else duration.get(nlice["duration"])
                        _onset_val = 1 if nlice["onset"] == "" else onset.get(nlice["onset"])
                        _excitation_val = 1 if nlice["exciation"] == "" else excitation.get(nlice["exciation"])
                        _frequency_val = 1 if nlice["frequency"] == "" else frequency.get(nlice["frequency"])               
                        temp_slot = [1,_nature_val,_location_main_val,_location_val,_intensity_val,_duration_val,_onset_val,_excitation_val,_frequency_val]
                    else:
                        temp_slot = [-1,-1,-1,-1,-1,-1,-1,-1,-1]

                else:
                    temp_slot = [0,0,0,0,0,0,0,0,1]
                current_slots_rep[slot_set[slot], :] = temp_slot
        state_rep = current_slots_rep.reshape(1,len(slot_set.keys())*9)[0]
            #print( temp_slot)
    else:
        current_slots_rep = np.zeros((len(slot_set.keys()),3))
        for slot in slot_set:
                if slot in current_slots.keys():
                    # if current_slots[slot] == False:
                    #     temp_slot = [-1,-1,-1,-1,-1,-1,-1,-1]
                    if isinstance(current_slots[slot],dict):                              
                        temp_slot = [1,0,0]
                    else:
                        temp_slot = [1,0,0]
                else:
                    temp_slot = [0,0,1]
                current_slots_rep[slot_set[slot], :] = temp_slot
        state_rep = current_slots_rep.reshape(1,len(slot_set.keys())*3)[0]

    return state_rep