## Description
Here are scripts used for thesis work [Automated Medical History Taking in Primary Care](https://repository.tudelft.nl/islandora/object/uuid%3Ab078d6b0-4cf2-4d1a-9221-9c0d571db5aa)

### useage
the running script run.py could be found in ./run/ folder.

for example:

To train HRL models:
```bash
 ./run/run.py --train true --file_all <path to all data> --nlice true --conditions <path to parsed conditions> --symptom <path to parsed symptoms> --dataset <path_to_patients_record> --context true
```

there are a few parameters needed for running this script:

**dataset**:

This dataset is generated by the same pipeline as [pipeline](https://github.com/Medvice/synthea-symcat-pipeline). You can custimize the dataset size. And all the NLICE feature encoding files are also be generated by the same manner as [enoding](https://github.com/Medvice/stanley-thesis-notebooks/tree/master/06_18_nlice_plus/prep-for-nlice-rl).After generating all those NLICE encoding files, you can set **excitation**,**frequency**,**vas**,**nature**,**duration**,**body**,etc parameters to your encoding file path.

**nlice**:
You can set this flag as true if you want to let agent ask patients NLICE questions and train models.

**piors**:
We need pior probabilities of race,age,and gender to perform context policy transformation. This parameter points to the extracted age,race,gender probability of each symptom.

**context**:
This flag should be set to true if want to perform policy transformation.

for other training configuration you can define as you need.

The following parameters state reward configuation:

**reward_for_not_come_yet**:
When no inquiry, the reward is set to 0.

**reward_success**
When successfully make correct disease prediction, the agent gets reward **reward_success**. And you can define your own reward value to set **reward_fail**,**reward_right**,**reward_wrong**,**reach_max_turn** and **reward_for_repeated_action**.

**max_turn**
the max number of interaction loops.

**nums_main_complaint**
we define that at the begining of diaglogue, the patient will tell the doctior his/her main symptom as main complaint. **nums_main_complaint** can define how many symptoms as main complaint.




