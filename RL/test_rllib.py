import rllib as rl
import json
import joblib

dataset = "./data/basic/data/test.csv"
symptom_map_file = "./data/basic/symptoms_db.json"
condition_map_file = "./data/basic/conditions_db.json"
clf_file = "./data/basic/data/ouput/rf/rf_serialized_sparse.joblib"
clf_data = joblib.load(clf_file)
clf = clf_data.get('clf')


env = rl.environment.AiBasicMedEnv(
    data_file=dataset,
    symptom_map_file=symptom_map_file,
    condition_map_file=condition_map_file,
    clf=clf
)
learning_start = 1280
batch_size = 128
target_update = 1280
replay_capacity = 1280
input_dim = 3 + 3*env.num_symptoms
output_dim = env.num_symptoms + env.num_conditions
layer_config = [
    [input_dim, 128],
    [128, 64],
    [64, 48],
    [48, 32],
    [32, 32],
    [32, 48],
    [48, 64],
    [64, output_dim]
]

learning_start = 1280
batch_size = 128
target_update = 1280
replay_capacity = 1280
agent = rl.agent.MedAgent(
    env,
    layer_config=layer_config,
    learning_start=learning_start,
    batch_size=batch_size,
    target_update=target_update,
    replay_capacity=replay_capacity,
    debug=False
)
bench = rl.bench.MedBench(agent, num_episodes=200)
bench.run_trial(debug=False)