
import pathlib
import json
import os
import numpy as np
from thesislib.utils.ml import runners, models
import pandas as pd
import scipy.sparse as sparse

RACE_CODE = {'white': 0, 'black':1, 'asian':2, 'native':3, 'other':4}
def _symptom_transform(val, labels, is_nlice=False):
    """
    Val is a string in the form: "symptom_0;symptom_1;...;symptom_n"
    :param val:
    :param labels:
    :return:
    """
    parts = val.split(";")
    if is_nlice:
        indices = []
        for item in parts:
            id, enc = item.split("|")
            label = labels.get(id)
            indices.append("|".join([label, enc]))
        res = ",".join(indices)
    else:
        indices = []
        for item in parts:
            symptom,_,_,_,_,_,_,_,_ = item.split(":")
            id = labels.get(symptom)
            if _ is None:
                raise ValueError("Unknown symptom")
            indices.append(id)
        res = ",".join(indices)
    return res


def parse_data(
        filepath,
        conditions_db_json,
        symptoms_db_json,
        output_path,
        is_nlice=False,
        transform_map=None,
        encode_map=None,
        reduce_map=None):

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(symptoms_db_json) as fp:
        symptoms_db = json.load(fp)

    with open(conditions_db_json) as fp:
        conditions_db = json.load(fp)

    condition_labels = {code: idx for idx, code in enumerate(sorted(conditions_db.keys()))}
    symptom_map = {code: str(idx) for idx, code in enumerate(sorted(symptoms_db.keys()))}

    usecols = ['GENDER', 'RACE', 'AGE_BEGIN', 'PATHOLOGY', 'NUM_SYMPTOMS', 'SYMPTOMS']

    df = pd.read_csv(filepath, usecols=usecols)

    filename = filepath.split("/")[-1]

    # drop the guys that have no symptoms
    df = df[df.NUM_SYMPTOMS > 0]
    df['LABEL'] = df.PATHOLOGY.apply(lambda v: condition_labels.get(v))
    df['RACE'] = df.RACE.apply(lambda v: RACE_CODE.get(v))
    df['GENDER'] = df.GENDER.apply(lambda gender: 0 if gender == 'F' else 1)
    df = df.rename(columns={'AGE_BEGIN': 'AGE'})
    # if is_nlice:
    #     df['SYMPTOMS'] = df.SYMPTOMS.apply(
    #         _tranform_symptoms,
    #         transformation_map=transform_map,
    #         symptom_combination_encoding_map=encode_map,
    #         reduction_map=reduce_map)
    df['SYMPTOMS'] = df.SYMPTOMS.apply(_symptom_transform, labels=symptom_map, is_nlice=is_nlice)
    ordered_keys = ['LABEL', 'GENDER', 'RACE', 'AGE', 'SYMPTOMS']
    df = df[ordered_keys]
    df.index.name = "Index"

    output_file = os.path.join(output_path, "%s_sparse.csv" % filename)
    df.to_csv(output_file)

    return output_file

def main():
    data_dir = os.curdir
    print(data_dir)
    op_data_dir = os.path.join(data_dir, "symtom_models")
    rf_dir = os.path.join(op_data_dir, "output/rf")
    rfparams = models.RFParams()
    rfparams.n_estimators = 200
    rfparams.max_depth = None
    parsed_train = "./data/basic/data/parsed/test.csv"
    symptom_map_file = "./data/basic/symptoms_db.json"
    condition_map_file = "./data/basic/conditions_db.json"
    pathlib.Path(rf_dir).mkdir(parents=True, exist_ok=True)
    nb_dir = os.path.join(op_data_dir, "output/nb")
    # # run_ok = runners.train_ai_med_nb(
    # #     parsed_train,
    # #     symptom_map_file,
    # #     nb_dir
    # # )
    # print(run_ok)
   
    # data = np.ones(len(rows))
    # symptoms_coo = sparse.coo_matrix((data, (rows, columns)), shape=(df.shape[0],self.num_symptoms))

    # data_coo = sparse.hstack([dense_matrix, symptoms_coo])

    # data_coo.tocsc()

    with open(symptom_map_file) as fp:
            symptoms_db = json.load(fp)
            num_symptoms = len(symptoms_db)
    data = pd.read_csv(parsed_train)
    classes = data.LABEL.unique().tolist()
    label_values = data.LABEL.values
    ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
    data = data[ordered_keys]
    sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
    data = sparsifier.fit_transform(data)

if __name__ == "__main__":
    main()
