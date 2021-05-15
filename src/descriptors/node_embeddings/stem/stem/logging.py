from pony.orm import *
import json

db = Database()
output_file = 'experiments.sqlite'
db.bind(provider='sqlite', filename=output_file, create_db=True)

class SignPrediction(db.Entity):
    method = Required(str)
    dataset = Required(str)
    auc = Required(float)
    f1 = Required(float)
    hyperparameters = Required(str)


db.generate_mapping(create_tables=True)

@db_session
def write_sign_prediction(method, dataset, auc, f1, hyperparmaters):
    h = json.dumps(hyperparmaters)
    sign = SignPrediction(method=method, dataset=dataset, auc=auc, f1=f1, hyperparmaters=h)
