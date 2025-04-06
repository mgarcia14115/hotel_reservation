from models import baseline
from configs.default import DEFAULTS 

def init_model(model_name):

    if model_name == "baseline":
        model = baseline.MGModel(DEFAULTS['input_dims'])
        return model
        
    