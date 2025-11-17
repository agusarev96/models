import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(""))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.data_utils.dataset import DataSet


class LinearRegressionModel:
    # нужно реализовать линейную регрессию:
    # - без регуляризации
    # - с L1
    # - с L2
    # - с разными лоссами
    # - с Elastic Net (Loss = MSE + α * [(1 - ρ) * L2_penalty + ρ * L1_penalty])
    
    def __init__(self):
        pass
    
    def train(self):
        pass
    
    def predict(self):
        pass
    