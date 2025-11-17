from typing import List, Union

import pandas as pd
import numpy as np


class DataSet:
    
    def __init__(self, data:Union[pd.DataFrame, np.ndarray], feature_names:List=[], target_column:Union[str, int]=None):
        if (type(data) == np.ndarray) and (len(feature_names) == 0):
            raise ValueError("If you pass numpy array you need to specify `feature_names`")
        
        if (type(data) == np.ndarray) and (len(data.shape) != 2):
            raise ValueError("Data need to be matrix M*N")
        
        if len(feature_names) != 0:
            if type(data) == pd.DataFrame:
                if len(feature_names) != len(data.columns):
                    raise ValueError("length of feature_names doesn't match length of df columns")
            else:
                if len(feature_names) != data.shape[1]:
                    raise ValueError("length of feature_names doesn't match length of data.shape[1]")
        
        self.feature_names = feature_names if feature_names else list(data.columns)
        self.data = data.values if type(data) == pd.DataFrame else data
        
        if target_column and (type(data) == pd.DataFrame):
            self.target = data[target_column].values
        elif target_column and (type(data)==np.ndarray):
            self.target = data[:, target_column]
        
    def __str__(self) -> str:
        return (
            f"columns: {self.feature_names}\n"
            f"shape: {self.data.shape}"
        )
        
    def __repr__(self) -> str:
        return (
            f"columns: {self.feature_names}\n"
            f"shape: {self.data.shape}"
        )