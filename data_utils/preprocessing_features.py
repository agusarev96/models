from typing import Union

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import train_test_split
import numpy as np

class FillNa:

    def __init__(self):
        """
        Returns class example with fit and transform interface
        that allows to fill nan values with <val>

        Args:
        """

    def fit(self, val:Union[str, int, float]):
        self.val = val
        return self
    
    def transform(self, arr:np.array):
        return np.where(np.isnan(arr), self.val, arr)

class Mice(IterativeImputer):

    
    def __init__(self, *args, **kwargs):
        """
        Returns class example with fit and transform interface
        that allows to predict missing values based on other features
        """
        super().__init__(*args, **kwargs)
        
class KNNImpute(KNNImputer):
    
    def __init__(self, *args, **kwargs):
        """
        Returns class example with fit and transform interface
        that allows to predict missing values based on other features
        """
        super().__init__(*args, **kwargs)
        
class RegularizedTargetEncoder:
    
    def __init__(self, cols, smoothing=10.0, kfolds=5, random_state=42):
        self.cols = cols
        self.smoothing = smoothing
        self.kfolds = kfolds
        self.random_state = random_state

    def fit(self, array, target):
        self.global_target_mean = target.mean()
        
        train_idx, _ = train_test_split(np.arange(0, array.shape[0]), test_size=.5, random_state=self.random_state)
        array = array[train_idx]
        target = array[train_idx]
            
        self.map_dict = {}
        for val in np.unique(array):
            idx = np.where(array==val)[0]
            target_mean = target[idx].mean()
            target_count = idx.shape[0]
            
            smooth_mean = (target_count * target_mean + 
                        10.0 * self.global_target_mean) / (target_count + 10.0)
            
            self.map_dict[val] = smooth_mean
    
    def transform(self, array):
        result = np.array(list(map(lambda x: self.map_dict[x], array)))
        return np.where(result.isnan(), self.global_target_mean, result)