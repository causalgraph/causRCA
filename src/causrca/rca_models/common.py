import sys
from pathlib import Path
project_path = Path(__file__).parents[3]
sys.path.insert(0, str(project_path))

import pandas as pd
from abc import ABC, abstractmethod


class RCAModel(ABC):
    """Base class for RCA Models."""
    
    def __init__(self, **kwargs):
        """Initialize the RCA_Model"""
        self.params = kwargs
    
    @abstractmethod
    def predict(self, dataset_csv_path: str, diagnosis_time: float, args) -> list:
        """Returns an ordered list of the most likely causes"""
        pass


###### Common Helper Functions for RCA Models ######

def limit_dataset_to_diagnosis_state(dataset_df: pd.DataFrame, diagnosis_time: float) -> pd.DataFrame:
    """
    Filters the dataset DataFrame to only include the last known value for each node before the specified diagnosis time.

    :param dataset_df: The dataset DataFrame containing columns 'time_s', 'node', and 'value'.
    :type dataset_df: pd.DataFrame
    :param diagnosis_time: The time at which to retrieve the last known state.
    :type diagnosis_time: float
    
    :return: Filtered DataFrame containing the last known values for each node before the diagnosis time.
    :rtype: pd.DataFrame
    """
    # Filter the dataset to include only rows with time less than or equal to the specified time
    filtered_df = dataset_df[dataset_df['time_s'] <= diagnosis_time]
    
    # Group by 'node' and get the last known value for each node
    last_known_state = filtered_df.groupby('node').last().reset_index()
    
    return last_known_state