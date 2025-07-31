import sys, os
from pathlib import Path
project_path = Path(__file__).parents[3]
sys.path.insert(0, str(project_path))

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm
import random
import networkx as nx


import causrca.rca_models.common as rcacom
import causrca.utils.utils as utils
from causrca.utils.discretize import transform_non_continuous_values_in_df


class UnsupervisedRCAModel(rcacom.RCAModel):
    """Base class for unsupervised RCA Models.
    
    Take as input a dataset and predictes the most likely faulty variable(s) based on the data."""
    
    def __init__(self, **kwargs):
        """Initialize the Unsupervised RCA_Model"""
        super().__init__(**kwargs)

    @abstractmethod
    def predict(self, dataset_csv_path: str, diagnosis_time: float, relevant_nodes: list = None) -> list:
        """Predicts the most likely faulty variable(s) based on the dataset and diagnosis time.
        
        :param dataset_csv_path: Path to the dataset CSV file.
        :type dataset_csv_path: str
        :param diagnosis_time: The time at which the diagnosis is made.
        :type diagnosis_time: float
        :param relevant_nodes: List of relevant nodes to consider for prediction.
        :type relevant_nodes: list, optional
        
        :return: An ordered list of the most likely variable(s).
        :rtype: list
        """
        pass


class TimeRecency_BaselineUnsupervisedRCA(UnsupervisedRCAModel):
    """Baseline Unsupervised RCA Model using time recency.
    
    This model takes the last variable changes before the last occurred alarm
    and predicts them from most recent to least recent."""
    
    def __init__(self, **kwargs):
        """Initialize the Time Recency RCA Model"""
        super().__init__(**kwargs)

    def predict(self, dataset_csv_path: str, diagnosis_time: float, relevant_nodes: list = None) -> list:
        """Returns an ordered list of the most recent variable changes before active alarms, interleaved.
        
        :param dataset_csv_path: Path to the dataset CSV file.
        :type dataset_csv_path: str
        :param diagnosis_time: The time at which the diagnosis is made.
        :type diagnosis_time: float
        :param relevant_nodes: List of relevant nodes to consider for prediction.
        :type relevant_nodes: list, optional
        
        :return: An ordered list of the most likely variable(s).
        :rtype: list        
        """
        # Load dataset and limit to provided relevant variables and last known state at the diagnosis time
        dataset_df = utils.load_dataset_csv_to_df(dataset_csv_path, limit_nodes_to=relevant_nodes)
        last_state = rcacom.limit_dataset_to_diagnosis_state(dataset_df, diagnosis_time)
        
        # Sort by time descending (highest on top) 
        sorted_state = last_state.sort_values('time_s', ascending=False)
        
        # Get all alarms (type=Alarm)
        alarm_rows = sorted_state[sorted_state['type'] == 'Alarm']
        
        if alarm_rows.empty:
            # If no alarms are present, simply return nodes in time order
            prediction = sorted_state[sorted_state['type'] != 'Alarm']['node'].tolist()
        else:
            # Sort alarms by time (most recent first)
            sorted_alarms = alarm_rows.sort_values('time_s', ascending=False)
            
            # For each alarm, find variable changes before it
            alarm_changes_lists = []
            for _, alarm_row in sorted_alarms.iterrows():
                alarm_time = alarm_row['time_s']
                
                # Find all variable changes before this alarm, sorted by time (most recent first)
                changes_before_alarm = sorted_state[(sorted_state['type'] != 'Alarm') & 
                                                (sorted_state['time_s'] < alarm_time)]
                changes_before_alarm = changes_before_alarm.sort_values('time_s', ascending=False)
                
                # Store the list of nodes that changed before this alarm
                alarm_changes_lists.append(changes_before_alarm['node'].tolist())
            
            # Sort as most recent before each alarm, second most recent before each alarm etc.
            prediction = []
            added_nodes = set()  # Track already added nodes to avoid duplicates
            
            max_length = max(len(changes) for changes in alarm_changes_lists) if alarm_changes_lists else 0
            
            for pos in range(max_length):
                for alarm_idx in range(len(alarm_changes_lists)):
                    if pos < len(alarm_changes_lists[alarm_idx]):
                        node = alarm_changes_lists[alarm_idx][pos]
                        if node not in added_nodes:
                            prediction.append(node)
                            added_nodes.add(node)
            
        return prediction


class CausalPrioTimeRecencyRCA(UnsupervisedRCAModel):
    """Causal-prioritized Time Recency RCA model.
    
    This model wraps the TimeRecency_BaselineUnsupervisedRCA model and uses
    causal structure to filter nodes to only include the alarm and its direct
    causal predecessors from the causal graph.
    """
    
    def __init__(self, causal_graph: nx.DiGraph = None, **kwargs):
        """Initialize the Causal Prioritized Time Recency RCA model.
        
        :param causal_graph: A directed graph representing causal relationships.
        :type causal_graph: nx.DiGraph
        """
        super().__init__(**kwargs)
        self.causal_graph = causal_graph
        self.time_recency_model = TimeRecency_BaselineUnsupervisedRCA(**kwargs)
    
    def __get_relevant_nodes_by_causal_structure(self, last_state: pd.DataFrame) -> list:
        """Determines the relevant nodes for all active alarms using the causal graph.
        Relevant nodes include all active alarms and their direct predecessors in the causal graph.
        
        :param last_state: The last state of the system with alarm information.
        :type last_state: pd.DataFrame
        
        :return: A list of relevant nodes (all alarms and their direct predecessors).
        :rtype: list
        """
        # Ensure that the causal graph is provided
        if self.causal_graph is None:
            raise ValueError("Causal graph not provided during initialization")
        
        # Get active alarms (type=Alarm and value=True)
        alarm_rows = last_state[(last_state['type'] == 'Alarm') & (last_state['value'] == 'True')]
        
        if alarm_rows.empty:
            # If no active alarms, return all nodes
            return last_state['node'].unique().tolist()
        
        # Filter for alarms that are actually in the causal graph
        causal_alarm_rows = alarm_rows[alarm_rows['node'].isin(self.causal_graph.nodes())]
    
        # If no active alarm is found within the causal graph, raise an error
        if causal_alarm_rows.empty:
            raise ValueError("No active alarm found in the causal graph.")
    
        # Get all active alarm nodes
        alarm_nodes = causal_alarm_rows['node'].tolist()
        
        # Find all direct parents (predecessors) of all alarms
        relevant_nodes = set(alarm_nodes)  # Start with alarm nodes
        for alarm_node in alarm_nodes:
            # Add all direct parents of this alarm
            parents = list(self.causal_graph.predecessors(alarm_node))
            relevant_nodes.update(parents)
        
        return list(relevant_nodes)

    def predict(self, dataset_csv_path: str, diagnosis_time: float, relevant_nodes: list = None) -> list:
        """Returns an ordered list of variables based on causal structure and temporal recency.
        
        :param dataset_csv_path: Path to the dataset CSV file.
        :type dataset_csv_path: str
        :param diagnosis_time: The time at which the diagnosis is made.
        :type diagnosis_time: float
        :param relevant_nodes: List of additional relevant nodes to consider for prediction.
        :type relevant_nodes: list, optional
        
        :return: An ordered list of the most likely variable(s).
        :rtype: list
        """
        # Load dataset 
        dataset_df = utils.load_dataset_csv_to_df(dataset_csv_path)
        last_state = rcacom.limit_dataset_to_diagnosis_state(dataset_df, diagnosis_time)
        
        try:
            # Get causally relevant nodes
            causal_nodes = self.__get_relevant_nodes_by_causal_structure(last_state)
            
            # Combine with user-provided relevant nodes if any
            if relevant_nodes:
                filtered_nodes = list(set(relevant_nodes).intersection(set(causal_nodes)))
            else:
                filtered_nodes = causal_nodes
                
            # Build Result with three predictions: 1) With filtered nodes (causal + relevant), 2) With all causal nodes and 3) With all relevant nodes
            prio1 = self.time_recency_model.predict(dataset_csv_path, diagnosis_time, filtered_nodes)
            prio2 = self.time_recency_model.predict(dataset_csv_path, diagnosis_time, causal_nodes)
            prio3 = self.time_recency_model.predict(dataset_csv_path, diagnosis_time, relevant_nodes)

            # Combine predictions, preserving order of first appearance and removing duplicates
            combined_prediction = []
            seen = set()
            for node in prio1 + prio2 + prio3:
                if node not in seen:
                    combined_prediction.append(node)
                    seen.add(node)
                        
            return combined_prediction
            
        except ValueError as e:
            # If there's an error with causal filtering, fall back to the base model
            print(f"Warning: {str(e)}. Falling back to standard TimeRecency prediction.")
            return self.time_recency_model.predict(dataset_csv_path, diagnosis_time, relevant_nodes)