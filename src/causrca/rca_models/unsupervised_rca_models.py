import sys
from pathlib import Path
project_path = Path(__file__).parents[3]
sys.path.insert(0, str(project_path))

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import networkx as nx
from sknetwork.ranking import PageRank
from sklearn.preprocessing import RobustScaler


import causrca.rca_models.common as rcacom
from causrca.rca_models.external.random_walk import random_walk
import causrca.utils.utils as utils
from causrca.utils.discretize import transform_non_continuous_values_in_df, discretize_dataset

CATEGORICAL_ENCODING_DICT = utils.get_encoding_dict()


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
        

class PageRankRCA(UnsupervisedRCAModel):
    """PageRank-based Unsupervised RCA model.
    
    This model uses the standard PageRank algorithm to rank nodes in the causal graph.
    Implementation adapted from pagerank usage in Paper "Root Cause Analysis for
    Microservices based on Causal Inference: How Far Are We?" by Pham et al.
    """
    
    def __init__(self, causal_graph: nx.DiGraph = None, **kwargs):
        """Initialize the PageRank RCA model.
        
        :param causal_graph: A directed graph representing causal relationships.
        :type causal_graph: nx.DiGraph
        """
        super().__init__(**kwargs)
        self.causal_graph = causal_graph
    
    def predict(self, dataset_csv_path: str, diagnosis_time: float = None, relevant_nodes: list = None) -> list:
        """Returns an ordered list of variables based on their PageRank scores in the causal graph.
        Note: This method ignores diagnosis_time as it focuses only on the static causal structure.
        
        :param dataset_csv_path: Path to the dataset CSV file.
        :type dataset_csv_path: str
        :param diagnosis_time: The time at which the diagnosis is made (not used in this model).
        :type diagnosis_time: float, optional
        :param relevant_nodes: List of relevant nodes to consider for prediction.
        :type relevant_nodes: list, optional
        
        :return: An ordered list of nodes ranked by PageRank scores.
        :rtype: list
        """
        # Ensure causal graph is provided
        if self.causal_graph is None:
            raise ValueError("Causal graph not provided during initialization")

        # Load dataset to get node names that appear in the dataset
        dataset_df = utils.load_dataset_csv_to_df(dataset_csv_path, limit_nodes_to=relevant_nodes)
        dataset_nodes = dataset_df['node'].unique().tolist()
        
        try:
            # Create a subgraph with only nodes that are in both the causal graph and dataset
            nodes_in_graph = set(self.causal_graph.nodes())
            common_nodes = list(set(dataset_nodes).intersection(nodes_in_graph))
            
            if not common_nodes:
                raise ValueError("No common nodes between causal graph and dataset")
            
            # Create subgraph with only common nodes
            subgraph = self.causal_graph.subgraph(common_nodes)
            
            # Convert graph to adjacency matrix for PageRank
            nodes = sorted(subgraph.nodes())
            adj = nx.to_numpy_array(subgraph, nodelist=nodes)
            
            # Apply PageRank algorithm
            pagerank = PageRank()
            scores = pagerank.fit_predict(adj.T)
            
            # Sort nodes by PageRank score
            ranks = list(zip(nodes, scores))
            ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
            
            # Exclude alarm nodes from the predictions (as in TimeRecency)
            non_alarm_nodes = dataset_df[dataset_df['type'] != 'Alarm']['node'].unique().tolist()
            filtered_ranks = [node for node, _ in ranks if node in non_alarm_nodes]
            
            # Return ordered node names excluding alarms
            return filtered_ranks
            
        except ValueError as e:
            print(f"Warning: {str(e)}. Returning nodes in alphabetical order.")
            return sorted(dataset_nodes)


class RandomWalkRCA(UnsupervisedRCAModel):
    """Random Walk-based Unsupervised RCA model.
    
    This model uses a random walk algorithm to rank nodes in the causal graph.
    Implementation based on the random walk approach described in "Root Cause Analysis for 
    Microservices based on Causal Inference: How Far Are We?" by Pham et al.
    """
    
    def __init__(self, causal_graph: nx.DiGraph = None, num_loop: int = None, random_seed: int = 42, **kwargs):
        """Initialize the Random Walk RCA model.
        
        :param causal_graph: A directed graph representing causal relationships.
        :type causal_graph: nx.DiGraph
        :param num_loop: Number of random walk steps (default: 10 * number of nodes).
        :type num_loop: int, optional
        :param random_seed: Random seed for reproducibility.
        :type random_seed: int
        """
        super().__init__(**kwargs)
        self.causal_graph = causal_graph
        self.num_loop = num_loop
        self.random_seed = random_seed
    
    def predict(self, dataset_csv_path: str, diagnosis_time: float = None, relevant_nodes: list = None) -> list:
        """Returns an ordered list of variables based on their random walk scores in the causal graph.
        
        :param dataset_csv_path: Path to the dataset CSV file.
        :type dataset_csv_path: str
        :param diagnosis_time: The time at which the diagnosis is made (not used in this model).
        :type diagnosis_time: float, optional
        :param relevant_nodes: List of relevant nodes to consider for prediction.
        :type relevant_nodes: list, optional
        
        :return: An ordered list of nodes ranked by random walk scores.
        :rtype: list
        """
        # Ensure causal graph is provided
        if self.causal_graph is None:
            raise ValueError("Causal graph not provided during initialization")

        # Load dataset to get node names that appear in the dataset
        dataset_df = utils.load_dataset_csv_to_df(dataset_csv_path, limit_nodes_to=relevant_nodes)
        dataset_nodes = dataset_df['node'].unique().tolist()
        
        try:
            # Create a subgraph with only nodes that are in both the causal graph and dataset
            nodes_in_graph = set(self.causal_graph.nodes())
            common_nodes = list(set(dataset_nodes).intersection(nodes_in_graph))
            
            if not common_nodes:
                raise ValueError("No common nodes between causal graph and dataset")
            
            # Create subgraph with only common nodes
            subgraph = self.causal_graph.subgraph(common_nodes)
            
            # Convert graph to adjacency matrix for random walk
            nodes = sorted(subgraph.nodes())
            adj = nx.to_numpy_array(subgraph, nodelist=nodes)
            
            # Apply simplified random walk algorithm
            ranks = random_walk(
                adj=adj,
                node_names=nodes,
                num_loop=self.num_loop,
                seed=self.random_seed
            )
            
            # Exclude alarm nodes from the predictions (as in other models)
            non_alarm_nodes = dataset_df[dataset_df['type'] != 'Alarm']['node'].unique().tolist()
            filtered_ranks = [node for node, _ in ranks if node in non_alarm_nodes]
            
            # Return ordered node names excluding alarms
            return filtered_ranks
            
        except ValueError as e:
            print(f"Warning: {str(e)}. Returning nodes in alphabetical order.")
            return sorted([node for node in dataset_nodes if node in dataset_df[dataset_df['type'] != 'Alarm']['node'].unique()])


class Baro(UnsupervisedRCAModel):
    """Baro Unsupervised RCA model based on robust scaling.
    
    This model implements the "Baro" method from "Root Cause Analysis for 
    Microservices based on Causal Inference: How Far Are We?" by Pham et al.
    It uses robust scaling to identify metrics that deviate most from normal behavior.
    
    The method is implemented as 'robust_scaler' in the original code at:
    https://anonymous.4open.science/r/ase24-cfm/cfm/e2e/__init__.py
    """
    
    def __init__(self, **kwargs):
        """Initialize the Baro RCA Model"""
        super().__init__(**kwargs)
    
    def _split_by_oldest_active_alarm(self, discrete_df, dataset_df, diagnosis_time):
        """Split data into normal and anomalous periods based on the oldest active alarm.
        
        Normal period: from start to the time of the oldest active alarm
        Anomalous period: from the oldest active alarm time to diagnosis time
        
        :param discrete_df: Discretized DataFrame with time as index and nodes as columns
        :param dataset_df: Original DataFrame with raw data
        :param diagnosis_time: The time at which the diagnosis is made
        :return: Tuple of (normal_discrete, anomal_discrete) DataFrames
        """
        # Get the last state at diagnosis time
        last_state = rcacom.limit_dataset_to_diagnosis_state(dataset_df, diagnosis_time)
        
        # Find active alarms at diagnosis time
        active_alarms = last_state[(last_state['type'] == 'Alarm') & 
                                ((last_state['value'] == 'True') | (last_state['value'] == True))]

        if active_alarms.empty:
            # If no active alarms, use diagnosis time as the split point
            print("Warning: No active alarms found. Using diagnosis time as split point.")
            normal_discrete = discrete_df[discrete_df.index < diagnosis_time]
            anomal_discrete = discrete_df[discrete_df.index >= diagnosis_time]
            return normal_discrete, anomal_discrete

        # Get the oldest active alarm time and substract 1s to estimate the normal data cut-off
        oldest_alarm_time = active_alarms['time_s'].min()
        split_time = oldest_alarm_time - 1.0
        
        # Find the closest timestamp in discrete_df index to the oldest alarm time
        # This handles cases where the exact alarm time isn't in the discretized index
        closest_idx = np.abs(np.array(discrete_df.index) - split_time).argmin()
        split_idx = discrete_df.index[closest_idx]
        
        # Split the data
        normal_discrete = discrete_df[discrete_df.index < split_idx]
        # Only include data up to diagnosis time in anomalous period
        anomal_discrete = discrete_df[(discrete_df.index >= split_idx) &
                                    (discrete_df.index <= diagnosis_time)]
        
        return normal_discrete, anomal_discrete
    
    def predict(self, dataset_csv_path: str, diagnosis_time: float, relevant_nodes: list = None) -> list:
        """Predicts root causes using robust scaling of metrics."""
        # Load and preprocess dataset
        dataset_df = utils.load_dataset_csv_to_df(dataset_csv_path, limit_nodes_to=relevant_nodes)
        if dataset_df.empty:
            print("Warning: Empty dataset.")
            return []
        
        # Discretize entire dataset at once
        try:
            _, discrete_df = discretize_dataset((0, dataset_df, CATEGORICAL_ENCODING_DICT))
            if discrete_df is None or discrete_df.empty:
                print("Warning: Failed to discretize data.")
                return []
                
            # Split based on oldest active alarm
            normal_discrete, anomal_discrete = self._split_by_oldest_active_alarm(
                discrete_df, dataset_df, diagnosis_time)
            
            if normal_discrete.empty or anomal_discrete.empty:
                print("Warning: No data for either normal or anomalous period.")
                return []
                
            # Calculate anomaly scores using RobustScaler/Baro by Pham et al.
            # Adaptation from Original Baro: uses Pandas Series directly instead of converting to numpy first
            ranks = []
            for col in discrete_df.columns:
                try:
                    normal_data = normal_discrete[col].dropna().astype(float)
                    anomal_data = anomal_discrete[col].dropna().astype(float)
                    
                    if len(normal_data) > 0 and len(anomal_data) > 0:
                        scaler = RobustScaler().fit(normal_data.values.reshape(-1, 1))
                        zscores = scaler.transform(anomal_data.values.reshape(-1, 1))[:, 0]
                        score = np.max(np.abs(zscores))
                        ranks.append((col, score))
                except (ValueError, TypeError):
                    continue
                    
            # Filter out alarm nodes and return ranked list
            non_alarm_nodes = dataset_df[dataset_df['type'] != 'Alarm']['node'].unique()
            ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
            return [node for node, _ in ranks if node in non_alarm_nodes]
            
        except Exception as e:
            print(f"Error in Baro prediction: {e}")
            return []
    