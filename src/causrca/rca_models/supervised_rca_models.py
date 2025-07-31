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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import causrca.rca_models.common as rcacom
import causrca.utils.utils as utils
from causrca.utils.discretize import transform_non_continuous_values_in_df


class SupervisedRCAModel(rcacom.RCAModel):
    """Base class for supervised RCA Models.
    
    Take as input a dataset and predicts the most likely diagnosis based on the data.
    
    Requires a labeled data for training that combine the dataset with ground truth labels (relevant alarm -> diagnosis mapping)."""
    
    def __init__(self, **kwargs):
        """Initialize the Supervised RCA_Model"""
        super().__init__(**kwargs)
    
    @abstractmethod
    def train(self, relevant_nodes: list, csv_path_to_truth_data: dict):
        """Train the RCA model on the datasets and ground truth labels (diagnosis).

        :param relevant_nodes: List of relevant nodes to consider for training.
        :type relevant_nodes: list
        :param csv_path_to_truth_data: Dictionary mapping CSV paths to ground truth diagnosis data (json with many relevant fields for diagnosis)
        :type csv_path_to_truth_data: dict
        """
        pass

class BaselineSupervisedRCA(SupervisedRCAModel):
    """Baseline for supervised RCA Models: Always predict order of diagnosis based on frequency of occurrence.
    
    Take as input a dataset and predicts, the most common diagnosis for the given set of alarms at diagnosis_time."""
    
    def __init__(self, **kwargs):
        """Initialize the Baseline Supervised RCA_Model"""
        super().__init__(**kwargs)
        self.known_alarms = set()  # Set to store known alarms, will be filled during training
    
    def __determine_oldest_known_active_alarm(self, dataset_df: pd.DataFrame, diagnosis_time: float) -> pd.Series:
        """Determine the oldest currently active alarm in the dataset at the given diagnosis time.

        :param dataset_df (pd.DataFrame): The dataset DataFrame containing columns 'time_s', 'node', and 'value'.
        :type diagnosis_time: float
        :param diagnosis_time: The time at which to retrieve the last known state.
        :type dataset_df: pd.DataFrame
        
        :return: The row corresponding to the oldest currently active alarm.
        :rtype: pd.Series
        """
        # Limit dataset to the last known state at the diagnosis time
        last_state = rcacom.limit_dataset_to_diagnosis_state(dataset_df, diagnosis_time)
        # Filter for nodes for known alarms and assert that have value = True
        alarms = last_state[last_state["node"].isin(self.known_alarms) & (last_state["value"] == "True")]
        if alarms.empty:
            raise ValueError(f"No active alarms found at diagnosis time {diagnosis_time}. Full table:\n{last_state}")
        # Return node value of the oldest alarm by time_s
        oldest_alarm_row = alarms.loc[alarms["time_s"].idxmin()]
        return oldest_alarm_row['node']
    
    def train(self, relevant_nodes: list, csv_path_to_truth_data: dict) -> dict:
        """Statistical based supervised RCA Model, that takes the input datasets, detects oldest currently active alarm and then predicts the most common diagnosis for it.
    
        :param relevant_nodes: List of relevant nodes to consider for training.
        :type relevant_nodes: list
        :param csv_path_to_truth_data: Dictionary mapping CSV paths to ground truth diagnosis data (json with many relevant fields for diagnosis)
        :type csv_path_to_truth_data: dict
        
        :return: A dictionary mapping alarms to a list of diagnosis IDs ordered by frequency (high to low).
        :rtype: dict
        """
        # Save relevant nodes for later use in prediction
        self.relevant_nodes = relevant_nodes
        
        # Create a dictionary to count occurrences of alarms with diagnoses
        alarm_diagnosis_counts = []
        unique_diagnosis = set()
    
        for _, true_data in tqdm(csv_path_to_truth_data.items(), desc="  |-- Gathering Statistics for Baseline Supervised RCA Model", unit="run"):
            # Extract alarms from true_data and process each alarm
            alarms_list = []
            if "alarms" in true_data:
                alarms = true_data["alarms"]
                if isinstance(alarms, str):
                    alarms_list.append(alarms)
                elif isinstance(alarms, list):
                    alarms_list.extend(alarms)
                else:
                    print(f"Warning: Unexpected type for 'alarms': {type(alarms)}. Skipping.")
            
            # For each diagnosis, add an alarm-diagnosis pair for each alarm
            if "diagnoses" in true_data and true_data["diagnoses"]:
                for diagnosis in true_data["diagnoses"]:
                    for alarm in alarms_list:
                        alarm_diagnosis_counts.append({
                            "alarm": alarm,
                            "diagnosis_id": diagnosis["id"],
                            "diagnosis_name": diagnosis["name"]
                        })
                    unique_diagnosis.add((diagnosis["id"], diagnosis["name"]))
    
        # Create a DataFrame from the collected data
        alarm_diagnosis_df = pd.DataFrame(alarm_diagnosis_counts)
    
        # Group by alarm and diagnosis_id to count occurrences
        self.statistics_df = alarm_diagnosis_df.groupby(["alarm", "diagnosis_id", "diagnosis_name"]).size().reset_index(name="num_instances")
        self.known_alarms = set(self.statistics_df["alarm"].unique())

        # Create a dict that maps from alarm to an ordered list of diagnosis_ids ordered by frequency (high to low)
        self.alarm_to_diagnosis_mapping = (
            self.statistics_df.groupby("alarm")
            .apply(lambda group: group.sort_values("num_instances", ascending=False)["diagnosis_id"].tolist())
            .to_dict()
        )

        # Ensure that every alarm has all possible diagnosis in list (even if no example exists) to ensure constant output vector size
        all_diagnoses = self.statistics_df["diagnosis_id"].unique().tolist()
        for alarm in self.alarm_to_diagnosis_mapping:
            # Find missing diagnoses for this alarm
            existing_diagnoses = self.alarm_to_diagnosis_mapping[alarm]
            missing_diagnoses = [d for d in all_diagnoses if d not in existing_diagnoses]
            if len(missing_diagnoses) > 0:
                random.shuffle(missing_diagnoses)
                self.alarm_to_diagnosis_mapping[alarm].extend(missing_diagnoses)

        return self.alarm_to_diagnosis_mapping
    
    def predict(self, dataset_csv_path: str, diagnosis_time: float) -> list:
        """Returns an ordered list of the most likely diagnosis"""
        # Load dataset, get oldest active alarm, return self.alarm_to_diagnosis_mapping for that alarm
        dataset_df = utils.load_dataset_csv_to_df(dataset_csv_path, limit_nodes_to=self.relevant_nodes)
        oldest_active_alarm = self.__determine_oldest_known_active_alarm(dataset_df, diagnosis_time)
        if oldest_active_alarm not in self.alarm_to_diagnosis_mapping:
            raise ValueError(f"Oldest active alarm {oldest_active_alarm} not found in training data. Available alarms: {list(self.alarm_to_diagnosis_mapping.keys())}")
        return self.alarm_to_diagnosis_mapping[oldest_active_alarm]
    

class LogisticRegressionRCA(SupervisedRCAModel):
    """Logistic Regression based supervised RCA model that predicts diagnosis probabilities.
    
    Uses the last known state of all variables at diagnosis time, with proper conversion of
    non-continuous values to float using the transform_non_continuous_values_in_df function.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Logistic Regression Supervised RCA_Model"""
        super().__init__(**kwargs)
        self.model = None
        self.default_values = {}  # Store default values for each variable
        self.feature_names = []
        self.diagnosis_classes = []
        self.categorical_encoding_dict = utils.get_encoding_dict()
    
    def _prepare_feature_vector(self, dataset_df, diagnosis_time) -> np.ndarray:
        """Create a feature vector from the last known state at diagnosis time.
        
        :param dataset_df: DataFrame containing the dataset with columns 'time_s', 'node', and 'value'.
        :type dataset_df: pd.DataFrame
        :param diagnosis_time: The time at which to retrieve the last known state.
        :type diagnosis_time: float
        
        :return: A numpy array representing the feature vector for the model.
        :rtype: np.ndarray 
        """
        # Get last known state at diagnosis time
        last_state = rcacom.limit_dataset_to_diagnosis_state(dataset_df, diagnosis_time)
        
        # Transform non-continuous values to float
        transformed_last_state = transform_non_continuous_values_in_df(
            last_state,
            self.categorical_encoding_dict,
            convert_to_float=True
        )
        
        # Check if there are nodes in the last state that weren't seen during training
        unknown_nodes = set(transformed_last_state['node'].unique()) - set(self.feature_names)
        if unknown_nodes:
            print(f"Warning: Found {len(unknown_nodes)} nodes in prediction data that weren't in training: {unknown_nodes}")
            # Remove the unknown nodes from consideration
            transformed_last_state = transformed_last_state[~transformed_last_state['node'].isin(unknown_nodes)]
        
        # Initialize features with default values
        features = {}
        for node in self.feature_names:
            features[node] = float(self.default_values.get(node, 0.0))
        
        # Update with actual values from transformed last state
        for _, row in transformed_last_state.iterrows():
            node = row["node"]
            if node in self.feature_names:
                try:
                    features[node] = float(row["value"])
                except (ValueError, TypeError):
                    # If conversion fails, use the default value
                    features[node] = float(self.default_values.get(node, 0.0))
        
        # Convert to numpy array in the correct order
        X = np.array([features[node] for node in self.feature_names]).reshape(1, -1)
        return X
    
    def train(self, relevant_nodes, csv_path_to_truth_data) -> Pipeline:
        """Train logistic regression model on the datasets and ground truth labels.
        
        :param relevant_nodes: List of relevant nodes to consider for training.
        :type relevant_nodes: list
        :param csv_path_to_truth_data: Dictionary mapping CSV paths to ground truth diagnosis data (json with many relevant fields for diagnosis)
        :type csv_path_to_truth_data: dict
        
        :return: The trained logistic regression model.
        :rtype: Pipeline        
        """
        self.relevant_nodes = relevant_nodes
        
        # Load all datasets once and store them in a dictionary for reuse
        transformed_dataframes = {}
        
        for csv_path, true_data in tqdm(csv_path_to_truth_data.items(), desc="  |-- Loading and transforming datasets for training", unit="run"):
            ds_df = utils.load_dataset_csv_to_df(csv_path, limit_nodes_to=self.relevant_nodes)
            transformed_df = transform_non_continuous_values_in_df(
                ds_df,
                self.categorical_encoding_dict,
                convert_to_float=True
            )
            transformed_dataframes[csv_path] = transformed_df
        
        # Calculate default values across all datasets
        self.default_values = utils.estimate_default_value_per_node_from_datasets(list(transformed_dataframes.values()))
        
        # Define feature names (all nodes that we've seen)
        self.feature_names = sorted(self.default_values.keys())
        
        # Collect training data
        X_train = []
        y_train = []
        all_diagnoses = set()
        
        # Process each training instance (reusing already loaded dataframes)
        for csv_path, true_data in tqdm(csv_path_to_truth_data.items(), desc="  |-- Preparing training data", unit="run"):
            try:
                # Get diagnosis time
                diagnosis_time = float(true_data["run_data"]["diagnosis_at"])
                
                # Use already loaded and transformed dataframe
                transformed_df = transformed_dataframes[csv_path]
                
                # Get last known state at diagnosis time
                last_state = rcacom.limit_dataset_to_diagnosis_state(transformed_df, diagnosis_time)
                
                # Initialize features with default values
                features = {}
                for node in self.feature_names:
                    features[node] = float(self.default_values.get(node, 0.0))
                
                # Update with actual values from transformed last state
                for _, row in last_state.iterrows():
                    node = row["node"]
                    if node in self.feature_names:
                        try:
                            features[node] = float(row["value"])
                        except (ValueError, TypeError):
                            # If conversion fails, use the default value
                            features[node] = float(self.default_values.get(node, 0.0))
                
                # Convert to numpy array in the correct order
                X = np.array([features[node] for node in self.feature_names]).reshape(1, -1)
                
                # Create a training sample for each diagnosis in the list
                if "diagnoses" in true_data and true_data["diagnoses"]:
                    for diagnosis in true_data["diagnoses"]:
                        diagnosis_id = diagnosis["id"]
                        all_diagnoses.add(diagnosis_id)
                        # Add to training data
                        X_train.append(X.flatten())
                        y_train.append(diagnosis_id)

            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
                continue
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Save diagnosis classes
        self.diagnosis_classes = sorted(all_diagnoses)
        
        # Create and train the logistic regression model
        print(f"  |-- Training logistic regression model on {len(X_train)} samples and {len(self.feature_names)} features....")
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                solver='lbfgs',
                C=1.0,
                max_iter=1000
            ))
        ])
        
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, dataset_csv_path, diagnosis_time) -> list:
        """Predict diagnosis probabilities for the given dataset at diagnosis time.
        
        :param dataset_csv_path: Path to the dataset CSV file
        :type dataset_csv_path: str
        :param diagnosis_time: Time at which to make the prediction
        :type diagnosis_time: float
        
        :return: Ordered list of diagnosis IDs based on predicted probabilities
        :rtype: list        
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Load dataset
        dataset_df = utils.load_dataset_csv_to_df(dataset_csv_path, limit_nodes_to=self.relevant_nodes)
        
        # Prepare input features
        X = self._prepare_feature_vector(dataset_df, diagnosis_time)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X)[0]
        
        # Map probabilities to diagnosis IDs
        diagnosis_probs = list(zip(self.diagnosis_classes, probabilities))
        diagnosis_probs.sort(key=lambda x: x[1], reverse=True)

        # NOTE: For debugging purposes, print the predicted probabilities
        # print(f"Predicted probs for {os.path.basename(dataset_csv_path)} at time={diagnosis_time}: {diagnosis_probs}")
        
        # Return ordered list of diagnosis IDs (for compatibility with existing evaluation)
        ordered_diagnoses = [diagnosis_id for diagnosis_id, _ in diagnosis_probs]
        
        return ordered_diagnoses
    

class CausalPrioLogisticRegressionRCA(SupervisedRCAModel):
    """ Wrapper for Logistic Regression RCA that uses the causal structure to limit the features (relevant nodes) for the logistic regressor."""
    
    def __init__(self, causal_graph:nx.DiGraph= None, **kwargs):
        """Initialize the Causal Prioritized Logistic Regression RCA model.
        
        :param causal_graph (nx.DiGraph): A directed graph representing causal relationships between nodes.
        :type causal_graph: nx.DiGraph
        """
        super().__init__(**kwargs)
        self.causal_graph = causal_graph
        self.logistic_model = None
        self.init_kwargs = kwargs  # Store initialization arguments for the logistic regression model

    def __get_relevant_nodes_by_causal_structure(self, alarms_list: list) -> list:
        """
        Determines the relevant nodes for a given list of alarms and a causal graph.
        Relevant nodes include the alarms themselves and their direct influences onto the alarm in the causal graph.

        :param alarms_list: List of alarm nodes (strings) for which to find relevant nodes.
        :type alarms_list: list
        
        :return: A list of relevant nodes (alarms and their direct predecessors) sorted alphabetically.
        :rtype: list
        """
        # Ensure that the causal graph is provided and all alarms are in the graph
        if self.causal_graph is None:
            raise ValueError("Causal graph not provided during initialization")
        if not all(alarm in self.causal_graph for alarm in alarms_list):
            missing_alarms = [alarm for alarm in alarms_list if alarm not in self.causal_graph]
            raise ValueError(f"Not all requested alarms are present in the causal graph. Alarms not in graph: {missing_alarms}")
        
        # Find all direct parents (predecessors) of alarms as a unique set
        parents_set = {pred for alarm in alarms_list for pred in self.causal_graph.predecessors(alarm)}
        
        # Combine alarms and their parents into a single set of relevant nodes
        relevant_nodes_set = set(alarms_list) | parents_set
        
        # Return the relevant nodes as a sorted list
        return sorted(relevant_nodes_set)
    
    def train(self, relevant_nodes=None, csv_path_to_truth_data=None) -> Pipeline:
        """Train model using causal prioritization to filter nodes.
        
        :param relevant_nodes: List of relevant nodes to consider for training. If None, uses nodes determined by causal structure.
        :type relevant_nodes: list, optional
        :param csv_path_to_truth_data: Dictionary mapping CSV paths to ground truth diagnosis data (json with many relevant fields for diagnosis)
        :type csv_path_to_truth_data: dict
        
        :return: The trained logistic regression model.
        :rtype: Pipeline
        """
        # Ensure that a causal graph is provided
        if self.causal_graph is None:
            raise ValueError("Causal graph not provided during initialization")
        
        # Extract all unique alarms from the truth data
        alarms_list = []
        for _, true_data in csv_path_to_truth_data.items():
            if "alarms" in true_data:
                alarms = true_data["alarms"]
                if isinstance(alarms, str):
                    alarms_list.append(alarms)
                elif isinstance(alarms, list):
                    alarms_list.extend(alarms)
                else:
                    print(f"Warning: Unexpected type for 'alarms': {type(alarms)}. Skipping.")
        alarms_list = sorted(set(alarms_list))
        
        # Get causally determined relevant nodes
        causally_determined_nodes = self.__get_relevant_nodes_by_causal_structure(alarms_list)
        
        # Use causally determined nodes or intersect with provided relevant nodes (from expert knowldge) if available
        if relevant_nodes is None:
            filtered_nodes = causally_determined_nodes
        else:
            filtered_nodes = [node for node in relevant_nodes if node in causally_determined_nodes]
        
        # Create and train a LogisticRegressionRCA model
        self.logistic_model = LogisticRegressionRCA(**self.init_kwargs)
        return self.logistic_model.train(filtered_nodes, csv_path_to_truth_data)
    
    def predict(self, dataset_csv_path: str, diagnosis_time: float) -> list:
        """Predict using the trained logistic regression model with causal prioritization.
        
        :param dataset_csv_path: Path to the dataset CSV file
        :type dataset_csv_path: str
        :param diagnosis_time: Time at which to make the prediction
        :type diagnosis_time: float
        
        :return: Ordered list of diagnosis IDs based on predicted probabilities
        :rtype: list
        """
        if self.logistic_model is None:
            raise ValueError("Model not trained yet")
        
        return self.logistic_model.predict(dataset_csv_path, diagnosis_time)