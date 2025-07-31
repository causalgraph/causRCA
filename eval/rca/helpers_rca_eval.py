import sys, os
from pathlib import Path
project_path = Path(__file__).parents[2]
sys.path.insert(0, str(project_path))

import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def aggregate_all_diagnosis_data(base_path: str) -> pd.DataFrame:
    """
    Extract diagnosis data from description JSON files and create a DataFrame.

    :param base_path: The base directory to search for JSON files.
    :type base_path: str or Path
    :return: DataFrame with unique values for alarm and diagnosis.id.
    :rtype: pd.DataFrame
    """
    base_path = Path(base_path)
        
    # Step 1: Find all *_description.json files recursively
    json_paths = list(base_path.glob("**/*_description.json"))
    
    # Step 2: Process each file
    rows = []
    for json_path in tqdm(json_paths, desc="  |-- Aggregating diagnosis overview from jsons", unit="file"):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            # Extract alarms (could be a single string or a list)
            alarms = data.get("alarms", [])
            if isinstance(alarms, str):
                alarms = [alarms]
            
            # Extract diagnosis details
            diagnosis = data.get("diagnosis", {})
            diag_id = diagnosis.get("id", "")
            diag_name = diagnosis.get("name", "")
            diag_comment = diagnosis.get("comment", "")
            
            # Add a row for each alarm
            for alarm in alarms:
                rows.append({
                    "alarm": alarm,
                    "diagnosis.id": diag_id,
                    "diagnosis.name": diag_name,
                    "diagnosis.comment": diag_comment
                })
    
    # Create initial DataFrame
    df = pd.DataFrame(rows)
    
    if df.empty:
        return pd.DataFrame(columns=["alarm", "diagnosis.id", "diagnosis.name", "diagnosis.variants"])
    
    # Group by alarm and diagnosis.id to create unique rows
    grouped_df = df.groupby(["alarm", "diagnosis.id"]).agg({
        "diagnosis.name": "first",  # Keep the first occurrence of diagnosis.name
        "diagnosis.comment": lambda x: list(set(filter(None, x)))  # Create a list of unique non-empty comments
    }).reset_index()
    
    # Rename the column for comments to diagnosis.variants
    grouped_df.rename(columns={"diagnosis.comment": "diagnosis.variants"}, inplace=True)
    
    return grouped_df


def get_run_csv_path_to_ground_truth_data(base_path: str) -> dict:
    """
    Get mapping from run CSV paths to ground truth data.

    :param base_path: The base directory to search for CSV files.
    :type base_path: str
    :return: Dictionary mapping CSV paths to ground truth data.
    :rtype: dict
    """
    base_path = Path(base_path)
    result_dict = {}
    
    # Find all exp_* directories recursively
    exp_folders = list(base_path.glob("**/exp_*"))
    exp_folders = [d for d in exp_folders if d.is_dir()]
    
    for exp_folder in tqdm(exp_folders, desc="  |-- Cataloging runs from experiments", unit="experiments"):
        exp_name = exp_folder.name  # e.g., "exp_1"
        
        # Find the description JSON file
        desc_json_path = exp_folder / f"{exp_name}_description.json"
        if not desc_json_path.exists():
            continue
        
        # Load experiment description
        with open(desc_json_path, 'r') as f:
            desc_data = json.load(f)
        
        # Find all run_* folders
        run_folders = [d for d in exp_folder.glob("run_*") if d.is_dir()]
        
        for run_folder in run_folders:
            # Find causes.json
            causes_json_path = run_folder / "causes.json"
            if not causes_json_path.exists():
                continue
                
            # Load run-specific cause data
            with open(causes_json_path, 'r') as f:
                causes_data = json.load(f)
            
            # Find CSV files in this run folder
            csv_files = list(run_folder.glob("*.csv"))
            if not csv_files:
                continue
            
            # For each CSV file, create a mapping to the merged data
            for csv_path in csv_files:
                # Merge the data
                merged_data = dict(desc_data)  # Create a copy of description data
                merged_data["run_data"] = causes_data
                
                # Add to result dictionary with CSV path as key
                result_dict[str(csv_path)] = merged_data
    
    return result_dict


def get_relevant_nodes_for_dataset_dir(base_path: str, all_nodes_path: Path = project_path / "data/expert_graph/all_nodes.csv") -> list:
    """
    Get relevant nodes for a dataset directory.

    :param base_path: Path to the dataset directory.
    :type base_path: str
    :param all_nodes_path: Path to the CSV file containing all nodes.
    :type all_nodes_path: Path
    :return: List of relevant nodes.
    :rtype: list
    """
    base_path = Path(base_path)
    
    # Find the first *_nodes.csv file in the directory
    csv_paths = list(base_path.glob("*_nodes.csv"))
    if not csv_paths:
        if not all_nodes_path.exists():
            raise FileNotFoundError(f"No '_nodes.csv' file found in directory: {base_path} and fallback 'all_nodes.csv' not found at {all_nodes_path}")
        print(f"  |-- ! No distinct '_nodes.csv' file found in directory: {base_path}. Using fallback: {all_nodes_path}")
        csv_path = all_nodes_path
    else:
        csv_path = csv_paths[0]
        print(f"  |-- Extracted relevant nodes from: {csv_path}")
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Extract labels if the 'label' column exists
    if 'label' in df.columns:
        labels = df['label'].tolist()
    else:
        labels = []
    
    return labels


def load_datasets_truth_data_and_relevant_nodes(path: str):
    """
    Load datasets' ground truth data and relevant nodes.

    :param path: Path to the dataset directory.
    :type path: str
    :return: Tuple of (csv_path_to_truth_data, relevant_nodes).
    :rtype: tuple
    """
    # Get number of experiments
    num_exps = len([p for p in Path(path).rglob('*') if p.is_dir() and p.name.startswith('exp_')])
    print(f"#### Preparing dataset: {os.path.basename(path)} with {num_exps} experiments ####")
    if num_exps == 0:
        raise ValueError(f"No experiments found in the directory: {path}")

    csv_path_to_truth_data = get_run_csv_path_to_ground_truth_data(path)
    rel_nodes = get_relevant_nodes_for_dataset_dir(path)
    
    if not csv_path_to_truth_data:
        raise ValueError(f"No ground truth data found in the directory: {path}")
    
    if not rel_nodes:
        raise ValueError(f"No relevant nodes found in the directory: {path}")
    
    return csv_path_to_truth_data, rel_nodes


def get_state_at_time(dataset_df: pd.DataFrame, time: float) -> dict:
    """
    Get the state of the dataset at a specific time.

    :param dataset_df: DataFrame of the dataset.
    :type dataset_df: pd.DataFrame
    :param time: Time to get the state at.
    :type time: float
    :return: Dictionary representing the state at the given time.
    :rtype: dict
    """
    # Filter the dataset to include only rows with time less than or equal to the specified time
    filtered_df = dataset_df[dataset_df['time_s'] <= time]
    
    # Sort the filtered DataFrame by time in descending order
    filtered_df = filtered_df.sort_values(by='time_s', ascending=False)
    
    # Get the last known value for each variable (node)
    state = filtered_df.groupby('node')['value'].first().to_dict()
    
    return state


def get_in_k_counters(ground_truth: list, predictions: list, max_k: int=5) -> dict:
    """
    Get counters for correct predictions in top-k.

    :param ground_truth: List of ground truth values.
    :type ground_truth: list
    :param predictions: List of predicted values.
    :type predictions: list
    :param max_k: Maximum k value to check.
    :type max_k: int
    :return: Dictionary with counts for in_1, in_2, ..., in_k.
    :rtype: dict
    """
    n_true_vars = len(ground_truth)
    in_k_counters = {}
    
    for k in range(1, max_k + 1):

        # Ensure that k does not exceed the length of predictions
        check_length = min(len(predictions), n_true_vars + (k - 1))
        
        # Check if all ground truth elements are in the first check_length predictions
        if all(true_var in predictions[:check_length] for true_var in ground_truth):
            in_k_counters[f"in_{k}"] = 1
        else:
            in_k_counters[f"in_{k}"] = 0
    
    return in_k_counters


def calculate_apk(y_true, y_pred, k=None):
    """
    Calculate average precision at k (AP@K).

    :param y_true: Ground truth values.
    :type y_true: list
    :param y_pred: Predicted values.
    :type y_pred: list
    :param k: Number of top predictions to consider.
    :type k: int, optional
    :return: AP@K score.
    :rtype: float
    """
    if not y_true:
        return 0.0
        
    if k is not None:
        y_pred = y_pred[:k]
        
    score = 0.0
    correct_predictions = 0
    used = set()  # to avoid double-counting
    
    for i, pred in enumerate(y_pred):
        rank = i + 1
        
        if pred in y_true and pred not in used:
            correct_predictions += 1
            precision_at_i = correct_predictions / rank
            score += precision_at_i
            used.add(pred)
    
    return score / min(len(y_true), k if k is not None else len(y_pred))


def calculate_mapk(y_true_list, y_pred_list, k=None):
    """
    Calculate mean average precision at k (MAP@K).

    :param y_true_list: List of ground truth values for each sample.
    :type y_true_list: list
    :param y_pred_list: List of predicted values for each sample.
    :type y_pred_list: list
    :param k: Number of top predictions to consider.
    :type k: int, optional
    :return: MAP@K score.
    :rtype: float
    """
    return np.mean([calculate_apk(y_true, y_pred, k) for y_true, y_pred in zip(y_true_list, y_pred_list)])