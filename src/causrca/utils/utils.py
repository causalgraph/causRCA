import random, os
import numpy as np
import torch
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# Fixed Path for Encoding json file
project_path = Path(__file__).parents[3]
ENCODING_FILE_PATH = f"{project_path}/data/categorical_encoding.json"

def get_encoding_dict(encoding_file_path: str = ENCODING_FILE_PATH) -> dict:
    """Loads the categorical encoding dictionary from a JSON file.

    :param encoding_file_path: Path to the JSON file containing the encoding dictionary.
    :type encoding_file_path: str
    :return: Dictionary containing the categorical encodings.
    :rtype: dict
    """
    with open(encoding_file_path, 'r') as f:
        return json.load(f)

def seed_everything(seed: int):
    """Set random seed for reproducibility across all libraries.

    :param seed: Random seed value.
    :type seed: int
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_dataset_csv_to_df(csv_path: str, limit_nodes_to: list = None) -> pd.DataFrame:
    """Loads a CSV file into a DataFrame.

    :param csv_path: Path to the CSV file.
    :type csv_path: str or Path
    :param limit_nodes_to: Optional list of nodes to filter by.
    :type limit_nodes_to: list, optional
    :return: DataFrame containing the CSV data.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If CSV file not found.
    :raises ValueError: If 'node' column doesn't exist when filtering.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)

    # Filter DataFrame to only nodes in limit_nodes_to
    if limit_nodes_to is not None:
        if "node" not in df.columns:
            raise ValueError("CSV does not contain 'node' column to filter by.")
        df = df[df["node"].isin(limit_nodes_to)]
    
    return df


def estimate_default_value_per_node_from_datasets(dataframes) -> dict:
    """Calculate the most common value for each node based on duration across all datasets.

    :param dataframes: List of pandas DataFrames to analyze.
    :type dataframes: list
    :return: Dictionary mapping node names to their most common value.
    :rtype: dict
    """
    # Dictionary to store time durations for each node and its values
    node_value_durations = defaultdict(lambda: defaultdict(float))
    
    # Process each DataFrame
    for df in tqdm(dataframes, desc="  |-- Calculating most common values per node from datasets", unit="dataset"):
        # Sort by time to ensure proper time calculation
        df = df.sort_values('time_s')
        
        # Group by node to process each node separately
        for node_name, node_group in df.groupby('node'):
            node_group = node_group.sort_values('time_s').reset_index(drop=True)
            
            # Calculate time duration for each value
            for i in range(len(node_group)):
                current_value = str(node_group.iloc[i]['value'])
                current_time = node_group.iloc[i]['time_s']
                
                # Calculate duration until next time point or end of dataset
                if i < len(node_group) - 1:
                    next_time = node_group.iloc[i + 1]['time_s']
                    duration = next_time - current_time
                else:
                    # For the last entry, use dataset end time or a default duration
                    if df['time_s'].max() > current_time:
                        duration = df['time_s'].max() - current_time
                    else:
                        # Use average interval or default duration
                        if len(node_group) > 1:
                            avg_interval = (node_group.iloc[-1]['time_s'] - node_group.iloc[0]['time_s']) / (len(node_group) - 1)
                            duration = avg_interval
                        else:
                            duration = 1.0  # Default duration for single entries
                
                # Accumulate time for this value
                node_value_durations[node_name][current_value] += duration
    
    # Get only the most common value for each node
    node_to_default_value_dict = {}
    for node_name, value_durations in node_value_durations.items():
        # Find the value with maximum duration
        most_common_value = max(value_durations.items(), key=lambda x: x[1])[0]
        node_to_default_value_dict[node_name] = most_common_value
    
    return node_to_default_value_dict