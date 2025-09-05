
import sys
from pathlib import Path
project_path = Path(__file__).parents[3]  # Go up 2 levels from eval/cd/ to causRCA/
sys.path.insert(0, str(project_path))

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union


# Helper function to discretize a dataset with multiple cores
def discretize_dataset(args):
    """Discretize a dataset using specified parameters and handle potential errors.
    This function takes a tuple of arguments containing dataset information and applies
    discretization with predefined parameters. It ensures the resulting dataset has
    more than one column before returning it. This function is using __discretize_dataset_helper
    to perforem the actual discretization, discretize_dataset is just a wrapper to handle multiprocessing.
    
    param args: A tuple containing:
        - dataset_index: Index or identifier for the dataset
        - dataset: The dataset to be discretized (as a pandas DataFrame)
        - categorical_encoding_dict: Dictionary for categorical variable encoding
    :type args: tuple
    :return: A tuple containing:
        - dataset_index: The original dataset index/identifier
        - data_dis: The discretized dataset if successful and contains more than one column,
          None otherwise.
    :type: tuple
    """
    dataset_index, dataset, categorical_encoding_dict = args
    
    try:
        data_dis = __discretize_dataset_helper(
            data=dataset,
            time_step_ms=500,
            max_time_s=None,
            convert_to_float=True,
            categorical_encoding_dict=categorical_encoding_dict,
            remove_constant_columns=False
        )
        
        # Check if discretized_data contains more than one column (node)
        if data_dis.shape[1] > 1:
            return (dataset_index, data_dis)
        else:
            return (dataset_index, None)
            
    except Exception as e:
        return (dataset_index, None)


def __discretize_dataset_helper(
    data: Union[str, pd.DataFrame],
    time_step_ms: int = 500,
    max_time_s: Optional[float] = None,
    convert_to_float: bool = False,
    categorical_encoding_dict: dict = {},
    remove_constant_columns: bool = True
) -> pd.DataFrame:
    """Discretizes a time series dataset from CSV format or DataFrame into a format for CD models.

    The input format contains changes at specific time points. This function creates
    a discretized time series with fixed time steps, where values remain constant
    between changes.

    :param data: Path to the CSV file or pandas DataFrame
    :type data: Union[str, pd.DataFrame]
    :param time_step_ms: Time step in milliseconds (Default: 500ms)
    :type time_step_ms: int
    :param max_time_s: Maximum time in seconds. If None, the maximum time from the data is used.
    :type max_time_s: float, optional
    :param convert_to_float: Whether to convert Binary, Alarm, and Categorical types to float
    :type convert_to_float: bool
    :param categorical_encoding_dict: Dictionary with categorical encodings for converting Categorical variables to float
    :type categorical_encoding_dict: dict, default to {}
    :param remove_constant_columns: Whether to remove columns with constant values. Defaults to True.
    :type remove_constant_columns: bool
    :return: Discretized data with columns as nodes and rows as time steps
    :rtype: pd.DataFrame
    """
    # Load data from CSV file or use DataFrame directly
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Data must be either a CSV file path (str) or a pandas DataFrame")
    
    # First encode the dataframe with proper types
    df = transform_non_continuous_values_in_df(df, categorical_encoding_dict, convert_to_float=convert_to_float)
    
    # Convert time step to seconds
    time_step_s = time_step_ms / 1000.0
    
    # Determine maximum time
    if max_time_s is None:
        max_time_s = df['time_s'].max()
    
    # Create time steps
    time_points = np.arange(0, max_time_s + time_step_s, time_step_s)
    
    # Create an empty DataFrame for results
    discretized_data = pd.DataFrame(index=time_points)
    discretized_data.index.name = 'time_s'
    
    # Process each node using pandas reindex and forward-fill to avoid loops
    for node in df['node'].unique():
        node_data = df[df['node'] == node].sort_values('time_s')
        
        if len(node_data) == 0:
            continue
            
        # Create a Series indexed by time_s
        node_series = pd.Series(
            data=node_data['value'].values,
            index=node_data['time_s']
        )
        
        # Reindex to all time points and forward fill / pad (propagate last value)
        resampled = node_series.reindex(
            time_points,
            method='pad'
        )
        
        # Handle values before first observation (back-fill from first value)
        if pd.isna(resampled.iloc[0]):
            resampled.fillna(node_data['value'].iloc[0], inplace=True)
            
        # Add to result DataFrame
        discretized_data[node] = resampled
    
    # Remove constant columns if requested - using vectorized operations
    if remove_constant_columns:
        nunique = discretized_data.nunique()
        constant_columns = nunique[nunique <= 1].index.tolist()
        
        if constant_columns:
            print(f"Number of removed constant nodes: {len(constant_columns)}")
            discretized_data = discretized_data.drop(columns=constant_columns)
    
    return discretized_data


def transform_non_continuous_values_in_df(df: pd.DataFrame, categorical_encoding_dict: dict, convert_to_float: bool = False) -> pd.DataFrame:
    """Transforms non-continuous values in a DataFrame to appropriate numeric representations.
    
    :param df: DataFrame with 'node', 'value', and 'type' columns
    :type df: pd.DataFrame
    :param categorical_encoding_dict: Dictionary mapping categorical values to numeric representations
    :type categorical_encoding_dict: dict
    :param convert_to_float: Whether to convert Binary, Alarm, and Categorical types to float
    :type convert_to_float: bool
    :return: Transformed DataFrame with properly typed values
    :rtype: pd.DataFrame
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get unique combinations of node and type - more efficient than groupby
    node_types = df[['node', 'type']].drop_duplicates().set_index('node')['type'].to_dict()
    
    # Group nodes by their types for vectorized operations
    binary_nodes = [node for node, type_ in node_types.items() if type_ == 'Binary']
    alarm_nodes = [node for node, type_ in node_types.items() if type_ == 'Alarm']
    counter_continuous_nodes = [node for node, type_ in node_types.items() if type_ in ['Counter', 'Continuous']]
    categorical_nodes = [node for node, type_ in node_types.items() if type_ == 'Categorical']
    
    if convert_to_float:
        # Create mapping dictionary for True/False values (used by both Binary and Alarm)
        bool_map = {True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0}
        
        # Process binary nodes all at once
        if binary_nodes:
            binary_mask = result_df['node'].isin(binary_nodes)
            result_df.loc[binary_mask, 'value'] = result_df.loc[binary_mask, 'value'].map(bool_map)
        
        # Process alarm nodes all at once (same mapping as binary)
        if alarm_nodes:
            alarm_mask = result_df['node'].isin(alarm_nodes)
            result_df.loc[alarm_mask, 'value'] = result_df.loc[alarm_mask, 'value'].map(bool_map)
        
        # Process continuous/counter nodes all at once
        if counter_continuous_nodes:
            cc_mask = result_df['node'].isin(counter_continuous_nodes)
            result_df.loc[cc_mask, 'value'] = pd.to_numeric(result_df.loc[cc_mask, 'value'], errors='coerce')
        
        # Process categorical nodes (these need individual treatment due to different mappings)
        for node in categorical_nodes:
            node_mask = result_df['node'] == node
            if node in categorical_encoding_dict:
                # Apply encoding in one step
                result_df.loc[node_mask, 'value'] = (
                    result_df.loc[node_mask, 'value'].astype(str)
                    .map(categorical_encoding_dict[node])
                    .fillna(0.0)
                )
            else:
                print(f"Warning: No encoding found for categorical variable '{node}'")
                result_df.loc[node_mask, 'value'] = result_df.loc[node_mask, 'value'].astype(str)
        
        # Check for unknown node types
        known_types = {'Binary', 'Alarm', 'Counter', 'Continuous', 'Categorical'}
        unknown_types = set(node_types.values()) - known_types
        if unknown_types:
            unknown_nodes = [node for node, type_ in node_types.items() if type_ in unknown_types]
            raise ValueError(f"Unknown node types '{unknown_types}' for nodes {unknown_nodes}")
            
    else:
        # Original behavior - keep values as they are in CSV
        # Binary nodes - keep values as they are (no processing needed)
        
        # Process continuous/counter nodes - try to convert to numeric
        if counter_continuous_nodes:
            cc_mask = result_df['node'].isin(counter_continuous_nodes)
            result_df.loc[cc_mask, 'value'] = pd.to_numeric(
                result_df.loc[cc_mask, 'value'], errors='ignore')
        
        # Process categorical and alarm nodes - convert to strings
        categorical_alarm_nodes = categorical_nodes + alarm_nodes
        if categorical_alarm_nodes:
            ca_mask = result_df['node'].isin(categorical_alarm_nodes)
            result_df.loc[ca_mask, 'value'] = result_df.loc[ca_mask, 'value'].astype(str)
        
        # Check for unknown node types
        known_types = {'Binary', 'Alarm', 'Counter', 'Continuous', 'Categorical'}
        unknown_types = set(node_types.values()) - known_types
        if unknown_types:
            unknown_nodes = [node for node, type_ in node_types.items() if type_ in unknown_types]
            raise ValueError(f"Unknown node types '{unknown_types}' for nodes {unknown_nodes}")
    
    return result_df


# Helper function for analyzing data structure
def analyze_dataset_structure(data: Union[str, pd.DataFrame]) -> Dict:
    """Analyzes the structure of a time series dataset.

    :param data: Path to the CSV file or pandas DataFrame
    :type data: Union[str, pd.DataFrame]
    :return: Information about the data structure
    :rtype: Dict
    """
    # Load data from CSV file or use DataFrame directly
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Data must be either a CSV file path (str) or a pandas DataFrame")
    
    # Calculate duration in seconds
    duration_seconds = df['time_s'].max() - df['time_s'].min()
    
    # Format duration as hh:mm:ss
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    
    analysis = {
        'total_records': len(df),
        'time_range': {
            'min': df['time_s'].min(),
            'max': df['time_s'].max(),
            'duration': duration_formatted,
            'duration_seconds': duration_seconds
        },
        'nodes': {
            'count': df['node'].nunique(),
            'names': df['node'].unique().tolist()
        },
        'types': df['type'].value_counts().to_dict(),
        'changes_per_node': df['node'].value_counts().to_dict()
    }
    
    return analysis
