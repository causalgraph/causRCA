
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
    
    # Get all unique nodes
    nodes = df['node'].unique()
    
    # Initialize discretized matrix
    discretized_data = pd.DataFrame(index=time_points, columns=nodes)
    discretized_data.index.name = 'time_s'
    
    # Propagate values over time for each node
    for node in nodes:
        node_data = df[df['node'] == node].sort_values('time_s')
        
        if len(node_data) == 0:
            continue
            
        # Set first value for all time points before the first change
        first_value = node_data.iloc[0]['value']
        current_value = first_value
        
        # Go through all time points
        for i, time_point in enumerate(time_points):
            # Check if there is a change at or before this time point
            changes_up_to_now = node_data[node_data['time_s'] <= time_point]
            
            if len(changes_up_to_now) > 0:
                # Take the last value before or at this time point
                current_value = changes_up_to_now.iloc[-1]['value']
            
            discretized_data.loc[time_point, node] = current_value
    
    # Remove constant columns if requested
    if remove_constant_columns:
        # Identify columns that are constant
        constant_columns = []
        for col in discretized_data.columns:
            if discretized_data[col].nunique() <= 1:
                constant_columns.append(col)
        
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
    
    # Get the type mapping for each node
    type_mapping = df.groupby('node')['type'].first().to_dict()
    
    # Process each unique node
    for node in type_mapping.keys():
        node_mask = result_df['node'] == node
        node_type = type_mapping.get(node, 'Continuous')
        
        if convert_to_float:
            if node_type == 'Binary':
                # Convert Binary True/False to 1.0/0.0
                result_df.loc[node_mask, 'value'] = result_df.loc[node_mask, 'value'].map(
                    {True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
            elif node_type == 'Alarm':
                # Convert Alarm True/False to 1.0/0.0
                result_df.loc[node_mask, 'value'] = result_df.loc[node_mask, 'value'].map(
                    {True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
            elif node_type == 'Categorical':
                # Convert Categorical using encoding JSON
                if node in categorical_encoding_dict:
                    # Convert values to string first, then map to float
                    result_df.loc[node_mask, 'value'] = result_df.loc[node_mask, 'value'].astype(str)
                    result_df.loc[node_mask, 'value'] = result_df.loc[node_mask, 'value'].map(
                        categorical_encoding_dict[node])
                    # Fill NaN values with 0.0 for unmapped categories
                    result_df.loc[node_mask, 'value'] = result_df.loc[node_mask, 'value'].fillna(0.0)
                else:
                    print(f"Warning: No encoding found for categorical variable '{node}'")
                    result_df.loc[node_mask, 'value'] = result_df.loc[node_mask, 'value'].astype(str)
            elif node_type in ['Counter', 'Continuous']:
                # Convert to numeric
                result_df.loc[node_mask, 'value'] = pd.to_numeric(result_df.loc[node_mask, 'value'], errors='coerce')
            else:
                raise ValueError(f"Unknown node type '{node_type}' for node '{node}'")
        else:
            # Original behavior - keep values as they are in CSV
            if node_type == 'Binary':
                # Keep binary values as they are
                pass
            elif node_type in ['Counter', 'Continuous']:
                # Try to convert to numeric, but keep as string if conversion fails
                result_df.loc[node_mask, 'value'] = pd.to_numeric(
                    result_df.loc[node_mask, 'value'], errors='ignore')
            elif node_type in ['Categorical', 'Alarm']:
                # Keep categorical values as strings
                result_df.loc[node_mask, 'value'] = result_df.loc[node_mask, 'value'].astype(str)
            else:
                raise ValueError(f"Unknown node type '{node_type}' for node '{node}'")
    
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
