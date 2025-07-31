import os
import pandas as pd
from collections import defaultdict, OrderedDict
import pandas as pd
import json

# Global configuration for encoding range
ENCODING_RANGE = {
    'min': 0,
    'max': 1000
}


def encode_categorical(data_dir: str, encoding_range: dict = None) -> dict:
    """
    Encode categorical variables into a configurable range based on time proportion.
    Only processes data with type='Categorical'.
    
    :param data_dir: Directory containing CSV files to process
    :type data_dir: str
    :param encoding_range: Dictionary with 'min' and 'max' keys for encoding range (optional)
    :type encoding_range: dict
    :return: Dictionary mapping node names to their encoded values
    :rtype: dict
    """
    # Use provided range or global default
    if encoding_range is None:
        encoding_range = ENCODING_RANGE
    
    range_min = encoding_range.get('min', 0)
    range_max = encoding_range.get('max', 100)
    range_span = range_max - range_min
    
    print(f"Using encoding range: [{range_min}, {range_max}]")
    
    # Dictionary to store time proportions for each node and its values
    node_time_counts = defaultdict(lambda: defaultdict(float))
    node_total_times = defaultdict(float)
    
    # Recursively find all CSV files in data_dir
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if not all(col in df.columns for col in ['time_s', 'node', 'value', 'type']):
                print(f"Skipping {csv_file}: Missing required columns")
                continue
            
            # Filter only categorical data types
            categorical_df = df[df['type'] == 'Categorical']
            if categorical_df.empty:
                print(f"Skipping {csv_file}: No categorical data found")
                continue
            
            # Sort by time to ensure proper time calculation
            categorical_df = categorical_df.sort_values('time_s')
            
            # Group by node to process each node separately
            for node_name, node_group in categorical_df.groupby('node'):
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
                        # For the last entry, we might need to estimate duration
                        # or use a default duration
                        if len(node_group) > 1:
                            avg_interval = (node_group.iloc[-1]['time_s'] - node_group.iloc[0]['time_s']) / (len(node_group) - 1)
                            duration = avg_interval
                        else:
                            duration = 1.0  # Default duration for single entries
                    
                    # Accumulate time for this value
                    node_time_counts[node_name][current_value] += duration
                    node_total_times[node_name] += duration
                    
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    # Calculate encoded values (time proportions normalized to configured range)
    encoded_values = {}
    
    for node_name in node_time_counts:
        total_time = node_total_times[node_name]
        if total_time > 0:
            # Calculate raw proportions for each value of this node
            value_proportions = {}
            for value, time_duration in node_time_counts[node_name].items():
                proportion = time_duration / total_time  # Get proportion [0, 1]
                value_proportions[value] = proportion
            
            # Find the maximum proportion to scale to range_max
            max_proportion = max(value_proportions.values()) if value_proportions else 1.0
            
            # Scale values with secondary sorting by value name for uniqueness
            scaled_proportions = {}
            
            # STEP 1: Sort values by proportion (descending) and by name (ascending) for tiebreaking
            # This ensures consistent ordering even when proportions are identical
            sorted_items = sorted(value_proportions.items(), 
                                key=lambda x: (-x[1], x[0]))
            
            # STEP 2: Calculate how much space we need to reserve for uniqueness offsets
            # We need small increments to differentiate values with identical proportions
            num_items = len(sorted_items)
            offset_increment = 0.001  # Small increment between identical values
            total_offset_space = (num_items - 1) * offset_increment  # Space needed for all offsets
            
            # STEP 3: Adjust the effective scaling range to accommodate the offsets
            # The highest value will get the full range_max after adding its offset
            effective_range_span = range_span - total_offset_space
            
            # STEP 4: Scale each value and add position-based offset for uniqueness
            for i, (value, proportion) in enumerate(sorted_items):
                if max_proportion > 0:
                    # Scale the proportion to the effective range (leaving room for offsets)
                    base_scaled_value = range_min + (proportion / max_proportion) * effective_range_span
                    
                    # Add position-based offset: highest values get largest offset
                    # This ensures the top value reaches exactly range_max (1000)
                    position_offset = (num_items - 1 - i) * offset_increment
                    scaled_value = base_scaled_value + position_offset
                else:
                    # Fallback for edge case where max_proportion is 0
                    scaled_value = range_min + (num_items - 1 - i) * offset_increment
                
                # Round to 5 decimal places for precision
                scaled_proportions[value] = round(scaled_value, 5)
            
            # Create OrderedDict maintaining the sorted order
            sorted_value_proportions = OrderedDict(
                (value, scaled_proportions[value]) for value, _ in sorted_items
            )
            # Store the sorted proportions (now in configured range)
            encoded_values[node_name] = sorted_value_proportions
        else:
            print(f"Warning: No valid time data for node {node_name}")
    
    return encoded_values


def apply_categorical_encoding(data, encoding_dict):
    """
    Apply the categorical encoding to new data.
    
    :param data: DataFrame with columns 'node' and 'value'
    :type data: pandas.DataFrame
    :param encoding_dict: Dictionary from encode_categorical function
    :type encoding_dict: dict
    :return: DataFrame with encoded values (0-100)
    :rtype: pandas.DataFrame
    """    
    encoded_data = data.copy()
    encoded_data['encoded_value'] = 0.0
    
    for idx, row in encoded_data.iterrows():
        node_name = row['node']
        value = str(row['value'])
        
        if node_name in encoding_dict and value in encoding_dict[node_name]:
            encoded_data.at[idx, 'encoded_value'] = encoding_dict[node_name][value]
        else:
            # Handle unknown values - could assign 0 or some default value
            encoded_data.at[idx, 'encoded_value'] = 0.0
            print(f"Warning: Unknown value '{value}' for node '{node_name}', assigned 0.0")
    
    return encoded_data


def save_encoding_dict(encoding_dict, filepath):
    """
    Save the encoding dictionary to a JSON file.
    
    :param encoding_dict: Dictionary from encode_categorical function
    :type encoding_dict: dict
    :param filepath: Path to save the encoding dictionary
    :type filepath: str
    """
    with open(filepath, 'w') as f:
        json.dump(encoding_dict, f, indent=2)
    print(f"Encoding dictionary saved to {filepath}")



# Example usage:
if __name__ == "__main__":
    
    data_directory = "./data"
    encoding_dict = encode_categorical(data_directory)

    # Save encoding dictionary
    save_encoding_dict(encoding_dict, "./data/categorical_encoding.json")

