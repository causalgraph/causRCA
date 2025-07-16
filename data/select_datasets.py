"""
Dataset Selection Module for Causal Discovery

This module provides functionality to select and filter CSV datasets for causal discovery.
It supports digital twin data (dig_twin) as main learning data and real operational 
data (real_op) for enrichment.

Returns individual pandas DataFrames (one per CSV file) allowing users to implement
their own learning strategies like concatenation or majority voting.

Author: Assistant
Date: June 2025
"""
import sys
from pathlib import Path
project_path = Path(__file__).parents[1]  # Go up 2 levels from eval/cd/ to causRCA/
sys.path.insert(0, str(project_path))

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetSelector:
    """Class for selecting and filtering CSV datasets for causal discovery analysis.
    
    The class handles two types of data:
    - Digital Twin Data (dig_twin): Main learning datasets for causal discovery
    - Real Operational Data (real_op): Enrichment datasets with real operational data
    
    Returns individual pandas DataFrames (one per CSV file) for flexible usage in:
    1. Concatenation strategies (combine all data and learn a single graph)
    2. Majority voting strategies (learn individual graphs and combine results)
    3. Any custom approach with the individual datasets
    """
    
    def __init__(self, dig_twin_dir: str, real_op_dir: str):
        """nitialize the DatasetSelector.
        
        :param dig_twin_dir: Directory containing digital twin datasets
        :type dig_twin_dir: str
        :param real_op_dir: Directory containing real operational datasets
        :type real_op_dir: str
        """
        self.dig_twin_dir = Path(dig_twin_dir).resolve()
        self.real_op_dir = Path(real_op_dir).resolve()
        
        # Validate directories exist
        self.__validate_directories()
        
    
    def __validate_directories(self):
        """Validate that required directories exist."""
        for dir_path in [self.dig_twin_dir, self.real_op_dir]:
            if not dir_path.exists():
                logger.warning(f"Directory does not exist: {dir_path}")
            else:
                logger.info(f"Directory validated: {dir_path}")
    
    
    def find_dig_twin_datasets(self,
                               subdirs: Optional[List[str]] = None, 
                               node_filter: Optional[List[str]] = None) -> List[Path]:
        """Recursively find digital twin CSV datasets (main learning data).
        
        :param subdirs: Specific subdirectories to search in (e.g., ['exp_cool_lubricant'])
        :type subdirs: Optional[List[str]]
        :param node_filter: List of node names to filter by
        :type node_filter: Optional[List[str]]
        
        :return: List of CSV dataset file paths   
        :rtype: List[Path]
        """
        datasets = []
        
        if subdirs is None:
            search_dirs = [self.dig_twin_dir]
        else:
            search_dirs = [self.dig_twin_dir / subdir for subdir in subdirs 
                          if (self.dig_twin_dir / subdir).exists()]
        
        for search_dir in search_dirs:
            # Find CSV files only
            csv_files = list(search_dir.rglob("*.csv"))
            datasets.extend(csv_files)
        
        if node_filter:
            datasets = self.__filter_csv_datasets_by_nodes(datasets, node_filter, min_coverage=0.33)
        
        logger.info(f"Found {len(datasets)} digital twin CSV datasets")
        return datasets
    
    
    def find_real_op_datasets(self, node_filter: Optional[List[str]] = None,
                             top_n: int = 20) -> List[Path]:
        """Find real operational datasets for enrichment.
        
        :param node_filter: List of node names to filter by (optional)
        :type node_filter: Optional[List[str]]
        :param top_n: Number of top datasets to return (based on scoring). Use -1 for all datasets.
        :type top_n: int
        
        :return: List of dataset file paths
        :rtype: List[Path]
        """
        datasets = list(self.real_op_dir.glob("*.csv"))
        
        # Always score datasets to get the best ones
        scored_datasets = self.__score_real_op_datasets(datasets, node_filter)
        # Sort by score (descending)
        scored_datasets.sort(key=lambda x: x[1], reverse=True)
        
        # Remove all datasets with score 0
        scored_datasets = [ds for ds in scored_datasets if ds[1] > 0]
        
        if not scored_datasets:
            logger.warning("No real operational datasets found with non-zero score")
            return []
        
        # Take top_n datasets (or all if top_n == -1)
        if top_n == -1:
            selected_datasets = [path for path, score in scored_datasets]
        else:
            selected_datasets = [path for path, score in scored_datasets[:top_n]]
        
        logger.info(f"Found {len(selected_datasets)} real operational CSV datasets for enrichment")
        return selected_datasets
    
    
    def __filter_csv_datasets_by_nodes(self,
                                       datasets: List[Path], 
                                       node_filter: List[str],
                                       min_coverage: float = 0.5) -> List[Path]:
        """Filter CSV datasets by checking if they contain ALL specified nodes.
        
        :param datasets: List of CSV dataset paths
        :type datasets: List[Path]
        :param node_filter: List of node names to filter by
        :type node_filter: List[str]
        :param min_coverage: Minimum percentage of nodes that must be present (0.0 to 1.0)
        :type min_coverage: float
        
        :return: Filtered list of dataset paths that contain ALL specified nodes
        :rtype: List[Path]
        """
        filtered_datasets = []
        min_required_nodes = max(1, int(len(node_filter) * min_coverage))
        
        for dataset_path in datasets:
            try:
                df = pd.read_csv(dataset_path, nrows=100)
                if 'node' in df.columns:
                    available_nodes = set(df['node'].unique())
                    matching_nodes = [node for node in node_filter if node in available_nodes]
                    
                    if len(matching_nodes) >= min_required_nodes:
                        filtered_datasets.append(dataset_path)
                        logger.info(f"Dataset {dataset_path.name}: {len(matching_nodes)}/{len(node_filter)} nodes matched")
                            
            except Exception as e:
                logger.warning(f"Error processing {dataset_path}: {e}")
                continue
        
        return filtered_datasets
    
    
    def __score_real_op_datasets(self, datasets: List[Path], 
                               node_filter: Optional[List[str]] = None) -> List[Tuple[Path, float]]:
        """Score real operational datasets based on node coverage AND data variability.
        
        :param datasets: List of dataset paths
        :type datasets: List[Path]
        :param node_filter: Optional list of node names to score by
        :type node_filter: Optional[List[str]]
        
        :return: List of tuples (dataset_path, score) where score is calculated as:
            - With node_filter: matching_nodes_count x variability_rate
            - Without node_filter: total_nodes_count x variability_rate
        :rtype: List[Tuple[Path, float]]
        """
        scored_datasets = []
        
        for dataset_path in datasets:
            try:
                # Load full dataset to analyze variability
                df = pd.read_csv(dataset_path)
                if 'node' in df.columns and 'value' in df.columns:
                    available_nodes = set(df['node'].unique())
                    
                    if node_filter:
                        # With node filter: score based on matching nodes
                        matching_nodes = [node for node in node_filter if node in available_nodes]
                        nodes_to_analyze = matching_nodes
                        node_count = len(matching_nodes)
                    else:
                        # Without node filter: score based on all nodes
                        nodes_to_analyze = list(available_nodes)
                        node_count = len(available_nodes)
                    
                    # Calculate variability rate across all relevant nodes
                    total_changes = 0
                    total_records = 0
                    
                    for node in nodes_to_analyze:
                        node_data = df[df['node'] == node].copy()
                        if len(node_data) > 1:
                            # Sort by time if available, otherwise by index
                            if 'time_s' in node_data.columns:
                                node_data = node_data.sort_values('time_s')
                            
                            values = node_data['value'].dropna()
                            if len(values) > 1:
                                # Count how many times the value changes
                                changes = (values != values.shift()).sum() - 1  # -1 because first value can't be a change
                                total_changes += max(changes, 0)
                                total_records += len(values)
                    
                    # Calculate variability rate (changes per record)
                    variability_rate = total_changes / max(total_records, 1) if total_records > 0 else 0
                    
                    # Simple scoring: node_count × variability_rate
                    # This gives 0 if no relevant nodes, higher scores for more nodes + more variability
                    total_score = node_count * variability_rate
                    
                    scored_datasets.append((dataset_path, total_score))
                else:
                    scored_datasets.append((dataset_path, 0.0))
                    
            except Exception as e:
                logger.warning(f"Error scoring {dataset_path}: {e}")
                scored_datasets.append((dataset_path, 0.0))
        
        return scored_datasets
    
    
    def select_datasets(self, 
                       include_dig_twin: bool = True,
                       include_real_op: bool = True,
                       dig_twin_subdirs: Optional[List[str]] = None,
                       node_filter: Optional[List[str]] = None,
                       real_op_top_n: int = 20,
                       prune_dig_twin_by_nodes: bool = False,
                       prune_real_op_by_nodes: bool = False) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Main method to select relevant CSV datasets and return them as individual DataFrames.
        
        :param include_dig_twin: Whether to include digital twin data (main learning data)
        :type include_dig_twin: bool
        :param include_real_op: Whether to include real operational data (enrichment data)
        :type include_real_op: bool
        :param dig_twin_subdirs: Specific digital twin subdirectories to include
        :type dig_twin_subdirs: Optional[List[str]]
        :param node_filter: List of nodes - ALL must be present in dig_twin datasets. If None, nodes from dig_twin datasets will be used for real_op scoring.
        :type node_filter: Optional[List[str]]
        :param real_op_top_n: Number of top real-op datasets to include for enrichment
        :type real_op_top_n: int
        :param prune_dig_twin_by_nodes: If True, remove all nodes from dig_twin datasets that are not in node_filter
        :type prune_dig_twin_by_nodes: bool
        :param prune_real_op_by_nodes: If True, remove all nodes from real_op datasets that are not in node_filter
        :type prune_real_op_by_nodes: bool
        
        :return: Tuple of (dig_twin_datasets, real_op_datasets) - two separate lists of DataFrames
        :rtype: Tuple[List[pd.DataFrame], List[pd.DataFrame]]
        """
        dig_twin_datasets = []
        real_op_datasets = []
        
        # Collect and load digital twin datasets (main learning data) FIRST
        if include_dig_twin:
            dig_twin_paths = self.find_dig_twin_datasets(
                subdirs=dig_twin_subdirs, 
                node_filter=node_filter
            )
            # Filter for CSV files only
            dig_twin_csv_paths = [p for p in dig_twin_paths if p.suffix == '.csv']
            
            for dataset_path in dig_twin_csv_paths:
                try:
                    df = pd.read_csv(dataset_path)
                    
                    # Apply node filter if specified
                    if node_filter and 'node' in df.columns:
                        if prune_dig_twin_by_nodes:
                            # Keep only rows with nodes that are in node_filter
                            df = df[df['node'].isin(node_filter)]
                        else:
                            # Check if ALL nodes are present (this is already done in filtering, but double-check)
                            available_nodes = set(df['node'].unique())
                            if not all(node in available_nodes for node in node_filter):
                                logger.warning(f"Dig twin dataset {dataset_path} doesn't contain all required nodes, skipping")
                                continue
                    
                    # Add metadata about the source
                    df.attrs['source_file'] = str(dataset_path)
                    df.attrs['source_type'] = 'dig_twin'
                    
                    if not df.empty:
                        dig_twin_datasets.append(df)
                        
                except Exception as e:
                    logger.warning(f"Skipping dig twin dataset {dataset_path}: {e}")
                    continue
        
        # Determine nodes for real_op scoring
        real_op_node_filter = node_filter
        if not node_filter and dig_twin_datasets:
            # Extract common nodes from dig_twin datasets if no explicit node_filter is given
            all_dig_twin_nodes = set()
            for df in dig_twin_datasets:
                if 'node' in df.columns:
                    all_dig_twin_nodes.update(df['node'].unique())
            
            real_op_node_filter = list(all_dig_twin_nodes)
            logger.info(f"No node_filter provided. Using {len(real_op_node_filter)} nodes from dig_twin datasets for real_op scoring.")
        
        # Collect and load real operational datasets (enrichment data)
        if include_real_op:
            real_op_paths = self.find_real_op_datasets(
                node_filter=real_op_node_filter, 
                top_n=real_op_top_n
            )
            
            for dataset_path in real_op_paths:
                try:
                    df = pd.read_csv(dataset_path)
                    
                    # Apply node filter if specified (but don't require ALL nodes for real_op)
                    if real_op_node_filter and 'node' in df.columns and prune_real_op_by_nodes:
                        # Keep only rows with nodes that are in the filter
                        df = df[df['node'].isin(real_op_node_filter)]
                    
                    # Add metadata about the source
                    df.attrs['source_file'] = str(dataset_path)
                    df.attrs['source_type'] = 'real_op'
                    
                    if not df.empty:
                        real_op_datasets.append(df)
                        
                except Exception as e:
                    logger.warning(f"Skipping real op dataset {dataset_path}: {e}")
                    continue
        
        logger.info(f"Successfully loaded {len(dig_twin_datasets)} dig_twin datasets and {len(real_op_datasets)} real_op datasets")
        return dig_twin_datasets, real_op_datasets
    
    
    def get_available_nodes(self, dataset_paths: Optional[List[Path]] = None) -> Dict[str, int]:
        """Get a summary of available nodes across CSV datasets.
        
        :param dataset_paths: Optional list of specific dataset paths to analyze
        :type dataset_paths: Optional[List[Path]]
        
        :return: Dictionary with node names as keys and frequencies as values
        :rtype: Dict[str, int]
        """
        if dataset_paths is None:
            # Get all available datasets
            dataset_paths = []
            dataset_paths.extend(self.find_dig_twin_datasets())
            dataset_paths.extend(self.find_real_op_datasets())
        
        node_counter = Counter()
        
        for dataset_path in dataset_paths:
            try:
                # Only process CSV files
                if dataset_path.suffix == '.csv':
                    df = pd.read_csv(dataset_path, nrows=100)
                    if 'node' in df.columns:
                        nodes = df['node'].unique()
                        node_counter.update(nodes)
                    
            except Exception as e:
                logger.warning(f"Error analyzing {dataset_path}: {e}")
                continue
        
        return dict(node_counter)


def print_selected_dataset_stats(datasets: List[pd.DataFrame], dataset_type: str = "datasets"):
    """Print statistics about selected datasets.
    
    :param datasets: List of pandas DataFrames representing the datasets
    :type datasets: List[pd.DataFrame]
    :param dataset_type: Type of datasets for display (e.g., "Dig Twin", "Real Op")
    :type dataset_type: str
    """
    print(f"\n{dataset_type} Datasets: {len(datasets)} datasets")
    if datasets:
        node_counts = [df['node'].nunique() if 'node' in df.columns else 0 for df in datasets]
        if node_counts:
            print(f"  Node counts ranging from {min(node_counts)} to {max(node_counts)}")
        
        for i, df in enumerate(datasets[:3]):  # Show first 3
            source_file = Path(df.attrs.get('source_file', 'unknown')).name
            unique_nodes = df['node'].nunique() if 'node' in df.columns else 0
            print(f"    Dataset {i+1}: {df.shape} - File: {source_file} - Nodes: {unique_nodes}")
        
        if len(datasets) > 3:
            print(f"    ... and {len(datasets) - 3} more datasets")
        



if __name__ == "__main__":
    # Initialize selector
    selector = DatasetSelector(dig_twin_dir=f"{project_path}/data/dig_twin",
                              real_op_dir=f"{project_path}/data/real_op")
            
    # Example 1: Select CSV datasets with specific nodes
    print("\n\n=== Example 1: Select CSV datasets with node filtering ===\n")
    selected_nodes = [
        "HP_Pump_isOff",
        "Hyd_A_700202",
        "Hyd_IsEnabled",
        "Hyd_Pressure",
        "Hyd_Pump_On",
    ]
    
    dig_twin_datasets, real_op_datasets = selector.select_datasets(
        # Dig Twin
        include_dig_twin=True,
        dig_twin_subdirs=["exp_hydraulics"],
        # Real Op
        include_real_op=True,
        real_op_top_n=-1,
        # Node filtering
        node_filter=selected_nodes, 
        prune_dig_twin_by_nodes=True,
        prune_real_op_by_nodes=True
    )
    
    # Use the function to print detailed stats
    print_selected_dataset_stats(dig_twin_datasets, "Dig Twin")
    print_selected_dataset_stats(real_op_datasets, "Real Op")
    
    
    # Example 2: Only dig_twin data from specific experiment
    #print("\n\n=== Example 2: Only Digital Twin Data from specific experiment ===\n")
    #dig_twin_only, _ = selector.select_datasets(
    #    # Dig Twin
    #    dig_twin_subdirs=['exp_cool_lubricant'],  # Specific experiment
    #    include_dig_twin=True,
    #    # Real Op
    #    include_real_op=False,  # No enrichment
    #    # Node filtering
    #    node_filter=None,  # Apply same node filter
    #    prune_dig_twin_by_nodes=True,
    #    prune_real_op_by_nodes=False  # Not needed since no real_op data
    #) 
    #print_selected_dataset_stats(dig_twin_only, "Dig Twin Only")
    


