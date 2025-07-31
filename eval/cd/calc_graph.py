

import sys
from pathlib import Path
project_path = Path(__file__).parents[2]  # Go up 2 levels from eval/cd/ to causRCA/
sys.path.insert(0, str(project_path))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import logging
import time

from data.select_datasets import DatasetSelector, print_selected_dataset_stats
from causrca.cd_models.cd_models import PCModel, FCIModel, FGESModel, PCMCIModel, CausalDiscoveryModel
from causrca.utils.utils import seed_everything
from causrca.utils.discretize import __discretize_dataset_helper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class CausalGraphLearner:
    
    def __init__(self, model: CausalDiscoveryModel):
        """Initialize the CausalGraphLearner with a specific causal discovery model.
        
        :param model: An instance of a causal discovery model (e.g., PCModel, FCIModel, etc.)
        :type model: CausalDiscoveryModel
        """
        self.model = model
        self.adj_matrix = None

    def fit(self, datasets: List[pd.DataFrame], approach: str = 'majority', threshold: float = 0.5) -> None:
        """Fit the model to the provided datasets using the specified approach.
        
        :param datasets: List of pandas DataFrames representing the datasets
        :type datasets: List[pd.DataFrame]
        :param approach: The approach to use for causal discovery ('majority' or 'concat')
        :type approach: str
        :param threshold: Threshold for majority voting (default is 0.5)
        :type threshold: float
        """
        start = time.time()
        if approach == 'majority':
            self.adj_matrix = self.__learn_by_majority_voting(datasets, threshold=threshold)
        elif approach == 'concat':
            self.adj_matrix = self.__learn_by_concatenation(datasets)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        end = time.time()
        logger.info(f"Learning completed in {time.strftime('%H:%M:%S', time.gmtime(end - start))}")  
    
    def __learn_by_majority_voting(self, datasets: List[pd.DataFrame], threshold: float = 0.5) -> pd.DataFrame:
        """Perform majority voting to learn the causal graph from multiple datasets.
        
        :param datasets: List of pandas DataFrames representing the datasets
        :type datasets: List[pd.DataFrame]
        :param threshold: Threshold for majority voting
        :type threshold: float
        :return: Adjacency matrix with node names as index and columns
        :rtype: pd.DataFrame
        """
        adj_matrices = []
        all_node_names = set()
        all_edges = set()
        
        # Iterate through each dataset and collect adjacency matrices
        for i, dataset in enumerate(datasets):
            logger.info(f"{self.model.__class__.__name__} - {i+1}/{len(datasets)}")
            adj_matrix = self.model.learn(dataset)
            adj_matrices.append(adj_matrix)
            
            # Collect all node names
            all_node_names.update(adj_matrix.index.tolist())
            
            # Collect all possible edges from this adjacency matrix
            for cause in adj_matrix.index:
                for effect in adj_matrix.columns:
                    if cause != effect:  # No self-loops
                        all_edges.add((cause, effect))

        
        # Convert to sorted lists for consistent ordering
        all_node_names = sorted(list(all_node_names))
        all_edges = sorted(list(all_edges))
                
        # Create final adjacency matrix with majority voting
        final_adj_matrix = np.zeros((len(all_node_names), len(all_node_names)))
        
        # For each possible edge, perform majority voting
        for cause, effect in all_edges:
            edge_votes = []
            
            # Check this edge in all adjacency matrices where both nodes exist
            for adj_matrix in adj_matrices:
                if cause in adj_matrix.index and effect in adj_matrix.columns:
                    edge_value = adj_matrix.loc[cause, effect]
                    edge_votes.append(edge_value)
            
            # Perform majority voting if we have votes
            if edge_votes:
                # Calculate the proportion of matrices that have this edge
                edge_proportion = sum(edge_votes) / len(edge_votes)
                
                # Get indices for the final matrix
                cause_idx = all_node_names.index(cause)
                effect_idx = all_node_names.index(effect)
                
                # Set the edge value as the proportion (0.0 to 1.0)
                final_adj_matrix[cause_idx, effect_idx] = edge_proportion
        
        # Apply threshold to final adjacency matrix
        final_adj_matrix[final_adj_matrix < threshold] = 0.0
        final_adj_matrix[final_adj_matrix >= threshold] = 1.0       
        
        # Create and return pandas DataFrame with node names as index and columns
        adj_matrix = pd.DataFrame(final_adj_matrix, index=all_node_names, columns=all_node_names)
        # only integer values (0 or 1) in the adjacency matrix
        adj_matrix = adj_matrix.astype(int)
        return adj_matrix
        
        
    
    def __learn_by_concatenation(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate multiple datasets and learn the causal graph.
        
        :param datasets: List of pandas DataFrames representing the datasets
        :type datasets: List[pd.DataFrame]
        :return: Adjacency matrix and node names
        :rtype: Tuple[np.ndarray, List[str]]
        """
        # Collect all columns (nodes) from all datasets
        all_node_names = set()
        for dataset in datasets:
            all_node_names.update(dataset.columns.tolist())
        
        # Sort columns to ensure consistent ordering
        all_node_names = sorted(list(all_node_names))
        
        # Prepare and concatenate datasets
        prepared_datasets = []
        buffer_steps = 10
        
        for i, dataset in enumerate(datasets):
            # Create new dataframe with all columns
            prepared_dataset = pd.DataFrame(index=dataset.index, columns=all_node_names)
            
            # Copy existing values
            for col in dataset.columns:
                prepared_dataset[col] = dataset[col]
            
            # Fill NaN values with 0.0
            prepared_dataset = prepared_dataset.fillna(0.0)
            
            # Add the prepared dataset to the list
            prepared_datasets.append(prepared_dataset)
            logger.info(f"Dataset {i+1}/{len(datasets)}: {dataset.shape[0]} rows, expanded to {len(all_node_names)} columns")
            
            # Add buffer rows between datasets (except after the last dataset)
            if i < len(datasets) - 1:
                # Get the last row values to use as buffer
                last_row_values = prepared_dataset.iloc[-1].values
                
                # Create buffer dataframe with constant values
                buffer_data = np.tile(last_row_values, (buffer_steps, 1))
                buffer_df = pd.DataFrame(buffer_data, columns=all_node_names)
                
                # Add buffer to the list
                prepared_datasets.append(buffer_df)
        
        # Concatenate all prepared datasets
        concatenated_dataset = pd.concat(prepared_datasets, ignore_index=True)
        total_rows = concatenated_dataset.shape[0]
        logger.info(f"Concatenated dataset: {len(all_node_names)} nodes with {total_rows} rows.")

        # Learn the causal graph using the model
        logger.info(f"Start learning causal graph with {self.model.__class__.__name__}")
        adj_matrix = self.model.learn(concatenated_dataset)
                
        # create and return pandas DataFrame with node names as index and columns
        adj_matrix = pd.DataFrame(adj_matrix, index=all_node_names, columns=all_node_names)
         # only integer values (0 or 1) in the adjacency matrix
        adj_matrix = adj_matrix.astype(int)
        return adj_matrix

