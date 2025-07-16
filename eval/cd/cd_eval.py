import sys
from pathlib import Path
project_path = Path(__file__).parents[2]  # Go up 2 levels from eval/cd/ to causRCA/
sys.path.insert(0, str(project_path))

import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from castle.metrics import MetricsDAG
import networkx as nx
import time

from data.select_datasets import DatasetSelector
from causrca.cd_models.cd_models import PCModel, FCIModel, FGESModel, PCMCIModel
from causrca.utils.utils import seed_everything, get_encoding_dict
from calc_graph import CausalGraphLearner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Predefined dataset configurations
DATASET_CONFIGS = {
    'coolant': {
        'description': 'Coolant dataset with key nodes',
        'selected_nodes': [
            "CLF_A_700307",
            "CLF_Filter_Ok",
            "F_A_700311",
            "F_A_700313",
            "F_Filter_Ok",
            "F_Transport_Ok",
            "HP_A_700304",
            "HP_Pump_isOff",
            "HP_Pump_Ok",
            "LT_A_700317",
            "LT_Level_Ok",
            "LT_Pump_Ok",
            "LP_A_700301",
            "LP_Pump_Ok",
            "LP_Pump_On",
            "CLT_A_700309",
            "CLT_A_700310",
            "CLT_Level_lt_Min",
            "CLT_Level_gt_Max",
            "Lubr_On",
            "Lubr_P_Ok"
        ],
        'include_dig_twin': True,
        'include_real_op': True,
        'dig_twin_subdirs': ['exp_coolant'],
        'real_op_top_n': 10,
        'prune_dig_twin_by_nodes': True,
        'prune_real_op_by_nodes': True
    },
    'hydraulic': {
        'description': 'Hydraulic dataset with key nodes',
        'selected_nodes': [
            "Hyd_A_700202",
            "Hyd_IsEnabled",
            "Hyd_Pressure",
            "Hyd_Valve_P_Up",
            "Hyd_A_700207",
            "Hyd_Filter_Ok",
            "Hyd_A_700205",
            "Hyd_A_700206",
            "Hyd_Level_Ok",
            "Hyd_A_700208",
            "Hyd_Pump_isOff",
            "Hyd_Pump_Ok",
            "Hyd_Pump_On",
            "Hyd_A_700203",
            "Hyd_A_700204",
            "Hyd_Temp_lt_70",
            "Hyd_Temp_lt_80"
        ],
        'include_dig_twin': True,
        'include_real_op': True,
        'dig_twin_subdirs': ['exp_hydraulics'],
        'real_op_top_n': 10,
        'prune_dig_twin_by_nodes': True,
        'prune_real_op_by_nodes': True
    },
    'probe': {
        'description': 'Small probe dataset with key nodes',
        'selected_nodes': [
            "MP_Inactive",
            "MPA_A_701124",
            "MPA_A_701125",
            "MPA_InitPos",
            "MPA_toInitPos",
            "MPA_toWorkPos",
            "MPA_WorkPos",
            "MPC_close",
            "MPC_Closed",
            "MPC_isOpen",
            "MPC_open"
        ],
        'include_dig_twin': True,
        'include_real_op': True,
        'dig_twin_subdirs': ['exp_probe'],
        'real_op_top_n': 10,
        'prune_dig_twin_by_nodes': True,
        'prune_real_op_by_nodes': True
    },
    'full': {
        'description': 'Full dataset with all nodes',
        'selected_nodes': None,  # No filtering, use all nodes
        'include_dig_twin': True,
        'include_real_op': True,
        'dig_twin_subdirs': None,
        'real_op_top_n': 10,
        'prune_dig_twin_by_nodes': False,  # Do not prune by nodes
        'prune_real_op_by_nodes': False  # Do not prune by nodes
    }
}

# Predefined model configurations
MODEL_CONFIGS = {
    'pc_default': {
        'description': 'PC algorithm with default parameters',
        'model_class': PCModel,
        'params': {
            'alpha': 0.05,
            'indep_test': 'gsq', # ["fisherz", "gsq", "chisq", "mv_fisherz", "kci"]
            'stable': True
        },
        'threshold': 0.5  # Default threshold for majority voting
    },
    'fci_default': {
        'description': 'FCI algorithm with default parameters',
        'model_class': FCIModel,
        'params': {
            'alpha': 0.05,
            'independence_test_method': 'gsq' # ["fisherz", "gsq", "chisq", "mv_fisherz", "kci"]
        },
        'threshold': 0.5  # Default threshold for majority voting
    },
    'fges_default': {
        'description': 'FGES algorithm with default parameters',
        'model_class': FGESModel,
        'params': {
            'score_func': 'linear'
        },
        'threshold': 0.4  # Default threshold for majority voting
    },
    'pcmci_default': {
        'description': 'PCMCI algorithm with default parameters',
        'model_class': PCMCIModel,
        'params': {
            'tau_max': 1,
            'alpha': 0.01
        },
        'threshold': 0.6  # Default threshold for majority voting
    }
}


def load_datasets(dataset_config: Dict[str, Any]) -> List[pd.DataFrame]:
    """Load datasets based on configuration.
    
    :param dataset_config: Dataset configuration dictionary
    :type dataset_config: Dict[str, Any]
    :return: List of loaded datasets
    :rtype: List[pd.DataFrame]
    """
    logger.info(f"Loading datasets with configuration: {dataset_config['description']}")
    
    # Get datasets using DatasetSelector
    selector = DatasetSelector(
        dig_twin_dir=f"{project_path}/data/dig_twin",
        real_op_dir=f"{project_path}/data/real_op"
    )
    
    dig_twin_datasets, real_op_datasets = selector.select_datasets(
        include_dig_twin=dataset_config['include_dig_twin'],
        include_real_op=dataset_config['include_real_op'],
        dig_twin_subdirs=dataset_config['dig_twin_subdirs'],
        node_filter=dataset_config['selected_nodes'],
        real_op_top_n=dataset_config['real_op_top_n'],
        prune_dig_twin_by_nodes=dataset_config['prune_dig_twin_by_nodes'],
        prune_real_op_by_nodes=dataset_config['prune_real_op_by_nodes']
    )
    
    # Merge datasets together and shuffle them randomly
    all_datasets = dig_twin_datasets + real_op_datasets
    np.random.shuffle(all_datasets)
    
    logger.info(f"Loaded {len(all_datasets)} datasets total")
    return all_datasets



def create_model(model_config: Dict[str, Any]):
    """Create causal discovery model based on configuration.
    
    :param model_config: Model configuration dictionary
    :type model_config: Dict[str, Any]
    :return: Instantiated model
    """
    model_class = model_config['model_class']
    params = model_config['params']
    
    logger.info(f"Creating model: {model_config['description']}")
    logger.info(f"Parameters: {params}")
    
    return model_class(**params)


def run_causal_discovery(dataset_name: str, model_name: str, approach: str = 'majority_voting', 
                        threshold: float = 0.5, n_jobs: int = None) -> pd.DataFrame:
    """Run causal discovery with specified dataset and model configurations.
    
    :param dataset_name: Name of the dataset configuration
    :type dataset_name: str
    :param model_name: Name of the model configuration  
    :type model_name: str
    :param approach: Learning approach ('majority' or 'concat')
    :type approach: str
    :param threshold: Threshold for majority voting
    :type threshold: float
    :param n_jobs: Number of parallel jobs for discretization. If None, uses all available cores.
    :type n_jobs: int
    :return: Learned adjacency matrix
    :rtype: pd.DataFrame
    """
    # Set seed for reproducibility
    seed_everything(42)
    
    # Validate configurations
    if dataset_name not in DATASET_CONFIGS:
        available_datasets = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset configuration '{dataset_name}'. Available: {available_datasets}")
    
    if model_name not in MODEL_CONFIGS:
        available_models = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model configuration '{model_name}'. Available: {available_models}")
    
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    # Load encoding dictionary
    categorical_encoding_dict = get_encoding_dict()
    
    # Load datasets and create model
    dataset_config = DATASET_CONFIGS[dataset_name]
    raw_datasets = load_datasets(dataset_config)
    
    # Create model early to use its prepare_dataset method
    model_config = MODEL_CONFIGS[model_name]
    model = create_model(model_config)
    
    # Prepare arguments for parallel processing
    discretization_args = [
        (i, dataset, categorical_encoding_dict) 
        for i, dataset in enumerate(raw_datasets)
    ]
    
    # Parallel discretization using model's prepare_dataset method
    datasets_discretized = []
    logger.info(f"Starting parallel discretization of {len(raw_datasets)} datasets")
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs using model's prepare_dataset method
        future_to_index = {
            executor.submit(model.prepare_dataset, args): args[0] 
            for args in discretization_args
        }
    
        # Collect results as they complete
        results = {}
        completed_count = 0
        total_count = len(raw_datasets)
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            completed_count += 1
            
            try:
                result_index, discretized_data = future.result()
                results[result_index] = discretized_data
                
                if discretized_data is not None:
                    logger.info(f"Successfully prepared dataset {result_index + 1}/{total_count} - Progress: {completed_count}/{total_count}")
                else:
                    logger.warning(f"Dataset {result_index + 1} was invalid (<=1 columns) - Progress: {completed_count}/{total_count}")
                    
            except Exception as e:
                logger.error(f"Error processing dataset {index + 1}: {str(e)} - Progress: {completed_count}/{total_count}")
                results[index] = None
    
    # Sort results by index and filter out None values
    for i in sorted(results.keys()):
        if results[i] is not None:
            datasets_discretized.append(results[i])
        
    if len(datasets_discretized) == 0:
        raise ValueError("No valid datasets found after preparation.")

    logger.info(f"Successfully discretized {len(datasets_discretized)} datasets")
    
    # Setup CausalGraphLearner (model was already created earlier)
    cg_learner = CausalGraphLearner(model=model)
    
    # Learn causal graph
    logger.info(f"Starting causal discovery with approach: {approach}")
    if approach == 'majority_voting':
        logger.info(f"Using threshold: {threshold}")
    
    cg_learner.fit(datasets=datasets_discretized, approach=approach, threshold=threshold)
    adj_matrix = cg_learner.adj_matrix
        
    return adj_matrix


def main():
    """Main function for command-line interface."""

    # Define ground truth graph gml file
    TRUTH_GML = f"{project_path}/data/expert_graph/expert_graph.gml"
    truth_graph = nx.read_gml(TRUTH_GML)
    
    # define possible combination of models, datasets and approaches
    available_approaches = ['majority']
    available_datasets = list(DATASET_CONFIGS.keys())
    available_models = list(MODEL_CONFIGS.keys())
    
    # Setup causal discovery runs for every possible combination and store resulting metrics
    # Create list to store all evaluation results
    all_results = []
    
    # Run evaluation for all combinations
    for dataset_name in available_datasets:
        for model_name in available_models:
            for approach in available_approaches:
                logger.info(f"Running evaluation: {dataset_name} + {model_name} + {approach}")
                
                try:
                    # start timing
                    start_time = time.time()
                    
                    # Run causal discovery
                    adj_matrix = run_causal_discovery(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        approach=approach,
                        threshold=MODEL_CONFIGS[model_name].get('threshold', 0.5)
                    )
                    
                    # end timing
                    end_time = time.time()
                    computation_time = end_time - start_time
                    
                    # Safe graph as gml file
                    gml_graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.MultiDiGraph)
                    gml_file = f"{project_path}/eval/cd/results/{model_name}_{dataset_name}_{approach}.gml"
                    nx.write_gml(gml_graph, gml_file)
                    
                    # Create subgraph of truth graph with nodes in adj_matrix
                    truth_nodes = set(adj_matrix.columns)
                    truth_subgraph = truth_graph.subgraph(truth_nodes)
                    truth_adj_matrix = nx.to_pandas_adjacency(truth_subgraph).astype(int)
                    
                    # Reorder truth_adj_matrix to match order of adj_matrix
                    truth_adj_matrix = truth_adj_matrix.reindex(index=adj_matrix.index, columns=adj_matrix.columns)
                    
                    # Convert matrices to numpy arrays
                    adj_matrix_np = adj_matrix.to_numpy()
                    truth_adj_matrix_np = truth_adj_matrix.to_numpy()
                    
                    # Calculate metrics
                    mt = MetricsDAG(adj_matrix_np, truth_adj_matrix_np)
                    metrics = mt.metrics
                    
                    # Store results
                    result = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'approach': approach,
                        'runtime': computation_time,
                        **metrics
                    }
                    all_results.append(result)
                    
                    logger.info(f"Completed: {dataset_name} + {model_name} + {approach}")
                    logger.info(f"Computation time: {computation_time:.2f} seconds")
                    logger.info(f"Metrics: {metrics}")
                    
                except Exception as e:
                    logger.error(f"Error in combination {dataset_name} + {model_name} + {approach}: {str(e)}")
                    # Store failed result
                    result = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'approach': approach,
                        'error': str(e)
                    }
                    all_results.append(result)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_file = f"{project_path}/eval/cd/results/evaluation_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Evaluation results saved to: {results_file}")

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(results_df.to_string(index=False))
    


if __name__ == "__main__":
    
    exit(main())

    # SCREENING for background exec
    
    # screen -S cd_eval
    # Crtl + A + D to detach
    # screen -ls to list screens
    # screen -r cd_eval to reattach
    