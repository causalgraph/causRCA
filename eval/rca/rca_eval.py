#%%
import sys, os
from pathlib import Path
project_path = Path(__file__).parents[2]
sys.path.insert(0, str(project_path))

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import networkx as nx

import eval.rca.helpers_rca_eval as helpers

from causrca.rca_models.supervised_rca_models import BaselineSupervisedRCA, SupervisedRCAModel, LogisticRegressionRCA, CausalPrioLogisticRegressionRCA
from causrca.rca_models.unsupervised_rca_models import  CausalPrioTimeRecencyRCA, TimeRecency_BaselineUnsupervisedRCA, PageRankRCA, RandomWalkRCA, Baro
from causrca.utils.utils import seed_everything, get_encoding_dict

# Set random seed for reproducibility
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)  # Set a random seed for reproducibility

# CONFIG PATHS for Datasets
DIG_TWIN_DS_PATH = Path(project_path, "data", "dig_twin")
PROBE_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_probe")
COOLANT_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_coolant")
HYDRAULICS_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_hydraulics")

# Full expert graph path (used when evaluating in 'full' mode)
FULL_EXPERT_GRAPH_PATH = Path(project_path, "data", "expert_graph")

# Load Categorical Encoding Dictionary ones at startup
CATEGORICAL_ENCODING_DICT = get_encoding_dict()

def evaluate_supervised_rca_model_with_k_folds(model: SupervisedRCAModel, path: str, k_folds: int = 5, mode: str = 'full'):
    """
    Evaluate a supervised RCA model using stratified k-fold cross-validation.

    :param model: A SupervisedRCAModel instance.
    :type model: SupervisedRCAModel
    :param path: Path to the dataset directory.
    :type path: str
    :param k_folds: Number of folds for cross-validation.
    :type k_folds: int
    :param mode: Evaluation mode - 'full' uses all nodes, 'sub' uses only relevant nodes.
    :type mode: str
    :return: Dictionary containing evaluation metrics.
    :rtype: dict
    """
    # Get dataset information
    csv_path_to_truth_data, rel_nodes = helpers.load_datasets_truth_data_and_relevant_nodes(path)
    # Use relevant nodes only in 'sub' mode
    if mode != 'sub':
        rel_nodes = None
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Metrics for each fold
    fold_metrics = []
    all_predictions = []

    # Prepare data for stratified k-fold
    all_csv_paths = list(csv_path_to_truth_data.keys())
    # Use only the first diagnosis for stratification
    all_diagnoses = [data["diagnoses"][0]["id"] for data in csv_path_to_truth_data.values() if data.get("diagnoses")]
    
    # Run k-fold cross-validation
    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(all_csv_paths, all_diagnoses)):
        print(f"\n#### Evaluating Fold {fold_idx+1}/{k_folds} ####")
        
        # Split data into train and test
        train_paths = [all_csv_paths[i] for i in train_indices]
        test_paths = [all_csv_paths[i] for i in test_indices]
        
        train_path_to_truth = {path: csv_path_to_truth_data[path] for path in train_paths}
        
        # Train the model
        print(f"  |-- Training on {len(train_paths)} runs...")
        model.train(
            relevant_nodes=rel_nodes,
            csv_path_to_truth_data=train_path_to_truth
        )
        
        # Evaluate on test set
        print(f"  |-- Testing on {len(test_paths)} runs...")
        
        correct_at_1 = 0
        correct_at_2 = 0
        
        # For AP@K and MAP@K calculation
        y_true_list = []
        y_pred_list = []
        
        for test_path in test_paths:
            true_data = csv_path_to_truth_data[test_path]
            diagnosis_time = true_data["run_data"]["diagnosis_at"]
            # Get all ground truth diagnoses for the run
            ground_truth = [d["id"] for d in true_data.get("diagnoses", [])]
            
            prediction = model.predict(test_path, diagnosis_time)
            
            # Record prediction details
            all_predictions.append({
                "fold": fold_idx + 1,
                "run": os.path.basename(test_path),
                "true_diagnosis": ground_truth,
                "predicted_order": prediction
            })
            
            # Store for MAP@K calculation
            y_true_list.append(ground_truth)
            y_pred_list.append(prediction)
            
            # Pass the list of ground truths directly
            correct_in_k = helpers.get_in_k_counters(ground_truth, prediction, max_k=2)
            correct_at_1 += correct_in_k["in_1"]
            correct_at_2 += correct_in_k["in_2"]
        
        # Calculate Precision@K metrics for this fold
        prec_at_1 = correct_at_1 / len(test_paths) if test_paths else 0
        prec_at_2 = correct_at_2 / len(test_paths) if test_paths else 0
        
        # Calculate MAP@K for this fold
        map_at_1 = helpers.calculate_mapk(y_true_list, y_pred_list, k=1)
        map_at_2 = helpers.calculate_mapk(y_true_list, y_pred_list, k=2)
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "prec_at_1": prec_at_1,
            "prec_at_2": prec_at_2,
            "map_at_1": map_at_1,
            "map_at_2": map_at_2,
            "test_size": len(test_paths)
        })
        
        print(f"Fold {fold_idx+1} metrics - Prec@1: {prec_at_1:.4f}, "
              f"Prec@2: {prec_at_2:.4f}, MAP@1: {map_at_1:.4f}, "
              f"MAP@2: {map_at_2:.4f}")
    
    # Calculate overall metrics
    avg_metrics = {
        "prec_at_1": np.mean([m["prec_at_1"] for m in fold_metrics]),
        "prec_at_2": np.mean([m["prec_at_2"] for m in fold_metrics]),
        "map_at_1": np.mean([m["map_at_1"] for m in fold_metrics]),
        "map_at_2": np.mean([m["map_at_2"] for m in fold_metrics])
    }
    
    # Create results table
    results_df = pd.DataFrame(fold_metrics)
    
    # Add average row
    avg_row = pd.DataFrame([{
        "fold": "Average",
        "prec_at_1": avg_metrics["prec_at_1"],
        "prec_at_2": avg_metrics["prec_at_2"],
        "map_at_1": avg_metrics["map_at_1"],
        "map_at_2": avg_metrics["map_at_2"],
        "test_size": np.mean([m["test_size"] for m in fold_metrics])
    }])
    
    results_table = pd.concat([results_df, avg_row])
    
    return {
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
        "results_table": results_table,
        "all_predictions": all_predictions
    }

def evaluate_unsupervised_rca_model(model, path: str, mode: str = 'full'):
    """
    Evaluate an unsupervised RCA model.

    :param model: An UnsupervisedRCAModel instance.
    :type model: object
    :param path: Path to the dataset directory.
    :type path: str
    :param mode: Evaluation mode - 'full' uses all nodes, 'sub' uses only relevant nodes.
    :type mode: str
    :return: Dictionary containing evaluation metrics.
    :rtype: dict
    """
    # Get dataset information
    csv_path_to_truth_data, rel_nodes = helpers.load_datasets_truth_data_and_relevant_nodes(path)
    # Use relevant nodes only in 'sub' mode
    if mode != 'sub':
        rel_nodes = None
    
    # Metrics
    all_predictions = []
    correct_at_1 = 0
    correct_at_3 = 0
    correct_at_5 = 0
    
    # For AP@K and MAP@K calculation
    y_true_list = []
    y_pred_list = []
    
    # Evaluate on all runs
    print(f"Evaluating {len(csv_path_to_truth_data)} runs...")
    
    for csv_path, true_data in csv_path_to_truth_data.items():
        diagnosis_time = true_data["run_data"]["diagnosis_at"]
        ground_truth_vars = true_data["manipulatedVars"]
        
        # Make prediction
        prediction = model.predict(
            dataset_csv_path=csv_path,
            diagnosis_time=diagnosis_time,
            relevant_nodes=rel_nodes
        )

        # Ensure prediction is a list and not empty
        if not isinstance(prediction, list) or not prediction:
            continue
        
        # Record prediction details
        all_predictions.append({
            "run": os.path.basename(csv_path),
            "true_manipulated_vars": ground_truth_vars,
            "predicted_order": prediction
        })
        
        # Store for MAP@K calculation
        y_true_list.append(ground_truth_vars)
        y_pred_list.append(prediction)
        
        # Get correct in k counters for the current prediction
        correct_in_k = helpers.get_in_k_counters(
            ground_truth_vars, prediction, max_k=5)
        correct_at_1 += correct_in_k["in_1"]
        correct_at_3 += correct_in_k["in_3"]
        correct_at_5 += correct_in_k["in_5"]

        # NOTE For debugging -> Print true and predicted values
        # print(f"Run: {os.path.basename(csv_path)} | "
        #       f"Correct@1: {correct_at_1}, Correct@3: {correct_at_3}, "
        #       f"Correct@5: {correct_at_5} | True: {ground_truth_vars} | "
        #       f"Predicted: {prediction[:n_true_vars+4]}")
            
    
    # Calculate Precision@K
    total_runs = len(csv_path_to_truth_data)
    prec_at_1 = correct_at_1 / total_runs if total_runs else 0
    prec_at_3 = correct_at_3 / total_runs if total_runs else 0
    prec_at_5 = correct_at_5 / total_runs if total_runs else 0
    
    # Calculate MAP@K
    map_at_1 = helpers.calculate_mapk(y_true_list, y_pred_list, k=1)
    map_at_3 = helpers.calculate_mapk(y_true_list, y_pred_list, k=3)
    map_at_5 = helpers.calculate_mapk(y_true_list, y_pred_list, k=5)
    
    return {
        "prec_at_1": prec_at_1,
        "prec_at_3": prec_at_3,
        "prec_at_5": prec_at_5,
        "map_at_1": map_at_1,
        "map_at_3": map_at_3,
        "map_at_5": map_at_5,
        "all_predictions": all_predictions
    }


def update_results_dataframe(
    results_df,
    model_name,
    dataset_name,
    suffix,
    last_result,
    k_fold_evaluation=True,
    in_k_to_track=[1, 2],
    include_map=True
):
    """
    Update results DataFrame with evaluation metrics.

    :param results_df: DataFrame to update.
    :type results_df: pd.DataFrame
    :param model_name: Name of the model.
    :type model_name: str
    :param dataset_name: Name of the dataset.
    :type dataset_name: str
    :param suffix: Suffix to add to column names (e.g. "-only_rel").
    :type suffix: str
    :param last_result: Result dictionary from evaluation.
    :type last_result: dict
    :param k_fold_evaluation: Whether this is a k-fold evaluation (supervised model).
    :type k_fold_evaluation: bool
    :param in_k_to_track: List of k values for Prec@k metrics to track.
    :type in_k_to_track: list
    :param include_map: Whether to include MAP@K metrics.
    :type include_map: bool
    :return: Updated DataFrame.
    :rtype: pd.DataFrame
    """
    # Add or update results in the DataFrame
    if model_name not in results_df["model"].values:
        new_row = pd.DataFrame([{"model": model_name}])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Find the row for this model
    row_idx = results_df.index[results_df["model"] == model_name].tolist()[0]
    
    # Add Precision@K
    for k in in_k_to_track:
        col_name = f"{dataset_name}{suffix}_Prec@{k}"
        metric_key = f"prec_at_{k}"
        
        # Get metric value from appropriate location in result dict
        if k_fold_evaluation:
            metric_value = last_result["avg_metrics"][metric_key]
        else:
            metric_value = last_result[metric_key]
            
        results_df.at[row_idx, col_name] = metric_value
    
    # Add MAP@K metrics if requested
    if include_map:
        for k in in_k_to_track:
            col_name = f"{dataset_name}{suffix}_MAP@{k}"
            metric_key = f"map_at_{k}"
            
            # Get metric value from appropriate location in result dict
            if k_fold_evaluation:
                metric_value = last_result["avg_metrics"][metric_key]
            else:
                metric_value = last_result[metric_key]
                
            results_df.at[row_idx, col_name] = metric_value
    
    return results_df


def find_and_load_gml_file(path) -> nx.DiGraph:
    """
    Find a .gml file in the specified path and load it as a NetworkX directed graph.

    :param path: Path to search for .gml files.
    :type path: str or Path
    :return: The loaded graph.
    :rtype: nx.DiGraph
    :raises FileNotFoundError: If no .gml file is found in the path.
    :raises ValueError: If multiple .gml files are found in the path.
    """
    gml_files = list(Path(path).glob("*.gml"))
    if not gml_files:
        raise FileNotFoundError(f"No causal graph in .gml file found in {path}")
    if len(gml_files) > 1:
        raise ValueError(f"Multiple .gml files found in {path}. Please ensure only one file is present.")
    
    # Use the first .gml file found
    graph_path = gml_files[0]
    print(f"Loading causal graph from: {graph_path}")
    
    # Load graph from file
    graph = nx.read_gml(graph_path)
    
    return graph
    


if __name__ == "__main__":
    #%%
    #############################################################
    #          SUPERVISED RCA EVALUATION WITH K-FOLDS           #
    #############################################################

    # NOTE Supervised models train on labeled data and predicts labeled "diagnosis"

    # Create DataFrames to store results of supervised models
    supervised_results = pd.DataFrame(columns=["model"])

    # Supervised models evaluation with results tracking
    for path in [PROBE_DS_PATH, COOLANT_DS_PATH, HYDRAULICS_DS_PATH]:
        # Load the graph, set name and init models with graph
        dataset_name = os.path.basename(path)
        for mode in ['sub', 'full']:
            if mode == 'full':
                print("Loading full expert graph for 'full' mode evaluation.")
                graph_in_path = find_and_load_gml_file(FULL_EXPERT_GRAPH_PATH)
            else:
                print("Loading dataset-specific graph for 'sub' mode evaluation.")
                graph_in_path = find_and_load_gml_file(path)
            # Init models with respective graph
            supervised_models = [
                BaselineSupervisedRCA(),
                LogisticRegressionRCA(),
                CausalPrioLogisticRegressionRCA(causal_graph=graph_in_path)]
            # Evaluate each supervised model
            suffix = "-sub" if mode == 'sub' else ""
            for supervised_model in supervised_models:
                model_name = supervised_model.__class__.__name__
                print(f"\nEvaluating RCA model '{model_name}' on {dataset_name} with mode={mode}:\n")
                current_result = evaluate_supervised_rca_model_with_k_folds(
                    model=supervised_model,
                    path=path,
                    k_folds=3,
                    mode=mode
                )
                # Update results dataframe
                supervised_results = update_results_dataframe(
                    results_df=supervised_results,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    suffix=suffix,
                    last_result=current_result,
                    k_fold_evaluation=True,  # Supervised models use k-fold eval
                    in_k_to_track=[1, 2],    # Track Prec@1 and Prec@2
                    include_map=True         # Include MAP@K metrics
                )
                print(f"\n### Evaluation Results for '{model_name}' on {dataset_name} with mode={mode} ###")
                print(current_result["results_table"])

    # Save supervised results
    supervised_results_path = Path(project_path, "eval", "rca", "results", "supervised_results.csv")
    os.makedirs(os.path.dirname(supervised_results_path), exist_ok=True)
    supervised_results.to_csv(supervised_results_path, index=False)
    print(f"\nSupervised results saved to {supervised_results_path}")

    # %%
    #############################################################
    #          UNSUPERVISED RCA EVALUATION                      #
    #############################################################

    # NOTE Unsupervised models try to directly identify anomalous data and
    # are compared to "manipulated variables" in ground truth

    # Create DataFrames to store results of unsupervised models
    unsupervised_results = pd.DataFrame(columns=["model"])

    # Unsupervised models evaluation with results tracking
    for path in [PROBE_DS_PATH, COOLANT_DS_PATH, HYDRAULICS_DS_PATH]:
        # Load the graph, set name and init models with graph
        dataset_name = os.path.basename(path)
        for mode in ['sub', 'full']:
            if mode == 'full':
                print("Loading full expert graph for 'full' mode evaluation.")
                graph_in_path = find_and_load_gml_file(FULL_EXPERT_GRAPH_PATH)
            else:
                print("Loading dataset-specific graph for 'sub' mode evaluation.")
                graph_in_path = find_and_load_gml_file(path)
            # Init models with respective graph
            unsupervised_models = [
                TimeRecency_BaselineUnsupervisedRCA(),
                Baro(),
                CausalPrioTimeRecencyRCA(causal_graph=graph_in_path),
                RandomWalkRCA(causal_graph=graph_in_path),
                PageRankRCA(causal_graph=graph_in_path)
            ]
            # Evaluate each unsupervised model
            suffix = "-sub" if mode == 'sub' else ""
            for unsupervised_model in unsupervised_models:
                model_name = unsupervised_model.__class__.__name__
                print(f"\nEvaluating RCA model '{model_name}' on "
                      f"{dataset_name} with mode={mode}:\n")
                result = evaluate_unsupervised_rca_model(
                    model=unsupervised_model,
                    path=path,
                    mode=mode
                )
                # Update results dataframe
                unsupervised_results = update_results_dataframe(
                    results_df=unsupervised_results,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    suffix=suffix,
                    last_result=result,
                    k_fold_evaluation=False,  # Unsupervised models no k-fold
                    in_k_to_track=[1, 3, 5],   # Track Prec@1, Prec@3, Prec@5
                    include_map=True          # Include MAP@K metrics
                )
                msg = (f"\n### Evaluation Results for '{model_name}' on "
                       f"{dataset_name} with mode={mode} ###")
                print(msg)
                print(f"Prec@1: {result['prec_at_1']:.4f}, "
                      f"Prec@3: {result['prec_at_3']:.4f}, "
                      f"Prec@5: {result['prec_at_5']:.4f}")
                print(f"MAP@1: {result['map_at_1']:.4f}, "
                      f"MAP@3: {result['map_at_3']:.4f}, "
                      f"MAP@5: {result['map_at_5']:.4f}")

    # Save unsupervised results to CSV
    unsupervised_results_path = Path(project_path, "eval", "rca", "results",
                                     "unsupervised_results.csv")
    os.makedirs(os.path.dirname(unsupervised_results_path), exist_ok=True)
    unsupervised_results.to_csv(unsupervised_results_path, index=False)
    print(f"\nUnsupervised results saved to {unsupervised_results_path}")