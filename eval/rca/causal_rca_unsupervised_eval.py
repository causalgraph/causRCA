
import sys
from pathlib import Path
project_path = Path(__file__).parents[2]
sys.path.insert(0, str(project_path))

import networkx as nx
import pandas as pd

from causrca.rca_models.unsupervised_rca_models import CausalPrioTimeRecencyRCA
from rca_eval import evaluate_unsupervised_rca_model
from causrca.utils.utils import seed_everything


# Set random seed for reproducibility
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)  # Set a random seed for reproducibility
DIG_TWIN_DS_PATH = Path(project_path, "data", "dig_twin")
CD_EVAL_GRAPH_PATH = Path(project_path, "eval", "cd", "results")


###############################
# PROBE DATASETS
###############################
PROBE_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_probe")
PROBE_CD_GRAPHS = {
    "PC": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "pc_default_probe_majority.gml")),
    "FCI": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "fci_default_probe_majority.gml")),
    "FGES": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "fges_default_probe_majority.gml")),
    "PCMCI": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "pcmci_default_probe_majority.gml")),
    "TRUE-GRAPH": nx.read_gml(Path(PROBE_DS_PATH, "probe_expert_graph.gml"))
}

# Define a dictionary to store results for each causal discovery method
probe_results = {
    "PC": {},
    "FCI": {},
    "FGES": {},
    "PCMCI": {},
    "TRUE-GRAPH": {}
}

# Run rca evaluation for each causal graph for probe datasets
for cd_method, graph in PROBE_CD_GRAPHS.items():
    print(f"\nEvaluating CausalPrioLogisticRegressionRCA with {cd_method}-based-CG on Probe Datasets:\n")    
    # Setup RCA unsupervised model
    unsupervised_model = CausalPrioTimeRecencyRCA(causal_graph=graph)
    # Run Evaluation 
    result =  evaluate_unsupervised_rca_model(
        model=unsupervised_model,
        path=PROBE_DS_PATH,
        use_relevant_nodes_info=False
    )
    # Print results
    print(f"\n### Evaluation Results for CausalPrioLogisticRegressionRCA on Probe Datasets with {cd_method}-based-CG ###\n")
    print(f"MAP@1: {result['map_at_1']:.4f}, MAP@3: {result['map_at_3']:.4f}, MAP@5: {result['map_at_5']:.4f}")
    # Store results in the dictionary
    probe_results[cd_method] = result
    
# Print result summary for probe datasets
print("\n### Summary of Results for Probe Datasets ###")
for cd_method, result in probe_results.items():
    print(f"{cd_method} - MAP@1: {result['map_at_1']:.4f}, MAP@3: {result['map_at_3']:.4f}, MAP@5: {result['map_at_5']:.4f}")



###############################
# COOLANT DATASETS
###############################
COOLANT_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_coolant")
COOLANT_CD_GRAPHS = {
    "PC": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "pc_default_coolant_majority.gml")),
    "FCI": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "fci_default_coolant_majority.gml")),
    "FGES": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "fges_default_coolant_majority.gml")),
    "PCMCI": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "pcmci_default_coolant_majority.gml")),
    "TRUE-GRAPH": nx.read_gml(Path(COOLANT_DS_PATH, "coolant_expert_graph.gml"))
}

# Define a dictionary to store results for each causal discovery method
coolant_results = {
    "PC": {},
    "FCI": {},
    "FGES": {},
    "PCMCI": {},
    "TRUE-GRAPH": {}
}

# Run rca evaluation for each causal graph for coolant datasets
for cd_method, graph in COOLANT_CD_GRAPHS.items():
    print(f"\nEvaluating CausalPrioTimeProximityRCA with {cd_method}-based-CG on Coolant Datasets:\n")    
    # Setup RCA unsupervised model
    unsupervised_model = CausalPrioTimeRecencyRCA(causal_graph=graph)
    # Run Evaluation 
    result =  evaluate_unsupervised_rca_model(
        model=unsupervised_model,
        path=COOLANT_DS_PATH,
        use_relevant_nodes_info=False
    )
    # Print results
    print(f"\n### Evaluation Results for CausalPrioTimeProximityRCA on Coolant Datasets with {cd_method}-based-CG ###\n")
    print(f"MAP@1: {result['map_at_1']:.4f}, MAP@3: {result['map_at_3']:.4f}, MAP@5: {result['map_at_5']:.4f}")
    # Store results in the dictionary
    coolant_results[cd_method] = result
# Print result summary for coolant datasets
print("\n### Summary of Results for Coolant Datasets ###")
for cd_method, result in coolant_results.items():
    print(f"{cd_method} - MAP@1: {result['map_at_1']:.4f}, MAP@3: {result['map_at_3']:.4f}, MAP@5: {result['map_at_5']:.4f}")
    


###############################
# HYDRAULICS DATASETS
###############################
HYDRAULICS_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_hydraulics")
HYDRAULICS_CD_GRAPHS = {
    "PC": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "pc_default_hydraulic_majority.gml")),
    "FCI": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "fci_default_hydraulic_majority.gml")),
    "FGES": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "fges_default_hydraulic_majority.gml")),
    "PCMCI": nx.read_gml(Path(CD_EVAL_GRAPH_PATH, "pcmci_default_hydraulic_majority.gml")),
    "TRUE-GRAPH": nx.read_gml(Path(HYDRAULICS_DS_PATH, "hydraulics_expert_graph.gml"))
}

# Define a dictionary to store results for each causal discovery method
hydraulics_results = {
    "PC": {},
    "FCI": {},
    "FGES": {},
    "PCMCI": {},
    "TRUE-GRAPH": {}
}

# Run rca evaluation for each causal graph for hydraulics datasets
for cd_method, graph in HYDRAULICS_CD_GRAPHS.items():
    print(f"\nEvaluating CausalPrioTimeProximityRCA with {cd_method}-based-CG on Hydraulics Datasets:\n")    
    # Setup RCA unsupervised model
    unsupervised_model = CausalPrioTimeRecencyRCA(causal_graph=graph)
    # Run Evaluation 
    result =  evaluate_unsupervised_rca_model(
        model=unsupervised_model,
        path=HYDRAULICS_DS_PATH,
        use_relevant_nodes_info=False
    )
    # Print results
    print(f"\n### Evaluation Results for CausalPrioTimeProximityRCA on Hydraulics Datasets with {cd_method}-based-CG ###\n")
    print(f"MAP@1: {result['map_at_1']:.4f}, MAP@3: {result['map_at_3']:.4f}, MAP@5: {result['map_at_5']:.4f}")
    # Store results in the dictionary
    hydraulics_results[cd_method] = result
# Print result summary for hydraulics datasets
print("\n### Summary of Results for Hydraulics Datasets ###")
for cd_method, result in hydraulics_results.items():
    print(f"{cd_method} - MAP@1: {result['map_at_1']:.4f}, MAP@3: {result['map_at_3']:.4f}, MAP@5: {result['map_at_5']:.4f}")
    

# Print out final table with relevant results
# Matches Table 4 in Paper
print("\n" + "="*51)
print("| RCA method     |       MAP@3 for datasets       |")
print("|                |--------------------------------|")
print("| /w base model  | Coolant | Hydraulics | Probe   |")
print("="*51)
for cd_method in ["PC", "FCI", "FGES", "PCMCI", "TRUE-GRAPH"]:
    coolant_map_at_3 = coolant_results[cd_method].get('map_at_3', -1.0) if coolant_results[cd_method] else -1.0
    probe_map_at_3 = probe_results[cd_method].get('map_at_3', -1.0) if probe_results[cd_method] else -1.0
    hydraulics_map_at_3 = hydraulics_results[cd_method].get('map_at_3', -1.0) if hydraulics_results[cd_method] else -1.0
    print(f"| {cd_method:<14} | {coolant_map_at_3:7.4f} | {hydraulics_map_at_3:10.4f} | {probe_map_at_3:7.4f} |")
print("="*51)

# Store results in /eval/rca/results as .csv-file
# Prepare data for DataFrame
results_data = []
for cd_method in ["PC", "FCI", "FGES", "PCMCI", "TRUE-GRAPH"]:
    coolant_map_at_3 = coolant_results[cd_method].get('map_at_3', -1.0) if coolant_results[cd_method] else -1.0
    probe_map_at_3 = probe_results[cd_method].get('map_at_3', -1.0) if probe_results[cd_method] else -1.0
    hydraulics_map_at_3 = hydraulics_results[cd_method].get('map_at_3', -1.0) if hydraulics_results[cd_method] else -1.0
    
    results_data.append({
        'RCA_method_with_base_model': cd_method,
        'Coolant_MAP@3': coolant_map_at_3,
        'Hydraulics_MAP@3': hydraulics_map_at_3,
        'Probe_MAP@3': probe_map_at_3
    })

results_df = pd.DataFrame(results_data)

# Create results directory if it doesn't exist
results_path = Path(project_path, "eval", "rca", "results")
results_path.mkdir(parents=True, exist_ok=True)

# Save to CSV file
csv_filename = Path(results_path, "cd_based_unsupervised_rca_eval_results.csv")
results_df.to_csv(csv_filename, index=False)
print(f"\nResults saved to: {csv_filename}")

