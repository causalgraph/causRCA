
import sys
from pathlib import Path
project_path = Path(__file__).parents[2]
sys.path.insert(0, str(project_path))

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
import networkx as nx
import numpy as np

from causrca.rca_models.unsupervised_rca_models import CausalPrioTimeRecencyRCA
from rca_eval import evaluate_unsupervised_rca_model
from causrca.utils.utils import seed_everything

# Set random seed for reproducibility
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)  # Set a random seed for reproducibility



def create_f1_variants_grouped_boxplot(f1_results_dict1, f1_results_dict2, f1_results_dict3, 
                                       dataset_names=None, save_path=None, width_cm=7.5, height_cm=4.50):
    """
    Creates a compact grouped box plot for three F1 variant evaluation results.
    
    :param f1_results_dict1: First dataset - Dictionary with F1 variants as keys and lists of MAP@3 values.
    :type f1_results_dict1: dict
    :param f1_results_dict2: Second dataset - Dictionary with F1 variants as keys and lists of MAP@3 values.
    :type f1_results_dict2: dict
    :param f1_results_dict3: Third dataset - Dictionary with F1 variants as keys and lists of MAP@3 values.
    :type f1_results_dict3: dict
    :param dataset_names: Names for the three datasets for legend. If None, defaults to ['Dataset 1', 'Dataset 2', 'Dataset 3'].
    :type dataset_names: list, optional
    :param save_path: Path to save the plot. If None, the plot is only displayed.
    :type save_path: str, optional
    :param width_cm: Width of the plot in centimeters.
    :type width_cm: float, optional
    :param height_cm: Height of the plot in centimeters.
    :type height_cm: float, optional
    """
    fm._load_fontmanager(try_read_cache=False)
    
    # Set font to Times New Roman with smaller default size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 8
    
    # Convert cm to inches (matplotlib uses inches)
    width_inches = width_cm / 2.54
    height_inches = height_cm / 2.54
    
    plt.figure(figsize=(width_inches, height_inches))
    
    # Get F1 variants (assuming all dicts have same keys)
    f1_variants = list(f1_results_dict1.keys())
    short_labels = [variant.replace('TG_', '') for variant in f1_variants]
    
    # Default dataset names if not provided
    if dataset_names is None:
        dataset_names = ['Dataset 1', 'Dataset 2', 'Dataset 3']
    
    # Define colors for the three datasets
    colors = ['#2E8B57', '#4169E1', '#DC143C']  # Green, Blue, Red
    dark_colors = ['#1F5F3F', '#2F4F8F', '#8B0000']  # Darker versions for medians
    
    # Calculate positions for grouped box plots
    n_groups = len(f1_variants)
    n_datasets = 3
    width = 0.25  # Width of each box
    positions = []
    
    # Create positions for each group
    for i in range(n_groups):
        group_center = i + 1
        positions.extend([group_center - width, group_center, group_center + width])
    
    # Prepare all data
    all_data = []
    all_colors = []
    all_dark_colors = []
    
    for i in range(n_groups):
        variant = f1_variants[i]
        # Add data for each dataset at this F1 variant
        all_data.extend([
            f1_results_dict1[variant],
            f1_results_dict2[variant], 
            f1_results_dict3[variant]
        ])
        all_colors.extend(colors)
        all_dark_colors.extend(dark_colors)
    
    # Create the grouped box plot
    box_plot = plt.boxplot(all_data, positions=positions, patch_artist=True, 
                          widths=width*0.8)
    
    # Color the boxes
    for i, (patch, color, dark_color) in enumerate(zip(box_plot['boxes'], all_colors, all_dark_colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    # Color the median lines
    for i, (median, dark_color) in enumerate(zip(box_plot['medians'], all_dark_colors)):
        median.set_color(dark_color)
        median.set_linewidth(2)
    
    # Set titles and labels
    plt.title('MAP@3 Performance Across Graph Quality Levels', fontsize=8, pad=8)
    plt.xlabel('Graph F1 score', fontsize=8, labelpad=3)
    plt.ylabel('MAP@3 values', fontsize=8, labelpad=3)
    plt.grid(True, alpha=0.3)
    
    # Set Y-axis range from 0.0 to 1.0 with 0.2 steps
    plt.ylim(0.0, 1.0)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Set x-axis ticks and labels
    plt.xticks(range(1, n_groups + 1), short_labels, fontsize=8)
    plt.yticks(fontsize=8)
    
    # Create legend outside the plot on the right
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=dataset_names[i]) 
                      for i in range(3)]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=7, framealpha=0.9)
    
    # Very tight layout adjustments
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.15, left=0.12, right=0.98, top=0.88)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def create_f1_variants_boxplot(f1_var_results, base_results=None, save_path=None, width_cm=6.5, height_cm=4.50):
    """
    Creates a compact box plot for F1 variant evaluation results.
    
    :param f1_var_results: Dictionary with F1 variants as keys and lists of MAP@3 values.
    :type f1_var_results: dict
    :param base_results: List of MAP@3 values from base unsupervised algorithm (TimeRank) for comparison. If None, no comparison lines are shown.
    :type base_results: list, optional
    :param save_path: Path to save the plot. If None, the plot is only displayed.
    :type save_path: str, optional
    :param width_cm: Width of the plot in centimeters.
    :type width_cm: float, optional
    :param height_cm: Height of the plot in centimeters.
    :type height_cm: float, optional
    """
    fm._load_fontmanager(try_read_cache=False)
    
    # Set font to Times New Roman with smaller default size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 8
    
    # Convert cm to inches (matplotlib uses inches)
    width_inches = width_cm / 2.54
    height_inches = height_cm / 2.54
    
    plt.figure(figsize=(width_inches, height_inches))
    
    # Prepare data for box plot
    f1_variants = list(f1_var_results.keys())
    map_at_3_values = [f1_var_results[variant] for variant in f1_variants]
    
    # Create shorter labels for more space
    short_labels = [variant.replace('TG_', '') for variant in f1_variants]
    
    # Create compact box plot
    box_plot = plt.boxplot(map_at_3_values, labels=short_labels, patch_artist=True, 
                          widths=0.4)
    
    # Compact titles and labels
    plt.title('MAP@3 Performance Across Graph Quality Levels', fontsize=8, pad=8)
    plt.xlabel('Graph F1 score', fontsize=8, labelpad=3)
    plt.ylabel('MAP@3 values', fontsize=8, labelpad=3)
    plt.grid(True, alpha=0.3)
    
    # Color all boxes uniformly in green
    green_color = '#2E8B57'  # Sea Green
    dark_green = '#1F5F3F'   # Dark Green for medians
    
    for patch in box_plot['boxes']:
        patch.set_facecolor(green_color)
        patch.set_alpha(0.7)
    
    # Make median lines dark green
    for median in box_plot['medians']:
        median.set_color(dark_green)
        median.set_linewidth(2)
    
    # Add base results comparison lines if provided
    if base_results is not None:
        base_mean = np.mean(base_results)
        base_min = np.min(base_results)
        base_max = np.max(base_results)
        base_std_dev = np.std(base_results)
        
        # Get x-axis limits for positioning
        x_min, x_max = plt.xlim()
        text_x = x_max - 0.1  # Position text slightly left of right edge
        
        # Fill areas between mean and min/max
        plt.axhspan(base_mean, base_max, color='#FFB6C1', alpha=0.3, zorder=1)
        plt.axhspan(base_min, base_mean, color='#FFB6C1', alpha=0.3, zorder=1)
        
        # Draw horizontal line for mean
        plt.axhline(y=base_mean, color='#DC143C', linestyle='-', linewidth=2, alpha=0.8, 
                   label=f'TimeRank (µ={base_mean:.2f}±{base_std_dev:.2f})', zorder=2)
                
        # Add legend for TimeRank baseline
        plt.legend(loc='lower right', fontsize=6, framealpha=0.8, handlelength=1.0)
    
    # Set Y-axis range from 0.0 to 1.0 with 0.2 steps
    plt.ylim(0.0, 1.0)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Compact axis labels
    plt.xticks(rotation=45, fontsize=8, ha='right')
    plt.yticks(fontsize=8)
    
    # Very tight layout adjustments
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.18, left=0.15, right=0.98, top=0.90)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


# Define paths for datasets
DIG_TWIN_DS_PATH = Path(project_path, "data", "dig_twin")
PROBE_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_probe")
COOLANT_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_coolant")
HYDRAULICS_DS_PATH = Path(DIG_TWIN_DS_PATH, "exp_hydraulics")

# Define paths for F1 variant graphs
F1_VARIANT_GRAPH_DIRS = {
    "TG_0.3": [
        Path(project_path, "data", "dig_twin", "exp_probe", "f1_variants", "f1_03"),
        Path(project_path, "data", "dig_twin", "exp_hydraulics", "f1_variants", "f1_03"),
        Path(project_path, "data", "dig_twin", "exp_coolant", "f1_variants", "f1_03")
    ],
    "TG_0.4": [
        Path(project_path, "data", "dig_twin", "exp_probe", "f1_variants", "f1_04"),
        Path(project_path, "data", "dig_twin", "exp_hydraulics", "f1_variants", "f1_04"),
        Path(project_path, "data", "dig_twin", "exp_coolant", "f1_variants", "f1_04")
    ],
    "TG_0.5": [
        Path(project_path, "data", "dig_twin", "exp_probe", "f1_variants", "f1_05"),
        Path(project_path, "data", "dig_twin", "exp_hydraulics", "f1_variants", "f1_05"),
        Path(project_path, "data", "dig_twin", "exp_coolant", "f1_variants", "f1_05")
    ],
    "TG_0.6": [
        Path(project_path, "data", "dig_twin", "exp_probe", "f1_variants", "f1_06"),
        Path(project_path, "data", "dig_twin", "exp_hydraulics", "f1_variants", "f1_06"),
        Path(project_path, "data", "dig_twin", "exp_coolant", "f1_variants", "f1_06")
    ],
    "TG_0.7": [
        Path(project_path, "data", "dig_twin", "exp_probe", "f1_variants", "f1_07"),
        Path(project_path, "data", "dig_twin", "exp_hydraulics", "f1_variants", "f1_07"),
        Path(project_path, "data", "dig_twin", "exp_coolant", "f1_variants", "f1_07")
    ],
    "TG_0.8": [
        Path(project_path, "data", "dig_twin", "exp_probe", "f1_variants", "f1_08"),
        Path(project_path, "data", "dig_twin", "exp_hydraulics", "f1_variants", "f1_08"),
        Path(project_path, "data", "dig_twin", "exp_coolant", "f1_variants", "f1_08")
    ],
    "TG_0.9": [
        Path(project_path, "data", "dig_twin", "exp_probe", "f1_variants", "f1_09"),
        Path(project_path, "data", "dig_twin", "exp_hydraulics", "f1_variants", "f1_09"),
        Path(project_path, "data", "dig_twin", "exp_coolant", "f1_variants", "f1_09")
    ],
    "TG_1.0": [
        Path(PROBE_DS_PATH),
        Path(HYDRAULICS_DS_PATH),
        Path(COOLANT_DS_PATH)
    ]
}

# Define a dictionary to store results for each F1 variant
f1_var_results = {
    "TG_0.3": [],
    "TG_0.4": [],
    "TG_0.5": [],
    "TG_0.6": [],
    "TG_0.7": [],
    "TG_0.8": [],
    "TG_0.9": [],
    "TG_1.0": []
}
f1_var_results_probe = {
    "TG_0.3": [],
    "TG_0.4": [],
    "TG_0.5": [],
    "TG_0.6": [],
    "TG_0.7": [],
    "TG_0.8": [],
    "TG_0.9": [],
    "TG_1.0": []
}
f1_var_results_coolant = {
    "TG_0.3": [],
    "TG_0.4": [],
    "TG_0.5": [],
    "TG_0.6": [],
    "TG_0.7": [],
    "TG_0.8": [],
    "TG_0.9": [],
    "TG_1.0": []
}
f1_var_results_hydraulics = {
    "TG_0.3": [],
    "TG_0.4": [],
    "TG_0.5": [],
    "TG_0.6": [],
    "TG_0.7": [],
    "TG_0.8": [],
    "TG_0.9": [],
    "TG_1.0": []
}



# Run RCA evaluation for each F1 variant over all sub datasets (probe, coolant, hydraulics) and
# caluculate MAP@1, MAP@3, MAP@5 by averaging over all sub datasets for each F1 variant
for f1_variant, paths in F1_VARIANT_GRAPH_DIRS.items():
    print(f"\nEvaluating CausalPrioTimeProximityRCA with F1 variant {f1_variant}:\n")
    
    for path in paths:
        
        # Calculate base dataset based on path sub string
        base_dataset = None
        if "probe" in str(path):
            base_dataset = PROBE_DS_PATH
        elif "coolant" in str(path):
            base_dataset = COOLANT_DS_PATH
        elif "hydraulics" in str(path):
            base_dataset = HYDRAULICS_DS_PATH
        else:
            raise ValueError(f"Unknown dataset path: {path}")        
        
        # Create list of graphs for the current F1 variant by loading all gml files in the path
        f1_variant_graphs = list(path.glob("*.gml"))
        
        # Run RCA evaluation for each graph in the F1 variant and append MAP@3 to f1_var_results
        for graph in f1_variant_graphs:
            print(f"Evaluating graph: {graph.name}")
            # Setup RCA unsupervised model
            unsupervised_model = CausalPrioTimeRecencyRCA(causal_graph=nx.read_gml(graph))
            # Run Evaluation
            result = evaluate_unsupervised_rca_model(
                model=unsupervised_model,
                path=base_dataset,
                mode='full'
            )
            # Print results
            print(f"\n### Evaluation Results for CausalPrioTimeRecencyRCA on {base_dataset.name} with F1 variant {f1_variant} ###\n")
            print(f"MAP@1: {result['map_at_1']:.4f}, MAP@3: {result['map_at_3']:.4f}, MAP@5: {result['map_at_5']:.4f}")
            
            ## Store MAP@3 result in the f1_var_results dictionary
            f1_var_results[f1_variant].append(result['map_at_3'])
            if "probe" in str(path):
                f1_var_results_probe[f1_variant].append(result['map_at_3'])
            elif "coolant" in str(path):
                f1_var_results_coolant[f1_variant].append(result['map_at_3'])
            elif "hydraulics" in str(path):
                f1_var_results_hydraulics[f1_variant].append(result['map_at_3'])
                

create_f1_variants_boxplot(f1_var_results=f1_var_results,
                           base_results=[0.35, 0.18, 0.19],
                           save_path=Path(project_path, "eval", "rca", "results", "f1_variants_boxplot_map.svg"),)

#create_f1_variants_grouped_boxplot(f1_results_dict1=f1_var_results_probe,
#                                   f1_results_dict2=f1_var_results_coolant,
#                                   f1_results_dict3=f1_var_results_hydraulics,
#                                   dataset_names=["Probe", "Coolant", "Hydraulics"],
#                                   save_path=Path(project_path, "eval", "rca", "results", "f1_variants_grouped_boxplot.svg"),
#                                   width_cm=7.5, height_cm=4.50)

