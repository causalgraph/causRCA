
import sys
from pathlib import Path
project_path = Path(__file__).parents[4]
sys.path.insert(0, str(project_path))

import warnings
import networkx as nx

from causrca.cd_models.libs.runner import fges_runner
from causrca.cd_models.pkgs.scores import linear_gaussian_score_iid, polynomial_2_gaussian_score_iid, polynomial_3_gaussian_score_iid

# from utils import compute_stats, read_data
warnings.filterwarnings("ignore")


def fges(data, score_func=None, **kwargs):
    assert score_func in [None, "linear", "p2", "p3"]
    if score_func is None or score_func == "linear":
        score_func = linear_gaussian_score_iid
    elif score_func == "p2":
        score_func = polynomial_2_gaussian_score_iid
    elif score_func == "p3":
        score_func = polynomial_3_gaussian_score_iid

    data.columns = range(data.shape[1])

    g = nx.DiGraph()
    g.add_nodes_from(list(data.columns))
    for col in data.columns:
        g.nodes[col]["type"] = "cont"
        g.nodes[col]["num_categories"] = "NA"

    result = fges_runner(
        data, g.nodes(data=True), n_bins=1, disc=None, score=score_func, knowledge=None
    )
    G = result["graph"]

    adj = nx.adjacency_matrix(G).todense()
    # adj = np.zeros((data.shape[1], data.shape[1]))
    # for n, neighbors in G.adjacency():
    #     for i in neighbors.keys():
    #         adj[i, n] = 1
    return adj


