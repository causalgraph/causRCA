import sys
from pathlib import Path
project_path = Path(__file__).parents[3]
sys.path.insert(0, str(project_path))

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from causallearn.graph import GeneralGraph
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci
from tigramite import data_processing
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from causrca.cd_models.pkgs.fges_base import fges
from causrca.utils.discretize import discretize_dataset


class CausalDiscoveryModel(ABC):
    """Base class for causal discovery models."""
    
    def __init__(self, **kwargs):
        """Initialize the causal discovery model."""
        self.params = kwargs
    
    def prepare_dataset(self, args):
        """Prepare and discretize a dataset for causal discovery.
        
        This method serves as a standardized interface for dataset preparation.
        Different models can override this method to use custom discretization
        or preprocessing approaches. For now, there are no custom implementations necessary,
        because all of the implemented models use the same discretization method.
        
        :param args: Arguments for dataset preparation, typically:
                    (dataset_index, dataset, categorical_encoding_dict)
        :type args: tuple
        :return: Tuple containing dataset index and prepared data
        :rtype: tuple
        """
        return discretize_dataset(args)
    
    @abstractmethod
    def learn(self, data, **kwargs):
        """Learn the causal structure from data.
        
        :param data: Input data as numpy array or pandas DataFrame
        :param kwargs: Additional parameters for the algorithm
        :return: Learned causal graph
        :rtype: object
        """
        pass
    


class PCModel(CausalDiscoveryModel):
    """PC algorithm for causal discovery."""
    
    def __init__(self, alpha=0.05, indep_test='chisq', stable=True, **kwargs):
        """Initialize PC model.
        
        :param alpha: Significance level for independence tests
        :type alpha: float
        :param indep_test: Independence test method ('fisherz', 'chisq', 'gsq', 'mv_fisherz', 'kci')
        :type indep_test: str
        :param stable: Whether to use stable version of PC algorithm
        :type stable: bool
        :param kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.indep_test = indep_test
        self.stable = stable
        self.indep_test_func = self.__get_independence_test(indep_test)
    
    def __get_independence_test(self, test_name):
        """Get the independence test function by name.
        
        :param test_name: Name of the independence test
        :type test_name: str
        :return: Independence test function
        :rtype: function
        """
        test_mapping = {
            'fisherz': fisherz,
            'chisq': chisq,
            'gsq': gsq,
            'mv_fisherz': mv_fisherz,
            'kci': kci
        }
        return test_mapping.get(test_name, fisherz)
    
    def learn(self, data, **kwargs):
        """Learn causal structure using PC algorithm.
        
        :param data: Input data as numpy array or pandas DataFrame
        :type data: pandas.DataFrame
        :param kwargs: Additional parameters (alpha, indep_test, stable, etc.)
        :return: Learned causal graph as adj matrix
        :rtype: pd.DataFrame
        """
        # Override default parameters with kwargs if provided
        alpha = kwargs.get('alpha', self.alpha)
        indep_test = kwargs.get('indep_test', self.indep_test_func)
        stable = kwargs.get('stable', self.stable)
        show_progress = kwargs.get('show_progress', False)

        # Fill NaN values
        data_clean = data.fillna(method='ffill')
        data_clean = data_clean.fillna(value=0)
        # Convert to numpy and take absolute values
        np_data = np.absolute(data_clean.to_numpy().astype(float))
        # Use the original DataFrame's column names as node names
        node_names = kwargs.get('node_names', data.columns.tolist())
        
        # Run PC algorithm
        G = pc(
            np_data,
            alpha=alpha,
            indep_test=indep_test,
            stable=stable,
            show_progress=show_progress,
            node_names=node_names
        ).G
        
        adj_matrix = G.graph
        adj_matrix = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)
        
        # Convert PC specific output to binary adjacency matrix
            # PC format:
            # cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates i --> j ,
            # cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
            # cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
        # Convert to binary adjacency matrix (1 indicates edge, 0 indicates no edge)
        binary_adj = np.zeros_like(adj_matrix.values)
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i != j:
                    # Check if there's any edge (directed or undirected)
                    if (adj_matrix.iloc[i, j] == 1 and adj_matrix.iloc[j, i] == -1) or \
                        (adj_matrix.iloc[i, j] == -1 and adj_matrix.iloc[j, i] == 1) or \
                        (adj_matrix.iloc[i, j] == -1 and adj_matrix.iloc[j, i] == -1) or \
                        (adj_matrix.iloc[i, j] == 1 and adj_matrix.iloc[j, i] == 1):
                        
                        binary_adj[i, j] = 1
        
        adj_matrix = pd.DataFrame(binary_adj, index=node_names, columns=node_names)
        
        return adj_matrix


class FCIModel(CausalDiscoveryModel):
    """FCI algorithm for causal discovery."""
    
    def __init__(self, alpha=0.05, independence_test_method='chisq', **kwargs):
        """Initialize FCI model.
        
        :param alpha: Significance level for independence tests
        :type alpha: float
        :param independence_test_method: Independence test method ('fisherz', 'chisq', 'gsq', 'mv_fisherz', 'kci')
        :type independence_test_method: str
        :param kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.independence_test_method = independence_test_method
        self.indep_test_func = self._get_independence_test(independence_test_method)
    
    def _get_independence_test(self, test_name):
        """Get the independence test function by name.
        
        :param test_name: Name of the independence test
        :type test_name: str
        :return: Independence test function
        :rtype: function
        """
        test_mapping = {
            'fisherz': fisherz,
            'chisq': chisq,
            'gsq': gsq,
            'mv_fisherz': mv_fisherz,
            'kci': kci
        }
        return test_mapping.get(test_name, fisherz)
    
    def learn(self, data, **kwargs):
        """Learn causal structure using FCI algorithm.
        
        :param data: Input data as numpy array or pandas DataFrame
        :type data: pandas.DataFrame
        :param kwargs: Additional parameters (alpha, independence_test_method, etc.)
        :return: Learned causal graph as adjacency matrix
        :rtype: pd.DataFrame
        """
        # Override default parameters with kwargs if provided
        alpha = kwargs.get('alpha', self.alpha)
        independence_test_method = kwargs.get('independence_test_method', self.indep_test_func)
        show_progress = kwargs.get('show_progress', False)
        verbose = kwargs.get('verbose', False)
        
        # Fill NaN values
        data_clean = data.fillna(method='ffill')
        data_clean = data_clean.fillna(value=0)
        # Convert to numpy and take absolute values
        np_data = np.absolute(data_clean.to_numpy().astype(float))
        # Use the original DataFrame's column names as node names
        node_names = kwargs.get('node_names', data.columns.tolist())
        
        # Run FCI algorithm
        G = fci(
            np_data,
            independence_test_method=independence_test_method,
            alpha=alpha,
            show_progress=show_progress,
            verbose=verbose,
            node_names=node_names
        )[0]
        
        adj_matrix = G.graph
        adj_matrix = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)
        # Convert FCI specific output to binary adjacency matrix
        # FCI format:
        #graph : a GeneralGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
        #        graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
        #        graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
        #        graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
        # Convert to binary adjacency matrix (1 indicates edge, 0 indicates no edge)
        binary_adj = np.zeros_like(adj_matrix.values)
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i != j:
                    # Any non-zero value indicates an edge exists
                    # FCI uses: 0=no edge, 1=arrowhead, -1=tail, 2=circle
                    if adj_matrix.iloc[i, j] != 0:
                        binary_adj[i, j] = 1
                        # If it's a circle (2), make it bidirectional
                        if adj_matrix.iloc[i, j] == 2:
                            binary_adj[j, i] = 1
        
        adj_matrix = pd.DataFrame(binary_adj, index=node_names, columns=node_names)
        
        return adj_matrix


class FGESModel(CausalDiscoveryModel):
    """FGES algorithm for causal discovery."""
    
    def __init__(self, score_func='linear', **kwargs):
        """Initialize FGES model.
        
        :param score_func: Score function ('linear', 'p2', 'p3')
        :type score_func: str
        :param kwargs: Additional parameters
        """        
        super().__init__(**kwargs)
        self.score_func = score_func
    
    def learn(self, data, **kwargs):
        """Learn causal structure using FGES algorithm.
        
        :param data: Input data as numpy array or pandas DataFrame
        :type data: numpy.ndarray or pandas.DataFrame
        :param kwargs: Additional parameters (score_func, etc.)
        :return: Adjacency matrix (numpy array)
        :rtype: numpy.ndarray
        """
        # Override default parameters with kwargs if provided
        score_func = kwargs.get('score_func', self.score_func)
        
        # Fill NaN values
        data_clean = data.fillna(method='ffill')
        data_clean = data_clean.fillna(value=0)
        
        # Convert to absolute values
        try:
            df_data = data_clean.abs()
        except Exception as e:
            # If there's an error in data processing, print detailed information
            print(f"Error processing data: {e}")
        # Node names
        node_names = df_data.columns.tolist()
        
        # Run FGES algorithm
        adj = fges(df_data, score_func=score_func)
        
        # Convert to numpy array if it's a matrix
        if hasattr(adj, 'A'):
            adj = np.array(adj.A)
        else:
            adj = np.array(adj)
        
        adj_matrix = pd.DataFrame(adj, index=node_names, columns=node_names)
        
        return adj_matrix


class PCMCIModel(CausalDiscoveryModel):
    """PCMCI algorithm for causal discovery."""
    
    def __init__(self, tau_max=3, alpha=0.2, **kwargs):
        """Initialize PCMCI model.
        
        :param tau_max: Maximum time lag to consider
        :type tau_max: int
        :param alpha: Significance level for conditional independence tests
        :type alpha: float
        :param kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.tau_max = tau_max
        self.alpha = alpha
    
    def learn(self, data, **kwargs):
        """Learn causal structure using PCMCI algorithm.
        
        :param data: Input data as numpy array or pandas DataFrame
        :type data: pandas.DataFrame
        :param kwargs: Additional parameters (tau_max, alpha, etc.)
        :return: Adjacency matrix (numpy array)
        :rtype: pd.DataFrame
        """
        # Override default parameters with kwargs if provided
        tau_max = kwargs.get('tau_max', self.tau_max)
        alpha = kwargs.get('alpha', self.alpha)
        
        # Fill NaN values
        data_clean = data.fillna(method='ffill')
        data_clean = data_clean.fillna(value=0)
        # Convert to absolute values
        data_array = np.absolute(data_clean.to_numpy().astype(float))
        node_names = data.columns.tolist()
        
        # Create tigramite dataframe
        dataframe = data_processing.DataFrame(data_array)
        
        # Initialize PCMCI with ParCorr independence test
        pcmci_instance = PCMCI(
            dataframe=dataframe, 
            cond_ind_test=ParCorr(significance="analytic"), 
            verbosity=0
        )
        
        # Run PCMCI algorithm
        report = pcmci_instance.run_pcmci(
            tau_max=tau_max,
            pc_alpha=alpha,
            max_conds_dim=None,
        )
        
        # Extract p-values and create adjacency matrix
        p_matrix = report["p_matrix"]
        num_vars = p_matrix.shape[0]
        
        # Create adjacency matrix by finding minimum p-value across time lags
        adj_matrix = np.zeros((num_vars, num_vars))
        for cause in range(num_vars):
            for effect in range(num_vars):
                if cause != effect:
                    # Find minimum p-value across all time lags
                    min_p_val = np.min(p_matrix[cause, effect, :])
                    if min_p_val < alpha:
                        adj_matrix[cause, effect] = 1
        
        # Convert to DataFrame with original node names
        adj_matrix = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)
        
        return adj_matrix

