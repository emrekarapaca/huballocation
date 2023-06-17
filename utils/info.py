class Info:
    """
    The Info class provides information and details about different clustering algorithms and heuristics.

    Attributes:
        ui (UI): An instance of the UI class.
        app (HubAllAPP): An instance of the HubAllAPP class.

    Methods:
        k_means_info(): Displays information about the K-means clustering algorithm.
        affinity_prop_info(): Displays information about the Affinity Propagation clustering algorithm.
        mean_shift_info(): Displays information about the Mean Shift clustering algorithm.
        spectral_clustering_info(): Displays information about the Spectral Clustering algorithm.
        hierarchical_clustering_info(): Displays information about the Hierarchical Clustering algorithm.
        dbscan_info(): Displays information about the DBSCAN algorithm.
        hill_climbing_info(): Displays information about the Hill Climbing heuristic algorithm.
        sim_annealing_info(): Displays information about the Simulated Annealing heuristic algorithm.
    """
    def __init__(self, ui, app):
        self.ui = ui
        self.app = app
    
    def k_means_info(self):
        """
        Displays information about the K-means clustering algorithm.

        Returns:
            None
        """
        info_text = """
        K-Means Clustering Algorithm
        ----------------------------
        K-Means Clustering is an iterative algorithm that aims to partition a dataset into K clusters,
        where each data point belongs to the cluster with the nearest mean (centroid).
        
        Parameters:
        - Number of Clusters: The desired number of clusters to be formed.
        - Maximum Iterations: The maximum number of iterations the algorithm will run.
        - Algorithm: The algorithm to be used for computing the centroids.
        
        Usage:
        1. Enter the number of clusters.
        2. Specify the maximum number of iterations.
        3. Choose the algorithm (auto, elkan, or full).
        4. Click the "K-Means Clustering" button to perform clustering.
        
        Note: The cluster labels and cluster centers will be displayed in the Results section.
        """
        self.ui.info_textBrowser.clear()
        self.ui.info_textBrowser.append(info_text)
    
    def affinity_prop_info(self):
        """
        Displays information about the Affinity Propagation clustering algorithm.

        Returns:
            None
        """
        info_text = """
        Affinity Propagation Algorithm
        ------------------------------
        Affinity Propagation is a clustering algorithm that does not require the number of clusters 
        to be specified in advance. It uses a message-passing approach to iteratively determine 
        exemplars (cluster centers) that represent data points in the dataset.
        
        Parameters:
        - Damping: Damping factor that controls the update of the responsibility and availability matrices.
        - Maximum Iterations: The maximum number of iterations the algorithm will run.
        - Convergence Iterations: The number of iterations with no change in the number of identified exemplars.
        - Affinity: The affinity metric to be used (e.g., Euclidean distance or precomputed affinity matrix).
        
        Usage:
        1. Enter the damping factor.
        2. Specify the maximum number of iterations.
        3. Set the convergence iterations.
        4. Choose the affinity metric (Euclidean or precomputed).
        5. Click the "Affinity Propagation" button to perform clustering.
        
        Note: The cluster labels and exemplars will be displayed in the Results section.
        """
        self.ui.info_textBrowser.clear()
        self.ui.info_textBrowser.append(info_text)
    
    def mean_shift_info(self):
        """
        Displays information about the Mean Shift clustering algorithm.

        Returns:
            None
        """
        info_text = """
        Mean Shift Algorithm
        --------------------
        Mean Shift is a non-parametric clustering algorithm that does not require the number of clusters 
        to be specified in advance. It iteratively shifts the data points towards the mode of the 
        underlying probability distribution to identify cluster centers.
        
        Parameters:
        - Bandwidth: The bandwidth or radius within which the algorithm searches for cluster centers.
        
        Usage:
        1. Enter the bandwidth value.
        2. Click the "Mean Shift" button to perform clustering.
        
        Note: The cluster labels and cluster centers will be displayed in the Results section.
        """
        self.ui.info_textBrowser.clear()
        self.ui.info_textBrowser.append(info_text)
    
    def spectral_clustering_info(self):
        """
        Displays information about the Spectral Clustering algorithm.

        Returns:
            None
        """
        info_text = """
        Spectral Clustering Algorithm
        ----------------------------
        Spectral Clustering is a technique that uses the eigenvalues of the similarity matrix of the data 
        to perform clustering. It is particularly effective in finding clusters in non-convex or 
        irregularly shaped datasets.
        
        Parameters:
        - Number of Clusters: The desired number of clusters to be identified.
        
        Usage:
        1. Enter the number of clusters.
        2. Click the "Spectral Clustering" button to perform clustering.
        
        Note: The cluster labels and cluster centers will be displayed in the Results section.
        """
        self.ui.info_textBrowser.clear()
        self.ui.info_textBrowser.append(info_text)
    
    def hierarchical_clustering_info(self):
        """
        Displays information about the Hierarchical Clustering algorithm.

        Returns:
            None
        """
        info_text = """
        Hierarchical Clustering Algorithm
        -------------------------------
        Hierarchical Clustering is a technique that creates a hierarchy of clusters by merging or 
        splitting them based on their similarity or distance. It builds a tree-like structure of 
        clusters, known as a dendrogram, to represent the data relationships.
        
        Parameters:
        - Linkage Method: The method used to calculate the distance between clusters.
        
        Usage:
        1. Select the desired linkage method from the dropdown menu.
        2. Click the "Hierarchical Clustering" button to perform clustering.
        
        Note: The cluster labels and cluster centers will be displayed in the Results section.
        """
        self.ui.info_textBrowser.clear()
        self.ui.info_textBrowser.append(info_text)
    
    def dbscan_info(self):
        """
        Displays information about the DBSCAN algorithm.

        Returns:
            None
        """
        info_text = """
        DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        ------------------------------------------------------------------
        DBSCAN is a density-based clustering algorithm that groups together data points that are 
        close to each other in the feature space and separates data points that are far from any 
        cluster. It is capable of discovering clusters of arbitrary shape and can handle noise points 
        as well.
        
        Parameters:
        - Epsilon (ε): The maximum distance between two samples for them to be considered as neighbors.
        - Minimum Samples: The minimum number of samples required to form a dense region.
        
        Usage:
        1. Set the values for Epsilon and Minimum Samples.
        2. Click the "DBSCAN" button to perform clustering.
        
        Note: The cluster labels will be displayed in the Results section.
        """
        self.ui.info_textBrowser.clear()
        self.ui.info_textBrowser.append(info_text)
    
    def hill_climbing_info(self):
        """
        Displays information about the Hill Climbing heuristic algorithm.

        Returns:
            None
        """
        
        info_text = """
        Hill Climbing
        -------------
        Hill Climbing is a local search algorithm used to find an optimal solution in optimization problems.
        It starts with an initial solution and iteratively moves to a neighboring solution that improves the objective function.
        The algorithm terminates when no better solution can be found in the neighborhood.
    
        Hill Climbing has the following key features:
        - It explores a single path by iteratively improving the current solution.
        - It does not maintain a search tree or keep track of previously visited states.
        - It can get stuck in local optima, which are suboptimal solutions that cannot be improved.
    
        Usage:
        1. Define an initial solution.
        2. Define a neighborhood function that generates neighboring solutions.
        3. Define an objective function to evaluate the quality of a solution.
        4. Start with the initial solution.
        5. Generate neighboring solutions.
        6. Select the best neighboring solution that improves the objective function.
        7. Repeat steps 5 and 6 until no better solution can be found or a termination condition is met.
    
        Note: Hill Climbing is a simple and intuitive algorithm but may not always find the global optimum.
    
        """
        self.ui.info_textBrowser.clear()
        self.ui.info_textBrowser.append(info_text)
    
    def sim_annealing_info(self):
        """
        Displays information about the Simulated Annealing heuristic algorithm.

        Returns:
            None
        """
        info_text = """
        Simulated Annealing
        -------------------
        Simulated Annealing is a metaheuristic algorithm inspired by the annealing process in metallurgy.
        It is used to find near-optimal solutions in optimization problems, especially when the search space is large and complex.
    
        Simulated Annealing has the following key features:
        - It explores the search space by iteratively accepting worse solutions based on a probabilistic criterion.
        - It allows for occasional "bad moves" to escape local optima and search for better solutions.
        - The probability of accepting a worse solution decreases as the algorithm progresses, simulating the cooling process in annealing.
    
        Usage:
        1. Define an initial solution.
        2. Define a neighborhood function that generates neighboring solutions.
        3. Define an objective function to evaluate the quality of a solution.
        4. Set an initial temperature and a cooling schedule.
        5. Start with the initial solution.
        6. Generate a neighboring solution.
        7. Evaluate the quality of the neighboring solution.
        8. Accept the neighboring solution based on a probability determined by the current temperature and the objective function difference.
        9. Update the temperature according to the cooling schedule.
        10. Repeat steps 6 to 9 until the termination condition is met.
    
        Note: Simulated Annealing is a stochastic algorithm and can be used to approximate global optima.
    
        """
        self.ui.info_textBrowser.clear()
        self.ui.info_textBrowser.append(info_text)
