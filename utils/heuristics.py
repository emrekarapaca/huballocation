import numpy as np
import random
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QDialog, QSpinBox, QLabel, QDialogButtonBox, QFormLayout, QDoubleSpinBox
from PyQt5.QtCore import QTimer
from utils.info import Info

class Heuristics:
    """
    The Heuristics class contains methods for implementing various heuristic operations.

    Attributes:
        ui (UI): An instance of the UI class.
        app (HubAllAPP): An instance of the HubAllAPP class.

    Methods:
        relocate_hub(): Performs the RelocateHub operation, which changes the location of the hub within a cluster to a different node in the cluster.
        reallocate_node(): Performs the ReallocateNode operation, which changes the allocation of a non-hub node to a different cluster.
        swap_nodes(): Performs the SwapNodes operation, which swaps the allocations of two non-hub nodes from different clusters.
        hill_climbing(): Implements the Hill Climbing heuristic algorithm.
        sim_annealing(): Implements the Simulated Annealing heuristic algorithm.
        objective_function(): Calculates the objective function value for a given solution.
        calculate_temperature(): It calculates temperature value.
        acceptance_probability() : It calcualtes acceptance prob. for sim. annealing algorithm. 
        plotCluster1(): It plots after heuristic operations.
    """
    def __init__(self, ui, app):
        self.ui = ui
        self.app = app
        self.current_solution = None
        self.current_objective = None
        self.lowest_objective = None
        self.lowest_solution = None
        self.info = Info(ui, app)
    
    def objective_function(self, solution=None):
        """
        Calculates the objective function value for a given solution.

        Args:
            solution (dict): The solution to calculate the objective function for.

        Returns:
            float: The objective function value.
        """
        if solution == None:
            solution = self.app.best_solution
            
        X = solution["X"]
        labels = solution["labels"]
        cluster_centers = solution["cluster_centers"]

        objective = 0.0
        num_clusters = len(cluster_centers)

        for cluster_label in range(num_clusters):
            cluster_points = X[labels == cluster_label]
            cluster_center = cluster_centers[cluster_label]

            if len(cluster_points) > 0:
                farthest_distance = np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))

                for other_cluster_label in range(num_clusters):
                    if other_cluster_label != cluster_label:
                        other_cluster_center = cluster_centers[other_cluster_label]
                        distance_to_other_center = np.linalg.norm(cluster_center - other_cluster_center)
                        
                        # Check if there are points in the other cluster
                        other_cluster_points = X[labels == other_cluster_label]
                        if len(other_cluster_points) > 0:
                            farthest_other_distance = np.max(np.linalg.norm(other_cluster_points - other_cluster_center, axis=1))
                        else:
                            farthest_other_distance = 0.0  # Set a default value for empty cluster

                        objective_value = farthest_distance + 0.75 * distance_to_other_center + farthest_other_distance
                        objective = max(objective, objective_value)

        return objective
    
    def relocate_hub(self, solution):
        """
        Performs the RelocateHub operation, which changes the location of the hub within a cluster
        to a different node in the cluster.
    
        Args:
            solution (dict): The current solution.
    
        Returns:
            None
        """
        
        cluster_labels = np.unique(solution["labels"])
        cluster_labels = cluster_labels[cluster_labels != -1]  #exclude outliers
        cluster_label = random.choice(cluster_labels)
    
        
        non_hub_indices = np.where((solution["labels"] == cluster_label) & (solution["labels"] != -1))[0]
    
        if len(non_hub_indices) > 0:
            #select a random non-hub node?
            node_index = random.choice(non_hub_indices)
    
            #update the hub location to the selected node
            solution["cluster_centers"][cluster_label] = solution["X"][node_index]

    def reallocate_node(self, solution):
        
        """
        Performs the ReallocateNode operation, which changes the allocation of a non-hub node to
        a different cluster.
    
        Args:
            solution (dict): The current solution.
    
        Returns:
            None
        """
       
        non_hub_indices = np.where(solution["labels"] != -1)[0]
        node_index = random.choice(non_hub_indices)
    
        
        current_cluster_label = solution["labels"][node_index]
    
        
        cluster_labels = np.unique(solution["labels"])
        cluster_labels = cluster_labels[cluster_labels != -1]  #exclude outliers
        cluster_label = random.choice(cluster_labels)
        while cluster_label == current_cluster_label:
            cluster_label = random.choice(cluster_labels)
    
        
        solution["labels"][node_index] = cluster_label

    def swap_nodes(self, solution):
        """
        Performs the SwapNodes operation, which swaps the allocations of two non-hub nodes from different clusters.
    
        Args:
            solution (dict): The current solution.
    
        Returns:
            None
        """
        
        cluster_labels = np.unique(solution["labels"])
        cluster_labels = cluster_labels[cluster_labels != -1]  #exclude outliers
        cluster_label_1, cluster_label_2 = random.sample(list(cluster_labels), k=2)
    
        
        non_hub_indices_1 = np.where((solution["labels"] == cluster_label_1) & (solution["labels"] != -1))[0]
        non_hub_indices_2 = np.where((solution["labels"] == cluster_label_2) & (solution["labels"] != -1))[0]
    
        if len(non_hub_indices_1) > 0 and len(non_hub_indices_2) > 0:
            
            node_index_1 = random.choice(non_hub_indices_1)
            node_index_2 = random.choice(non_hub_indices_2)
    
            
            solution["labels"][node_index_1] = cluster_label_2
            solution["labels"][node_index_2] = cluster_label_1
    
    def hill_climbing(self):
        
        self.info.hill_climbing_info()
        dialog = QDialog(self.ui.centralwidget)
        dialog.setWindowTitle("Hill Climbing Method")
    
        max_iter_label = QLabel("Maximum Iterations:")
        max_iter_input = QSpinBox()
        max_iter_input.setMinimum(1)
        max_iter_input.setMaximum(1000)
        max_iter_input.setValue(100)
    
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout = QFormLayout()
        layout.addRow(max_iter_label, max_iter_input)
        layout.addRow(button_box)
        dialog.setLayout(layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        self.app.statusBar().showMessage("Hill Climbing method running.")
        QTimer.singleShot(5000, self.app.statusBar().clearMessage)
    
        if dialog.exec() == QDialog.Accepted:
            max_iter = max_iter_input.value()
    
            if self.current_solution is None:
                self.current_solution = self.app.best_solution
                self.current_objective = self.objective_function(self.current_solution)
                
    
                self.lowest_objective = self.current_objective
                self.lowest_solution = self.current_solution
    
            found_lower = False
            
            for iteration in range(max_iter):
                new_solution = self.generate_new_solution(self.current_solution)
                new_objective = self.objective_function(new_solution)
    
                if new_objective < self.current_objective:
                    self.current_solution = new_solution
                    self.current_objective = new_objective
    
                    if new_objective < self.lowest_objective:
                        self.lowest_objective = new_objective
                        self.lowest_solution = new_solution
                        found_lower = True
    
                if found_lower:
                    
                    X = self.lowest_solution["X"]
                    labels = self.lowest_solution["labels"]
                    cluster_centers = self.lowest_solution["cluster_centers"]
                    self.plotClusters1(X, labels, cluster_centers)
                else:
                # X = self.current_solution["X"]
                # labels = self.current_solution["labels"]
                # cluster_centers = self.current_solution["cluster_centers"]
                    X = self.app.best_solution["X"]
                    labels = self.app.best_solution["labels"]
                    cluster_centers = self.app.best_solution["cluster_centers"]
                    #print("else")
                    self.plotClusters1(X, labels, cluster_centers)
                
    
            
            #self.plotClusters1(X, labels, cluster_centers)
            self.app.edit.history.append(self.current_solution)
            self.app.edit.history_index += 1
    
            # Show the lowest objective function value
            message_box = QMessageBox(self.ui.centralwidget)
            message_box.setWindowTitle("Hill Climbing Results")
            message_box.setText(f"Lowest Objective Function Value: {self.lowest_objective}")
            message_box.exec_()
    
            self.app.afterHeuristics()
            self.show_cluster_results(labels, cluster_centers)
    
    def show_cluster_results(self, labels, cluster_centers):
        """
        Displays the cluster results using labels and cluster centers.

        Args:
            labels (array-like): The cluster labels for each data point.
            cluster_centers (array-like): The cluster centers.

        Returns:
            None
        """
        n_clusters = len(np.unique(labels))
        obj = self.lowest_objective
        self.ui.results_textBrowser.clear()
        result_text = "Number of Cluster:" +str(n_clusters) + "\nCluster Labels:\n" +  "\nObjective Function Result:" +str(obj) + "\n\nCluster Centers:\n" + str(cluster_centers)
        self.ui.results_textBrowser.setText(result_text)
        
        
        
        
        
        
    def generate_new_solution(self, current_solution):
        """
        Generates a new solution by applying one of the operators.
    
        Args:
            current_solution (dict): The current solution.
    
        Returns:
            dict: The new solution.
        """
        new_solution = dict(current_solution)  # Copy the current solution
    
        # Select an operator randomly
        operator = random.choice(["relocate_hub", "reallocate_node", "swap_nodes"])
    
        # Apply the selected operator
        if operator == "relocate_hub":
            self.relocate_hub(new_solution)
        elif operator == "reallocate_node":
            self.reallocate_node(new_solution)
        elif operator == "swap_nodes":
            self.swap_nodes(new_solution)
    
        return new_solution


        
    
    def sim_annealing(self):
        self.app.statusBar().clearMessage()
        self.info.sim_annealing_info()
        dialog = QDialog(self.ui.centralwidget)
        dialog.setWindowTitle("Simulated Annealing Method")
    
        max_iter_label = QLabel("Maximum Iterations:")
        max_iter_input = QSpinBox()
        max_iter_input.setMinimum(1)
        max_iter_input.setMaximum(1000)
        max_iter_input.setValue(100)
    
        initial_temp_label = QLabel("Initial Temperature:")
        initial_temp_input = QDoubleSpinBox()
        initial_temp_input.setMinimum(0.0)
        initial_temp_input.setMaximum(1000.0)
        initial_temp_input.setValue(100.0)
    
        cooling_rate_label = QLabel("Cooling Rate:")
        cooling_rate_input = QDoubleSpinBox()
        cooling_rate_input.setMinimum(0.0)
        cooling_rate_input.setMaximum(1.0)
        cooling_rate_input.setSingleStep(0.01)
        cooling_rate_input.setValue(0.1)
    
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout = QFormLayout()
        layout.addRow(max_iter_label, max_iter_input)
        layout.addRow(initial_temp_label, initial_temp_input)
        layout.addRow(cooling_rate_label, cooling_rate_input)
        layout.addRow(button_box)
        dialog.setLayout(layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        self.app.statusBar().showMessage("Simulated Annealing method running.")
        QTimer.singleShot(5000, self.app.statusBar().clearMessage)
    
        if dialog.exec() == QDialog.Accepted:
            max_iter = max_iter_input.value()
            initial_temp = initial_temp_input.value()
            cooling_rate = cooling_rate_input.value()
    
            current_solution = self.app.best_solution
            current_objective = self.objective_function(current_solution)
            best_solution = current_solution
            best_objective = current_objective
    
            for iteration in range(max_iter):
                temperature = self.calculate_temperature(initial_temp, cooling_rate, iteration)
    
                new_solution = self.generate_new_solution(current_solution)
                new_objective = self.objective_function(new_solution)
    
                if self.acceptance_probability(current_objective, new_objective, temperature) > random.random():
                    current_solution = new_solution
                    current_objective = new_objective
    
                if new_objective < best_objective:
                    best_solution = new_solution
                    best_objective = new_objective
    
                X = best_solution["X"]
                labels = best_solution["labels"]
                cluster_centers = best_solution["cluster_centers"]
                self.plotClusters1(X, labels, cluster_centers)
    
            self.app.best_solution = best_solution
    
            # Show the best objective function value
            message_box = QMessageBox(self.ui.centralwidget)
            message_box.setWindowTitle("Simulated Annealing Results")
            message_box.setText(f"Best Objective Function Value: {best_objective}")
            message_box.exec_()
    
            self.app.afterHeuristics()
            self.show_cluster_results(labels, cluster_centers)


    def calculate_temperature(self, initial_temp, cooling_rate, iteration):
        """
        It calculates temperature value.

        Parameters
        ----------
        initial_temp : float
        cooling_rate : float
        iteration : float

        Returns: temperature value as initial_temp * (cooling_rate ** iteration)
        -------
        TYPE
            DESCRIPTION.

        """
        return initial_temp * (cooling_rate ** iteration)


    def acceptance_probability(self, current_objective, new_objective, temperature):
        """
        It calculates acceptance probability for simulated annealing method.

        Parameters
        ----------
        current_objective : float
        new_objective : flaot
        temperature : float

        Returns: acceptance probability value as np.exp((current_objective - new_objective) / temperature) OR 1.0
        -------
        TYPE
            DESCRIPTION.

        """
        if new_objective < current_objective:
            return 1.0
        else:
            return np.exp((current_objective - new_objective) / temperature)

    def plotClusters1(self, X, labels, cluster_centers):
        """
        Plots the cluster centers and labels after performing clustering operations using the heuristic method.
    
        :param X: numpy array, input data
        :param labels: array-like, cluster labels for each data point
        :param cluster_centers: array-like, cluster centers
    
        Returns: 
            None
        """
        X = np.array(X)
        fig = plt.figure() 
        plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', alpha=1)
        plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='x', color='red', s=100, label="Cluster Centers")
        plt.legend()
        for i, point in enumerate(self.app.initial_data):
            plt.text(point[0], point[1], str(i), verticalalignment='bottom', horizontalalignment='right')
        plt.close(fig)
        figure = fig
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        image = figure.canvas.tostring_rgb()
        pixmap = QPixmap.fromImage(QImage(image, width, height, QImage.Format_RGB888))
        label = self.ui.final_solution_label
        label.setPixmap(pixmap)
        label.setMinimumSize(1, 1)
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(True)
    