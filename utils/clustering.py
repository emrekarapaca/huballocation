from utils.info import Info
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QDialog, QSpinBox, QDoubleSpinBox, QLabel, QDialogButtonBox, QComboBox, QFormLayout
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
import numpy as np
import sys
from PyQt5.QtCore import QTimer

class Clustering:
    """
    The Clustering class contains methods for performing different clustering algorithms and displaying cluster results.

    Attributes:
        ui (UI): An instance of the UI class.
        app (HubAllAPP): An instance of the HubAllAPP class.
        info (Info): An instance of the Info class.

    Methods:
        k_means(): Performs the k-means clustering algorithm.
        affinity_prop(): Performs the affinity propagation clustering algorithm.
        mean_shift(): Performs the mean shift clustering algorithm.
        spectral_clustering(): Performs spectral clustering.
        hierar_clustering(): Performs hierarchical clustering.
        dbscan(): Performs DBSCAN clustering.
        calculate_cluster_centers(X, labels): Calculates the cluster centers based on the data and labels.
        show_cluster_results(labels, cluster_centers): Displays the cluster results using labels and cluster centers.
        plotClusters(X, labels, cluster_centers): Plots the cluster centers and labels
    """
    def __init__(self, ui, app):
        self.ui = ui
        self.app = app
        self.info = Info(ui, app)
        
    
    def k_means(self):
        """
        Performs the k-means clustering algorithm.

        Returns:
            None
        """
        try:
            self.info.k_means_info()
            dialog = QDialog(self.ui.centralwidget)
            dialog.setWindowTitle("K-Means Clustering")
            
            num_clusters_label = QLabel("Number of Clusters:")
            num_clusters_input = QSpinBox()
            num_clusters_input.setMinimum(1)
            num_clusters_input.setValue(8)
            
            max_iter_label = QLabel("Maximum Iterations:")
            max_iter_input = QSpinBox()
            max_iter_input.setMinimum(1)
            max_iter_input.setMaximum(1000)
            max_iter_input.setValue(100)
            
            algorithm_label = QLabel("Algorithm:")
            algorithm_input = QComboBox()
            algorithm_input.addItem("auto")
            algorithm_input.addItem("elkan")
            algorithm_input.addItem("full")    
        
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout = QFormLayout()
            layout.addRow(num_clusters_label, num_clusters_input)
            layout.addRow(max_iter_label, max_iter_input)
            layout.addRow(algorithm_label, algorithm_input)
            layout.addRow(button_box)
            dialog.setLayout(layout)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            if dialog.exec() == QDialog.Accepted:
                    num_clusters = num_clusters_input.value()
                    max_iter = max_iter_input.value()
                    algorithm = algorithm_input.currentText()
                    
                    X = self.app.initial_data
                    X = np.array(X)
                    kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, algorithm=algorithm)
                    kmeans.fit(X)
    
                    labels = kmeans.labels_
                    cluster_centers = kmeans.cluster_centers_
                    
                    self.app.edit.history_index += 1
                    solution = {"X": X, "labels": labels, "cluster_centers": cluster_centers}
                    self.solx = solution
                    self.app.edit.history.append(solution)
                    
                    self.app.best_solution = solution.copy()
            
                    self.app.cluster_center = cluster_centers
                    self.app.labels = labels
                    self.show_cluster_results(labels, cluster_centers)
                    self.app.statusBar().showMessage("K-Means completed.")
                    QTimer.singleShot(5000, self.app.statusBar().clearMessage)
                    self.plotClusters(X, labels, cluster_centers)
                    self.app.afterClustering()
        except Exception:
                error_message = str(sys.exc_info()[1])
                QMessageBox.warning(self.ui.centralwidget, "Warning", error_message)
                    
            
    def affinity_prop(self):
        """
        Performs the affinity propagation clustering algorithm.

        Returns:
            None
        """
        try:
            self.info.affinity_prop_info()
            dialog = QDialog(self.ui.centralwidget)
            dialog.setWindowTitle("Affinity Propagation Parameters")
            
            damping_label = QLabel("Damping factor in the range [0.5, 1.0):")
            
            damping_input = QDoubleSpinBox()
            damping_input.setRange(0.5, 0.99)
            damping_input.setValue(0.5)
            
            max_iter_label = QLabel("Maximum Iterations:")
            max_iter_input = QSpinBox()
            max_iter_input.setMinimum(1)
            max_iter_input.setMaximum(1000)
            max_iter_input.setValue(200)
            
            con_iter_label = QLabel("Convergence Iterations:")
            con_iter_input = QSpinBox()
            con_iter_input.setMinimum(1)
            con_iter_input.setMaximum(300)
            con_iter_input.setValue(15)
            
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            
            layout = QFormLayout()
            layout.addRow(damping_label, damping_input)
            layout.addRow(max_iter_label, max_iter_input)
            layout.addRow(con_iter_label, con_iter_input)
            layout.addRow(button_box)
            dialog.setLayout(layout)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            if dialog.exec() == QDialog().Accepted:
                damping = damping_input.value()
                max_iter = max_iter_input.value()
                con_iter = con_iter_input.value()
                
                X= self.app.initial_data
                X = np.array(X)
                affinity_propagation = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=con_iter)
                affinity_propagation.fit(self.app.initial_data)
                
                labels = affinity_propagation.labels_
                cluster_centers = affinity_propagation.cluster_centers_
                
                self.app.edit.history_index += 1
                solution = {"X": X, "labels": labels, "cluster_centers": cluster_centers}
                self.app.edit.history.append(solution)
                
                self.app.best_solution = solution.copy()
                
                self.app.cluster_center = cluster_centers
                self.app.labels = labels
                
                self.show_cluster_results(labels, cluster_centers)
                self.app.statusBar().showMessage("Affinity Propagation completed.")
                QTimer.singleShot(5000, self.app.statusBar().clearMessage)
                self.plotClusters(X, labels, cluster_centers)
                self.app.afterClustering()
        except Exception:
            error_message = str(sys.exc_info()[1])
            QMessageBox.warning(self.ui.centralwidget, "Warning", error_message)
    
    
    
    def mean_shift(self):
        """
        Performs the mean shift clustering algorithm.

        Returns:
            None
        """
        try:
            self.info.mean_shift_info()
            dialog = QDialog(self.ui.centralwidget)
            dialog.setWindowTitle("Mean-Shift Clustering")
            
            bandwidth_label = QLabel("Bandwidth")
            bandwidth_input = QDoubleSpinBox()
            bandwidth_input.setMinimum(1000)
            bandwidth_input.setMaximum(20000)
            bandwidth_input.setValue(10000)
            
            max_iter_label = QLabel("Maximum Iterations:")
            max_iter_input = QSpinBox()
            max_iter_input.setMinimum(1)
            max_iter_input.setMaximum(1000)
            max_iter_input.setValue(300)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

            layout = QFormLayout()
            layout.addRow(bandwidth_label, bandwidth_input)
            layout.addRow(max_iter_label, max_iter_input)
            layout.addRow(button_box)

            dialog.setLayout(layout)

            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            if dialog.exec() == QDialog().Accepted:
                bandwidth = bandwidth_input.value()
                max_iter = max_iter_input.value()
                
                X = self.app.initial_data
                X = np.array(X)
                
                mean_shift = MeanShift(bandwidth=bandwidth, max_iter=max_iter)
                mean_shift.fit(X)
            
                labels = mean_shift.labels_
                cluster_centers = mean_shift.cluster_centers_
                
                
                
                self.app.edit.history_index += 1
                solution = {"X": X, "labels": labels, "cluster_centers": cluster_centers}
                self.app.edit.history.append(solution)
                
                self.app.best_solution = solution.copy()
                
                self.app.cluster_center = cluster_centers
                self.app.labels = labels
                
                self.show_cluster_results(labels, cluster_centers)
                self.app.statusBar().showMessage("Mean-Shift completed.")
                QTimer.singleShot(5000, self.app.statusBar().clearMessage)
                self.plotClusters(X, labels, cluster_centers)
                self.app.afterClustering()
        
        except Exception:
            error_message = str(sys.exc_info()[1])
            QMessageBox.warning(self.ui.centralwidget, "Warning", error_message)     
    
    def spectral_clustering(self):
        
        
        """
        Performs spectral clustering.

        Returns:
            None
        """
        try:
            self.info.spectral_clustering_info()
            dialog = QDialog(self.ui.centralwidget)
            dialog.setWindowTitle("Spectral Clustering")
            
            num_clusters_label = QLabel("Number of Cluster")
            num_clusters_input = QDoubleSpinBox()
            num_clusters_input.setMinimum(1)
            num_clusters_input.setMaximum(1000)
            num_clusters_input.setValue(8)
            
            solver_label = QLabel("Eigen Solver:")
            solver_input = QComboBox()
            solver_input.addItem('arpack')
            solver_input.addItem('lobpcg')
             
           
            assign_label = QLabel("Assign Labels:")
            assign_input = QComboBox()
            assign_input.addItem('kmeans')
            assign_input.addItem('discretize')
            
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

            layout = QFormLayout()
            layout.addRow(num_clusters_label, num_clusters_input)
            layout.addRow(solver_label, solver_input)
            layout.addRow(assign_label, assign_input)
            layout.addRow(button_box)

            dialog.setLayout(layout)

            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            if dialog.exec() == QDialog().Accepted:
                num_clusters = num_clusters_input.value()
                solver = solver_input.currentText()
                assign = assign_input.currentText()
                num_clusters = int(num_clusters)
                
                
                X = self.app.initial_data
                X = np.array(X)
                
                spectral_clustering = SpectralClustering(n_clusters=num_clusters, eigen_solver=solver, assign_labels=assign, affinity="nearest_neighbors")
                spectral_clustering.fit(X)

                labels = spectral_clustering.labels_
            
                cluster_centers = self.calculate_cluster_centers(X, labels)
                self.app.cluster_center = cluster_centers
                self.app.labels = labels
                
                self.plotClusters(X, labels, cluster_centers)
                self.app.afterClustering()
                
                self.app.edit.history_index += 1
                solution = {"X": X, "labels": labels, "cluster_centers": cluster_centers}
                self.app.edit.history.append(solution)
                self.app.best_solution = solution.copy()
                
                result_text = "Cluster Labels:\n" + str(labels) + "\nNumber of Cluster:\n" +str(num_clusters) +  "\n\nCluster Centers:\n" + str(cluster_centers)
                self.ui.results_textBrowser.setText(result_text)
                
                self.app.statusBar().showMessage("Spectral Clustering completed.")
                QTimer.singleShot(5000, self.app.statusBar().clearMessage)
                self.plotClusters(X, labels, cluster_centers)
                self.app.afterClustering()
                
        except Exception:
            error_message = str(sys.exc_info()[1])
            QMessageBox.warning(self.ui.centralwidget, "Warning", error_message)
        
            
        
            
    
    
    def hierar_clustering(self):
        """
        Performs hierarchical clustering.
        (Agglomerative Clusteringz)
        Returns:
            None
        """
        try:
            self.info.hierarchical_clustering_info()
            dialog = QDialog(self.ui.centralwidget)
            dialog.setWindowTitle("Spectral Clustering")
            
            num_clusters_label = QLabel("Number of Cluster")
            num_clusters_input = QDoubleSpinBox()
            num_clusters_input.setMinimum(1)
            num_clusters_input.setMaximum(1000)
            num_clusters_input.setValue(8)
            
            linkage_label = QLabel("Eigen Solver:")
            linkage_input = QComboBox()
            linkage_input.addItem('ward')
            linkage_input.addItem('complete')
            linkage_input.addItem('average')
            linkage_input.addItem('single')
           
            
            
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

            layout = QFormLayout()
            layout.addRow(num_clusters_label, num_clusters_input)
            layout.addRow(linkage_label, linkage_input)
            layout.addRow(button_box)

            dialog.setLayout(layout)

            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            if dialog.exec() == QDialog().Accepted:
                num_clusters = num_clusters_input.value()
                linkage = linkage_input.currentText()
                num_clusters = int(num_clusters)
                
                
                X = self.app.initial_data
                X = np.array(X)
                
                hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage)
                hierarchical_clustering.fit(X)
                

                labels = hierarchical_clustering.labels_
            
                cluster_centers = self.calculate_cluster_centers(X, labels)
                self.app.cluster_center = cluster_centers
                self.app.labels = labels
                
                self.plotClusters(X, labels, cluster_centers)
                self.app.afterClustering()
                
                self.app.edit.history_index += 1
                solution = {"X": X, "labels": labels, "cluster_centers": cluster_centers}
                self.app.edit.history.append(solution)
                self.app.best_solution = solution.copy()
                
                
                result_text = "Cluster Labels:\n" + str(labels) + "\nNumber of Cluster:\n" +str(num_clusters) +  "\n\nCluster Centers:\n" + str(cluster_centers)
                self.ui.results_textBrowser.setText(result_text)
                
                self.app.statusBar().showMessage("Hierarchical Clustering completed.")
                QTimer.singleShot(5000, self.app.statusBar().clearMessage)
                self.plotClusters(X, labels, cluster_centers)
                self.app.afterClustering()
                
        except Exception:
            error_message = str(sys.exc_info()[1])
            QMessageBox.warning(self.ui.centralwidget, "Warning", error_message)
        
    
    def dbscan(self):
        """
        Performs DBSCAN clustering.

        Returns:
            None
        """
        try: 
            self.info.dbscan_info()
            dialog = QDialog(self.ui.centralwidget)
            dialog.setWindowTitle("DBSCAN")
            
            eps_label = QLabel("eps: (Maximum distance btw 2 sample:)")
            eps_input = QDoubleSpinBox()
            eps_input.setMinimum(0.1)
            eps_input.setMaximum(50000)
            eps_input.setValue(7000)
            
            min_samples_label = QLabel("Minimum Samples:")
            min_samples_input = QDoubleSpinBox()
            min_samples_input.setMinimum(1)
            min_samples_input.setMaximum(100)
            min_samples_input.setValue(5)
            
            alg_label = QLabel("Algorithm:")
            alg_input = QComboBox()
            alg_input.addItem('auto')
            alg_input.addItem('ball_tree')
            alg_input.addItem('kd_tree')
            alg_input.addItem('brute')
           
            
            
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

            layout = QFormLayout()
            layout.addRow(eps_label, eps_input)
            layout.addRow(min_samples_label, min_samples_input)
            layout.addRow(alg_label, alg_input)
            layout.addRow(button_box)

            dialog.setLayout(layout)

            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            if dialog.exec() == QDialog().Accepted:
                eps = eps_input.value()
                min_samples = min_samples_input.value()
                alg = alg_input.currentText()
                #eps = int(eps)
                #min_samples = int(min_samples)
                
                
                X = self.app.initial_data
                X = np.array(X)
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm=alg)
                dbscan.fit(X)
                
                

                labels = dbscan.labels_
                
            
                cluster_centers = self.calculate_cluster_centers(X, labels)
                self.app.cluster_center = cluster_centers
                self.app.labels = labels
                
                self.plotClusters(X, labels, cluster_centers)
                self.app.afterClustering()
                
                self.app.edit.history_index += 1
                solution = {"X": X, "labels": labels, "cluster_centers": cluster_centers}
                self.app.edit.history.append(solution)
                self.app.best_solution = solution.copy()
                
                num_clusters =  len(np.unique(labels))
                result_text = "Cluster Labels:\n" + str(labels) + "\nNumber of Cluster:\n" +str(num_clusters) +  "\n\nCluster Centers:\n" + str(cluster_centers)
                self.ui.results_textBrowser.setText(result_text)
                
                self.app.statusBar().showMessage("DBSCAN completed.")
                QTimer.singleShot(5000, self.app.statusBar().clearMessage)
                self.plotClusters(X, labels, cluster_centers)
                self.app.afterClustering()
                
        except Exception:
            error_message = str(sys.exc_info()[1])
            QMessageBox.warning(self.ui.centralwidget, "Warning", error_message)
    
    def calculate_cluster_centers(self, X, labels):
        """
        Calculates the cluster centers based on the data and labels.

        Args:
            X (array-like): The data points.
            labels (array-like): The cluster labels for each data point.

        Returns:
            None
        """
        cluster_centers = []
        unique_labels = np.unique(labels)
    
        for label in unique_labels:
            cluster_points = X[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
    
        return np.array(cluster_centers)
    
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
        obj = self.app.heuristic.objective_function(self.app.best_solution)
        result_text = "Number of Cluster:" +str(n_clusters) + "\nCluster Labels:\n" +str(labels)+  "\nObjective Function Result:" +str(obj) + "\n\nCluster Centers:\n" + str(cluster_centers)
        self.ui.results_textBrowser.setText(result_text)
        

    def plotClusters(self, X, labels, cluster_centers):
        """
        Plots the cluster centers and labels after performing clustering operations using the K-Means method.
    
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
        label = self.ui.initial_solution_label
        label.setPixmap(pixmap)
        label.setMinimumSize(1, 1)
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(True)
        
    