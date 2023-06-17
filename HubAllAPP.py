from ui.hubAppUI import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QPushButton
import sys
from utils.file import File
from utils.edit import Edit
from utils.clustering import Clustering
from utils.heuristics import Heuristics
from utils.info import Info



class HubAllAPP(QMainWindow):
    """
    Main application class that inherits from QMainWindow.

    Attributes
    ----------
    ui : Ui_MainWindow
        User interface object.
    cluster_center : list
        List to store cluster centers.
    labels : list
        List to store labels.
    initial_data : list
        List to store initial data.

    file : File
        File-related operations object.
    edit : Edit
        Edit-related operations object.
    cluster : Clustering
        Clustering-related operations object.
    heuristic : Heuristics
        Heuristic-related operations object.
    info : Info
        Info-related operations object.
    """

    def __init__(self):
        """
        Constructor for HubAllAPP class.

        Initializes the main application and sets up the UI.

        Returns
        -------
        None.
        """
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cluster_center = []
        self.labels = []
        self.initial_data = []
        
        self.best_solution = []

        self.file = File(self.ui, self)
        self.edit = Edit(self.ui, self)
        self.cluster = Clustering(self.ui, self)
        self.heuristic = Heuristics(self.ui, self)
        self.info = Info(self.ui, self)

        self.connections()
        self.firstScreen()

    def connections(self):
        """
        Establishes connections between buttons, actions, and functions.

        Returns
        -------
        None.
        """
        self.ui.open_data_action.triggered.connect(self.file.open_data)
        self.ui.save_initial_solution_action.triggered.connect(self.file.save_initial_solution)
        self.ui.save_final_solution_action.triggered.connect(self.file.save_final_solution)
        self.ui.exp_as_initial_solution_action.triggered.connect(self.file.export_as_initial)
        self.ui.exp_as_final_solution_action.triggered.connect(self.file.export_as_final)
        self.ui.close_action.triggered.connect(self.file.close_app)
        self.ui.clear_initial_solution_action.triggered.connect(self.edit.clear_initial)
        self.ui.clear_final_solution_action.triggered.connect(self.edit.clear_final)
        self.ui.undo_initial_solution_action.triggered.connect(self.edit.undo_initial)
        self.ui.undo_final_solution_action.triggered.connect(self.edit.undo_final)
        self.ui.redo_initial_solution_action.triggered.connect(self.edit.redo_initial)
        self.ui.redo_final_solution_action.triggered.connect(self.edit.redo_final)
        self.ui.k_means_action.triggered.connect(self.cluster.k_means)        
        self.ui.affinity_prop_action.triggered.connect(self.cluster.affinity_prop)
        self.ui.mean_shift_action.triggered.connect(self.cluster.mean_shift)
        self.ui.spectral_clustering_action.triggered.connect(self.cluster.spectral_clustering)
        self.ui.hierar_clustering_action.triggered.connect(self.cluster.hierar_clustering)
        self.ui.dbscan_action.triggered.connect(self.cluster.dbscan) 
        self.ui.hill_climbing_action.triggered.connect(self.heuristic.hill_climbing)    
        self.ui.simulated_annealing_action.triggered.connect(self.heuristic.sim_annealing)
        self.ui.about_action.triggered.connect(self.file.about_me)   
        
        
        self.ui.open_data_button.clicked.connect(self.file.open_data)
        self.ui.save_initial_solution_button.clicked.connect(self.file.save_initial_solution)
        self.ui.save_final_solution_button.clicked.connect(self.file.save_final_solution)
        self.ui.exp_as_initial_solution_button.clicked.connect(self.file.export_as_initial)
        self.ui.exp_as_final_solution_button.clicked.connect(self.file.export_as_final)
        self.ui.clear_initial_solution_button.clicked.connect(self.edit.clear_initial)
        self.ui.clear_final_solution_button.clicked.connect(self.edit.clear_final)
        self.ui.undo_initial_solution_button.clicked.connect(self.edit.undo_initial)
        self.ui.undo_final_solution_button.clicked.connect(self.edit.undo_final)
        self.ui.redo_initial_solution_button.clicked.connect(self.edit.redo_initial)
        self.ui.redo_final_solution_button.clicked.connect(self.edit.redo_final)
        self.ui.k_means_button.clicked.connect(self.cluster.k_means)    
        self.ui.affinity_prop_button.clicked.connect(self.cluster.affinity_prop)
        self.ui.mean_shift_button.clicked.connect(self.cluster.mean_shift)
        self.ui.spectral_clustering_button.clicked.connect(self.cluster.spectral_clustering)
        self.ui.hierar_clustering_button.clicked.connect(self.cluster.hierar_clustering)
        self.ui.dbscan_button.clicked.connect(self.cluster.dbscan) 
        self.ui.hill_climbing_button.clicked.connect(self.heuristic.hill_climbing)  
        self.ui.simulated_annealing_button.clicked.connect(self.heuristic.sim_annealing)
        

    def firstScreen(self):
        """
        Function to handle the first screen of the application.

        Returns
        -------
        None.
        """
        for menu in self.menuBar().findChildren(QMenu):
            for action in menu.actions():
                action.setEnabled(False)
        
        for button in self.findChildren(QPushButton):
            button.setEnabled(False)
        
        self.ui.open_data_action.setEnabled(True)
        self.ui.open_data_button.setEnabled(True)
        self.ui.close_action.setEnabled(True)
        self.ui.about_action.setEnabled(True)
        self.statusBar().showMessage("Ready")
        
        ##
        self.ui.hill_climbing_button.setEnabled(False)
        self.ui.simulated_annealing_button.setEnabled(False)

    def afterOpenData(self):
        """
        Function to handle operations after opening data.

        Returns
        -------
        None.
        """
        for action in self.ui.clustering_menu.actions():
            action.setEnabled(True)
        for button in self.ui.clustering_groupBox.findChildren(QPushButton):
            button.setEnabled(True)
        
        for action in self.ui.heuristics_menu.actions():
            action.setEnabled(True)
        for button in self.ui.heuristics_groupBox.findChildren(QPushButton):
            button.setEnabled(True)
            
        for action in self.ui.edit_menu.actions():
            action.setEnabled(True) 
            
        self.ui.clear_initial_solution_action.setEnabled(True)
        self.ui.clear_initial_solution_button.setEnabled(True)
        
        ##
        self.ui.hill_climbing_button.setEnabled(False)
        self.ui.simulated_annealing_button.setEnabled(False)

    def afterClustering(self):
        """
        Function to handle operations after clustering.

        Returns
        -------
        None.
        """
        for action in self.ui.file_menu.actions():
            action.setEnabled(True)
        self.ui.save_final_solution_action.setEnabled(False)
        self.ui.exp_as_initial_solution_action.setEnabled(True)
        self.ui.save_initial_solution_button.setEnabled(True)
        self.ui.exp_as_initial_solution_button.setEnabled(True)
        self.ui.undo_initial_solution_action.setEnabled(True)
        self.ui.undo_initial_solution_button.setEnabled(True)
        self.ui.redo_initial_solution_action.setEnabled(True)
        self.ui.redo_initial_solution_button.setEnabled(True)
        
        ##
        self.ui.hill_climbing_button.setEnabled(True)
        self.ui.simulated_annealing_button.setEnabled(True)

    def afterHeuristics(self):
        """
        Function to handle operations after applying heuristics.

        Returns
        -------
        None.
        """
        self.ui.exp_as_final_solution_action.setEnabled(True)
        self.ui.exp_as_final_solution_button.setEnabled(True)
        self.ui.save_final_solution_action.setEnabled(True)
        self.ui.save_final_solution_button.setEnabled(True)
        self.ui.undo_final_solution_action.setEnabled(True)
        self.ui.undo_final_solution_button.setEnabled(True)
        self.ui.redo_final_solution_action.setEnabled(True)
        self.ui.redo_final_solution_button.setEnabled(True)
        self.ui.clear_final_solution_action.setEnabled(True)
        self.ui.clear_final_solution_button.setEnabled(True)

    
    
        



    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HubAllAPP()
    window.show()
    sys.exit(app.exec())