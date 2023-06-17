from PyQt5.QtCore import QTimer
class Edit:
    """
    The Edit class contains methods for performing edit-related operations.

    Attributes:
        ui (UI): An instance of the UI class.
        app (HubAllAPP): An instance of the HubAllAPP class.

    Methods:
        clear_initial(): Clears the initial solution.
        clear_final(): Clears the final solution.
        undo_initial(): Performs an undo operation on the initial solution.
        undo_final(): Performs an undo operation on the final solution.
        redo_initial(): Performs a redo operation on the initial solution.
        redo_final(): Performs a redo operation on the final solution.
    """
    def __init__(self, ui, app):
        self.ui = ui
        self.app = app
        
        self.history = []  
        self.history_index = -1
    
    def clear_initial(self):
        """
        Clears the initial solution.

        Returns:
            None
        """
        self.ui.initial_solution_label.clear()
        self.clear_final()
        self.app.firstScreen()
        self.ui.results_textBrowser.clear()
        self.ui.info_textBrowser.clear()
        self.history = []
        self.history_index = -1
    
    def clear_final(self):
        """
        Clears the final solution.

        Returns:
            None
        """
        self.ui.final_solution_label.clear()
    
    def undo_initial(self):
        """
        Performs an undo operation on the initial solution.
        """
        if self.history_index > 0:
            self.history_index -= 1
            solution = self.history[self.history_index]
            X = solution["X"]
            labels = solution["labels"]
            cluster_centers = solution["cluster_centers"]
            self.app.cluster.plotClusters(X, labels, cluster_centers)
            self.app.statusBar().showMessage("Undo operation performed.")
            QTimer.singleShot(5000, self.app.statusBar().clearMessage)
        else:
            self.app.statusBar().showMessage("No previous changes.")
            QTimer.singleShot(5000, self.app.statusBar().clearMessage)
            
    
    def undo_final(self):
        """
        Performs an undo operation on the final solution.
        """
        if self.history_index > 0:
            self.history_index -= 1
            solution = self.history[self.history_index]
            X = solution["X"]
            labels = solution["labels"]
            cluster_centers = solution["cluster_centers"]
            self.app.heuristic.plotClusters1(X, labels, cluster_centers)
            self.app.statusBar().showMessage("Undo operation performed.")
            QTimer.singleShot(5000, self.app.statusBar().clearMessage)
        else:
            self.app.statusBar().showMessage("No previous changes.")
            QTimer.singleShot(5000, self.app.statusBar().clearMessage)
    
    def redo_initial(self):
        """
        Performs a redo operation on the initial solution.
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            solution = self.history[self.history_index]
            X = solution["X"]
            labels = solution["labels"]
            cluster_centers = solution["cluster_centers"]
            self.app.cluster.plotClusters(X, labels, cluster_centers)
            self.app.statusBar().showMessage("Redo operation performed.")
            QTimer.singleShot(5000, self.app.statusBar().clearMessage)
        else:
            self.app.statusBar().showMessage("No changes.")
            QTimer.singleShot(5000, self.app.statusBar().clearMessage)
    
    def redo_final(self):
        """
        Performs a redo operation on the final solution.
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            solution = self.history[self.history_index]
            X = solution["X"]
            labels = solution["labels"]
            cluster_centers = solution["cluster_centers"]
            self.app.heuristic.plotClusters1(X, labels, cluster_centers)
            self.app.statusBar().showMessage("Redo operation performed.")
            QTimer.singleShot(5000, self.app.statusBar().clearMessage)
        else:
            self.app.statusBar().showMessage("No changes.")
            QTimer.singleShot(5000, self.app.statusBar().clearMessage)
