from PyQt5.QtWidgets import QMessageBox, QFileDialog
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys

class File:
    """
    Represents a file object.

    This class provides methods to read, write, and manipulate files.

    Attributes:
        ui (UI): An instance of the UI class.
        app (HubAllAPP): An instance of the HubAllAPP class.

    Methods:
        open_data(): Performs the operation of opening data.
        save_initial_solution(): Performs the operation of saving the initial solution.
        save_final_solution(): Performs the operation of saving the final solution.
        export_as_initial(): Performs the operation of exporting as the initial solution.
        export_as_final(): Performs the operation of exporting as the final solution.
        plot_initial_data():  Function to plot the initial data.
    """
    def __init__(self, ui, app):
        self.ui = ui
        self.app = app
    
    def open_data(self):
        """
        Performs the operation of opening data.

        Returns:
            None
        """
        try:
            
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("Text Files (*.txt)")
            file_dialog.setWindowTitle("Open the Data")
            
            if file_dialog.exec():
                selected_file = file_dialog.selectedFiles()[0]
                with open(selected_file, 'r') as file:
                    self.app.initial_data.clear()
                    
                    for line in file:
                        coords = line.split()
                        if len(coords) == 2:
                            x = float(coords[0])
                            y = float(coords[1])
                            point = (x,y)
                            self.app.initial_data.append(point)
                self.plot_initial_data()
                self.app.afterOpenData()
        except Exception:
            error_message = str(sys.exc_info()[1])
            QMessageBox.warning(self.ui.centralwidget, "Warning", error_message)
                        
    def save_initial_solution(self):
        """
        Performs the operation of saving the initial solution.
        It saves with .txt suffix
        Returns:
            None
        """
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Text Files (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                solution = self.app.initial_data
                data_string = '\n'.join(str(point).replace("(", "").replace(")", "").replace(",", "") for point in solution)
                with open(file_path, 'w') as file:
                    file.write(data_string)
        
    
    def save_final_solution(self):
        """
        Performs the operation of saving the final solution.
        It saves with .txt suffix
        Returns:
            None
        """
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Text Files (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                solution = self.app.initial_data
                data_string = '\n'.join(str(point).replace("(", "").replace(")", "").replace(",", "") for point in solution)
                with open(file_path, 'w') as file:
                    file.write(data_string)
    def export_as_initial(self):
        """
        Performs the operation of exporting as the initial solution.
        It exports as JPG file. 
        Returns:
            None
        """
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("JPG Files (*jpg)")
        file_dialog.setDefaultSuffix("jpg")
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                pixmap = self.ui.initial_solution_label.pixmap()
                pixmap.save(file_path, "JPG")
            else:
                pass
    
    def export_as_final(self):
        """
        Performs the operation of exporting as the final solution.
        It exports as JPG file.
        Returns:
            None
        """
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("JPG Files (*jpg)")
        file_dialog.setDefaultSuffix("jpg")
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                pixmap = self.ui.final_solution_label.pixmap()
                pixmap.save(file_path, "JPG")
            else:
                pass
    
    def close_app(self):
        """
        Function to handle closing the application.
    
        Returns
        -------
        None.
        """
        reply = QMessageBox.question(self.app, "Close Application", "Are you sure you want to close the application?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)  
        
        if reply == QMessageBox.Yes:
            self.app.close()
    
    def about_me(self):
        """
        Function to display information about the developer.
    
        Returns
        -------
        None.
        """
        message_box = QMessageBox()
        message_box.setWindowTitle("About me")
        message_box.setText("Emre Karapaça\n151220184005")
        message_box.setIcon(QMessageBox.Information)
        message_box.exec()
        
    
    def plot_initial_data(self):
        """
        Function to plot the initial data.
    
        Returns
        -------
        None.
        """
        
            
        x = [point[0] for point in self.app.initial_data]
        y = [point[1] for point in self.app.initial_data]
        self.ui.initial_solution_label.clear()
        
        fig = plt.figure()
        plt.scatter(x, y, color='black', marker='o')
        
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
        label.setMinimumSize(1,1)
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(True)


        
        
