import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

import numpy as np
import trimesh
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

print("Imports successful")


class RoboWatchGUI(QMainWindow):
    def __init__(self):
        print("Initializing RoboWatchGUI...")
        super().__init__()
        self.setWindowTitle("RoboWatch - UR5e STL Analyzer")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        central_widget.setLayout(layout)

        # Setup menu bar
        self.create_menu_bar()

        self.current_mesh = None
        self.ax = None

        print("RoboWatchGUI initialization complete")

    def create_menu_bar(self):
        """Create the menu bar with File menu"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Load action
        load_action = QAction("Load STL", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_stl_file)
        file_menu.addAction(load_action)

        # Separator
        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def load_stl_file(self):
        """Open file dialog and load STL file"""
        file_dialog = QFileDialog()
        stl_dir = Path(__file__).parent / "STL"

        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Load STL File",
            str(stl_dir),
            "STL Files (*.stl);;All Files (*)"
        )

        if file_path:
            try:
                print(f"Loading: {file_path}")

                # Load mesh using trimesh
                self.current_mesh = trimesh.load(file_path)

                print(f"Mesh loaded successfully")
                print(f"Mesh bounds: {self.current_mesh.bounds}")
                print(f"Mesh volume: {self.current_mesh.volume}")

                # Display the mesh with matplotlib
                self.display_mesh_matplotlib()

                # Update window title
                self.setWindowTitle(f"RoboWatch - {Path(file_path).name}")

                print("Mesh displayed successfully")

            except Exception as e:
                print(f"Error loading file: {e}")
                import traceback
                traceback.print_exc()

    def display_mesh_matplotlib(self):
        """Display the mesh using matplotlib embedded in Qt"""
        if self.current_mesh is None:
            return

        mesh = self.current_mesh
        vertices = mesh.vertices
        faces = mesh.faces

        # Clear previous plot
        self.figure.clear()

        # Create 3D subplot
        self.ax = self.figure.add_subplot(111, projection='3d')

        # Plot the mesh
        self.ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            color='lightblue',
            edgecolor='darkblue',
            linewidth=0.1,
            alpha=0.9
        )

        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"STL Viewer - Rotate, Zoom, Pan")

        # Get mesh bounds and set equal aspect ratio
        bounds = mesh.bounds
        center = mesh.centroid
        max_range = np.max(bounds[1] - bounds[0]) / 2.0

        self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
        self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
        self.ax.set_zlim(center[2] - max_range, center[2] + max_range)

        # Draw and show
        self.figure.tight_layout()
        self.canvas.draw()

        print("Mesh displayed in application window")
        print("Controls:")
        print("  - Rotate: Click and drag")
        print("  - Zoom: Scroll wheel or use toolbar")
        print("  - Pan: Right-click and drag (or use toolbar)")


def main():
    print("Creating QApplication...")
    app = QApplication(sys.argv)

    print("Creating window...")
    window = RoboWatchGUI()

    print("Showing window...")
    window.show()

    print("Starting event loop...")
    print("GUI is running. Use File > Load STL to load an STL file.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
