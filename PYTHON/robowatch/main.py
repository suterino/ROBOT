import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

import numpy as np
import trimesh
from vispy import scene, app
from vispy.scene import Mesh
from vispy.geometry import MeshData

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
        layout.setContentsMargins(0, 0, 0, 0)

        # Create Vispy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=False)
        self.view = self.canvas.central_widget.add_view()

        # Set up camera with arcball for interactive navigation
        self.view.camera = 'arcball'

        # Enable mouse wheel zoom
        self.view.camera.set_range()

        # Bind keyboard events for zoom
        self.canvas.events.key_press.connect(self.on_key_press)

        # Convert canvas to Qt widget
        native_canvas = self.canvas.native
        layout.addWidget(native_canvas)

        central_widget.setLayout(layout)

        # Setup menu bar
        self.create_menu_bar()

        self.current_mesh = None
        self.mesh_actor = None

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

                # Display the mesh with Vispy
                self.display_mesh_vispy()

                # Update window title
                self.setWindowTitle(f"RoboWatch - {Path(file_path).name}")

                print("Mesh displayed successfully")

            except Exception as e:
                print(f"Error loading file: {e}")
                import traceback
                traceback.print_exc()

    def display_mesh_vispy(self):
        """Display the mesh using Vispy"""
        if self.current_mesh is None:
            return

        # Clear previous mesh
        if self.mesh_actor is not None:
            self.mesh_actor.parent = None

        mesh = self.current_mesh
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)

        # Create MeshData from vertices and faces
        mesh_data = MeshData(
            vertices=vertices,
            faces=faces
        )

        # Create mesh actor
        self.mesh_actor = Mesh(
            meshdata=mesh_data,
            color=(0.5, 0.8, 1.0, 0.9),
            shading='flat'
        )

        self.view.add(self.mesh_actor)

        # Center and fit camera
        mesh_center = mesh.centroid
        mesh_bounds = mesh.bounds
        mesh_size = np.max(mesh_bounds[1] - mesh_bounds[0])

        # Set camera to view the mesh
        self.view.camera.center = mesh_center
        self.view.camera.distance = mesh_size * 1.5

        print("Mesh displayed in Vispy")
        print("Controls:")
        print("  - Rotate: Left-click and drag")
        print("  - Zoom: Right-click drag up/down OR Scroll wheel OR +/- keys")
        print("  - Pan: Middle-click and drag")
        print("  - Reset: Press 'Home' or 'r' key")

    def on_key_press(self, event):
        """Handle keyboard events for zoom and other controls"""
        if event.key == '+' or event.key == '=':
            # Zoom in
            self.view.camera.distance *= 0.8
            self.canvas.update()
            print("Zoomed in")
        elif event.key == '-' or event.key == '_':
            # Zoom out
            self.view.camera.distance *= 1.25
            self.canvas.update()
            print("Zoomed out")
        elif event.key == 'f':
            # Fit to view
            if self.current_mesh is not None:
                mesh_center = self.current_mesh.centroid
                mesh_bounds = self.current_mesh.bounds
                mesh_size = np.max(mesh_bounds[1] - mesh_bounds[0])
                self.view.camera.center = mesh_center
                self.view.camera.distance = mesh_size * 1.5
                self.canvas.update()
                print("Fit to view")


def main():
    print("Creating QApplication...")
    app_qt = QApplication(sys.argv)

    print("Creating window...")
    window = RoboWatchGUI()

    print("Showing window...")
    window.show()

    print("Starting event loop...")
    print("GUI is running. Use File > Load STL to load an STL file.")
    sys.exit(app_qt.exec())


if __name__ == "__main__":
    main()
