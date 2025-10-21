import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QListWidget,
                             QListWidgetItem, QDockWidget)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

import numpy as np
import trimesh
from vispy import scene, app
from vispy.scene import Mesh, Markers
from vispy.geometry import MeshData
from vispy.color import Color

print("Imports successful")


class RoboWatchGUI(QMainWindow):
    def __init__(self):
        print("Initializing RoboWatchGUI...")
        super().__init__()
        self.setWindowTitle("RoboWatch - UR5e STL Analyzer")
        self.setGeometry(100, 100, 1600, 900)

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

        # Bind keyboard and mouse events
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)

        # Convert canvas to Qt widget
        native_canvas = self.canvas.native
        layout.addWidget(native_canvas)

        central_widget.setLayout(layout)

        # Setup menu bar
        self.create_menu_bar()

        # Create left dock panel
        self.create_left_panel()

        self.current_mesh = None
        self.mesh_actor = None
        self.markers_actor = None
        self.picked_points = []
        self.point_picking_mode = False

        print("RoboWatchGUI initialization complete")

    def create_left_panel(self):
        """Create the left dock panel with commands"""
        dock = QDockWidget("Commands", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)

        # Create dock widget content
        dock_widget = QWidget()
        dock_layout = QVBoxLayout()

        # Title
        title = QLabel("Path Planning")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        dock_layout.addWidget(title)

        # Start button
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.start_btn.clicked.connect(self.toggle_point_picking)
        dock_layout.addWidget(self.start_btn)

        # Points list label
        points_label = QLabel("Picked Points:")
        points_label.setStyleSheet("margin-top: 15px; font-weight: bold;")
        dock_layout.addWidget(points_label)

        # Points list
        self.points_list = QListWidget()
        dock_layout.addWidget(self.points_list)

        # Clear points button
        clear_btn = QPushButton("Clear Points")
        clear_btn.clicked.connect(self.clear_points)
        dock_layout.addWidget(clear_btn)

        # Add stretch to bottom
        dock_layout.addStretch()

        # Info label
        info_label = QLabel("Click 'Start' then click on\nthe mesh to add points")
        info_label.setStyleSheet("font-size: 10px; color: gray;")
        dock_layout.addWidget(info_label)

        dock_widget.setLayout(dock_layout)
        dock.setWidget(dock_widget)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

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

    def toggle_point_picking(self):
        """Toggle point picking mode"""
        self.point_picking_mode = not self.point_picking_mode
        if self.point_picking_mode:
            self.start_btn.setStyleSheet(
                "background-color: #f44336; color: white; font-weight: bold; padding: 8px;"
            )
            self.start_btn.setText("Stop")
            print("Point picking mode ON - Click on mesh to add points")
        else:
            self.start_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
            )
            self.start_btn.setText("Start")
            print("Point picking mode OFF")

    def on_mouse_press(self, event):
        """Handle mouse press events for point picking"""
        if not self.point_picking_mode or self.current_mesh is None:
            return

        # Get mouse position
        if event.button != 1:  # Left click only
            return

        # Get the pick position
        pos = event.pos
        if pos is None:
            return

        # Cast a ray from camera through the mouse position
        try:
            # Get the 3D position using the camera
            size = self.canvas.size
            x, y = pos[0], pos[1]

            # Normalize to -1 to 1
            norm_x = (x / size[0]) * 2 - 1
            norm_y = 1 - (y / size[1]) * 2

            # Get ray from camera
            ray_origin = self.view.camera.eye
            ray_direction = self.get_ray_direction(norm_x, norm_y)

            # Ray-mesh intersection using trimesh
            locations, index_ray, index_tri = self.current_mesh.ray.intersects_location(
                [ray_origin],
                [ray_direction],
                multiple_hits=False
            )

            if len(locations) > 0:
                # Get the closest intersection point
                point = locations[0]
                self.picked_points.append(point)

                # Add to list widget
                point_str = f"Point {len(self.picked_points)}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
                self.points_list.addItem(QListWidgetItem(point_str))

                print(f"Added point: {point}")

                # Update visualization
                self.update_markers()
        except Exception as e:
            print(f"Error picking point: {e}")

    def get_ray_direction(self, norm_x, norm_y):
        """Get ray direction from normalized coordinates"""
        # This is a simplified ray direction calculation
        # In a real implementation, you might use camera projection matrices
        fov = self.view.camera.fov[0]
        aspect = self.canvas.size[0] / self.canvas.size[1]

        # Simple calculation
        ray_dir = np.array([norm_x, norm_y, -1])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        return ray_dir

    def update_markers(self):
        """Update marker visualization"""
        if len(self.picked_points) == 0:
            if self.markers_actor is not None:
                self.markers_actor.parent = None
                self.markers_actor = None
            return

        # Remove old markers
        if self.markers_actor is not None:
            self.markers_actor.parent = None

        # Create new markers
        positions = np.array(self.picked_points)
        self.markers_actor = Markers(
            pos=positions,
            size=10,
            color='red',
            parent=self.view.scene
        )

    def clear_points(self):
        """Clear all picked points"""
        self.picked_points = []
        self.points_list.clear()
        self.update_markers()
        print("Points cleared")

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
