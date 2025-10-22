import sys
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QListWidget,
                             QListWidgetItem, QDockWidget, QCheckBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

import numpy as np
import pyvista as pv

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
        layout.setContentsMargins(5, 5, 5, 5)

        # Add info label about PyVista window
        info = QLabel("3D View: PyVista opens in a separate window\nUse controls below to manage the view")
        info.setStyleSheet("color: #666; font-size: 10px; padding: 10px; background: #f0f0f0; border-radius: 4px;")
        layout.addWidget(info)

        layout.addStretch()

        central_widget.setLayout(layout)

        # Create PyVista plotter with its own window
        # This will open in a separate window, not embedded in Qt
        self.plotter = pv.Plotter(off_screen=False)
        self.plotter.background_color = 'white'

        # Setup menu bar
        self.create_menu_bar()

        # Create left dock panel
        self.create_left_panel()

        # Initialize state variables
        self.current_mesh = None
        self.original_mesh = None
        self.mesh_actor = None
        self.axis_actors = {}  # Store axis actors
        self.markers_actor = None
        self.picked_points = []
        self.point_picking_mode = False
        self.top_view_mode = False

        # Store camera positions for view control
        self.saved_camera_state = None

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

        # Axes label
        axes_label = QLabel("Axes:")
        axes_label.setStyleSheet("margin-top: 15px; font-weight: bold;")
        dock_layout.addWidget(axes_label)

        # Axes checkboxes
        self.x_axis_checkbox = QCheckBox("X")
        self.x_axis_checkbox.setChecked(True)
        self.x_axis_checkbox.setStyleSheet("color: red; font-weight: bold;")
        self.x_axis_checkbox.stateChanged.connect(self.toggle_x_axis)
        dock_layout.addWidget(self.x_axis_checkbox)

        self.y_axis_checkbox = QCheckBox("Y")
        self.y_axis_checkbox.setChecked(True)
        self.y_axis_checkbox.setStyleSheet("color: green; font-weight: bold;")
        self.y_axis_checkbox.stateChanged.connect(self.toggle_y_axis)
        dock_layout.addWidget(self.y_axis_checkbox)

        self.z_axis_checkbox = QCheckBox("Z")
        self.z_axis_checkbox.setChecked(True)
        self.z_axis_checkbox.setStyleSheet("color: blue; font-weight: bold;")
        self.z_axis_checkbox.stateChanged.connect(self.toggle_z_axis)
        dock_layout.addWidget(self.z_axis_checkbox)

        # View Control label
        view_label = QLabel("View Control:")
        view_label.setStyleSheet("margin-top: 15px; font-weight: bold;")
        dock_layout.addWidget(view_label)

        # Horizontal layout for Top button and rotation controls
        view_control_layout = QHBoxLayout()

        # Top view button
        self.top_btn = QPushButton("Top")
        self.top_btn.setMaximumWidth(50)
        self.top_btn.setStyleSheet(
            "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        self.top_btn.clicked.connect(self.toggle_top_view)
        view_control_layout.addWidget(self.top_btn)

        # Counter-clockwise rotation button
        ccw_btn = QPushButton("↶ CCW")
        ccw_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 6px;"
        )
        ccw_btn.clicked.connect(self.rotate_view_ccw)
        view_control_layout.addWidget(ccw_btn)

        # Clockwise rotation button
        cw_btn = QPushButton("CW ↷")
        cw_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 6px;"
        )
        cw_btn.clicked.connect(self.rotate_view_cw)
        view_control_layout.addWidget(cw_btn)

        dock_layout.addLayout(view_control_layout)

        # Add stretch to bottom
        dock_layout.addStretch()

        # Temporary debug button
        temp_btn = QPushButton("load temp")
        temp_btn.setStyleSheet("background-color: #808080; color: white; padding: 6px;")
        temp_btn.clicked.connect(self.load_temp_file)
        dock_layout.addWidget(temp_btn)

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

    def load_temp_file(self):
        """Load temporary debug file"""
        file_path = "/Users/massimo/GitHub/ROBOT/PYTHON/robowatch/STL/watch_case_rebuiding_v24_v1.stl"
        self._load_stl(file_path)

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
            self._load_stl(file_path)

    def _load_stl(self, file_path):
        """Internal method to load STL file"""
        try:
            print(f"Loading: {file_path}")

            # Load mesh using PyVista
            self.current_mesh = pv.read(file_path)
            self.original_mesh = self.current_mesh.copy()

            print(f"Mesh loaded successfully")
            print(f"Mesh bounds: {self.current_mesh.bounds}")

            # Display the mesh
            self.display_mesh()

            # Update window title
            self.setWindowTitle(f"RoboWatch - {Path(file_path).name}")

            print("Mesh displayed successfully")

        except Exception as e:
            print(f"Error loading file: {e}")
            import traceback
            traceback.print_exc()

    def display_mesh(self):
        """Display the mesh using PyVista"""
        if self.current_mesh is None:
            return

        try:
            # Clear previous mesh
            self.plotter.clear()

            # Add mesh
            self.mesh_actor = self.plotter.add_mesh(
                self.current_mesh,
                color=(0.5, 0.8, 1.0),
                opacity=0.9
            )

            # Create and display axes
            self.create_axes()

            # Fit camera to mesh
            self.plotter.reset_camera()

            # Render
            self.plotter.render()

            print("Mesh displayed in PyVista")
            print("Controls:")
            print("  - Rotate: Left-click and drag")
            print("  - Zoom: Scroll wheel or right-click drag")
            print("  - Pan: Middle-click and drag")

        except Exception as e:
            print(f"Error displaying mesh: {e}")
            import traceback
            traceback.print_exc()

    def create_axes(self):
        """Create and display X, Y, Z axes"""
        try:
            # Clear previous axes
            for axis_name in ['x', 'y', 'z']:
                if axis_name in self.axis_actors:
                    self.plotter.remove_actor(self.axis_actors[axis_name])
            self.axis_actors = {}

            # Get mesh center and size for axis scaling
            mesh_center = self.current_mesh.center
            bounds = self.current_mesh.bounds
            mesh_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            axis_length = mesh_size * 0.3

            # X axis (red)
            x_points = [mesh_center, mesh_center + np.array([axis_length, 0, 0])]
            x_line = pv.Line(x_points[0], x_points[1])
            self.axis_actors['x'] = self.plotter.add_mesh(x_line, color='red', line_width=3)

            # Y axis (green)
            y_points = [mesh_center, mesh_center + np.array([0, axis_length, 0])]
            y_line = pv.Line(y_points[0], y_points[1])
            self.axis_actors['y'] = self.plotter.add_mesh(y_line, color='green', line_width=3)

            # Z axis (blue)
            z_points = [mesh_center, mesh_center + np.array([0, 0, axis_length])]
            z_line = pv.Line(z_points[0], z_points[1])
            self.axis_actors['z'] = self.plotter.add_mesh(z_line, color='blue', line_width=3)

            print("Axes created: Red=X, Green=Y, Blue=Z")

        except Exception as e:
            print(f"Error creating axes: {e}")
            import traceback
            traceback.print_exc()

    def toggle_x_axis(self, state):
        """Toggle X axis visibility"""
        if 'x' in self.axis_actors:
            self.axis_actors['x'].SetVisibility(state != 0)
            self.plotter.render()

    def toggle_y_axis(self, state):
        """Toggle Y axis visibility"""
        if 'y' in self.axis_actors:
            self.axis_actors['y'].SetVisibility(state != 0)
            self.plotter.render()

    def toggle_z_axis(self, state):
        """Toggle Z axis visibility"""
        if 'z' in self.axis_actors:
            self.axis_actors['z'].SetVisibility(state != 0)
            self.plotter.render()

    def toggle_top_view(self):
        """Toggle top view mode"""
        self.top_view_mode = not self.top_view_mode
        if self.top_view_mode:
            self.top_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 2px solid white;"
            )
            self.top_btn.setText("Top")
            self.set_top_view()
            print("Top View mode ON")
        else:
            self.top_btn.setStyleSheet(
                "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )
            self.top_btn.setText("Top")
            self.restore_normal_view()
            print("Top View mode OFF")

    def set_top_view(self):
        """Set camera to top view - looking straight down Z axis"""
        try:
            # Get mesh center
            mesh_center = self.current_mesh.center

            # Position camera on Z axis looking down at mesh center
            bounds = self.current_mesh.bounds
            mesh_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            camera_distance = mesh_size * 2.0

            # Set camera position (on Z axis, above the object)
            self.plotter.camera.position = (
                mesh_center[0],
                mesh_center[1],
                mesh_center[2] + camera_distance
            )

            # Set focal point to mesh center
            self.plotter.camera.focal_point = mesh_center

            # Set view up direction (Y pointing up)
            self.plotter.camera.up = (0, 1, 0)

            self.plotter.render()
            print("Top view set - camera on Z axis looking at origin, Y up, X right")

        except Exception as e:
            print(f"Error setting top view: {e}")
            import traceback
            traceback.print_exc()

    def rotate_view_cw(self):
        """Rotate camera 90 degrees clockwise around Z axis"""
        if not self.top_view_mode:
            return

        try:
            # Get mesh center
            mesh_center = self.current_mesh.center
            bounds = self.current_mesh.bounds
            mesh_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

            # Current camera position relative to mesh center
            current_pos = np.array(self.plotter.camera.position)
            relative_pos = current_pos - mesh_center

            # Rotate around Z axis by -90 degrees (clockwise)
            angle = np.radians(-90)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            rotated_pos = rot_matrix @ relative_pos
            new_pos = mesh_center + rotated_pos

            # Update camera position
            self.plotter.camera.position = new_pos
            self.plotter.camera.focal_point = mesh_center
            self.plotter.camera.up = (0, 1, 0)

            self.plotter.render()
            print("Rotated CW")

        except Exception as e:
            print(f"Error rotating CW: {e}")
            import traceback
            traceback.print_exc()

    def rotate_view_ccw(self):
        """Rotate camera 90 degrees counter-clockwise around Z axis"""
        if not self.top_view_mode:
            return

        try:
            # Get mesh center
            mesh_center = self.current_mesh.center
            bounds = self.current_mesh.bounds
            mesh_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

            # Current camera position relative to mesh center
            current_pos = np.array(self.plotter.camera.position)
            relative_pos = current_pos - mesh_center

            # Rotate around Z axis by +90 degrees (counter-clockwise)
            angle = np.radians(90)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            rotated_pos = rot_matrix @ relative_pos
            new_pos = mesh_center + rotated_pos

            # Update camera position
            self.plotter.camera.position = new_pos
            self.plotter.camera.focal_point = mesh_center
            self.plotter.camera.up = (0, 1, 0)

            self.plotter.render()
            print("Rotated CCW")

        except Exception as e:
            print(f"Error rotating CCW: {e}")
            import traceback
            traceback.print_exc()

    def restore_normal_view(self):
        """Restore normal interactive view"""
        try:
            # Reset camera
            self.plotter.reset_camera()
            self.plotter.render()
            print("Normal view restored")

        except Exception as e:
            print(f"Error restoring normal view: {e}")
            import traceback
            traceback.print_exc()

    def toggle_point_picking(self):
        """Toggle point picking mode"""
        self.point_picking_mode = not self.point_picking_mode
        if self.point_picking_mode:
            self.start_btn.setStyleSheet(
                "background-color: #f44336; color: white; font-weight: bold; padding: 8px;"
            )
            self.start_btn.setText("Stop")
            print("Point picking mode ON")
            # TODO: Implement point picking with PyVista
        else:
            self.start_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
            )
            self.start_btn.setText("Start")
            print("Point picking mode OFF")

    def add_picked_point(self, point):
        """Add a point to the picked points list"""
        self.picked_points.append(point)
        point_str = f"Point {len(self.picked_points)}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
        self.points_list.addItem(QListWidgetItem(point_str))
        print(f"Added point: {point}")
        self.update_markers()

    def update_markers(self):
        """Update marker visualization"""
        if len(self.picked_points) == 0:
            if self.markers_actor is not None:
                self.plotter.remove_actor(self.markers_actor)
                self.markers_actor = None
            return

        # Remove old markers
        if self.markers_actor is not None:
            self.plotter.remove_actor(self.markers_actor)

        # Create new markers
        points = np.array(self.picked_points)
        self.markers_actor = self.plotter.add_points(points, color='red', point_size=10)
        self.plotter.render()

    def clear_points(self):
        """Clear all picked points"""
        self.picked_points = []
        self.points_list.clear()
        self.update_markers()
        print("Points cleared")


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
