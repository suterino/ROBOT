import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QListWidget,
                             QListWidgetItem, QDockWidget, QCheckBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer

import numpy as np
import trimesh
from vispy import scene, app
from vispy.scene import Mesh, Markers, Line
from vispy.geometry import MeshData
from vispy.color import Color
from vispy.util.transforms import perspective, translate, rotate as vispy_rotate

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
        self.canvas.events.mouse_move.connect(self.on_mouse_move)

        # Convert canvas to Qt widget
        native_canvas = self.canvas.native
        layout.addWidget(native_canvas)

        central_widget.setLayout(layout)

        # Setup menu bar
        self.create_menu_bar()

        # Create left dock panel
        self.create_left_panel()

        self.current_mesh = None
        self.original_mesh = None  # Store original mesh before any rotations
        self.mesh_actor = None
        self.markers_actor = None
        self.preview_marker = None
        self.axis_x_line = None
        self.axis_y_line = None
        self.axis_z_line = None
        self.picked_points = []
        self.point_picking_mode = False
        self.top_view_mode = False
        self.current_rotation = 0  # Track rotation in degrees (0, 90, 180, 270)
        self.current_rotation_matrix = None  # Store current rotation matrix to apply to axes

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

        # Top view button (small, like a radio button)
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

            # Load mesh using trimesh
            self.current_mesh = trimesh.load(file_path)

            # Store a copy of the original mesh for rotation reference
            self.original_mesh = self.current_mesh.copy()

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

        # Create and display axes (with rotation if in top view mode)
        self.create_axes(mesh_center, mesh_size * 0.3, self.current_rotation_matrix)

        print("Mesh displayed in Vispy")
        print("Controls:")
        print("  - Rotate: Left-click and drag")
        print("  - Zoom: Right-click drag up/down OR Scroll wheel OR +/- keys")
        print("  - Pan: Middle-click and drag")
        print("  - Reset: Press 'Home' or 'r' key")

    def create_axes(self, origin, axis_length, rotation_matrix=None):
        """Create and display X, Y, Z axes with optional rotation"""
        # Clear previous axes
        if self.axis_x_line is not None:
            self.axis_x_line.parent = None
        if self.axis_y_line is not None:
            self.axis_y_line.parent = None
        if self.axis_z_line is not None:
            self.axis_z_line.parent = None

        # Define axis directions
        x_dir = np.array([axis_length, 0, 0])
        y_dir = np.array([0, axis_length, 0])
        z_dir = np.array([0, 0, axis_length])

        # Apply rotation if provided
        if rotation_matrix is not None:
            x_dir = x_dir @ rotation_matrix.T
            y_dir = y_dir @ rotation_matrix.T
            z_dir = z_dir @ rotation_matrix.T

        # X axis (red)
        x_points = np.array([
            origin,
            origin + x_dir
        ])
        self.axis_x_line = Line(pos=x_points, color='red', width=2, parent=self.view.scene)

        # Y axis (green)
        y_points = np.array([
            origin,
            origin + y_dir
        ])
        self.axis_y_line = Line(pos=y_points, color='green', width=2, parent=self.view.scene)

        # Z axis (blue)
        z_points = np.array([
            origin,
            origin + z_dir
        ])
        self.axis_z_line = Line(pos=z_points, color='blue', width=2, parent=self.view.scene)

        print("Axes created: Red=X, Green=Y, Blue=Z")

    def toggle_x_axis(self, state):
        """Toggle X axis visibility"""
        if self.axis_x_line is not None:
            self.axis_x_line.parent = self.view.scene if state else None
            self.canvas.update()

    def toggle_y_axis(self, state):
        """Toggle Y axis visibility"""
        if self.axis_y_line is not None:
            self.axis_y_line.parent = self.view.scene if state else None
            self.canvas.update()

    def toggle_z_axis(self, state):
        """Toggle Z axis visibility"""
        if self.axis_z_line is not None:
            self.axis_z_line.parent = self.view.scene if state else None
            self.canvas.update()

    def toggle_top_view(self):
        """Toggle top view mode"""
        self.top_view_mode = not self.top_view_mode
        if self.top_view_mode:
            self.top_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 2px solid white;"
            )
            self.top_btn.setText("Top")
            self.set_top_view()
            print("Top View mode ON - Axes frozen, zoom in/out still working")
        else:
            self.top_btn.setStyleSheet(
                "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )
            self.top_btn.setText("Top")
            # Re-enable interactive mode
            self.view.interactive = True

            # Restore the original mesh to clean state
            # This ensures that if the user rotates the camera and clicks Top again,
            # everything is in a consistent state
            if self.original_mesh is not None:
                self.current_mesh = self.original_mesh.copy()
                self.display_mesh_vispy()

            # Reset rotation matrix
            self.current_rotation_matrix = None

            self.canvas.update()
            print("Top View mode OFF - Normal interaction enabled (mesh restored to original)")

    def set_top_view(self):
        """Set top view - rotate mesh so Z toward viewer, Y up, X right"""
        if self.current_mesh is None or self.original_mesh is None:
            return

        try:
            print("Setting top view...")

            # Create rotation matrix: +90 degrees around X axis
            # This makes:
            # - Z point toward viewer (from pointing away)
            # - Y stay pointing up
            # - X stay pointing right
            angle_x = np.radians(90)
            cos_x = np.cos(angle_x)
            sin_x = np.sin(angle_x)

            # Rotation matrix around X axis (3x3)
            rot_matrix = np.array([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], dtype=np.float32)

            # Store rotation matrix for axes visualization
            self.current_rotation_matrix = rot_matrix

            # Apply rotation to mesh vertices
            self.current_mesh = self.original_mesh.copy()
            rotated_vertices = self.current_mesh.vertices @ rot_matrix.T
            self.current_mesh.vertices = rotated_vertices

            print("Mesh vertices rotated +90 degrees around X axis")

            # Redisplay mesh with rotated axes
            self.display_mesh_vispy()

            # Set up camera for top view
            mesh_center = self.current_mesh.centroid
            mesh_bounds = self.current_mesh.bounds
            mesh_size = np.max(mesh_bounds[1] - mesh_bounds[0])

            # Reset camera to normal state
            self.view.camera.transform.reset()
            self.view.camera.center = mesh_center
            self.view.camera.distance = mesh_size * 2.0

            # Disable interactive camera rotation
            self.view.interactive = False
            print("Camera interactive mode disabled")

            self.current_rotation = 0  # Reset rotation counter
            self.canvas.update()
            print("Top view set - mesh rotated so Z points toward viewer, Y up, X right, axes stay relative to object")

        except Exception as e:
            print(f"Error setting top view: {e}")
            import traceback
            traceback.print_exc()

    def rotate_view_cw(self):
        """Rotate mesh 90 degrees clockwise (around Z axis) in top view"""
        if not self.top_view_mode or self.current_mesh is None:
            return

        try:
            # Update rotation counter
            self.current_rotation = (self.current_rotation + 90) % 360

            # Create combined rotation: X rotation (for top view) + Z rotation (for user rotation)
            # X-axis rotation (+90 degrees for top view)
            angle_x = np.radians(90)
            cos_x = np.cos(angle_x)
            sin_x = np.sin(angle_x)
            rot_x = np.array([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], dtype=np.float32)

            # Z-axis rotation (clockwise is negative)
            angle_z = np.radians(-self.current_rotation)
            cos_z = np.cos(angle_z)
            sin_z = np.sin(angle_z)
            rot_z = np.array([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1]
            ], dtype=np.float32)

            # Combine: apply Z rotation first, then X rotation
            combined = rot_x @ rot_z

            # Store rotation matrix
            self.current_rotation_matrix = combined

            # Apply rotation to mesh vertices
            self.current_mesh = self.original_mesh.copy()
            rotated_vertices = self.current_mesh.vertices @ combined.T
            self.current_mesh.vertices = rotated_vertices

            # Redisplay mesh with rotated axes
            self.display_mesh_vispy()

            self.canvas.update()
            print(f"Rotated CW: {self.current_rotation}°")
        except Exception as e:
            print(f"Error rotating CW: {e}")
            import traceback
            traceback.print_exc()

    def rotate_view_ccw(self):
        """Rotate mesh 90 degrees counter-clockwise (around Z axis) in top view"""
        if not self.top_view_mode or self.current_mesh is None:
            return

        try:
            # Update rotation counter
            self.current_rotation = (self.current_rotation - 90) % 360

            # Create combined rotation: X rotation (for top view) + Z rotation (for user rotation)
            # X-axis rotation (+90 degrees for top view)
            angle_x = np.radians(90)
            cos_x = np.cos(angle_x)
            sin_x = np.sin(angle_x)
            rot_x = np.array([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], dtype=np.float32)

            # Z-axis rotation
            angle_z = np.radians(-self.current_rotation)
            cos_z = np.cos(angle_z)
            sin_z = np.sin(angle_z)
            rot_z = np.array([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1]
            ], dtype=np.float32)

            # Combine: apply Z rotation first, then X rotation
            combined = rot_x @ rot_z

            # Store rotation matrix
            self.current_rotation_matrix = combined

            # Apply rotation to mesh vertices
            self.current_mesh = self.original_mesh.copy()
            rotated_vertices = self.current_mesh.vertices @ combined.T
            self.current_mesh.vertices = rotated_vertices

            # Redisplay mesh with rotated axes
            self.display_mesh_vispy()

            self.canvas.update()
            print(f"Rotated CCW: {self.current_rotation}°")
        except Exception as e:
            print(f"Error rotating CCW: {e}")
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
            print("Point picking mode ON - Move mouse over mesh to see preview, click to add points")
        else:
            self.start_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
            )
            self.start_btn.setText("Start")
            print("Point picking mode OFF")
            # Clear preview marker
            if self.preview_marker is not None:
                self.preview_marker.parent = None
                self.preview_marker = None

    def compute_ray_intersection(self, x, y):
        """Compute ray intersection with mesh at screen coordinates (x, y)"""
        if self.current_mesh is None:
            return None

        try:
            size = self.canvas.size
            camera = self.view.camera

            # Get camera properties
            cam_center = np.array(camera.center)
            cam_distance = camera.distance

            # Normalize screen coordinates to -1 to 1
            norm_x = (2.0 * x) / size[0] - 1.0
            norm_y = 1.0 - (2.0 * y) / size[1]

            # Simple ray computation
            ray_origin = cam_center + np.array([0, 0, cam_distance * 0.5])
            ray_direction = np.array([
                norm_x * cam_distance * 0.1,
                norm_y * cam_distance * 0.1,
                -cam_distance
            ])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            # Ray-mesh intersection using trimesh
            locations, index_ray, index_tri = self.current_mesh.ray.intersects_location(
                [ray_origin],
                [ray_direction],
                multiple_hits=False
            )

            if len(locations) > 0:
                return locations[0]
            return None
        except Exception as e:
            return None

    def on_mouse_move(self, event):
        """Handle mouse move events for preview"""
        # Disable mouse interaction in top view mode
        if self.top_view_mode:
            return

        if not self.point_picking_mode or self.current_mesh is None:
            return

        pos = event.pos
        if pos is None:
            return

        # Compute intersection
        intersection_point = self.compute_ray_intersection(pos[0], pos[1])

        # Update preview marker
        if intersection_point is not None:
            # Remove old preview marker
            if self.preview_marker is not None:
                self.preview_marker.parent = None

            # Create new preview marker (yellow/gold color)
            self.preview_marker = Markers(
                pos=np.array([intersection_point]),
                size=8,
                color='yellow',
                parent=self.view.scene
            )
            self.canvas.update()
        else:
            # No intersection - remove preview marker
            if self.preview_marker is not None:
                self.preview_marker.parent = None
                self.preview_marker = None
                self.canvas.update()

    def on_mouse_press(self, event):
        """Handle mouse press events for point picking"""
        # Disable mouse interaction in top view mode
        if self.top_view_mode:
            return

        if not self.point_picking_mode or self.current_mesh is None:
            return

        # Get mouse position
        if event.button != 1:  # Left click only
            return

        # Get the pick position
        pos = event.pos
        if pos is None:
            return

        try:
            # Use the same ray computation as preview
            intersection_point = self.compute_ray_intersection(pos[0], pos[1])

            if intersection_point is not None:
                self.add_picked_point(intersection_point)
            else:
                print("No intersection with mesh at this point")

        except Exception as e:
            print(f"Error picking point: {e}")
            import traceback
            traceback.print_exc()

    def add_picked_point(self, point):
        """Add a point to the picked points list"""
        self.picked_points.append(point)

        # Add to list widget
        point_str = f"Point {len(self.picked_points)}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
        self.points_list.addItem(QListWidgetItem(point_str))

        print(f"Added point: {point}")

        # Update visualization
        self.update_markers()

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
