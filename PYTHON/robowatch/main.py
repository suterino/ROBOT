import sys
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QListWidget,
                             QListWidgetItem, QDockWidget, QCheckBox, QSlider, QSpinBox)
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
        self.setGeometry(100, 100, 350, 700)  # Compact window size for control panel only

        # Create minimal central widget (mostly hidden)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        central_widget.setLayout(layout)

        # Set central widget to be hidden/minimal
        central_widget.setMaximumWidth(1)
        central_widget.setMaximumHeight(1)

        # Don't create plotter yet - create it when mesh is loaded
        # This avoids creating an empty window upfront
        self.plotter = None

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
        self.mesh_edges_visible = False
        self.mesh_opacity = 0.9
        self.zoom_level = 1.0  # Default zoom level

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
        title = QLabel("RoboWatch Controls")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        dock_layout.addWidget(title)

        # Controls info box
        controls_label = QLabel("3D View Controls:")
        controls_label.setStyleSheet("margin-top: 8px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(controls_label)

        controls_info = QLabel(
            "🖱 Scroll: Zoom\n"
            "🖱 Left-drag: Rotate\n"
            "🖱 Middle-drag: Pan\n"
            "⌨ +/-: Zoom in/out"
        )
        controls_info.setStyleSheet(
            "font-size: 9px; color: #555; padding: 6px; "
            "background: #f9f9f9; border-radius: 3px; border: 1px solid #e0e0e0;"
        )
        dock_layout.addWidget(controls_info)

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

        # Camera Control label
        camera_label = QLabel("Camera Control:")
        camera_label.setStyleSheet("margin-top: 15px; font-weight: bold;")
        dock_layout.addWidget(camera_label)

        # Zoom label
        zoom_label = QLabel("Zoom: 1.0x")
        zoom_label.setStyleSheet("font-size: 10px; color: #666;")
        self.zoom_label = zoom_label
        dock_layout.addWidget(zoom_label)

        # Zoom slider (0.1x to 5x)
        zoom_layout = QHBoxLayout()
        zoom_slider = QSlider(Qt.Orientation.Horizontal)
        zoom_slider.setMinimum(10)  # 0.1x
        zoom_slider.setMaximum(500)  # 5.0x
        zoom_slider.setValue(100)  # 1.0x (default)
        zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        zoom_slider.setTickInterval(10)
        # Use sliderMoved for smooth dragging feedback
        zoom_slider.sliderMoved.connect(self.on_zoom_slider_change)
        zoom_slider.valueChanged.connect(self.on_zoom_slider_change)
        self.zoom_slider = zoom_slider
        zoom_layout.addWidget(zoom_slider)
        dock_layout.addLayout(zoom_layout)

        # Mesh Display label
        mesh_label = QLabel("Mesh Display:")
        mesh_label.setStyleSheet("margin-top: 15px; font-weight: bold;")
        dock_layout.addWidget(mesh_label)

        # Opacity label
        opacity_label = QLabel("Opacity: 90%")
        opacity_label.setStyleSheet("font-size: 10px; color: #666;")
        self.opacity_label = opacity_label
        dock_layout.addWidget(opacity_label)

        # Opacity slider (0-100)
        opacity_layout = QHBoxLayout()
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(90)
        opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        opacity_slider.setTickInterval(10)
        # Use sliderMoved for smooth dragging feedback
        opacity_slider.sliderMoved.connect(self.on_opacity_slider_change)
        opacity_slider.valueChanged.connect(self.on_opacity_slider_change)
        self.opacity_slider = opacity_slider
        opacity_layout.addWidget(opacity_slider)
        dock_layout.addLayout(opacity_layout)

        # Edges button
        edges_btn = QPushButton("Edges")
        edges_btn.setStyleSheet("background-color: #FF5722; color: white; padding: 6px; font-size: 9px;")
        edges_btn.clicked.connect(self.toggle_mesh_edges)
        dock_layout.addWidget(edges_btn)

        # Add stretch to bottom
        dock_layout.addStretch()

        # Temporary debug button
        temp_btn = QPushButton("load temp")
        temp_btn.setStyleSheet("background-color: #808080; color: white; padding: 6px;")
        temp_btn.clicked.connect(self.load_temp_file)
        dock_layout.addWidget(temp_btn)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 10px; color: #666; padding: 5px; background: #f5f5f5; border-radius: 3px;")
        dock_layout.addWidget(self.status_label)

        # Info label
        info_label = QLabel("Click 'Start' then click on\nthe mesh to add points")
        info_label.setStyleSheet("font-size: 10px; color: gray;")
        dock_layout.addWidget(info_label)

        dock_widget.setLayout(dock_layout)
        dock_widget.setMaximumWidth(280)  # Limit dock width
        dock.setWidget(dock_widget)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [280], Qt.Orientation.Horizontal)

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
        self.status_label.setText("Loading temp file...")
        print("load_temp_file() called")
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
            self.status_label.setText("Reading STL file...")
            print(f"Loading: {file_path}")

            # Load mesh using PyVista
            self.current_mesh = pv.read(file_path)
            self.original_mesh = self.current_mesh.copy()

            self.status_label.setText("Mesh loaded, creating viewer...")
            print(f"Mesh loaded successfully")
            print(f"Mesh bounds: {self.current_mesh.bounds}")

            # Display the mesh
            self.display_mesh()

            # Update window title
            self.setWindowTitle(f"RoboWatch - {Path(file_path).name}")
            self.status_label.setText("Mesh ready! Check PyVista window")

            print("Mesh displayed successfully")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:50]}")
            print(f"Error loading file: {e}")
            import traceback
            traceback.print_exc()

    def display_mesh(self):
        """Display the mesh using PyVista"""
        if self.current_mesh is None:
            return

        try:
            # Create plotter if it doesn't exist
            if self.plotter is None:
                self.status_label.setText("Creating PyVista window...")
                print("Creating PyVista plotter window...")
                self.plotter = pv.Plotter(off_screen=False)
                self.plotter.background_color = 'white'
                print("  ✓ PyVista window created")

            # Clear previous mesh
            self.plotter.clear()
            self.status_label.setText("Clearing old mesh...")
            print("  ✓ Previous mesh cleared")

            # Add mesh
            self.status_label.setText("Adding mesh...")
            print("  ✓ Adding mesh to plotter...")
            self.mesh_actor = self.plotter.add_mesh(
                self.current_mesh,
                color=(0.5, 0.8, 1.0),
                opacity=0.9
            )
            print("  ✓ Mesh added")

            # Create and display axes
            self.status_label.setText("Creating axes...")
            print("  ✓ Creating axes...")
            self.create_axes()

            # Set camera to top view (Z toward viewer, X horizontal, Y vertical)
            self.status_label.setText("Setting camera to top view...")
            print("  ✓ Setting camera to top view...")

            mesh_center = self.current_mesh.center
            bounds = self.current_mesh.bounds
            mesh_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            camera_distance = mesh_size * 2.0

            # Position camera on Z axis looking down at mesh
            # This gives us: Z toward viewer (blue axis as a point), X horizontal (red), Y vertical (green)
            self.plotter.camera.position = (
                mesh_center[0],
                mesh_center[1],
                mesh_center[2] + camera_distance
            )
            self.plotter.camera.focal_point = mesh_center
            self.plotter.camera.up = (0, 1, 0)  # Y points up

            # Save the initial camera state as the "top view" state
            self.saved_camera_state = {
                'position': tuple(self.plotter.camera.position),
                'focal_point': tuple(self.plotter.camera.focal_point),
                'up': tuple(self.plotter.camera.up)
            }
            print(f"  ✓ Saved initial camera state for Top View")
            print(f"    Position: {self.saved_camera_state['position']}")
            print(f"    Focal Point: {self.saved_camera_state['focal_point']}")
            print(f"    Up: {self.saved_camera_state['up']}")

            # Automatically activate top view mode on load (view is already top view, freeze it)
            self.top_view_mode = True
            self.top_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 2px solid white;"
            )

            # Freeze the interaction immediately since we're in top view
            if hasattr(self.plotter, 'iren') and self.plotter.iren:
                try:
                    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleNone
                    frozen_style = vtkInteractorStyleNone()
                    self.plotter.iren.SetInteractorStyle(frozen_style)
                    print("  ✓ Top view mode ACTIVATED on load - interaction FROZEN")
                except Exception as e:
                    print(f"  ! Warning: Could not freeze interaction: {e}")

            # Render
            self.status_label.setText("Rendering...")
            print("  ✓ Rendering mesh...")
            self.plotter.render()

            # Initialize interactor to make window visible
            print("  ✓ Initializing interactor...")
            self.plotter.iren.initialize()
            print("  ✓ Interactor initialized - window should be visible now")

            self.status_label.setText("Done! Mesh displayed")
            print("\nMesh displayed in PyVista!")
            print("Controls:")
            print("  - Rotate: Left-click and drag")
            print("  - Zoom: Scroll wheel or right-click drag")
            print("  - Pan: Middle-click and drag")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:40]}")
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
        if self.plotter and 'x' in self.axis_actors:
            self.axis_actors['x'].SetVisibility(state != 0)
            self.plotter.render()

    def toggle_y_axis(self, state):
        """Toggle Y axis visibility"""
        if self.plotter and 'y' in self.axis_actors:
            self.axis_actors['y'].SetVisibility(state != 0)
            self.plotter.render()

    def toggle_z_axis(self, state):
        """Toggle Z axis visibility"""
        if self.plotter and 'z' in self.axis_actors:
            self.axis_actors['z'].SetVisibility(state != 0)
            self.plotter.render()

    def on_opacity_slider_change(self, value):
        """Handle opacity slider change (0-100)"""
        if not self.plotter or not self.mesh_actor:
            return

        # Convert 0-100 slider value to 0.0-1.0 opacity
        self.mesh_opacity = value / 100.0
        self.mesh_actor.GetProperty().SetOpacity(self.mesh_opacity)

        # Force render window update
        self.plotter.render_window.Render()

        # Process both Qt and VTK events for smooth updates
        QApplication.instance().processEvents()

        # Update label
        self.opacity_label.setText(f"Opacity: {value}%")

    def on_zoom_slider_change(self, value):
        """Handle zoom slider change (10-500 = 0.1x-5.0x)"""
        if not self.plotter:
            return

        # Convert slider value (10-500) to zoom factor (0.1-5.0)
        target_zoom = value / 100.0

        # Calculate the zoom factor needed to achieve this
        zoom_factor = target_zoom / self.zoom_level

        # Apply the zoom
        self.plotter.camera.zoom(zoom_factor)

        # Force render window update
        self.plotter.render_window.Render()

        # Process both Qt and VTK events for smooth updates
        QApplication.instance().processEvents()

        # Update state
        self.zoom_level = target_zoom

        # Update label
        self.zoom_label.setText(f"Zoom: {target_zoom:.1f}x")

    def toggle_mesh_edges(self):
        """Toggle mesh edges visibility"""
        if not self.plotter or not self.mesh_actor:
            return

        self.mesh_edges_visible = not self.mesh_edges_visible
        if self.mesh_edges_visible:
            self.mesh_actor.GetProperty().EdgeVisibilityOn()
            self.mesh_actor.GetProperty().SetEdgeColor([0, 0, 0])  # Black edges
            print("Mesh edges ON")
        else:
            self.mesh_actor.GetProperty().EdgeVisibilityOff()
            print("Mesh edges OFF")

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
        """Set camera to top view - restore initial camera position and freeze interaction"""
        if not self.plotter or not self.saved_camera_state:
            print("Error: No mesh loaded or camera state not saved. Click 'load temp' first.")
            return

        try:
            # Restore the saved camera state from when mesh was loaded
            self.plotter.camera.position = self.saved_camera_state['position']
            self.plotter.camera.focal_point = self.saved_camera_state['focal_point']
            self.plotter.camera.up = self.saved_camera_state['up']

            # Freeze mouse interaction by setting a None style
            if hasattr(self.plotter, 'iren') and self.plotter.iren:
                try:
                    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleNone
                    frozen_style = vtkInteractorStyleNone()
                    self.plotter.iren.SetInteractorStyle(frozen_style)
                    print("  ✓ Mouse interaction FROZEN")
                except Exception as freeze_error:
                    print(f"  ! Warning: Could not freeze interaction: {freeze_error}")

            # Force render update
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

            print("Top view restored - camera position:")
            print(f"  Position: {self.plotter.camera.position}")
            print(f"  Focal Point: {self.plotter.camera.focal_point}")
            print(f"  Up: {self.plotter.camera.up}")

        except Exception as e:
            print(f"Error setting top view: {e}")
            import traceback
            traceback.print_exc()

    def rotate_view_cw(self):
        """Rotate camera 90 degrees clockwise around Z axis"""
        if not self.top_view_mode or not self.plotter:
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
        if not self.top_view_mode or not self.plotter:
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
        """Restore normal interactive view - keep camera position, allow interaction"""
        if not self.plotter:
            return

        try:
            # Re-enable mouse interaction by setting trackball style (default PyVista style)
            if hasattr(self.plotter, 'iren') and self.plotter.iren:
                try:
                    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
                    trackball_style = vtkInteractorStyleTrackballCamera()
                    self.plotter.iren.SetInteractorStyle(trackball_style)
                    print("  ✓ Mouse interaction UNFROZEN")
                except Exception as unfreeze_error:
                    print(f"  ! Warning: Could not unfreeze interaction: {unfreeze_error}")

            # Render and allow interaction again
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()
            print("Normal view restored - interaction enabled, camera position kept")
            print(f"  Position: {self.plotter.camera.position}")

        except Exception as e:
            print(f"Error restoring normal view: {e}")
            import traceback
            traceback.print_exc()

    def zoom_in(self):
        """Zoom in using camera zoom"""
        if not self.plotter:
            return

        try:
            self.plotter.camera.zoom(1.2)  # Zoom in by 20%
            self.plotter.render()
            print("Zoomed in")
        except Exception as e:
            print(f"Error zooming in: {e}")

    def zoom_out(self):
        """Zoom out using camera zoom"""
        if not self.plotter:
            return

        try:
            self.plotter.camera.zoom(0.8)  # Zoom out by 20%
            self.plotter.render()
            print("Zoomed out")
        except Exception as e:
            print(f"Error zooming out: {e}")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()

        # Plus/Equals key for zoom in
        if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.zoom_in()
            event.accept()
        # Minus key for zoom out
        elif key == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
        else:
            super().keyPressEvent(event)

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
