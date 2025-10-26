import sys
import time
import json
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QListWidget,
                             QListWidgetItem, QDockWidget, QCheckBox, QSlider, QSpinBox, QRadioButton, QComboBox)
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

        # Position window on the largest monitor (usually external monitor on laptop)
        self._position_menu_on_largest_monitor()

        # Set window size for compact control panel (reduced height by 30%)
        self.resize(420, 490)

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
        self.path_lines_actor = None  # Store path lines connecting points
        self.torch_segments_actor = None  # Store torch distance segments
        self.picked_points = []
        self.point_path_id = []  # Track which path each point belongs to
        self.point_normals = []  # Store surface normal at each point
        self.current_path_id = 0  # ID of current path being created
        self.point_picking_mode = False
        self.top_view_mode = False
        self.side_view_mode = False
        self.mesh_edges_visible = False
        self.mesh_opacity = 0.3
        self.zoom_level = 1.0  # Default zoom level
        self.last_pick_time = time.time() - 1  # For debouncing point picks (start in past)
        self.torch_distance = 1.0  # Default torch distance in mm

        # Simulation mode variables
        self.torch_actor = None  # The torch cylinder actor
        self.torch_orientation_line_actor = None  # The white orientation line actor
        self.simulation_mode = False  # Whether we're in simulation
        self.selected_path_id = None  # Which path is selected
        self.current_point_index = 0  # Current point in the path

        # Lighting properties
        self.ambient_light = 0.3  # Default ambient light
        self.diffuse_light = 0.7  # Default diffuse light
        self.specular_light = 0.3  # Default specular light

        # Store camera positions for view control
        self.saved_camera_state = None  # Top view camera state
        self.saved_side_camera_state = None  # Side view camera state

        print("RoboWatchGUI initialization complete")

    def create_left_panel(self):
        """Create the left dock panel with commands"""
        dock = QDockWidget("Commands", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)

        # Create dock widget content
        dock_widget = QWidget()
        dock_layout = QVBoxLayout()
        dock_layout.setSpacing(4)
        dock_layout.setContentsMargins(6, 4, 6, 4)

        # Title
        title = QLabel("RoboWatch Controls")
        title.setStyleSheet("font-weight: bold; font-size: 11px;")
        dock_layout.addWidget(title)

        # Create path button
        self.add_point_btn = QPushButton("create path")
        self.add_point_btn.setStyleSheet(
            "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; font-size: 10px;"
        )
        self.add_point_btn.clicked.connect(self.toggle_point_picking)
        self.add_point_btn.setEnabled(False)
        dock_layout.addWidget(self.add_point_btn)

        # Points list label
        points_label = QLabel("Picked Points:")
        points_label.setStyleSheet("margin-top: 6px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(points_label)

        # Points list (limited height)
        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(80)
        dock_layout.addWidget(self.points_list)

        # Clear point button with "all" radio button
        clear_points_layout = QHBoxLayout()

        # Clear point button (narrow)
        clear_btn = QPushButton("Clear Point")
        clear_btn.setMaximumWidth(90)
        clear_btn.clicked.connect(self.clear_points)
        clear_points_layout.addWidget(clear_btn)

        # "all" radio button to control clear behavior
        self.clear_all_radio = QRadioButton("all")
        self.clear_all_radio.setChecked(True)  # Default: clear all points
        clear_points_layout.addWidget(self.clear_all_radio)

        clear_points_layout.addStretch()  # Add stretch to keep button and radio on left
        dock_layout.addLayout(clear_points_layout)

        # Axes label and checkboxes (combined)
        axes_label = QLabel("Axes:")
        axes_label.setStyleSheet("margin-top: 6px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(axes_label)

        # Axes checkboxes in one row
        axes_layout = QHBoxLayout()
        axes_layout.setSpacing(6)

        self.x_axis_checkbox = QCheckBox("X")
        self.x_axis_checkbox.setChecked(True)
        self.x_axis_checkbox.setStyleSheet("color: red; font-weight: bold; font-size: 10px;")
        self.x_axis_checkbox.stateChanged.connect(self.toggle_x_axis)
        axes_layout.addWidget(self.x_axis_checkbox)

        self.y_axis_checkbox = QCheckBox("Y")
        self.y_axis_checkbox.setChecked(True)
        self.y_axis_checkbox.setStyleSheet("color: green; font-weight: bold; font-size: 10px;")
        self.y_axis_checkbox.stateChanged.connect(self.toggle_y_axis)
        axes_layout.addWidget(self.y_axis_checkbox)

        self.z_axis_checkbox = QCheckBox("Z")
        self.z_axis_checkbox.setChecked(True)
        self.z_axis_checkbox.setStyleSheet("color: blue; font-weight: bold; font-size: 10px;")
        self.z_axis_checkbox.stateChanged.connect(self.toggle_z_axis)
        axes_layout.addWidget(self.z_axis_checkbox)

        axes_layout.addStretch()
        dock_layout.addLayout(axes_layout)

        # View Control label
        view_label = QLabel("View Control:")
        view_label.setStyleSheet("margin-top: 6px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(view_label)

        # Vertical layout for view controls (2 rows)
        view_control_layout = QVBoxLayout()

        # First row: Top and Side buttons
        view_buttons_layout = QHBoxLayout()

        # Top view button
        self.top_btn = QPushButton("Top")
        self.top_btn.setStyleSheet(
            "background-color: #808080; color: white; font-weight: bold; padding: 4px; border-radius: 3px; font-size: 9px;"
        )
        self.top_btn.clicked.connect(self.toggle_top_view)
        view_buttons_layout.addWidget(self.top_btn)

        # Side view button
        self.side_btn = QPushButton("Side")
        self.side_btn.setStyleSheet(
            "background-color: #808080; color: white; font-weight: bold; padding: 4px; border-radius: 3px; font-size: 9px;"
        )
        self.side_btn.clicked.connect(self.toggle_side_view)
        view_buttons_layout.addWidget(self.side_btn)

        view_control_layout.addLayout(view_buttons_layout)

        # Second row: CCW and CW buttons
        rotation_buttons_layout = QHBoxLayout()

        # Counter-clockwise rotation button
        self.ccw_btn = QPushButton("CW ↷")
        self.ccw_btn.setStyleSheet(
            "background-color: #888888; color: #cccccc; font-weight: bold; padding: 4px; font-size: 9px;"
        )
        self.ccw_btn.clicked.connect(self.rotate_view_ccw)
        self.ccw_btn.setEnabled(False)
        rotation_buttons_layout.addWidget(self.ccw_btn)

        # Clockwise rotation button
        self.cw_btn = QPushButton("↶ CCW")
        self.cw_btn.setStyleSheet(
            "background-color: #888888; color: #cccccc; font-weight: bold; padding: 4px; font-size: 9px;"
        )
        self.cw_btn.clicked.connect(self.rotate_view_cw)
        self.cw_btn.setEnabled(False)
        rotation_buttons_layout.addWidget(self.cw_btn)

        view_control_layout.addLayout(rotation_buttons_layout)
        dock_layout.addLayout(view_control_layout)

        # Camera Control label
        camera_label = QLabel("Camera Control:")
        camera_label.setStyleSheet("margin-top: 6px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(camera_label)

        # Zoom label
        zoom_label = QLabel("Zoom: 1.0x")
        zoom_label.setStyleSheet("font-size: 9px; color: #666;")
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
        mesh_label.setStyleSheet("margin-top: 6px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(mesh_label)

        # Opacity label
        opacity_label = QLabel("Opacity: 30%")
        opacity_label.setStyleSheet("font-size: 9px; color: #666;")
        self.opacity_label = opacity_label
        dock_layout.addWidget(opacity_label)

        # Opacity slider (0-100)
        opacity_layout = QHBoxLayout()
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(30)
        opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        opacity_slider.setTickInterval(10)
        # Use sliderMoved for smooth dragging feedback
        opacity_slider.sliderMoved.connect(self.on_opacity_slider_change)
        opacity_slider.valueChanged.connect(self.on_opacity_slider_change)
        self.opacity_slider = opacity_slider
        opacity_layout.addWidget(opacity_slider)
        dock_layout.addLayout(opacity_layout)

        # Torch Distance label
        torch_label = QLabel("Torch Distance:")
        torch_label.setStyleSheet("margin-top: 6px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(torch_label)

        # Torch distance label
        torch_distance_label = QLabel("Torch Distance: 1.0mm")
        torch_distance_label.setStyleSheet("font-size: 9px; color: #666;")
        self.torch_distance_label = torch_distance_label
        dock_layout.addWidget(torch_distance_label)

        # Torch distance slider (0-10mm, step 0.1)
        torch_layout = QHBoxLayout()
        torch_slider = QSlider(Qt.Orientation.Horizontal)
        torch_slider.setMinimum(0)  # 0.0 mm
        torch_slider.setMaximum(100)  # 10.0 mm (100 * 0.1)
        torch_slider.setValue(10)  # 1.0 mm (10 * 0.1) - default
        torch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        torch_slider.setTickInterval(10)
        # Use sliderMoved for smooth dragging feedback
        torch_slider.sliderMoved.connect(self.on_torch_distance_change)
        torch_slider.valueChanged.connect(self.on_torch_distance_change)
        self.torch_slider = torch_slider
        torch_layout.addWidget(torch_slider)
        dock_layout.addLayout(torch_layout)

        # Lighting controls section
        lighting_label = QLabel("Lighting:")
        lighting_label.setStyleSheet("margin-top: 6px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(lighting_label)

        # Ambient light slider
        ambient_label = QLabel("Ambient: 30%")
        ambient_label.setStyleSheet("font-size: 9px; color: #666;")
        self.ambient_label = ambient_label
        dock_layout.addWidget(ambient_label)

        ambient_layout = QHBoxLayout()
        ambient_slider = QSlider(Qt.Orientation.Horizontal)
        ambient_slider.setMinimum(0)
        ambient_slider.setMaximum(100)
        ambient_slider.setValue(30)
        ambient_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        ambient_slider.setTickInterval(10)
        ambient_slider.sliderMoved.connect(self.on_ambient_light_change)
        ambient_slider.valueChanged.connect(self.on_ambient_light_change)
        self.ambient_slider = ambient_slider
        ambient_layout.addWidget(ambient_slider)
        dock_layout.addLayout(ambient_layout)

        # Diffuse light slider
        diffuse_label = QLabel("Diffuse: 70%")
        diffuse_label.setStyleSheet("font-size: 9px; color: #666;")
        self.diffuse_label = diffuse_label
        dock_layout.addWidget(diffuse_label)

        diffuse_layout = QHBoxLayout()
        diffuse_slider = QSlider(Qt.Orientation.Horizontal)
        diffuse_slider.setMinimum(0)
        diffuse_slider.setMaximum(100)
        diffuse_slider.setValue(70)
        diffuse_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        diffuse_slider.setTickInterval(10)
        diffuse_slider.sliderMoved.connect(self.on_diffuse_light_change)
        diffuse_slider.valueChanged.connect(self.on_diffuse_light_change)
        self.diffuse_slider = diffuse_slider
        diffuse_layout.addWidget(diffuse_slider)
        dock_layout.addLayout(diffuse_layout)

        # Specular light slider
        specular_label = QLabel("Specular: 30%")
        specular_label.setStyleSheet("font-size: 9px; color: #666;")
        self.specular_label = specular_label
        dock_layout.addWidget(specular_label)

        specular_layout = QHBoxLayout()
        specular_slider = QSlider(Qt.Orientation.Horizontal)
        specular_slider.setMinimum(0)
        specular_slider.setMaximum(100)
        specular_slider.setValue(30)
        specular_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        specular_slider.setTickInterval(10)
        specular_slider.sliderMoved.connect(self.on_specular_light_change)
        specular_slider.valueChanged.connect(self.on_specular_light_change)
        self.specular_slider = specular_slider
        specular_layout.addWidget(specular_slider)
        dock_layout.addLayout(specular_layout)

        # Bottom buttons layout
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(4)

        # Edges button
        edges_btn = QPushButton("Edges")
        edges_btn.setStyleSheet("background-color: #FF5722; color: white; padding: 4px; font-size: 9px;")
        edges_btn.clicked.connect(self.toggle_mesh_edges)
        bottom_layout.addWidget(edges_btn)

        # Temporary debug button
        temp_btn = QPushButton("load temp")
        temp_btn.setStyleSheet("background-color: #808080; color: white; padding: 4px; font-size: 9px;")
        temp_btn.clicked.connect(self.load_temp_file)
        bottom_layout.addWidget(temp_btn)

        dock_layout.addLayout(bottom_layout)

        # Simulation section (at bottom)
        simulation_label = QLabel("Simulation:")
        simulation_label.setStyleSheet("margin-top: 6px; font-weight: bold; font-size: 10px;")
        dock_layout.addWidget(simulation_label)

        # Simulation button
        self.simulation_btn = QPushButton("Simulation")
        self.simulation_btn.setStyleSheet(
            "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; font-size: 10px;"
        )
        self.simulation_btn.clicked.connect(self.toggle_simulation_mode)
        self.simulation_btn.setEnabled(False)
        dock_layout.addWidget(self.simulation_btn)

        # Path selection dropdown for simulation
        path_list_label = QLabel("Select Path:")
        path_list_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        dock_layout.addWidget(path_list_label)

        self.simulation_path_dropdown = QComboBox()
        self.simulation_path_dropdown.currentIndexChanged.connect(self.on_simulation_path_selected)
        self.simulation_path_dropdown.setEnabled(False)
        self.simulation_path_dropdown.setStyleSheet("font-size: 9px;")
        dock_layout.addWidget(self.simulation_path_dropdown)

        # FWD and BACK buttons
        nav_layout = QHBoxLayout()

        self.back_btn = QPushButton("BACK")
        self.back_btn.setStyleSheet(
            "background-color: #888888; color: #cccccc; font-weight: bold; padding: 4px; font-size: 9px;"
        )
        self.back_btn.clicked.connect(self.on_simulation_back)
        self.back_btn.setEnabled(False)
        nav_layout.addWidget(self.back_btn)

        self.fwd_btn = QPushButton("FWD")
        self.fwd_btn.setStyleSheet(
            "background-color: #888888; color: #cccccc; font-weight: bold; padding: 4px; font-size: 9px;"
        )
        self.fwd_btn.clicked.connect(self.on_simulation_fwd)
        self.fwd_btn.setEnabled(False)
        nav_layout.addWidget(self.fwd_btn)

        dock_layout.addLayout(nav_layout)

        # Add stretch to bottom
        dock_layout.addStretch()

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 8px; color: #666; padding: 3px; background: #f5f5f5; border-radius: 3px;")
        dock_layout.addWidget(self.status_label)

        dock_widget.setLayout(dock_layout)
        dock_widget.setMaximumWidth(420)  # Limit dock width
        dock.setWidget(dock_widget)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [420], Qt.Orientation.Horizontal)

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

        # Save action
        save_action = QAction("Save STL", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_stl_file)
        file_menu.addAction(save_action)

        # Separator
        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _position_menu_on_largest_monitor(self):
        """Position the menu window on the largest monitor (external monitor on laptop setup)"""
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            screens = app.screens()

            if not screens:
                print("  ! No screens detected")
                return

            # Find the largest monitor (usually the external one)
            largest_screen = max(screens, key=lambda s: s.geometry().width() * s.geometry().height())
            largest_geom = largest_screen.geometry()

            print(f"  ✓ Found {len(screens)} monitor(s)")
            print(f"  ✓ Largest monitor: {largest_geom.width()}x{largest_geom.height()} at ({largest_geom.x()}, {largest_geom.y()})")

            # Position menu window at top-left of largest monitor with some padding
            menu_x = largest_geom.x() + 20
            menu_y = largest_geom.y() + 50

            self.move(menu_x, menu_y)
            print(f"  ✓ Menu window positioned on largest monitor at ({menu_x}, {menu_y})")

        except Exception as e:
            print(f"  ! Error positioning menu window: {e}")

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

            # Close old plotter if it exists
            if self.plotter is not None:
                try:
                    self.plotter.close()
                    print("  ✓ Old plotter window closed")
                except Exception as e:
                    print(f"  ! Warning: Could not close old plotter: {e}")
                self.plotter = None

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

            # Try to load associated JSON file with points and paths
            json_path = Path(file_path).with_suffix('.json')
            if json_path.exists():
                print(f"Found JSON file: {json_path}")
                self.load_paths_from_json(str(json_path))
                self.status_label.setText("Mesh and paths loaded!")
                print("✓ Points and paths loaded into view")
            else:
                print(f"No JSON file found at: {json_path}")
                self.status_label.setText("Mesh ready! Check PyVista window")

            print("Mesh displayed successfully")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:50]}")
            print(f"Error loading file: {e}")
            import traceback
            traceback.print_exc()

    def save_stl_file(self):
        """Save the current STL mesh and path data"""
        if self.current_mesh is None:
            print("No mesh loaded to save")
            return

        try:
            file_dialog = QFileDialog()
            stl_dir = Path(__file__).parent / "STL"
            stl_dir.mkdir(exist_ok=True)

            file_path, _ = file_dialog.getSaveFileName(
                self,
                "Save STL File",
                str(stl_dir),
                "STL Files (*.stl);;All Files (*)"
            )

            if not file_path:
                return

            file_path = Path(file_path)

            # Ensure .stl extension
            if file_path.suffix.lower() != '.stl':
                file_path = file_path.with_suffix('.stl')

            # Save the mesh as STL
            self.current_mesh.save(str(file_path))
            print(f"Mesh saved to: {file_path}")

            # Save points and paths data as JSON
            json_path = file_path.with_suffix('.json')
            paths_data = {
                'paths': [],
                'all_points': [],
                'torch_distance': float(self.torch_distance)
            }

            # Group points by path
            for path_id in range(1, self.current_path_id + 1):
                path_points = []
                for i, point in enumerate(self.picked_points):
                    if self.point_path_id[i] == path_id:
                        normal = self.point_normals[i] if i < len(self.point_normals) else [0, 0, 1]
                        path_points.append({
                            'x': float(point[0]),
                            'y': float(point[1]),
                            'z': float(point[2]),
                            'normal_x': float(normal[0]),
                            'normal_y': float(normal[1]),
                            'normal_z': float(normal[2])
                        })

                if path_points:
                    paths_data['paths'].append({
                        'path_id': path_id,
                        'points': path_points
                    })

            # Also store all points with their path IDs and normals
            for i, point in enumerate(self.picked_points):
                normal = self.point_normals[i] if i < len(self.point_normals) else [0, 0, 1]
                paths_data['all_points'].append({
                    'index': i,
                    'path_id': int(self.point_path_id[i]),
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'z': float(point[2]),
                    'normal_x': float(normal[0]),
                    'normal_y': float(normal[1]),
                    'normal_z': float(normal[2])
                })

            # Write JSON file
            with open(json_path, 'w') as f:
                json.dump(paths_data, f, indent=2)

            print(f"Path data saved to: {json_path}")
            self.status_label.setText(f"Saved: {file_path.name} and {json_path.name}")
            print("✓ Save complete!")

        except Exception as e:
            self.status_label.setText(f"Error saving: {str(e)[:50]}")
            print(f"Error saving file: {e}")
            import traceback
            traceback.print_exc()

    def load_paths_from_json(self, json_file_path):
        """Load points and paths from a JSON file"""
        try:
            print(f"Loading paths from: {json_file_path}")

            # Make sure plotter exists and is ready
            if not self.plotter:
                print("  ! Error: Plotter not initialized yet")
                return

            with open(json_file_path, 'r') as f:
                paths_data = json.load(f)

            # Clear existing points and paths
            self.picked_points = []
            self.point_path_id = []
            self.point_normals = []
            self.current_path_id = 0

            # Load torch distance if available
            if 'torch_distance' in paths_data:
                self.torch_distance = float(paths_data['torch_distance'])
                # Update slider to reflect loaded torch distance
                slider_value = int(self.torch_distance * 10)
                self.torch_slider.blockSignals(True)
                self.torch_slider.setValue(slider_value)
                self.torch_slider.blockSignals(False)
                self.torch_distance_label.setText(f"Torch Distance: {self.torch_distance:.1f}mm")
                print(f"  ✓ Loaded torch distance: {self.torch_distance:.1f}mm")

            # Load all points
            if 'all_points' in paths_data:
                for point_data in paths_data['all_points']:
                    point = [point_data['x'], point_data['y'], point_data['z']]
                    self.picked_points.append(point)
                    self.point_path_id.append(point_data['path_id'])

                    # Load normal if available
                    if 'normal_x' in point_data:
                        normal = np.array([point_data['normal_x'], point_data['normal_y'], point_data['normal_z']])
                    else:
                        normal = np.array([0, 0, 1])
                    self.point_normals.append(normal)

                    # Update current_path_id to track highest path ID
                    if point_data['path_id'] > self.current_path_id:
                        self.current_path_id = point_data['path_id']

                    # Add to points list in UI
                    points_in_path = sum(1 for pid in self.point_path_id if pid == point_data['path_id'])
                    if points_in_path == 1:
                        point_str = f"Start point... (Path {point_data['path_id']}): ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
                    else:
                        point_str = f"Point {points_in_path} (Path {point_data['path_id']}): ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"

                    self.points_list.addItem(QListWidgetItem(point_str))

                # Update visualization
                self.update_markers()
                self.update_torch_segments()  # Update torch segments
                self.update_path()

                # Enable simulation button now that we have points from JSON
                self.simulation_btn.setEnabled(True)
                self.simulation_btn.setStyleSheet(
                    "background-color: #FF9800; color: white; font-weight: bold; padding: 6px; font-size: 10px;"
                )

                # Force a complete render to display the loaded points and paths
                if self.plotter:
                    self.plotter.render_window.Render()
                    QApplication.instance().processEvents()
                    print("  ✓ Render complete - points, paths, and torch segments displayed")

                # Scroll to bottom of points list
                self.points_list.scrollToBottom()

                print(f"✓ Loaded {len(self.picked_points)} points from {len(set(self.point_path_id))} paths")
            else:
                print("No points found in JSON file")

        except Exception as e:
            print(f"Error loading paths from JSON: {e}")
            import traceback
            traceback.print_exc()

    def toggle_simulation_mode(self):
        """Toggle simulation mode on/off"""
        if not self.picked_points:
            print("No points to simulate - create a path first")
            return

        self.simulation_mode = not self.simulation_mode

        if self.simulation_mode:
            # Entering simulation mode
            self.simulation_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 6px; border: 2px solid white;"
            )
            self.add_point_btn.setEnabled(False)
            self.add_point_btn.setStyleSheet("background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; font-size: 10px;")

            # Populate path dropdown
            self.update_simulation_path_list()

            # Enable path dropdown
            self.simulation_path_dropdown.setEnabled(True)

            # Enable BACK/FWD buttons
            self.back_btn.setEnabled(True)
            self.fwd_btn.setEnabled(True)
            self.back_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 4px; font-size: 9px;"
            )
            self.fwd_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 4px; font-size: 9px;"
            )

            print("Simulation mode ON")
        else:
            # Exiting simulation mode
            self.simulation_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; font-size: 10px;"
            )
            self.add_point_btn.setEnabled(True)
            self.add_point_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
            )

            # Clear simulation
            self.selected_path_id = None
            self.current_point_index = 0
            self.simulation_path_dropdown.clear()
            self.simulation_path_dropdown.setEnabled(False)
            self.back_btn.setEnabled(False)
            self.fwd_btn.setEnabled(False)
            self.back_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 4px; font-size: 9px;"
            )
            self.fwd_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 4px; font-size: 9px;"
            )

            # Remove torch and orientation line
            if self.torch_actor is not None and self.plotter:
                self.plotter.remove_actor(self.torch_actor)
                self.torch_actor = None
            if self.torch_orientation_line_actor is not None and self.plotter:
                self.plotter.remove_actor(self.torch_orientation_line_actor)
                self.torch_orientation_line_actor = None
            if self.plotter:
                self.plotter.render_window.Render()
                QApplication.instance().processEvents()

            print("Simulation mode OFF")

    def update_simulation_path_list(self):
        """Update the path dropdown in simulation"""
        # Block signals to avoid triggering selection during update
        self.simulation_path_dropdown.blockSignals(True)

        self.simulation_path_dropdown.clear()

        # Get unique path IDs
        unique_paths = sorted(set(self.point_path_id))

        for path_id in unique_paths:
            # Count points in this path
            point_count = sum(1 for pid in self.point_path_id if pid == path_id)
            item_text = f"Path {path_id} ({point_count} points)"
            self.simulation_path_dropdown.addItem(item_text)

        # Set first path as current (without triggering signal yet)
        self.simulation_path_dropdown.setCurrentIndex(0)

        # Unblock signals
        self.simulation_path_dropdown.blockSignals(False)

        # Manually call the selection handler to display the torch
        self.on_simulation_path_selected()

    def on_simulation_path_selected(self):
        """Handle path selection in simulation"""
        text = self.simulation_path_dropdown.currentText()
        if not text:
            return

        # Extract path ID from item text (e.g., "Path 1 (5 points)" -> 1)
        path_id_str = text.split()[1]  # Get the number after "Path"
        self.selected_path_id = int(path_id_str)
        self.current_point_index = 0

        # Position torch at first point of path
        self.update_torch_position()

        print(f"Selected Path {self.selected_path_id}")

    def update_torch_position(self):
        """Update torch position based on selected path and point index"""
        if self.selected_path_id is None or not self.plotter:
            return

        # Find all points in the selected path
        path_point_indices = [i for i, pid in enumerate(self.point_path_id) if pid == self.selected_path_id]

        if not path_point_indices:
            print(f"No points found in path {self.selected_path_id}")
            return

        # Clamp current_point_index to valid range
        if self.current_point_index >= len(path_point_indices):
            self.current_point_index = len(path_point_indices) - 1
        if self.current_point_index < 0:
            self.current_point_index = 0

        # Get the current point index in the global list
        global_index = path_point_indices[self.current_point_index]
        point = np.array(self.picked_points[global_index])
        normal = self.point_normals[global_index] if global_index < len(self.point_normals) else np.array([0, 0, 1])

        # Calculate torch endpoint (at the tip of the vertical segment)
        torch_endpoint = point + normal * self.torch_distance

        # Create or update torch
        self.create_or_update_torch(torch_endpoint, normal)

        # Print info
        point_num = self.current_point_index + 1
        total_points = len(path_point_indices)
        print(f"Path {self.selected_path_id}, Point {point_num}/{total_points}")
        print(f"  Position: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
        print(f"  Normal: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})")

    def on_simulation_fwd(self):
        """Move to next point in path"""
        if self.selected_path_id is None:
            return

        # Find total points in this path
        path_point_count = sum(1 for pid in self.point_path_id if pid == self.selected_path_id)

        # Move to next point
        if self.current_point_index < path_point_count - 1:
            self.current_point_index += 1
            self.update_torch_position()
        else:
            print(f"Already at last point of path {self.selected_path_id}")

    def on_simulation_back(self):
        """Move to previous point in path"""
        if self.selected_path_id is None:
            return

        # Move to previous point
        if self.current_point_index > 0:
            self.current_point_index -= 1
            self.update_torch_position()
        else:
            print(f"Already at first point of path {self.selected_path_id}")

    def create_or_update_torch(self, position, normal):
        """Create or update the torch cylinder at given position and orientation"""
        if not self.plotter:
            return

        # Remove old torch if exists
        if self.torch_actor is not None:
            self.plotter.remove_actor(self.torch_actor)
            self.torch_actor = None

        # Remove old orientation line if exists
        if self.torch_orientation_line_actor is not None:
            self.plotter.remove_actor(self.torch_orientation_line_actor)
            self.torch_orientation_line_actor = None

        # Torch parameters - small cylinder extending from the vertical line
        torch_radius = 0.8  # mm - slightly bigger than the vertical lines
        torch_height = 4.0  # mm - 1/5 of original 20mm
        torch_color = 'green'

        try:
            # Create cylinder aligned with the normal direction
            # Default cylinder is aligned with Z axis
            torch_cyl = pv.Cylinder(radius=torch_radius, height=torch_height, direction=(0, 0, 1))

            # Rotate cylinder to align with normal
            # Calculate rotation axis and angle
            default_normal = np.array([0, 0, 1])
            normal_normalized = normal / np.linalg.norm(normal)

            # Calculate rotation
            rotation_axis = np.cross(default_normal, normal_normalized)
            rotation_magnitude = np.linalg.norm(rotation_axis)

            if rotation_magnitude > 1e-6:  # Only rotate if axis is significant
                rotation_axis = rotation_axis / rotation_magnitude
                rotation_angle = np.arccos(np.clip(np.dot(default_normal, normal_normalized), -1.0, 1.0))
                rotation_angle_deg = np.degrees(rotation_angle)

                # Rotate the cylinder
                torch_cyl = torch_cyl.rotate_vector(rotation_axis, rotation_angle_deg, point=torch_cyl.center)

            # Position the torch so it starts at the tip of the vertical line and extends along the normal
            # The cylinder's center should be at: position + (torch_height / 2) * normal_normalized
            torch_center = position + normal_normalized * (torch_height / 2)
            torch_cyl = torch_cyl.translate(torch_center - torch_cyl.center)

            # Add white orientation line on the side of the cylinder
            line_start = torch_center
            line_end = torch_center + np.array([torch_radius * 0.6, 0, 0])  # Line extends radially
            orientation_line = pv.Line(line_start, line_end)

            # Add torch to plotter and store references
            self.torch_actor = self.plotter.add_mesh(torch_cyl, color=torch_color, opacity=0.8)
            self.torch_orientation_line_actor = self.plotter.add_mesh(orientation_line, color='white', line_width=2)

            # Render
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

            print(f"  ✓ Torch positioned at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")

        except Exception as e:
            print(f"Error creating torch: {e}")
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
                opacity=0.3
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

            # Also calculate and save the side view camera state
            # Side view: X axis toward viewer, Z is up, Y is horizontal
            self.saved_side_camera_state = {
                'position': (mesh_center[0] + camera_distance, mesh_center[1], mesh_center[2]),
                'focal_point': tuple(mesh_center),
                'up': (0, 0, 1)  # Z points up
            }
            print(f"  ✓ Saved initial camera state for Side View")
            print(f"    Position: {self.saved_side_camera_state['position']}")
            print(f"    Focal Point: {self.saved_side_camera_state['focal_point']}")
            print(f"    Up: {self.saved_side_camera_state['up']}")

            # Keep both top_view_mode and side_view_mode as False on load - buttons start disabled (gray)
            # View is positioned at top, but interaction is NOT frozen until user clicks "Top" or "Side"
            self.top_view_mode = False
            self.side_view_mode = False
            self.top_btn.setStyleSheet(
                "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )
            self.side_btn.setStyleSheet(
                "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Allow interaction initially - user can click "Top" to freeze the view
            if hasattr(self.plotter, 'iren') and self.plotter.iren:
                try:
                    from vtkmodules.vtkRenderingCore import vtkInteractorStyleTrackballCamera
                    trackball_style = vtkInteractorStyleTrackballCamera()
                    self.plotter.iren.SetInteractorStyle(trackball_style)
                    print("  ✓ Interaction ENABLED on load - click 'Top' to freeze view")
                except Exception as e:
                    print(f"  ! Warning: Could not set interaction style: {e}")

            # Add lighting for shadows and depth
            self.status_label.setText("Setting up lighting...")
            print("  ✓ Adding point light source for shadows...")
            try:
                from vtkmodules.vtkRenderingCore import vtkLight

                # Create a new light from upper-left-front position
                light = vtkLight()
                light.SetPosition(
                    mesh_center[0] - camera_distance * 0.5,  # Left side
                    mesh_center[1] + camera_distance * 0.5,  # Above
                    mesh_center[2] + camera_distance * 0.8   # Toward viewer
                )
                light.SetFocalPoint(mesh_center[0], mesh_center[1], mesh_center[2])
                light.SetIntensity(1.0)
                light.PositionalOn()  # Make it a point light (not directional)

                # Add the light to the renderer
                self.plotter.renderer.AddLight(light)

                print("  ✓ Point light added - shadows enabled")
            except Exception as e:
                print(f"  ! Warning: Could not add point light: {e}")

            # Render
            self.status_label.setText("Rendering...")
            print("  ✓ Rendering mesh...")
            self.plotter.render()

            # Initialize interactor to make window visible
            print("  ✓ Initializing interactor...")
            self.plotter.iren.initialize()

            # Force window to be shown and on top
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

            print("  ✓ Interactor initialized - window should be visible now")

            # Note on macOS: VTK windows cannot be repositioned after creation due to platform limitations
            # The PyVista window may appear on a different monitor than the menu
            # You can manually drag it to the desired monitor if needed

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
            # Force immediate render update
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

    def toggle_y_axis(self, state):
        """Toggle Y axis visibility"""
        if self.plotter and 'y' in self.axis_actors:
            self.axis_actors['y'].SetVisibility(state != 0)
            # Force immediate render update
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

    def toggle_z_axis(self, state):
        """Toggle Z axis visibility"""
        if self.plotter and 'z' in self.axis_actors:
            self.axis_actors['z'].SetVisibility(state != 0)
            # Force immediate render update
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

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

    def on_ambient_light_change(self, value):
        """Handle ambient light slider change (0-100)"""
        if not self.plotter or not self.mesh_actor:
            return

        # Convert 0-100 slider value to 0.0-1.0
        self.ambient_light = value / 100.0
        self.mesh_actor.GetProperty().SetAmbient(self.ambient_light)

        # Force render window update
        self.plotter.render_window.Render()
        QApplication.instance().processEvents()

        # Update label
        self.ambient_label.setText(f"Ambient: {value}%")

    def on_diffuse_light_change(self, value):
        """Handle diffuse light slider change (0-100)"""
        if not self.plotter or not self.mesh_actor:
            return

        # Convert 0-100 slider value to 0.0-1.0
        self.diffuse_light = value / 100.0
        self.mesh_actor.GetProperty().SetDiffuse(self.diffuse_light)

        # Force render window update
        self.plotter.render_window.Render()
        QApplication.instance().processEvents()

        # Update label
        self.diffuse_label.setText(f"Diffuse: {value}%")

    def on_specular_light_change(self, value):
        """Handle specular light slider change (0-100)"""
        if not self.plotter or not self.mesh_actor:
            return

        # Convert 0-100 slider value to 0.0-1.0
        self.specular_light = value / 100.0
        self.mesh_actor.GetProperty().SetSpecular(self.specular_light)

        # Force render window update
        self.plotter.render_window.Render()
        QApplication.instance().processEvents()

        # Update label
        self.specular_label.setText(f"Specular: {value}%")

    def on_torch_distance_change(self, value):
        """Handle torch distance slider change (0-100 = 0.0-10.0mm)"""
        # Convert slider value (0-100) to mm (0.0-10.0)
        self.torch_distance = value / 10.0

        # Update label
        self.torch_distance_label.setText(f"Torch Distance: {self.torch_distance:.1f}mm")

        # Update torch segments in the viewer
        self.update_torch_segments()

        # If in simulation mode with a path selected, update torch position
        if self.simulation_mode and self.selected_path_id is not None:
            self.update_torch_position()

        # Render only once after updating segments
        if self.plotter:
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

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

        # Force immediate render update
        self.plotter.render_window.Render()
        QApplication.instance().processEvents()

    def toggle_top_view(self):
        """Toggle top view mode - disable Side view if Top is enabled"""
        self.top_view_mode = not self.top_view_mode
        if self.top_view_mode:
            # Activate Top view
            self.top_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 2px solid white;"
            )

            # Disable Side button when Top is active
            self.side_view_mode = False
            self.side_btn.setEnabled(False)
            self.side_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Enable CW/CCW buttons with active styling
            self.cw_btn.setEnabled(True)
            self.ccw_btn.setEnabled(True)
            self.cw_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )
            self.ccw_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Enable add point button
            self.add_point_btn.setEnabled(True)
            self.add_point_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
            )

            self.set_top_view()
            print("Top View mode ON - Side view disabled - CW/CCW buttons enabled - add point enabled")
        else:
            # Deactivate Top view
            self.top_btn.setStyleSheet(
                "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Enable Side button again
            self.side_btn.setEnabled(True)
            self.side_btn.setStyleSheet(
                "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Disable CW/CCW buttons with inactive styling
            self.cw_btn.setEnabled(False)
            self.ccw_btn.setEnabled(False)
            self.cw_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; border-radius: 4px;"
            )
            self.ccw_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Disable add point button and stop picking if active
            if self.point_picking_mode:
                self.point_picking_mode = False
                self._remove_point_picking()
            self.add_point_btn.setEnabled(False)
            self.add_point_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 8px;"
            )
            self.add_point_btn.setText("add point")

            self.restore_normal_view()
            print("Top View mode OFF - Side view re-enabled - CW/CCW buttons disabled - add point disabled")

    def set_top_view(self):
        """Set camera to top view - restore initial camera position and freeze interaction"""
        if not self.plotter or not self.saved_camera_state:
            print("Error: No mesh loaded or camera state not saved. Click 'load temp' first.")
            return

        try:
            # Make sure point picking observer is removed before freezing
            try:
                self.plotter.iren.remove_observer('LeftButtonPressEvent')
                print("  ✓ Removed point picking observer")
            except:
                pass  # Observer might not exist

            # Restore the saved camera state from when mesh was loaded
            self.plotter.camera.position = self.saved_camera_state['position']
            self.plotter.camera.focal_point = self.saved_camera_state['focal_point']
            self.plotter.camera.up = self.saved_camera_state['up']

            # Freeze mouse interaction by setting a None style
            if hasattr(self.plotter, 'iren') and self.plotter.iren:
                try:
                    from vtkmodules.vtkRenderingCore import vtkInteractorStyleNone
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

    def toggle_side_view(self):
        """Toggle side view mode - disable Top view if Side is enabled"""
        self.side_view_mode = not self.side_view_mode
        if self.side_view_mode:
            # Activate Side view
            self.side_btn.setStyleSheet(
                "background-color: #9C27B0; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 2px solid white;"
            )

            # Disable Top button when Side is active
            self.top_view_mode = False
            self.top_btn.setEnabled(False)
            self.top_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Enable CW/CCW buttons with active styling
            self.cw_btn.setEnabled(True)
            self.ccw_btn.setEnabled(True)
            self.cw_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )
            self.ccw_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Enable add point button
            self.add_point_btn.setEnabled(True)
            self.add_point_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
            )

            self.set_side_view()
            print("Side View mode ON - Top view disabled - CW/CCW buttons enabled - add point enabled")
        else:
            # Deactivate Side view
            self.side_btn.setStyleSheet(
                "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Enable Top button again
            self.top_btn.setEnabled(True)
            self.top_btn.setStyleSheet(
                "background-color: #808080; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Disable CW/CCW buttons with inactive styling
            self.cw_btn.setEnabled(False)
            self.ccw_btn.setEnabled(False)
            self.cw_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; border-radius: 4px;"
            )
            self.ccw_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; border-radius: 4px;"
            )

            # Disable add point button and stop picking if active
            if self.point_picking_mode:
                self.point_picking_mode = False
                self._remove_point_picking()
            self.add_point_btn.setEnabled(False)
            self.add_point_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 8px;"
            )
            self.add_point_btn.setText("add point")

            self.restore_normal_view()
            print("Side View mode OFF - Top view re-enabled - CW/CCW buttons disabled - add point disabled")

    def set_side_view(self):
        """Set camera to side view - restore initial side view camera position and freeze interaction"""
        if not self.plotter or not self.saved_side_camera_state:
            print("Error: No mesh loaded or side camera state not saved.")
            return

        try:
            # Make sure point picking observer is removed before freezing
            try:
                self.plotter.iren.remove_observer('LeftButtonPressEvent')
                print("  ✓ Removed point picking observer")
            except:
                pass  # Observer might not exist

            # Restore the saved side camera state
            self.plotter.camera.position = self.saved_side_camera_state['position']
            self.plotter.camera.focal_point = self.saved_side_camera_state['focal_point']
            self.plotter.camera.up = self.saved_side_camera_state['up']

            # Freeze mouse interaction by setting a None style
            if hasattr(self.plotter, 'iren') and self.plotter.iren:
                try:
                    from vtkmodules.vtkRenderingCore import vtkInteractorStyleNone
                    frozen_style = vtkInteractorStyleNone()
                    self.plotter.iren.SetInteractorStyle(frozen_style)
                    print("  ✓ Mouse interaction FROZEN")
                except Exception as freeze_error:
                    print(f"  ! Warning: Could not freeze interaction: {freeze_error}")

            # Force render update
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

            print("Side view restored - camera position:")
            print(f"  Position: {self.plotter.camera.position}")
            print(f"  Focal Point: {self.plotter.camera.focal_point}")
            print(f"  Up: {self.plotter.camera.up}")

        except Exception as e:
            print(f"Error setting side view: {e}")
            import traceback
            traceback.print_exc()

    def rotate_view_cw(self):
        """Rotate view 90 degrees clockwise around Z axis"""
        # Valid if either Top or Side view is active
        if not (self.top_view_mode or self.side_view_mode) or not self.plotter:
            return

        try:
            mesh_center = self.current_mesh.center

            if self.top_view_mode:
                # Top view: rotate the up vector around Z axis
                current_up = np.array(self.plotter.camera.up)

                # Rotate around Z axis by -90 degrees (clockwise)
                angle = np.radians(-90)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                # Apply 2D rotation to X,Y components (Z stays the same)
                new_up_x = current_up[0] * cos_a - current_up[1] * sin_a
                new_up_y = current_up[0] * sin_a + current_up[1] * cos_a
                new_up_z = current_up[2]

                new_up = (new_up_x, new_up_y, new_up_z)
                self.plotter.camera.up = new_up

                print(f"Rotated CW (90 degrees clockwise) - Top view")
                print(f"  New up vector: {self.plotter.camera.up}")

            else:  # side_view_mode
                # Side view: rotate camera position around Z axis
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

                self.plotter.camera.position = new_pos
                self.plotter.camera.focal_point = mesh_center
                self.plotter.camera.up = (0, 0, 1)  # Z points up

                print(f"Rotated CW (90 degrees clockwise) - Side view")
                print(f"  New camera position: {self.plotter.camera.position}")

            # Force immediate render update
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

        except Exception as e:
            print(f"Error rotating CW: {e}")
            import traceback
            traceback.print_exc()

    def rotate_view_ccw(self):
        """Rotate view 90 degrees counter-clockwise around Z axis"""
        # Valid if either Top or Side view is active
        if not (self.top_view_mode or self.side_view_mode) or not self.plotter:
            return

        try:
            mesh_center = self.current_mesh.center

            if self.top_view_mode:
                # Top view: rotate the up vector around Z axis
                current_up = np.array(self.plotter.camera.up)

                # Rotate around Z axis by +90 degrees (counter-clockwise)
                angle = np.radians(90)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                # Apply 2D rotation to X,Y components (Z stays the same)
                new_up_x = current_up[0] * cos_a - current_up[1] * sin_a
                new_up_y = current_up[0] * sin_a + current_up[1] * cos_a
                new_up_z = current_up[2]

                new_up = (new_up_x, new_up_y, new_up_z)
                self.plotter.camera.up = new_up

                print(f"Rotated CCW (90 degrees counter-clockwise) - Top view")
                print(f"  New up vector: {self.plotter.camera.up}")

            else:  # side_view_mode
                # Side view: rotate camera position around Z axis
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

                self.plotter.camera.position = new_pos
                self.plotter.camera.focal_point = mesh_center
                self.plotter.camera.up = (0, 0, 1)  # Z points up

                print(f"Rotated CCW (90 degrees counter-clockwise) - Side view")
                print(f"  New camera position: {self.plotter.camera.position}")

            # Force immediate render update
            self.plotter.render_window.Render()
            QApplication.instance().processEvents()

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
                    from vtkmodules.vtkRenderingCore import vtkInteractorStyleTrackballCamera
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
        """Toggle point picking mode - only available when Top or Side view is active"""
        # Only allow toggling if a view mode is active
        if not (self.top_view_mode or self.side_view_mode):
            print("Point picking requires Top or Side view to be active")
            return

        self.point_picking_mode = not self.point_picking_mode
        if self.point_picking_mode:
            # Start a new path - increment path ID (don't clear old points)
            self.current_path_id += 1
            print(f"Starting new path (ID: {self.current_path_id})")

            self.add_point_btn.setStyleSheet(
                "background-color: #f44336; color: white; font-weight: bold; padding: 8px;"
            )
            self.add_point_btn.setText("picking...")
            print("Path picking mode ON - Click on mesh to create path points")
            # Reset the pick timer to ensure first click works
            self.last_pick_time = time.time() - 1
            # Setup mouse click callback for picking
            self._setup_point_picking()
        else:
            self.add_point_btn.setStyleSheet(
                "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
            )
            self.add_point_btn.setText("create path")
            print("Path picking mode OFF")
            # Remove mouse click callback
            self._remove_point_picking()

    def add_picked_point(self, point, normal=None):
        """Add a point to the picked points list and connect with previous point"""
        self.picked_points.append(point)
        self.point_path_id.append(self.current_path_id)

        # Store the normal at this point (default to upward if not provided)
        if normal is None:
            normal = np.array([0, 0, 1])
        self.point_normals.append(normal)

        # Count how many points are in the current path
        points_in_current_path = sum(1 for pid in self.point_path_id if pid == self.current_path_id)

        # First point of current path is labeled as "Start point..."
        if points_in_current_path == 1:
            point_str = f"Start point... (Path {self.current_path_id}): ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
        else:
            point_str = f"Point {points_in_current_path} (Path {self.current_path_id}): ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"

        self.points_list.addItem(QListWidgetItem(point_str))
        # Scroll to show the newly added point
        self.points_list.scrollToBottom()
        print(f"Added point: {point}")

        self.update_markers()
        self.update_torch_segments()  # Update torch segments
        self.update_path()  # Update path lines between consecutive points

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

        # Create new markers: first point green, rest red
        points = np.array(self.picked_points)

        # Create color array: first point of each path is dark green, rest are red (255, 0, 0)
        colors = []
        for i in range(len(points)):
            # Check if this is the first point of its path
            is_first_in_path = True
            current_path = self.point_path_id[i]
            for j in range(i):
                if self.point_path_id[j] == current_path:
                    is_first_in_path = False
                    break

            if is_first_in_path:
                colors.append([0, 128, 0])  # Dark green for start point of path
            else:
                colors.append([255, 0, 0])  # Red for subsequent points

        colors = np.array(colors)

        self.markers_actor = self.plotter.add_points(
            points,
            scalars=colors,
            rgb=True,
            point_size=10,
            render_points_as_spheres=True
        )
        # Force immediate render update
        self.plotter.render_window.Render()
        QApplication.instance().processEvents()

    def update_path(self):
        """Update path lines connecting consecutive points"""
        # Remove old path lines
        if self.path_lines_actor is not None:
            self.plotter.remove_actor(self.path_lines_actor)
            self.path_lines_actor = None

        # Need at least 2 points to draw a line
        if len(self.picked_points) < 2:
            return

        # Create lines connecting consecutive points (only within same path)
        points = np.array(self.picked_points)

        # Create a polyline connecting all points in sequence
        # Only draw lines between consecutive points in the same path
        lines = []
        for i in range(len(points) - 1):
            # Only connect if they're in the same path
            if self.point_path_id[i] == self.point_path_id[i + 1]:
                lines.append([points[i], points[i + 1]])

        if lines:
            lines_array = np.array(lines)
            # Flatten to (n*2, 3) for add_lines format
            line_points = lines_array.reshape(-1, 3)

            # Create connectivity array: [2, p0, p1, 2, p2, p3, ...]
            # This tells PyVista to draw lines between consecutive pairs
            connectivity = []
            for i in range(0, len(line_points), 2):
                connectivity.extend([2, i, i + 1])

            # Create a polydata object with the line segments
            from pyvista import PolyData
            path_polydata = PolyData(line_points, connectivity)

            # Add the path lines to the plotter
            self.path_lines_actor = self.plotter.add_mesh(
                path_polydata,
                color='yellow',
                line_width=3,
                style='wireframe'
            )

        # Force immediate render update
        self.plotter.render_window.Render()
        QApplication.instance().processEvents()

    def update_torch_segments(self):
        """Update torch distance segments (perpendicular to surface at each point)"""
        # Skip if no plotter
        if not self.plotter:
            return

        # Remove old torch segments
        if self.torch_segments_actor is not None:
            self.plotter.remove_actor(self.torch_segments_actor)
            self.torch_segments_actor = None

        # Need at least 1 point to draw segments
        if len(self.picked_points) == 0:
            return

        # Create line segments from each point along its normal
        torch_lines = []
        for i, point in enumerate(self.picked_points):
            if i < len(self.point_normals):
                normal = self.point_normals[i]
                # Calculate the end point of the torch segment
                end_point = point + normal * self.torch_distance
                torch_lines.append([point, end_point])

        if torch_lines:
            torch_lines_array = np.array(torch_lines)
            # Flatten to (n*2, 3) for add_lines format
            line_points = torch_lines_array.reshape(-1, 3)

            # Create connectivity array: [2, p0, p1, 2, p2, p3, ...]
            # This tells PyVista to draw lines between consecutive pairs
            connectivity = []
            for i in range(0, len(line_points), 2):
                connectivity.extend([2, i, i + 1])

            # Create a polydata object with the torch line segments
            from pyvista import PolyData
            torch_polydata = PolyData(line_points, connectivity)

            # Add the torch lines to the plotter (red color)
            self.torch_segments_actor = self.plotter.add_mesh(
                torch_polydata,
                color='red',
                line_width=2,
                style='wireframe'
            )

    def clear_points(self):
        """Clear points based on 'all' radio button state"""
        if self.clear_all_radio.isChecked():
            # Clear all points
            self.picked_points = []
            self.point_normals = []
            self.points_list.clear()
            print("All points cleared")
        else:
            # Clear only the last point
            if len(self.picked_points) > 0:
                removed_point = self.picked_points.pop()
                if self.point_normals:
                    self.point_normals.pop()
                self.points_list.takeItem(self.points_list.count() - 1)
                print(f"Removed last point: ({removed_point[0]:.2f}, {removed_point[1]:.2f}, {removed_point[2]:.2f})")
            else:
                print("No points to clear")

        # Disable simulation button if no points
        if len(self.picked_points) == 0:
            self.simulation_btn.setEnabled(False)
            self.simulation_btn.setStyleSheet(
                "background-color: #888888; color: #cccccc; font-weight: bold; padding: 6px; font-size: 10px;"
            )
            # Exit simulation mode if active
            if self.simulation_mode:
                self.toggle_simulation_mode()

        # Update visualization
        self.update_markers()
        self.update_torch_segments()  # Update torch segments after clearing points
        self.update_path()  # Update path lines after clearing points
        # Force immediate render update
        self.plotter.render_window.Render()
        QApplication.instance().processEvents()

    def _calculate_surface_normal(self, point):
        """Calculate the surface normal at a given point on the mesh"""
        try:
            # Recompute normals to ensure they're accurate (not just relying on cached normals)
            self.current_mesh.compute_normals(inplace=True, cell_normals=False, point_normals=True)

            # Find the closest point on the mesh
            closest_point_id = self.current_mesh.find_closest_point(point)

            # Get the normal at that point
            normals = self.current_mesh.active_normals
            if normals is not None and closest_point_id < len(normals):
                normal = np.array(normals[closest_point_id])

                # Normalize the normal vector
                norm_magnitude = np.linalg.norm(normal)
                if norm_magnitude > 0:
                    normal = normal / norm_magnitude
                else:
                    print(f"  ! Warning: Normal magnitude is zero at point {point}")
                    return np.array([0, 0, 1])

                print(f"  ✓ Calculated normal at point {point}: {normal}")
                return normal
            else:
                # Fallback: return a default upward normal
                print(f"  ! Warning: Could not get normal from mesh at point {point}, using default (0, 0, 1)")
                return np.array([0, 0, 1])

        except Exception as e:
            print(f"Error calculating surface normal: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return a default upward normal
            return np.array([0, 0, 1])

    def _setup_point_picking(self):
        """Setup mouse click callback for point picking on the mesh"""
        if not self.plotter or not self.plotter.iren:
            print("Error: Cannot setup point picking without plotter")
            return

        try:
            # Register left click event on the render window
            self.plotter.iren.add_observer('LeftButtonPressEvent', self._on_mesh_pick)
            print("Point picking callback registered")
        except Exception as e:
            print(f"Error setting up point picking: {e}")

    def _remove_point_picking(self):
        """Remove mouse click callback for point picking"""
        if not self.plotter or not self.plotter.iren:
            return

        try:
            # Remove the observer
            self.plotter.iren.remove_observer('LeftButtonPressEvent')
            print("Point picking callback removed")
        except Exception as e:
            print(f"Error removing point picking: {e}")

    def _on_mesh_pick(self, obj, event):
        """Callback for mesh click - picks a point on the surface"""
        if not self.point_picking_mode or not self.plotter:
            return

        try:
            # Debounce: prevent multiple picks from the same click event (within 100ms)
            current_time = time.time()
            if current_time - self.last_pick_time < 0.1:
                return
            self.last_pick_time = current_time

            # Get the click position using snake_case method
            click_pos = self.plotter.iren.get_event_position()

            # Create a picker to find the closest point on the mesh
            from vtkmodules.vtkRenderingCore import vtkCellPicker
            picker = vtkCellPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)

            # Get the picked position in world coordinates
            if picker.GetCellId() >= 0:
                # Get the picked position
                picked_position = picker.GetPickPosition()

                # Calculate surface normal at the picked point
                normal = self._calculate_surface_normal(picked_position)

                # Add the point
                self.add_picked_point(picked_position, normal)

                # Force render update to show the point
                self.plotter.render_window.Render()
                QApplication.instance().processEvents()
                print(f"Point picked at: ({picked_position[0]:.2f}, {picked_position[1]:.2f}, {picked_position[2]:.2f})")
        except Exception as e:
            print(f"Error picking point: {e}")
            import traceback
            traceback.print_exc()


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
