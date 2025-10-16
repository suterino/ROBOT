import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


class RobotPath:
    """
    A class to calculate and visualize intermediate points along a path from start to end.
    """

    def __init__(self):
        self.points = []

    def calculate_path(self, start, end, step, type="straight", back=0, width=0):
        """
        Calculate intermediate points from start to end with specified step distance.

        Parameters:
        -----------
        start : tuple
            Starting point (x, y)
        end : tuple
            Ending point (x, y)
        step : float
            Distance between consecutive points
        type : str, optional
            Path type: "straight" (default), "backAndForth", or "zigzag"
        back : float, optional
            Distance to move back after each forward step (only for "backAndForth" type).
            Must be smaller than step. Default is 0.
        width : float, optional
            Width of the zigzag pattern perpendicular to the main direction (only for "zigzag" type).
            Default is 0.

        Returns:
        --------
        list
            List of all points including start, intermediate points, and end

        Raises:
        -------
        ValueError
            If back is specified with type="straight" or if back >= step
        """
        self.points = []

        # Validate inputs
        if type == "straight" and back != 0:
            raise ValueError("Parameter 'back' must be 0 when type='straight'")

        if type == "straight" and width != 0:
            raise ValueError("Parameter 'width' must be 0 when type='straight'")

        if type == "backAndForth" and width != 0:
            raise ValueError("Parameter 'width' must be 0 when type='backAndForth'")

        if type == "zigzag" and back != 0:
            raise ValueError("Parameter 'back' must be 0 when type='zigzag'")

        if back >= step:
            raise ValueError(f"Parameter 'back' ({back}) must be smaller than 'step' ({step})")

        if back < 0:
            raise ValueError(f"Parameter 'back' ({back}) must be non-negative")

        if width < 0:
            raise ValueError(f"Parameter 'width' ({width}) must be non-negative")

        # Calculate total distance
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        total_distance = math.sqrt(dx**2 + dy**2)

        # Calculate unit vector direction
        if total_distance > 0:
            ux = dx / total_distance
            uy = dy / total_distance
        else:
            # Start and end are the same point
            self.points.append(start)
            return self.points

        # Add start point
        self.points.append(start)

        if type == "straight":
            # Original straight line behavior
            num_steps = int(total_distance / step)

            for i in range(1, num_steps + 1):
                distance = i * step
                x = start[0] + ux * distance
                y = start[1] + uy * distance
                self.points.append((x, y))

            # Add end point if not already reached
            if num_steps * step < total_distance:
                self.points.append(end)

        elif type == "backAndForth":
            # Back and forth pattern: forward by step, back by back, net progress = step - back
            net_progress = step - back
            current_distance = 0

            while current_distance < total_distance:
                # Move forward by step
                forward_distance = current_distance + step

                # Check if we've reached or passed the end
                if forward_distance >= total_distance:
                    # Add final end point
                    self.points.append(end)
                    break

                # Add forward point
                x_forward = start[0] + ux * forward_distance
                y_forward = start[1] + uy * forward_distance
                self.points.append((x_forward, y_forward))

                # Move back by back distance
                if back > 0:
                    backward_distance = forward_distance - back
                    x_back = start[0] + ux * backward_distance
                    y_back = start[1] + uy * backward_distance
                    self.points.append((x_back, y_back))

                # Update current distance (net progress)
                current_distance = forward_distance - back

        elif type == "zigzag":
            # Zigzag pattern perpendicular to the main direction
            # Calculate perpendicular unit vector (rotate 90 degrees counter-clockwise)
            perp_ux = -uy
            perp_uy = ux

            # Start at left side of the start point (perpendicular offset)
            current_distance = 0
            side = -1  # Start on left side (-1), then alternate to right (+1)

            # First point: offset to the left by width/2
            x_offset = start[0] + perp_ux * (width / 2) * side
            y_offset = start[1] + perp_uy * (width / 2) * side
            self.points.append((x_offset, y_offset))

            while current_distance < total_distance:
                # Move forward by step
                current_distance += step

                # Check if we've reached or passed the end
                if current_distance >= total_distance:
                    # Move to the final position on the end line
                    # Position on the opposite side from current side
                    side = -side
                    x_final = end[0] + perp_ux * (width / 2) * side
                    y_final = end[1] + perp_uy * (width / 2) * side
                    self.points.append((x_final, y_final))
                    break

                # Switch to opposite side
                side = -side

                # Calculate new position: forward along main direction + perpendicular offset
                x_new = start[0] + ux * current_distance + perp_ux * (width / 2) * side
                y_new = start[1] + uy * current_distance + perp_uy * (width / 2) * side
                self.points.append((x_new, y_new))

        else:
            raise ValueError(f"Invalid type '{type}'. Must be 'straight', 'backAndForth', or 'zigzag'")

        return self.points

    def visualize(self, start, end):
        """
        Visualize the calculated path on a Cartesian plot.

        Parameters:
        -----------
        start : tuple
            Starting point (x, y)
        end : tuple
            Ending point (x, y)
        """
        if not self.points:
            print("No points to visualize. Please run calculate_path first.")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Determine plot boundaries with some padding
        all_x = [p[0] for p in self.points]
        all_y = [p[1] for p in self.points]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Add padding
        x_padding = (x_max - x_min) * 0.1 if x_max != x_min else 1
        y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 1

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Draw rectangle (workspace boundary)
        rect_width = (x_max - x_min) + 2 * x_padding
        rect_height = (y_max - y_min) + 2 * y_padding
        rect = patches.Rectangle(
            (x_min - x_padding, y_min - y_padding),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor='black',
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

        # Plot the path line
        ax.plot(all_x, all_y, 'b-', linewidth=1, alpha=0.5, label='Path')

        # Plot intermediate points
        if len(self.points) > 2:
            intermediate_x = [p[0] for p in self.points[1:-1]]
            intermediate_y = [p[1] for p in self.points[1:-1]]
            ax.scatter(intermediate_x, intermediate_y, c='blue', s=50,
                      marker='o', label='Intermediate Points', zorder=3)

        # Plot start point
        ax.scatter(start[0], start[1], c='green', s=200, marker='o',
                  label='Start', zorder=4, edgecolors='black', linewidths=2)

        # Plot end point
        ax.scatter(end[0], end[1], c='red', s=200, marker='s',
                  label='End', zorder=4, edgecolors='black', linewidths=2)

        # Add labels to start and end points
        ax.annotate(f'Start\n({start[0]:.1f}, {start[1]:.1f})',
                   xy=start, xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.3))

        ax.annotate(f'End\n({end[0]:.1f}, {end[1]:.1f})',
                   xy=end, xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3))

        # Set labels and title
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Robot Path Visualization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.show()

    def visualize_time(self, start, end, interval):
        """
        Visualize the calculated path with animation, showing points appearing over time.

        Parameters:
        -----------
        start : tuple
            Starting point (x, y)
        end : tuple
            Ending point (x, y)
        interval : float
            Time interval in milliseconds between points appearing
        """
        if not self.points:
            print("No points to visualize. Please run calculate_path first.")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Determine plot boundaries with some padding
        all_x = [p[0] for p in self.points]
        all_y = [p[1] for p in self.points]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Add padding
        x_padding = (x_max - x_min) * 0.1 if x_max != x_min else 1
        y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 1

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Draw rectangle (workspace boundary)
        rect_width = (x_max - x_min) + 2 * x_padding
        rect_height = (y_max - y_min) + 2 * y_padding
        rect = patches.Rectangle(
            (x_min - x_padding, y_min - y_padding),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor='black',
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

        # Set labels and title
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Robot Path Animation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Initialize plot elements
        path_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5, label='Path')
        intermediate_scatter = ax.scatter([], [], c='blue', s=50, marker='o',
                                         label='Intermediate Points', zorder=3)
        current_point_scatter = ax.scatter([], [], c='orange', s=150, marker='o',
                                          label='Current Point', zorder=5,
                                          edgecolors='black', linewidths=2)

        # Plot start point (always visible)
        ax.scatter(start[0], start[1], c='green', s=200, marker='o',
                  label='Start', zorder=4, edgecolors='black', linewidths=2)

        # Plot end point (always visible)
        ax.scatter(end[0], end[1], c='red', s=200, marker='s',
                  label='End', zorder=4, edgecolors='black', linewidths=2)

        # Add labels to start and end points
        ax.annotate(f'Start\n({start[0]:.1f}, {start[1]:.1f})',
                   xy=start, xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.3))

        ax.annotate(f'End\n({end[0]:.1f}, {end[1]:.1f})',
                   xy=end, xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3))

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

        # Animation update function
        def update(frame):
            # Update path line up to current frame
            current_points = self.points[:frame+1]
            path_x = [p[0] for p in current_points]
            path_y = [p[1] for p in current_points]
            path_line.set_data(path_x, path_y)

            # Update intermediate points (exclude first and last)
            if frame > 0:
                intermediate_points = current_points[1:]
                intermediate_x = [p[0] for p in intermediate_points]
                intermediate_y = [p[1] for p in intermediate_points]
                intermediate_scatter.set_offsets(list(zip(intermediate_x, intermediate_y)))

            # Update current point
            if frame < len(self.points):
                current_point_scatter.set_offsets([self.points[frame]])

            return path_line, intermediate_scatter, current_point_scatter

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(self.points),
                           interval=interval, blit=True, repeat=False)

        plt.tight_layout()
        plt.show()


def main():
    """
    Example usage of the RobotPath class.
    """
    # Create instance
    robot_path = RobotPath()

    # Define start and end points
    start_point = (0.5, 5)
    end_point = (10, 3)
    step_distance = 0.1
    back_distance = 0.0
    width_distance =0.25

    print("=" * 50)
    print("ROBOT PATH CALCULATION")
    print("=" * 50)
    print(f"Start Point: {start_point}")
    print(f"End Point: {end_point}")
    print(f"Step Distance: {step_distance}")
    print(f"Back Distance: {back_distance}")
    print(f"Width Distance: {width_distance}")
    print("=" * 50)
    print()

    # Calculate path - choose one of the following options:

    # Option 1: Straight path (default)
    # points = robot_path.calculate_path(start_point, end_point, step_distance)

    # Option 2: Back and forth path
    # points = robot_path.calculate_path(start_point, end_point, step_distance,
    #                                   type="backAndForth", back=back_distance)

    # Option 3: Zigzag path
    points = robot_path.calculate_path(start_point, end_point, step_distance,
                                      type="zigzag", width=width_distance)

    print()
    print(f"Total points calculated: {len(points)}")
    print("=" * 50)
    print()

    # Print all points
    for i, point in enumerate(points):
        if i == 0:
            print(f"Point {i} (Start): ({point[0]:.2f}, {point[1]:.2f})")
        elif i == len(points) - 1:
            print(f"Point {i} (End): ({point[0]:.2f}, {point[1]:.2f})")
        else:
            print(f"Point {i}: ({point[0]:.2f}, {point[1]:.2f})")

    print()
    print("=" * 50)

    # Visualize (static)
    #robot_path.visualize(start_point, end_point)

    # Visualize with animation (time interval in milliseconds)
    robot_path.visualize_time(start_point, end_point, interval=400)


if __name__ == "__main__":
    main()
