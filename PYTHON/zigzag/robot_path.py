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

    def calculate_path(self, start, end, step):
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

        Returns:
        --------
        list
            List of all points including start, intermediate points, and end
        """
        self.points = []

        # Calculate total distance
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        total_distance = math.sqrt(dx**2 + dy**2)

        # Calculate number of steps
        num_steps = int(total_distance / step)

        # Calculate unit vector direction
        if total_distance > 0:
            ux = dx / total_distance
            uy = dy / total_distance
        else:
            # Start and end are the same point
            self.points.append(start)
            print(f"Point 0: ({start[0]:.2f}, {start[1]:.2f})")
            return self.points

        # Add start point
        self.points.append(start)
        print(f"Point 0 (Start): ({start[0]:.2f}, {start[1]:.2f})")

        # Calculate and add intermediate points
        for i in range(1, num_steps + 1):
            distance = i * step
            x = start[0] + ux * distance
            y = start[1] + uy * distance
            self.points.append((x, y))
            print(f"Point {i}: ({x:.2f}, {y:.2f})")

        # Add end point if not already reached
        if num_steps * step < total_distance:
            self.points.append(end)
            print(f"Point {num_steps + 1} (End): ({end[0]:.2f}, {end[1]:.2f})")
        else:
            print(f"(End point reached at Point {num_steps})")

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
        ax.legend(loc='best')
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

        ax.legend(loc='best')

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
                           interval=interval, blit=True, repeat=True)

        plt.tight_layout()
        plt.show()


def main():
    """
    Example usage of the RobotPath class.
    """
    # Create instance
    robot_path = RobotPath()

    # Define start and end points
    start_point = (1, 3)
    end_point = (10, 5)
    step_distance = 0.3

    print("=" * 50)
    print("ROBOT PATH CALCULATION")
    print("=" * 50)
    print(f"Start Point: {start_point}")
    print(f"End Point: {end_point}")
    print(f"Step Distance: {step_distance}")
    print("=" * 50)
    print()

    # Calculate path
    points = robot_path.calculate_path(start_point, end_point, step_distance)

    print()
    print(f"Total points calculated: {len(points)}")
    print("=" * 50)

    # Visualize (static)
    #robot_path.visualize(start_point, end_point)

    # Visualize with animation (time interval in milliseconds)
    robot_path.visualize_time(start_point, end_point, interval=400)


if __name__ == "__main__":
    main()
