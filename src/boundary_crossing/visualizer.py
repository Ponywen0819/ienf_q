"""
Crossing Visualizer

Visualization functions for boundary crossing detection results.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
from .crossing_analyzer import CrossingCandidate, CrossingResult


class CrossingVisualizer:
    """Visualize boundary crossing detection results"""

    def __init__(self, config: dict):
        self.config = config

        # Color scheme (BGR format for OpenCV)
        self.colors = {
            'boundary': (0, 255, 255),  # Yellow
            'candidate': (255, 255, 0),  # Cyan
            'success_high': (0, 255, 0),  # Green (high confidence)
            'success_low': (0, 165, 255),  # Orange (low confidence)
            'failure': (0, 0, 255),  # Red
            'crossing_point': (255, 0, 255),  # Magenta
            'text': (255, 255, 255),  # White
        }

    def visualize_all(
        self,
        image: np.ndarray,
        boundary_detector,
        candidates: List[CrossingCandidate],
        results: List[CrossingResult],
        statistics: Dict,
        output_path: Path = None
    ) -> np.ndarray:
        """
        Create comprehensive visualization of crossing analysis

        Args:
            image: Input image
            boundary_detector: BoundaryDetector instance
            candidates: List of crossing candidates
            results: List of crossing results
            statistics: Statistics dictionary
            output_path: Optional path to save the visualization

        Returns:
            Visualization image
        """
        # Create base visualization
        vis_image = self._prepare_image(image)

        # Draw boundary line
        vis_image = self._draw_boundary(vis_image, boundary_detector)

        # Draw candidate endpoints
        vis_image = self._draw_candidates(vis_image, candidates)

        # Draw extension paths and crossing points
        vis_image = self._draw_results(vis_image, results)

        # Add statistics text
        vis_image = self._add_statistics_text(vis_image, statistics)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)

        return vis_image

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for visualization (convert to BGR if needed)"""
        vis_image = image.copy()

        # Convert to BGR if grayscale
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        # Ensure BGR format
        elif vis_image.shape[2] == 3:
            # Assume RGB, convert to BGR
            if np.max(vis_image) <= 1.0:
                vis_image = (vis_image * 255).astype(np.uint8)

        return vis_image

    def _draw_boundary(
        self,
        image: np.ndarray,
        boundary_detector
    ) -> np.ndarray:
        """Draw boundary line on image"""
        boundary_points = boundary_detector.get_boundary_points()

        if len(boundary_points) > 1:
            pts = np.array(boundary_points, dtype=np.int32)
            cv2.polylines(
                image,
                [pts],
                isClosed=False,
                color=self.colors['boundary'],
                thickness=2
            )

        return image

    def _draw_candidates(
        self,
        image: np.ndarray,
        candidates: List[CrossingCandidate]
    ) -> np.ndarray:
        """Draw candidate endpoint markers"""
        for candidate in candidates:
            x, y = candidate.position
            cv2.circle(
                image,
                (x, y),
                radius=3,
                color=self.colors['candidate'],
                thickness=-1  # Filled circle
            )

            # Draw direction arrow
            arrow_length = 10
            end_x = int(x + candidate.direction[0] * arrow_length)
            end_y = int(y + candidate.direction[1] * arrow_length)

            cv2.arrowedLine(
                image,
                (x, y),
                (end_x, end_y),
                color=self.colors['candidate'],
                thickness=1,
                tipLength=0.3
            )

        return image

    def _draw_results(
        self,
        image: np.ndarray,
        results: List[CrossingResult]
    ) -> np.ndarray:
        """Draw extension paths and crossing points"""
        min_confidence = self.config.get('min_confidence', 0.7)

        for result in results:
            # Determine color based on result
            if result.success:
                if result.confidence >= min_confidence:
                    color = self.colors['success_high']
                else:
                    color = self.colors['success_low']
            else:
                color = self.colors['failure']

            # Draw path
            if len(result.path) > 1:
                for i in range(len(result.path) - 1):
                    pt1 = result.path[i]
                    pt2 = result.path[i + 1]
                    cv2.line(image, pt1, pt2, color, thickness=2)

            # Draw crossing point for successful crossings
            if result.success and result.crossing_point:
                cv2.circle(
                    image,
                    result.crossing_point,
                    radius=5,
                    color=self.colors['crossing_point'],
                    thickness=2
                )

                # Add confidence label
                label = f"{result.confidence:.2f}"
                self._draw_text_with_background(
                    image,
                    label,
                    (result.crossing_point[0] + 8, result.crossing_point[1] - 8),
                    font_scale=0.3,
                    color=self.colors['text']
                )

        return image

    def _add_statistics_text(
        self,
        image: np.ndarray,
        statistics: Dict
    ) -> np.ndarray:
        """Add statistics text overlay"""
        # Prepare text lines
        lines = [
            f"Candidates: {statistics.get('total_candidates', 0)}",
            f"Successful: {statistics.get('successful_crossings', 0)}",
            f"High Confidence: {statistics.get('high_confidence_crossings', 0)}",
        ]

        if 'mean_confidence' in statistics:
            lines.append(f"Mean Conf: {statistics['mean_confidence']:.3f}")

        if 'mean_length' in statistics:
            lines.append(f"Mean Length: {statistics['mean_length']:.1f}px")

        # Draw text box
        x, y = 10, 30
        line_height = 20

        for i, line in enumerate(lines):
            text_y = y + i * line_height
            self._draw_text_with_background(
                image,
                line,
                (x, text_y),
                font_scale=0.5,
                color=self.colors['text']
            )

        return image

    def _draw_text_with_background(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.5,
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = 1
    ):
        """Draw text with background rectangle for better visibility"""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = position
        padding = 2

        # Draw background rectangle
        cv2.rectangle(
            image,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1
        )

        # Draw text
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def create_comparison_grid(
        self,
        images: List[np.ndarray],
        titles: List[str] = None,
        output_path: Path = None
    ) -> np.ndarray:
        """
        Create a grid of images for comparison

        Args:
            images: List of images to display
            titles: Optional titles for each image
            output_path: Optional path to save the grid

        Returns:
            Grid image
        """
        if not images:
            return None

        # Determine grid layout
        n = len(images)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        # Get dimensions (assume all images same size)
        h, w = images[0].shape[:2]

        # Create grid
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            # Ensure image is BGR
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Place image in grid
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

            # Add title if provided
            if titles and idx < len(titles):
                text_y = row * h + 20
                text_x = col * w + 10
                self._draw_text_with_background(
                    grid,
                    titles[idx],
                    (text_x, text_y),
                    font_scale=0.6
                )

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), grid)

        return grid

    def create_legend(self, height: int = 150, width: int = 300) -> np.ndarray:
        """Create a legend image explaining the color coding"""
        legend = np.zeros((height, width, 3), dtype=np.uint8)

        # Legend items
        items = [
            ("Boundary Line", self.colors['boundary']),
            ("Candidate Endpoint", self.colors['candidate']),
            ("High Confidence Cross", self.colors['success_high']),
            ("Low Confidence Cross", self.colors['success_low']),
            ("Failed Extension", self.colors['failure']),
            ("Crossing Point", self.colors['crossing_point']),
        ]

        y_start = 20
        line_height = 20

        for i, (label, color) in enumerate(items):
            y = y_start + i * line_height

            # Draw color sample
            cv2.rectangle(legend, (10, y - 10), (30, y + 5), color, -1)

            # Draw label
            cv2.putText(
                legend,
                label,
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.colors['text'],
                1,
                cv2.LINE_AA
            )

        return legend

    def visualize_single_candidate_detailed(
        self,
        candidate_idx: int,
        candidate: 'CrossingCandidate',
        result: 'CrossingResult',
        image: np.ndarray,
        boundary_detector,
        epidermis_stats: Dict,
        config: Dict,
        output_path: Path
    ) -> np.ndarray:
        """
        Create detailed multi-panel visualization for a single candidate

        Args:
            candidate_idx: Index of the candidate (for labeling)
            candidate: CrossingCandidate object
            result: CrossingResult object
            image: Original image
            boundary_detector: BoundaryDetector instance
            epidermis_stats: Statistics dictionary
            config: Configuration dictionary
            output_path: Path to save the visualization

        Returns:
            Combined visualization image
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        # Create figure with 3x2 grid
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Extract green channel
        if len(image.shape) == 3:
            green_channel = image[:, :, 1]
        else:
            green_channel = image

        # Panel A: Local region view
        ax_local = fig.add_subplot(gs[0, 0])
        self._draw_local_region_panel(
            ax_local, candidate, result, image, boundary_detector
        )

        # Panel B: Green heatmap
        ax_heatmap = fig.add_subplot(gs[0, 1])
        self._draw_green_heatmap_panel(
            ax_heatmap, candidate, result, green_channel, boundary_detector
        )

        # Panel C: Direction vectors
        ax_direction = fig.add_subplot(gs[1, 0])
        self._draw_direction_panel(
            ax_direction, candidate, result, image
        )

        # Panel D: Intensity curve
        ax_curve = fig.add_subplot(gs[1, 1])
        self._draw_intensity_curve_panel(
            ax_curve, result, green_channel, epidermis_stats, config
        )

        # Panel E: Statistics table
        ax_stats = fig.add_subplot(gs[2, 0])
        self._draw_statistics_panel(
            ax_stats, candidate, result, epidermis_stats, config
        )

        # Panel F: Global context
        ax_global = fig.add_subplot(gs[2, 1])
        self._draw_global_context_panel(
            ax_global, candidate, image, boundary_detector
        )

        # Overall title
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        fig.suptitle(
            f'Candidate #{candidate_idx} - {status} (Confidence: {result.confidence:.3f})',
            fontsize=16,
            fontweight='bold'
        )

        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Detailed visualization saved: {output_path}")

        return None

    def _draw_local_region_panel(
        self,
        ax,
        candidate,
        result,
        image: np.ndarray,
        boundary_detector
    ):
        """Draw local region around candidate with path"""
        # Define region (200x200 around candidate)
        cx, cy = candidate.position
        half_size = 100

        x_min = max(0, cx - half_size)
        x_max = min(image.shape[1], cx + half_size)
        y_min = max(0, cy - half_size)
        y_max = min(image.shape[0], cy + half_size)

        # Extract region
        if len(image.shape) == 3:
            region = image[y_min:y_max, x_min:x_max]
        else:
            region = cv2.cvtColor(image[y_min:y_max, x_min:x_max], cv2.COLOR_GRAY2RGB)

        # Draw on region
        region_viz = region.copy()

        # Draw boundary line in this region
        boundary_y = boundary_detector.get_boundary_y(cx)
        if boundary_y is not None and y_min <= boundary_y < y_max:
            local_by = boundary_y - y_min
            cv2.line(region_viz, (0, local_by), (region_viz.shape[1], local_by),
                    (0, 255, 255), 2)

        # Draw candidate point
        local_cx = cx - x_min
        local_cy = cy - y_min
        cv2.circle(region_viz, (local_cx, local_cy), 5, (255, 0, 0), -1)

        # Draw path with gradient color
        if len(result.path) > 1:
            for i in range(len(result.path) - 1):
                pt1 = result.path[i]
                pt2 = result.path[i + 1]

                # Convert to local coordinates
                local_pt1 = (pt1[0] - x_min, pt1[1] - y_min)
                local_pt2 = (pt2[0] - x_min, pt2[1] - y_min)

                # Color gradient: green -> yellow -> red
                progress = i / max(len(result.path) - 1, 1)
                if progress < 0.5:
                    # Green to yellow
                    r = int(255 * (progress * 2))
                    g = 255
                    b = 0
                else:
                    # Yellow to red
                    r = 255
                    g = int(255 * (1 - (progress - 0.5) * 2))
                    b = 0

                cv2.line(region_viz, local_pt1, local_pt2, (b, g, r), 2)

        # Display
        ax.imshow(cv2.cvtColor(region_viz, cv2.COLOR_BGR2RGB))
        ax.set_title('A. Local Region View', fontweight='bold')
        ax.set_xlabel(f'Region: [{x_min},{y_min}] to [{x_max},{y_max}]')
        ax.axis('off')

    def _draw_green_heatmap_panel(
        self,
        ax,
        candidate,
        result,
        green_channel: np.ndarray,
        boundary_detector
    ):
        """Draw green channel heatmap with path overlay"""
        import matplotlib.pyplot as plt

        # Define same region as local view
        cx, cy = candidate.position
        half_size = 100

        x_min = max(0, cx - half_size)
        x_max = min(green_channel.shape[1], cx + half_size)
        y_min = max(0, cy - half_size)
        y_max = min(green_channel.shape[0], cy + half_size)

        # Extract region
        region = green_channel[y_min:y_max, x_min:x_max]

        # Display heatmap
        im = ax.imshow(region, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label='Green Intensity')

        # Draw boundary line
        boundary_y = boundary_detector.get_boundary_y(cx)
        if boundary_y is not None and y_min <= boundary_y < y_max:
            local_by = boundary_y - y_min
            ax.axhline(y=local_by, color='yellow', linestyle='--', linewidth=2, label='Boundary')

        # Draw candidate point
        local_cx = cx - x_min
        local_cy = cy - y_min
        ax.plot(local_cx, local_cy, 'r*', markersize=15, label='Start')

        # Draw path points with intensities
        if len(result.path) > 0:
            path_xs = [pt[0] - x_min for pt in result.path]
            path_ys = [pt[1] - y_min for pt in result.path]
            ax.plot(path_xs, path_ys, 'r-', linewidth=2, alpha=0.7, label='Path')

            # Annotate intensities at key points
            step = max(1, len(result.path) // 5)
            for i in range(0, len(result.path), step):
                pt = result.path[i]
                if y_min <= pt[1] < y_max and x_min <= pt[0] < x_max:
                    intensity = green_channel[pt[1], pt[0]]
                    ax.text(pt[0] - x_min, pt[1] - y_min, f'{intensity}',
                           color='white', fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        ax.set_title('B. Green Channel Heatmap', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('X (local)')
        ax.set_ylabel('Y (local)')

    def _draw_direction_panel(
        self,
        ax,
        candidate,
        result,
        image: np.ndarray
    ):
        """Draw direction vectors along the path"""
        # Same region
        cx, cy = candidate.position
        half_size = 100

        x_min = max(0, cx - half_size)
        x_max = min(image.shape[1], cx + half_size)
        y_min = max(0, cy - half_size)
        y_max = min(image.shape[0], cy + half_size)

        # Extract and display region
        if len(image.shape) == 3:
            region = image[y_min:y_max, x_min:x_max, 1]  # Green channel
        else:
            region = image[y_min:y_max, x_min:x_max]

        ax.imshow(region, cmap='gray', alpha=0.5)

        # Draw initial direction
        arrow_len = 20
        end_x = cx + candidate.direction[0] * arrow_len
        end_y = cy + candidate.direction[1] * arrow_len

        ax.arrow(cx - x_min, cy - y_min,
                candidate.direction[0] * arrow_len,
                candidate.direction[1] * arrow_len,
                head_width=5, head_length=3, fc='cyan', ec='cyan',
                linewidth=2, label='Initial Direction')

        # Draw direction changes along path
        if len(result.path) >= 3:
            for i in range(2, len(result.path), 2):
                # Calculate local direction
                pt_prev = result.path[i-2]
                pt_curr = result.path[i]

                dx = pt_curr[0] - pt_prev[0]
                dy = pt_curr[1] - pt_prev[1]
                length = np.sqrt(dx**2 + dy**2)

                if length > 0:
                    dx /= length
                    dy /= length

                    # Draw arrow
                    ax.arrow(pt_curr[0] - x_min, pt_curr[1] - y_min,
                            dx * 10, dy * 10,
                            head_width=3, head_length=2, fc='yellow', ec='yellow',
                            linewidth=1, alpha=0.7)

        # Mark candidate
        ax.plot(cx - x_min, cy - y_min, 'r*', markersize=15)

        ax.set_title('C. Direction Vectors', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('X (local)')
        ax.set_ylabel('Y (local)')
        ax.set_xlim(0, x_max - x_min)
        ax.set_ylim(y_max - y_min, 0)  # Invert y-axis

    def _draw_intensity_curve_panel(
        self,
        ax,
        result,
        green_channel: np.ndarray,
        epidermis_stats: Dict,
        config: Dict
    ):
        """Draw intensity curve along the path"""
        # Extract intensities along path
        intensities = []
        for pt in result.path:
            x, y = pt
            if 0 <= y < green_channel.shape[0] and 0 <= x < green_channel.shape[1]:
                intensities.append(green_channel[y, x])
            else:
                intensities.append(0)

        if not intensities:
            ax.text(0.5, 0.5, 'No path data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('D. Intensity Curve', fontweight='bold')
            return

        # Plot intensity curve
        steps = list(range(len(intensities)))
        ax.plot(steps, intensities, 'b-o', linewidth=2, markersize=4, label='Path Intensity')

        # Plot threshold
        mean_intensity = epidermis_stats.get('green_intensity_mean', 128)
        std_intensity = epidermis_stats.get('green_intensity_std', 30)
        sigma_multiplier = config.get('intensity_sigma_threshold', 2.0)
        threshold = mean_intensity - sigma_multiplier * std_intensity

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold ({threshold:.1f})')

        # Plot mean line
        ax.axhline(y=mean_intensity, color='green', linestyle='--', linewidth=1,
                  label=f'Mean ({mean_intensity:.1f})')

        # Mark points below threshold
        below_threshold = [(i, val) for i, val in enumerate(intensities) if val < threshold]
        if below_threshold:
            bt_steps, bt_vals = zip(*below_threshold)
            ax.plot(bt_steps, bt_vals, 'rx', markersize=10, label='Below Threshold')

        ax.set_title('D. Intensity Curve Along Path', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Green Intensity')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_statistics_panel(
        self,
        ax,
        candidate,
        result,
        epidermis_stats: Dict,
        config: Dict
    ):
        """Draw statistics comparison table"""
        ax.axis('off')

        # Diagnose failure reason
        failure_reason = self._diagnose_failure_reason(result, epidermis_stats, config)

        # Prepare data
        stats_data = [
            ['Metric', 'Value', 'Expected/Threshold'],
            ['─' * 20, '─' * 15, '─' * 20],
            ['Status', '✓ Success' if result.success else '✗ Failed', ''],
            ['Confidence', f'{result.confidence:.3f}', f'≥ {config.get("min_confidence", 0.7)}'],
            ['Path Length', f'{result.length} px', f'≥ {config.get("min_crossing_depth", 3)} px'],
            ['Mean Intensity', f'{result.mean_intensity:.1f}',
             f'{epidermis_stats.get("green_intensity_mean", 128):.1f} ± {epidermis_stats.get("green_intensity_std", 30):.1f}'],
            ['Start Position', f'({candidate.position[0]}, {candidate.position[1]})', ''],
            ['Distance to Boundary', f'{candidate.distance_to_boundary:.1f} px', '≤ 5 px'],
            ['', '', ''],
            ['Failure Reason:', failure_reason, ''],
        ]

        # Create table
        table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                        colWidths=[0.35, 0.3, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style failure reason row
        table[(9, 0)].set_text_props(weight='bold')
        table[(9, 1)].set_facecolor('#FFE699')

        # Color code status
        status_cell = table[(2, 1)]
        if result.success:
            status_cell.set_facecolor('#C6E0B4')
        else:
            status_cell.set_facecolor('#F4B084')

        ax.set_title('E. Statistics & Diagnosis', fontweight='bold')

    def _draw_global_context_panel(
        self,
        ax,
        candidate,
        image: np.ndarray,
        boundary_detector
    ):
        """Draw global view showing candidate location"""
        import matplotlib.pyplot as plt

        # Downsample image for global view
        scale = 0.2
        small_h = int(image.shape[0] * scale)
        small_w = int(image.shape[1] * scale)

        if len(image.shape) == 3:
            small_img = cv2.resize(image, (small_w, small_h))
            small_img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        else:
            small_img = cv2.resize(image, (small_w, small_h))
            small_img_rgb = cv2.cvtColor(small_img, cv2.COLOR_GRAY2RGB)

        ax.imshow(small_img_rgb)

        # Draw candidate location with box
        cx, cy = candidate.position
        box_size = 50
        rect = plt.Rectangle(
            ((cx - box_size) * scale, (cy - box_size) * scale),
            box_size * 2 * scale, box_size * 2 * scale,
            fill=False, edgecolor='red', linewidth=3
        )
        ax.add_patch(rect)

        # Draw boundary line
        boundary_pts = boundary_detector.get_boundary_points()
        if boundary_pts:
            xs = [pt[0] * scale for pt in boundary_pts[::10]]  # Subsample
            ys = [pt[1] * scale for pt in boundary_pts[::10]]
            ax.plot(xs, ys, 'y-', linewidth=2, alpha=0.7, label='Boundary')

        ax.set_title('F. Global Context', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')

    def _diagnose_failure_reason(
        self,
        result,
        epidermis_stats: Dict,
        config: Dict
    ) -> str:
        """Diagnose why the crossing detection failed"""
        if result.success:
            return "N/A (Success)"

        reasons = []

        # Check path length
        min_depth = config.get('min_crossing_depth', 3)
        if result.length < min_depth:
            reasons.append(f"Path too short ({result.length} < {min_depth} px)")

        # Check intensity
        mean_intensity = epidermis_stats.get('green_intensity_mean', 128)
        std_intensity = epidermis_stats.get('green_intensity_std', 30)
        sigma = config.get('intensity_sigma_threshold', 2.0)
        threshold = mean_intensity - sigma * std_intensity

        if result.mean_intensity < threshold:
            reasons.append(f"Intensity too low ({result.mean_intensity:.1f} < {threshold:.1f})")

        # Check confidence
        min_conf = config.get('min_confidence', 0.7)
        if result.confidence < min_conf:
            reasons.append(f"Low confidence ({result.confidence:.3f} < {min_conf})")

        if not reasons:
            reasons.append("Unknown (check individual metrics)")

        return "; ".join(reasons)
