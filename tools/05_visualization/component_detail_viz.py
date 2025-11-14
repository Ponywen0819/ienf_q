"""Component Detail Visualizer for displaying components with skeletons and seeds."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class ComponentDetailVisualizer:
    """Visualize individual components with skeleton and seed overlays."""

    def __init__(
        self,
        image_path: str | Path,
        components_path: str | Path,
        seeds_path: str | Path,
        skeletons_path: str | Path,
        labeled_skeletons_path: str | Path | None = None,
    ):
        """Initialize the visualizer.

        Args:
            image_path: Path to the original .tif image
            components_path: Path to components.json
            seeds_path: Path to seeds.json
            skeletons_path: Path to skeletons.json
            labeled_skeletons_path: Path to labeled_skeletons.png (optional)
        """
        self.image_path = Path(image_path)
        self.components_path = Path(components_path)
        self.seeds_path = Path(seeds_path)
        self.skeletons_path = Path(skeletons_path)
        self.labeled_skeletons_path = (
            Path(labeled_skeletons_path) if labeled_skeletons_path else None
        )

        # Load data
        self.image = self._load_image()
        self.components_data = self._load_json(self.components_path)
        self.seeds_data = self._load_json(self.seeds_path)
        self.skeletons_data = self._load_json(self.skeletons_path)
        self.labeled_skeletons = self._load_labeled_skeletons()

    def _load_image(self) -> np.ndarray:
        """Load the original TIF image.

        Returns:
            Loaded image as numpy array
        """
        img = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        # Extract green channel for RGB images, or use grayscale
        if len(img.shape) == 3:
            return img[:, :, 1]  # Green channel
        return img

    def _load_json(self, path: Path) -> Dict:
        """Load JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Loaded JSON data as dictionary
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_labeled_skeletons(self) -> np.ndarray | None:
        """Load the labeled skeletons image.

        Returns:
            Labeled skeletons image as numpy array, or None if not provided
        """
        if self.labeled_skeletons_path is None or not self.labeled_skeletons_path.exists():
            return None

        # Load as 16-bit image
        labeled_skeletons = cv2.imread(
            str(self.labeled_skeletons_path), cv2.IMREAD_UNCHANGED
        )
        if labeled_skeletons is None:
            print(f"Warning: Failed to load labeled skeletons: {self.labeled_skeletons_path}")
            return None

        return labeled_skeletons

    def select_component(
        self, min_area: int = 100, component_id: Optional[int] = None
    ) -> Dict:
        """Select a component for visualization.

        Args:
            min_area: Minimum area threshold for random selection
            component_id: Specific component ID to select (overrides random selection)

        Returns:
            Selected component data
        """
        components = self.components_data["components"]

        if component_id is not None:
            # Find specific component by ID
            for comp in components:
                if comp["id"] == component_id:
                    return comp
            raise ValueError(f"Component ID {component_id} not found")

        # Filter components by minimum area
        large_components = [c for c in components if c["area"] >= min_area]

        if not large_components:
            raise ValueError(
                f"No components found with area >= {min_area}. "
                f"Total components: {len(components)}"
            )

        # Select random component
        selected = random.choice(large_components)
        print(f"Selected component ID: {selected['id']}, Area: {selected['area']}")
        return selected

    def get_component_data(
        self, component_id: int
    ) -> Tuple[Dict, List[Dict], List[Dict]]:
        """Get all related data for a component.

        Args:
            component_id: Component ID

        Returns:
            Tuple of (component, seeds, skeleton_points)
        """
        # Find component
        component = None
        for comp in self.components_data["components"]:
            if comp["id"] == component_id:
                component = comp
                break

        if component is None:
            raise ValueError(f"Component {component_id} not found")

        # Find seeds for this component
        seeds = [
            s for s in self.seeds_data["seeds"] if s["component_id"] == component_id
        ]

        # Find skeleton for this component
        skeleton = None
        for skel in self.skeletons_data["skeletons"]:
            if skel["component_id"] == component_id:
                skeleton = skel
                break

        return component, seeds, skeleton

    def visualize_component(
        self,
        component_id: Optional[int] = None,
        min_area: int = 100,
        output_path: Optional[str | Path] = None,
        padding: int = 20,
    ) -> None:
        """Create visualization for a component.

        Args:
            component_id: Specific component ID (None for random selection)
            min_area: Minimum area for random selection
            output_path: Output file path (None for default)
            padding: Padding around component bounding box
        """
        # Select component
        if component_id is None:
            selected_comp = self.select_component(min_area=min_area)
            component_id = selected_comp["id"]

        # Get all related data
        component, seeds, skeleton = self.get_component_data(component_id)

        # Extract bounding box
        bbox = component["bbox"]
        x_min = max(0, bbox["x_min"] - padding)
        x_max = min(self.image.shape[1], bbox["x_max"] + padding)
        y_min = max(0, bbox["y_min"] - padding)
        y_max = min(self.image.shape[0], bbox["y_max"] + padding)

        # Extract component region
        component_region = self.image[y_min:y_max, x_min:x_max]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(component_region, cmap="gray", origin="upper")

        # Draw skeleton if available
        if skeleton and skeleton.get("skeleton_pixels", 0) > 0:
            self._draw_skeleton(ax, skeleton, x_min, y_min, component_id)

        # Draw seeds
        if seeds:
            self._draw_seeds(ax, seeds, x_min, y_min)

        # Add title and labels
        ax.set_title(
            f"Component {component_id} Detail View\n"
            f"Area: {component['area']}, "
            f"Seeds: {len(seeds)}, "
            f"Skeleton Length: {skeleton.get('skeleton_length', 0):.2f} if skeleton else 0",
            fontsize=14,
            pad=20,
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        # Add legend
        self._add_legend(ax)

        # Remove grid for cleaner look
        ax.grid(False)

        # Set aspect ratio to equal
        ax.set_aspect("equal")

        plt.tight_layout()

        # Save or show
        if output_path is None:
            output_path = (
                Path("output/visualization")
                / f"component_{component_id}_detail.png"
            )
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")
        plt.close()

    def _draw_skeleton(
        self, ax: plt.Axes, skeleton: Dict, x_offset: int, y_offset: int, component_id: int
    ) -> None:
        """Draw skeleton on the plot.

        Args:
            ax: Matplotlib axes
            skeleton: Skeleton data
            x_offset: X offset for coordinate transformation
            y_offset: Y offset for coordinate transformation
            component_id: Component ID for extracting skeleton pixels
        """
        # Draw actual skeleton pixels if labeled_skeletons image is available
        if self.labeled_skeletons is not None:
            # Extract all skeleton pixels for this component
            skeleton_mask = (self.labeled_skeletons == component_id)
            skeleton_coords = np.argwhere(skeleton_mask)  # Returns [[y, x], [y, x], ...]

            if len(skeleton_coords) > 0:
                # Transform coordinates to local plot coordinates
                y_coords = skeleton_coords[:, 0] - y_offset
                x_coords = skeleton_coords[:, 1] - x_offset

                # Draw skeleton pixels as scatter plot
                ax.scatter(
                    x_coords,
                    y_coords,
                    c="lime",
                    s=20,
                    alpha=0.8,
                    marker="s",
                    linewidths=0,
                    label="Skeleton",
                )

        # Draw endpoints (larger markers on top of skeleton)
        endpoints = skeleton.get("endpoints", [])
        for ep in endpoints:
            x = ep["x"] - x_offset
            y = ep["y"] - y_offset
            ax.plot(
                x,
                y,
                "o",
                color="red",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
            )

        # Draw branchpoints (larger markers on top of skeleton)
        branchpoints = skeleton.get("branchpoints", [])
        for bp in branchpoints:
            x = bp["x"] - x_offset
            y = bp["y"] - y_offset
            ax.plot(
                x,
                y,
                "s",
                color="cyan",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
            )

    def _draw_seeds(
        self, ax: plt.Axes, seeds: List[Dict], x_offset: int, y_offset: int
    ) -> None:
        """Draw seeds on the plot.

        Args:
            ax: Matplotlib axes
            seeds: List of seed data
            x_offset: X offset for coordinate transformation
            y_offset: Y offset for coordinate transformation
        """
        seed_styles = {
            "endpoint": {"marker": "o", "color": "red", "size": 10, "label": "Endpoint"},
            "branchpoint": {"marker": "s", "color": "blue", "size": 10, "label": "Branchpoint"},
            "curvature": {"marker": "*", "color": "white", "size": 12, "label": "Curvature"},
            "centroid": {"marker": "D", "color": "orange", "size": 10, "label": "Centroid"},
            "regular": {"marker": ".", "color": "lightblue", "size": 8, "label": "Regular"},
        }

        for seed in seeds:
            x = seed["position"]["x"] - x_offset
            y = seed["position"]["y"] - y_offset
            seed_type = seed.get("type", "regular")

            style = seed_styles.get(seed_type, seed_styles["regular"])

            ax.plot(
                x,
                y,
                marker=style["marker"],
                color=style["color"],
                markersize=style["size"],
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

    def _add_legend(self, ax: plt.Axes) -> None:
        """Add legend to the plot.

        Args:
            ax: Matplotlib axes
        """
        legend_elements = [
            mpatches.Patch(color="lime", label="Skeleton Endpoint"),
            mpatches.Patch(color="cyan", label="Skeleton Branchpoint"),
            mpatches.Patch(color="red", label="Seed: Endpoint"),
            mpatches.Patch(color="blue", label="Seed: Branchpoint"),
            mpatches.Patch(color="white", label="Seed: Curvature"),
            mpatches.Patch(color="orange", label="Seed: Centroid"),
            mpatches.Patch(color="lightblue", label="Seed: Regular"),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
        )
