"""
Node extraction utilities for image analysis.

This module provides functions to extract representative nodes from labeled images
by finding connected components and their highest pixel value positions.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any


def extract_nodes_from_label_image(
    input_image: np.ndarray,
    label_image: np.ndarray,
    mask_image: Optional[np.ndarray] = None,
    min_component_size: int = 1,
    max_components: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract representative nodes from a labeled image by finding connected components
    and their highest pixel value positions.

    Args:
        input_image: Original grayscale input image (2D or 3D array)
        label_image: Binary or labeled image for component detection (2D array)
        mask_image: Optional mask to filter components (2D array)
        min_component_size: Minimum size of components to consider
        max_components: Maximum number of components to extract (None for all)

    Returns:
        List of dictionaries containing node information:
        - 'label': Component label number
        - 'position': (y, x) coordinates of highest pixel value
        - 'pixel_value': Highest pixel value in the component
        - 'component_size': Number of pixels in the component
        - 'all_positions': List of all (y, x) positions with highest value

    Example:
        >>> input_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        >>> label_img = np.zeros((100, 100), dtype=np.uint8)
        >>> label_img[20:30, 20:30] = 255  # Create a labeled region
        >>> nodes = extract_nodes_from_label_image(input_img, label_img)
        >>> print(f"Found {len(nodes)} nodes")
    """

    # Handle 3D input images by taking the appropriate channel
    if len(input_image.shape) == 3:
        if input_image.shape[2] > 1:
            # Take the second channel (index 1) as in the original code
            working_image = input_image[:, :, 1]
        else:
            working_image = input_image[:, :, 0]
    else:
        working_image = input_image.copy()

    # Ensure label image is uint8 for connected components
    label_uint8 = label_image.astype(np.uint8)

    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(label_uint8)

    print(f"Number of connected components found: {num_labels - 1}")

    nodes = []

    # Process each component (skip background label 0)
    for label in range(1, num_labels):
        # Create mask for current component
        component_mask = (labels_im == label).astype(np.uint8)

        # Check component size
        component_size = np.sum(component_mask)
        if component_size < min_component_size:
            continue

        # Apply additional mask if provided
        if mask_image is not None:
            component_mask = cv2.bitwise_and(component_mask, mask_image.astype(np.uint8))
            if np.sum(component_mask) == 0:
                continue

        # Apply mask to input image
        masked_input = cv2.bitwise_and(working_image, working_image, mask=component_mask)

        # Find highest pixel value in the component
        highest_pixel_value = np.max(masked_input)

        # Find all positions with the highest value
        highest_positions = np.where(masked_input == highest_pixel_value)

        if len(highest_positions[0]) == 0:
            continue

        # Use the first position as the representative position
        representative_position = (highest_positions[0][0], highest_positions[1][0])

        # Store all positions for reference
        all_positions = list(zip(highest_positions[0], highest_positions[1]))

        node_info = {
            'label': label,
            'position': representative_position,
            'pixel_value': highest_pixel_value,
            'component_size': component_size,
            'all_positions': all_positions
        }

        nodes.append(node_info)

        print(f"Label: {label}, Highest pixel value: {highest_pixel_value}, "
              f"Position: {representative_position}, Component size: {component_size}")

    # Limit number of components if specified
    if max_components is not None and len(nodes) > max_components:
        # Sort by pixel value (descending) and take top components
        nodes.sort(key=lambda x: x['pixel_value'], reverse=True)
        nodes = nodes[:max_components]
        print(f"Limited to top {max_components} components by pixel value")

    return nodes


def extract_nodes_from_cropped_region(
    input_image: np.ndarray,
    label_image: np.ndarray,
    crop_bounds: Tuple[int, int, int, int],
    mask_image: Optional[np.ndarray] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Extract nodes from a specific cropped region of the images.

    Args:
        input_image: Original input image
        label_image: Labeled image
        crop_bounds: (y_start, y_end, x_start, x_end) for cropping
        mask_image: Optional mask image
        **kwargs: Additional arguments passed to extract_nodes_from_label_image

    Returns:
        List of node dictionaries with positions relative to the original image
    """
    y_start, y_end, x_start, x_end = crop_bounds

    # Crop the images
    if len(input_image.shape) == 3:
        cropped_input = input_image[y_start:y_end, x_start:x_end, :]
    else:
        cropped_input = input_image[y_start:y_end, x_start:x_end]

    cropped_label = label_image[y_start:y_end, x_start:x_end]

    cropped_mask = None
    if mask_image is not None:
        cropped_mask = mask_image[y_start:y_end, x_start:x_end]

    # Extract nodes from cropped region
    nodes = extract_nodes_from_label_image(
        cropped_input, cropped_label, cropped_mask, **kwargs
    )

    # Adjust positions to be relative to original image
    for node in nodes:
        y, x = node['position']
        node['position'] = (y + y_start, x + x_start)

        # Adjust all positions as well
        adjusted_positions = [(y + y_start, x + x_start) for y, x in node['all_positions']]
        node['all_positions'] = adjusted_positions

    return nodes


def filter_nodes_by_distance(
    nodes: List[Dict[str, Any]],
    min_distance: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Filter nodes to remove those that are too close to each other.
    Keeps the node with higher pixel value when nodes are too close.

    Args:
        nodes: List of node dictionaries
        min_distance: Minimum distance between nodes

    Returns:
        Filtered list of nodes
    """
    if len(nodes) <= 1:
        return nodes

    # Sort by pixel value (descending)
    sorted_nodes = sorted(nodes, key=lambda x: x['pixel_value'], reverse=True)
    filtered_nodes = []

    for current_node in sorted_nodes:
        current_pos = current_node['position']

        # Check if current node is too close to any already accepted node
        too_close = False
        for accepted_node in filtered_nodes:
            accepted_pos = accepted_node['position']
            distance = np.sqrt((current_pos[0] - accepted_pos[0])**2 +
                             (current_pos[1] - accepted_pos[1])**2)

            if distance < min_distance:
                too_close = True
                break

        if not too_close:
            filtered_nodes.append(current_node)

    print(f"Filtered from {len(nodes)} to {len(filtered_nodes)} nodes "
          f"with minimum distance {min_distance}")

    return filtered_nodes


def get_node_positions(nodes: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    Extract just the positions from a list of nodes.

    Args:
        nodes: List of node dictionaries

    Returns:
        List of (y, x) position tuples
    """
    return [node['position'] for node in nodes]


def print_node_summary(nodes: List[Dict[str, Any]]) -> None:
    """
    Print a summary of extracted nodes.

    Args:
        nodes: List of node dictionaries
    """
    print(f"\n=== Node Extraction Summary ===")
    print(f"Total nodes extracted: {len(nodes)}")

    if nodes:
        pixel_values = [node['pixel_value'] for node in nodes]
        component_sizes = [node['component_size'] for node in nodes]

        print(f"Pixel value range: {min(pixel_values)} - {max(pixel_values)}")
        print(f"Average pixel value: {np.mean(pixel_values):.2f}")
        print(f"Component size range: {min(component_sizes)} - {max(component_sizes)}")
        print(f"Average component size: {np.mean(component_sizes):.2f}")

        print("\nNode details:")
        for i, node in enumerate(nodes):
            print(f"  Node {i}: Label={node['label']}, "
                  f"Pos={node['position']}, "
                  f"Value={node['pixel_value']}, "
                  f"Size={node['component_size']}")
    print("================================\n")