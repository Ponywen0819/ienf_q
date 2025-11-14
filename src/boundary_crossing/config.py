"""
Configuration parameters for boundary crossing detection
"""

CROSSING_CONFIG = {
    # === Boundary Detection ===
    'boundary_tolerance': 5,  # Maximum distance (px) from boundary to consider as candidate
    'boundary_smoothing': True,  # Whether to smooth the boundary line
    'smoothing_window': 5,  # Window size for boundary smoothing

    # === Short-distance Extension ===
    'max_extension_length': 15,  # Maximum extension distance in pixels
    'step_size': 2,  # Step size for each extension iteration (pixels)
    'min_crossing_depth': 3,  # Minimum depth below boundary to confirm crossing (pixels)

    # === Direction Estimation ===
    'direction_window': 5,  # Number of points to use for direction estimation
    'min_direction_points': 3,  # Minimum points needed for direction estimation

    # === Image-based Filtering ===
    'intensity_sigma_threshold': 2.0,  # Number of standard deviations for intensity threshold
    'use_adaptive_threshold': True,  # Use adaptive threshold based on path progression

    # === Confidence Scoring ===
    'min_confidence': 0.7,  # Minimum confidence score to accept a crossing
    'min_path_length': 3,  # Minimum path length (points) for confidence calculation

    # === Statistical Features ===
    'extract_width': True,  # Whether to extract nerve width statistics
    'extract_curvature': True,  # Whether to extract curvature statistics
    'width_estimation_window': 5,  # Window size for width estimation

    # === Output ===
    'save_intermediate_results': True,  # Save intermediate processing results
    'verbose': True,  # Print detailed progress information
}


def get_config():
    """Get a copy of the configuration dictionary"""
    return CROSSING_CONFIG.copy()


def update_config(updates: dict):
    """Update configuration with custom parameters"""
    CROSSING_CONFIG.update(updates)
