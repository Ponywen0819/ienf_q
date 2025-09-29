"""
Edge connection utilities for neural fiber analysis.

This module provides functions to connect nodes using pathfinding algorithms
and graph theory, extracting the logic from process_test.ipynb.
"""

import numpy as np
import networkx as nx
import heapq
import math
from typing import List, Tuple, Dict, Any, Optional, Union


def optimized_dijkstra_pathfinding(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_path_length: Optional[int] = None
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Optimized Dijkstra pathfinding algorithm for image-based navigation.

    Args:
        image: 2D image array to navigate through
        start: Starting position (y, x)
        end: Ending position (y, x)
        max_path_length: Maximum allowed path length

    Returns:
        Tuple of (path as list of coordinates, total cost)
    """
    rows, cols = image.shape

    # Initialize data structures
    costs = np.full((rows, cols), np.inf)
    costs[start] = 0
    priority_queue = [(0, start)]
    came_from = {start: None}
    path_lengths = {start: 1}

    # 8-directional movement
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Calculate maximum path length if not provided
    start_to_end_distance = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    if max_path_length is None:
        max_path_length = math.ceil(start_to_end_distance ** 2)
    else:
        max_path_length = min(math.ceil(start_to_end_distance * 5), max_path_length)

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        # Check if we reached the target
        if current_node[0] == end[0] and current_node[1] == end[1]:
            break

        for direction in directions:
            neighbor = (current_node[0] + direction[0],
                       current_node[1] + direction[1])

            # Boundary and obstacle checks
            if (0 <= neighbor[0] < rows and
                0 <= neighbor[1] < cols and
                image[neighbor] != 0):

                new_length = path_lengths[current_node] + 1

                # Path length constraint
                if new_length > max_path_length:
                    continue

                # Cost function: pixel value difference + path length
                new_cost = abs(int(image[current_node]) - int(image[neighbor])) + path_lengths[current_node]

                if new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    came_from[neighbor] = current_node
                    path_lengths[neighbor] = new_length
                    heapq.heappush(priority_queue, (new_cost, neighbor))

    # Path reconstruction
    if end not in came_from:
        return [], np.inf

    path = []
    current = end

    while current is not None:
        if len(path) > max_path_length:
            print("Exceeded max path length during reconstruction.")
            return [], np.inf
        path.append(current)
        current = came_from.get(current)

    path.reverse()
    return path, costs[end]


def astar_image_pathfinding(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_path_length: Optional[int] = None
) -> Tuple[List[Tuple[int, int]], float]:
    """
    A* pathfinding algorithm optimized for image navigation.
    Cost function: pixel value difference + path length + brightness penalty.

    Args:
        image: 2D image array to navigate through
        start: Starting position (y, x)
        end: Ending position (y, x)
        max_path_length: Maximum allowed path length

    Returns:
        Tuple of (path as list of coordinates, total cost)
    """
    rows, cols = image.shape

    def heuristic(node: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    # Initialize data structures
    open_set = [(heuristic(start, end), start)]
    came_from = {}
    g_score = {start: 0}
    path_lengths = {start: 1}

    # Set path length limit
    if max_path_length is None:
        distance = math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        max_path_length = min(math.ceil(distance ** 2), 20)

    # 8-directional movement
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        current_f, current = heapq.heappop(open_set)

        # Goal reached
        if current == end:
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1], g_score[current]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            # Boundary and obstacle checks
            if (not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols) or
                image[neighbor] == 0):
                continue

            new_length = path_lengths[current] + 1

            # Path length constraint
            if new_length > max_path_length:
                continue

            # Cost function: pixel difference + path length weight + brightness penalty
            pixel_diff = abs(int(image[current]) - int(image[neighbor]))
            tentative_g = (g_score[current] + pixel_diff +
                          new_length * 0.1 + (255 - image[neighbor]) * 0.01)

            # Update if better path found
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                path_lengths[neighbor] = new_length

                # f = g + h
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score, neighbor))

    return [], float('inf')


def calculate_node_connections(
    nodes: List[Dict[str, Any]],
    image: np.ndarray,
    max_distance: float = 20.0,
    pathfinding_algorithm: str = "astar",
    max_path_length: Optional[int] = None
) -> Tuple[nx.Graph, Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    """
    Calculate connections between nodes using pathfinding algorithms.

    Args:
        nodes: List of node dictionaries with 'position' key
        image: 2D image array for pathfinding
        max_distance: Maximum Euclidean distance to consider connections
        pathfinding_algorithm: "astar" or "dijkstra"
        max_path_length: Maximum path length for pathfinding

    Returns:
        Tuple of (NetworkX graph, dictionary of paths)
    """
    # Convert image to float64 for processing
    image_float = image.astype(np.float64)

    # Create graph
    G = nx.Graph()
    path_dict = {}

    # Extract node positions
    node_positions = [node['position'] for node in nodes]

    # Add nodes to graph
    for idx, pos in enumerate(node_positions):
        G.add_node(idx, position=pos)

    # Choose pathfinding algorithm
    if pathfinding_algorithm.lower() == "astar":
        pathfind_func = astar_image_pathfinding
    else:
        pathfind_func = optimized_dijkstra_pathfinding

    print(f"Calculating connections using {pathfinding_algorithm} algorithm...")

    # Calculate connections between all node pairs
    for i in range(len(node_positions)):
        for j in range(i + 1, len(node_positions)):
            pos_i = node_positions[i]
            pos_j = node_positions[j]

            # Calculate Euclidean distance
            pos_distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)

            # Skip if nodes are too far apart
            if pos_distance > max_distance:
                continue

            val_i = image_float[pos_i]
            val_j = image_float[pos_j]

            print(f"Calculating path from node {i} at {pos_i} (value={val_i:.1f}) "
                  f"to node {j} at {pos_j} (value={val_j:.1f})")

            # Find path between nodes
            path, weight = pathfind_func(image_float, pos_i, pos_j, max_path_length)

            if path and weight != np.inf:
                G.add_edge(i, j, weight=weight, distance=pos_distance)
                path_dict[(i, j)] = path
                print(f"  -> Path found with weight {weight:.2f}")
            else:
                print(f"  -> No path found")

    return G, path_dict


def find_optimal_connections(
    graph: nx.Graph,
    connection_strategy: str = "nearest_neighbor"
) -> Dict[int, Dict[str, Any]]:
    """
    Find optimal connections for each node based on the specified strategy.

    Args:
        graph: NetworkX graph with weighted edges
        connection_strategy: "nearest_neighbor", "minimum_spanning_tree", or "all_pairs"

    Returns:
        Dictionary mapping node IDs to their optimal connections
    """
    optimal_connections = {}

    if connection_strategy == "nearest_neighbor":
        # For each node, find its nearest neighbor
        for node in graph.nodes:
            min_weight = float('inf')
            min_target = None
            min_path = None

            for target in graph.nodes:
                if node != target:
                    try:
                        path = nx.dijkstra_path(graph, source=node, target=target, weight='weight')
                        weight = nx.dijkstra_path_length(graph, source=node, target=target, weight='weight')

                        if weight < min_weight:
                            min_weight = weight
                            min_target = target
                            min_path = path
                    except nx.NetworkXNoPath:
                        continue

            if min_path is not None:
                optimal_connections[node] = {
                    'target': min_target,
                    'path': min_path,
                    'weight': min_weight
                }

    elif connection_strategy == "minimum_spanning_tree":
        # Create minimum spanning tree
        mst = nx.minimum_spanning_tree(graph, weight='weight')
        for edge in mst.edges(data=True):
            node1, node2, data = edge
            weight = data['weight']

            # Add bidirectional connections
            for source, target in [(node1, node2), (node2, node1)]:
                if source not in optimal_connections:
                    optimal_connections[source] = []
                optimal_connections[source].append({
                    'target': target,
                    'weight': weight
                })

    elif connection_strategy == "all_pairs":
        # Keep all connections
        for node in graph.nodes:
            optimal_connections[node] = []
            for neighbor in graph.neighbors(node):
                weight = graph[node][neighbor]['weight']
                optimal_connections[node].append({
                    'target': neighbor,
                    'weight': weight
                })

    return optimal_connections


def visualize_connections_data(
    nodes: List[Dict[str, Any]],
    connections: Dict[int, Dict[str, Any]],
    path_dict: Dict[Tuple[int, int], List[Tuple[int, int]]]
) -> Dict[str, Any]:
    """
    Prepare visualization data for connections.

    Args:
        nodes: List of node dictionaries
        connections: Optimal connections from find_optimal_connections
        path_dict: Dictionary of calculated paths

    Returns:
        Dictionary with visualization data
    """
    viz_data = {
        'nodes': [],
        'edges': [],
        'paths': []
    }

    # Prepare node data
    for i, node in enumerate(nodes):
        viz_data['nodes'].append({
            'id': i,
            'position': node['position'],
            'pixel_value': node.get('pixel_value', 0)
        })

    # Prepare edge data
    for source, connection in connections.items():
        if isinstance(connection, dict) and 'target' in connection:
            # Single target (nearest neighbor)
            target = connection['target']
            weight = connection['weight']

            # Find the path
            path_key = (source, target) if (source, target) in path_dict else (target, source)
            path = path_dict.get(path_key, [])

            viz_data['edges'].append({
                'source': source,
                'target': target,
                'weight': weight,
                'path': path
            })

        elif isinstance(connection, list):
            # Multiple targets (MST or all pairs)
            for conn in connection:
                target = conn['target']
                weight = conn['weight']

                path_key = (source, target) if (source, target) in path_dict else (target, source)
                path = path_dict.get(path_key, [])

                viz_data['edges'].append({
                    'source': source,
                    'target': target,
                    'weight': weight,
                    'path': path
                })

    return viz_data


def connect_neural_fiber_nodes(
    nodes: List[Dict[str, Any]],
    image: np.ndarray,
    max_distance: float = 20.0,
    pathfinding_algorithm: str = "astar",
    connection_strategy: str = "nearest_neighbor",
    max_path_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main function to connect neural fiber nodes.

    Args:
        nodes: List of node dictionaries from node extraction
        image: 2D image array for pathfinding
        max_distance: Maximum distance to consider connections
        pathfinding_algorithm: "astar" or "dijkstra"
        connection_strategy: "nearest_neighbor", "minimum_spanning_tree", or "all_pairs"
        max_path_length: Maximum path length for pathfinding

    Returns:
        Dictionary with connection results and visualization data
    """
    print(f"Connecting {len(nodes)} nodes...")
    print(f"Parameters: max_distance={max_distance}, algorithm={pathfinding_algorithm}, "
          f"strategy={connection_strategy}")

    # Calculate all possible connections
    graph, path_dict = calculate_node_connections(
        nodes, image, max_distance, pathfinding_algorithm, max_path_length
    )

    print(f"Found {graph.number_of_edges()} potential connections")

    # Find optimal connections
    optimal_connections = find_optimal_connections(graph, connection_strategy)

    print(f"Selected {len(optimal_connections)} optimal connections")

    # Prepare visualization data
    viz_data = visualize_connections_data(nodes, optimal_connections, path_dict)

    return {
        'graph': graph,
        'optimal_connections': optimal_connections,
        'path_dict': path_dict,
        'visualization_data': viz_data,
        'statistics': {
            'total_nodes': len(nodes),
            'potential_connections': graph.number_of_edges(),
            'optimal_connections': len(optimal_connections),
            'algorithm_used': pathfinding_algorithm,
            'strategy_used': connection_strategy
        }
    }