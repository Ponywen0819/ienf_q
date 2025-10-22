from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import heapq
import math

def get_centroids(input_img, label_img):
    num_labels, labels_im = cv2.connectedComponents(label_img.astype(np.uint8))
    print("Number of labels: ", num_labels - 1)

    # get highest pixel value in each label, and record position of them in an array
    highest_pixel_values = []
    for label in range(1, num_labels):
        mask = np.zeros(label_img.shape, dtype=np.uint8)
        mask[labels_im == label] = 1
        # apply mask to clipped_input
        masked_input = cv2.bitwise_and(input_img, input_img, mask=mask)
        # get highest pixel value
        highest_pixel_value = np.max(masked_input)
        target = np.where(masked_input == highest_pixel_value)
        print("Label: ", label, " Highest pixel value: ", highest_pixel_value, " Position: ", target)
        highest_pixel_values.append((label, highest_pixel_value, target))
    print("Highest pixel values: ", highest_pixel_values)

    return highest_pixel_values

def astar_image_pathfinding(image, start, end, max_path_length=None):
    """
    針對圖像的 A* 路徑搜索
    成本函數：像素值差異 + 路徑長度
    """
    
    def heuristic(node, goal):
        # 歐幾里得距離作為啟發式
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
    
    # 初始化
    open_set = [(heuristic(start, end), start)]
    came_from = {}
    g_score = {start: 0}
    path_lengths = {start: 1}
    
    # 設定路徑長度限制
    if max_path_length is None:
        distance = math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        # max_path_length = min(math.ceil(distance ** 2) , 20)
        max_path_length = math.ceil(distance * 5)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        # 到達目標
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
            
            # 邊界和障礙物檢查
            if image[neighbor] == 0:
                continue
            
            new_length = path_lengths[current] + 1
            
            # 路徑長度限制
            if new_length > max_path_length:
                continue
            
            # 計算實際成本：像素差異 + 路徑長度權重
            pixel_diff = abs(int(image[current]) - int(image[neighbor]))
            tentative_g = g_score[current] + pixel_diff + new_length * 0.1 + (255 - image[neighbor]) * 0.05
            
            # 如果找到更好的路徑
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                path_lengths[neighbor] = new_length
                
                # f = g + h
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score, neighbor))
    
    return [], float('inf')

def construct_graph(image, centroids):
    G = nx.Graph()

    # Extract node positions from highest_pixel_values
    node_positions = [ (pos[0][0], pos[1][0]) for _, _, pos in centroids ]

    # Add nodes to graph
    for idx, pos in enumerate(node_positions):
      G.add_node(idx, position=pos)

    path_dict = {}

    # Add edges with weights (pixel value difference)
    for i in range(len(node_positions)):
      for j in range(i+1, len(node_positions)):
        pos_i = node_positions[i]
        pos_j = node_positions[j]
        val_i = image[pos_i]
        val_j = image[pos_j]

        pos_dis = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
        if pos_dis > 40:
          continue  # skip if distance is too far

        print(f"Calculating path from node {i} at {pos_i} with value {val_i} to node {j} at {pos_j} with value {val_j}")
        # res = optimized_std_dijkstra(clipped_input, pos_i, pos_j)
        res = astar_image_pathfinding(image, pos_i, pos_j)
        path, weight = res
        # print(f"Result: {path[0]} -> {path[1]} nodes, weight {weight}")
        G.add_edge(i, j, weight=weight)
        path_dict[(i, j)] = path
    
    return G, path_dict

def pick_shortest_path(G):
    shortest_paths = {}
    for node in G.nodes:
      min_weight = float('inf')
      min_target = None
      min_path = None
      for target in G.nodes:
        if node != target:
          try:
            path = nx.dijkstra_path(G, source=node, target=target, weight='weight')
            weight = nx.dijkstra_path_length(G, source=node, target=target, weight='weight')
            if weight < min_weight:
              min_weight = weight
              min_target = target
              min_path = path
          except nx.NetworkXNoPath:
            continue
      if min_path is not None:
        shortest_paths[node] = {'target': min_target, 'path': min_path, 'weight': min_weight}
    return shortest_paths

def draw_paths(image, path_dict, shortest_paths):
    for node, path_info in shortest_paths.items():
        path_nodes = path_info['path']
        print(path_info)
        for i in range(len(path_nodes) - 1):
            n1 = path_nodes[i]
            n2 = path_nodes[i + 1]
            if (n1, n2) in path_dict:
                path = path_dict[(n1, n2)]
            elif (n2, n1) in path_dict:
                path = path_dict[(n2, n1)]
            else:
                continue
            
            for (x, y) in path:
                if path_info['weight'] < 100:  # Only draw paths with weight less than 1000
                  image[x, y, 0] = 128
                  image[x, y, 1] = 0  # Draw the path in white
                  image[x,y,2] = 128
                elif path_info['weight'] < 200:
                  image[x, y, 0] = 0
                  image[x, y, 1] = 128  # Draw the path in white
                  image[x,y,2] = 128
                else:
                  image[x, y, 0] = 0
                  image[x, y, 1] = 255  # Draw the path in white
                  image[x,y,2] = 255  
    return image

def draw_centroids(image, centroids):
    for _, _, pos in centroids:
        # draw a yellow circle at the centroid position
        cv2.circle(image, (pos[1][0], pos[0][0]), 1, (255, 255, 0), -1)
    return image

def main(path: str):
    input_img = Image.open(path)
    input_img = np.array(input_img)[:, :, 1]

    label_image = Image.open(path.replace(".tif", "_label.tif"))
    label_image = np.array(label_image)

    centroids = get_centroids(input_img, label_image)
    G, path_dict = construct_graph(input_img, centroids)
    shortest_paths = pick_shortest_path(G)
    result_image = draw_paths(input_img.copy(), path_dict, shortest_paths)
    # Turn result to RGB image for better visualization
    result_image = np.stack([result_image]*3, axis=-1)
    result_image = draw_centroids(result_image, centroids)

    Image.fromarray(result_image).save("result_image.png")



if __name__ == "__main__":
    main("/home/bl515-ml/Documents/shaio_jie/ienf_q/Centered/processed_S163-2_a.tif")