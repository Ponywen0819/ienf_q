"""
神經網路疊加在 Viridis 影像上

將 MST 重建的神經網路拓樸疊加到 Viridis 色彩映射的背景影像上

使用範例:
python tools/overlay_network_on_viridis.py \\
    --network together/boundary_connection_output/merged_mst_forest.json \\
    --image data/Original/S163-2_a.tif \\
    --output output/overlay.png
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from matplotlib import cm
import networkx as nx
import json


def load_network_from_json(network_path: str) -> nx.Graph:
    """
    載入 JSON 格式的 MST 網路

    Args:
        network_path: JSON 檔案路徑

    Returns:
        NetworkX Graph
    """
    print(f"  載入 JSON 網路: {network_path}")

    with open(network_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立空圖
    G = nx.Graph()

    # 添加節點
    for node in data['nodes']:
        node_id = str(node['id'])
        G.add_node(node_id,
                   x=int(node['x']),
                   y=int(node['y']),
                   component_id=node.get('component_id'),
                   seed_type=node.get('seed_type'))

    # 添加邊
    for edge in data['edges']:
        source = str(edge['source']) if 'source' in edge else str(edge['source_id'])
        target = str(edge['target']) if 'target' in edge else str(edge['target_id'])

        # 處理 path 資料
        path_data = edge.get('path')
        if path_data:
            if isinstance(path_data, str):
                try:
                    import ast
                    path_list = ast.literal_eval(path_data)
                except:
                    path_list = None
            else:
                path_list = path_data
        else:
            path_list = None

        # 添加邊及其屬性
        G.add_edge(source, target,
                   weight=edge.get('weight'),
                   edge_type=edge.get('edge_type'),
                   path=path_list)

    print(f"    節點數: {G.number_of_nodes()}, 邊數: {G.number_of_edges()}")
    return G


def load_network_from_graphml(network_path: str) -> nx.Graph:
    """
    載入 GraphML 網路

    Args:
        network_path: GraphML 檔案路徑

    Returns:
        NetworkX Graph
    """
    print(f"  載入 GraphML 網路: {network_path}")
    G = nx.read_graphml(network_path)

    # 轉換節點座標為整數
    for node in G.nodes():
        if 'x' in G.nodes[node]:
            G.nodes[node]['x'] = int(float(G.nodes[node]['x']))
        if 'y' in G.nodes[node]:
            G.nodes[node]['y'] = int(float(G.nodes[node]['y']))

    # 轉換邊的 path 字串為列表
    for u, v in G.edges():
        edge_data = G[u][v]
        if 'path' in edge_data and isinstance(edge_data['path'], str):
            try:
                import ast
                path_list = ast.literal_eval(edge_data['path'])
                G[u][v]['path'] = path_list
            except:
                G[u][v]['path'] = None

    print(f"    節點數: {G.number_of_nodes()}, 邊數: {G.number_of_edges()}")
    return G


def load_network(network_path: str) -> nx.Graph:
    """
    載入網路檔案（自動偵測格式：JSON 或 GraphML）

    Args:
        network_path: 網路檔案路徑（.json 或 .graphml）

    Returns:
        NetworkX Graph
    """
    path = Path(network_path)

    if path.suffix.lower() == '.json':
        return load_network_from_json(network_path)
    elif path.suffix.lower() == '.graphml':
        return load_network_from_graphml(network_path)
    else:
        raise ValueError(f"不支援的網路檔案格式: {path.suffix}，請使用 .json 或 .graphml")




def extract_green_channel(image_path: str) -> np.ndarray:
    """
    提取影像的綠色通道

    Args:
        image_path: 影像路徑

    Returns:
        綠色通道灰階影像
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"無法讀取影像: {image_path}")

    # 如果是灰階影像，直接返回
    if len(image.shape) == 2:
        return image

    # 如果是彩色影像，提取綠色通道
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return image[:, :, 1]  # G 通道 (BGR)

    raise ValueError(f"不支援的影像格式: {image.shape}")






def draw_network_simple(
    green_channel: np.ndarray,
    network: nx.Graph,
    output_path: str,
    network_alpha: float = 0.8,
    edge_width: float = 1.5,
    edge_color: tuple = (0, 255, 255)   # BGR: 青色
):
    """
    簡化版：直接在影像上繪製網路（無邊框、軸等）

    Args:
        green_channel: 綠色通道灰階影像
        network: NetworkX Graph
        output_path: 輸出路徑
        network_alpha: 網路透明度 (0-1)
        edge_width: 邊線寬度（像素）
        edge_color: 邊顏色 (B, G, R)
    """
    # 1. 生成 Viridis 背景
    normalized = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
    normalized = np.clip(normalized, 0, 1)

    viridis_cmap = cm.get_cmap('viridis')
    colored = viridis_cmap(normalized)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)

    # 2. 創建疊加層（透明）
    overlay = colored_bgr.copy()

    # 3. 繪製網路的邊
    components = list(nx.connected_components(network))
    print(f"  連通分量數: {len(components)}")

    for component_nodes in components:
        subgraph = network.subgraph(component_nodes)

        for u, v, data in subgraph.edges(data=True):
            # 從節點屬性獲取座標
            node_u = network.nodes[u]
            node_v = network.nodes[v]

            x_u = node_u.get('x')
            y_u = node_u.get('y')
            x_v = node_v.get('x')
            y_v = node_v.get('y')

            if x_u is None or y_u is None or x_v is None or y_v is None:
                continue

            # 嘗試沿著路徑繪製
            path_data = data.get('path', None)
            if path_data and path_data != 'None':
                try:
                    if isinstance(path_data, str):
                        import ast
                        path = ast.literal_eval(path_data)
                    else:
                        path = path_data

                    if path and len(path) >= 2:
                        points = np.array([[pos[1], pos[0]] for pos in path], dtype=np.int32)
                        cv2.polylines(overlay, [points], isClosed=False,
                                    color=edge_color, thickness=int(edge_width))
                        continue
                except:
                    pass

            # 否則繪製直線
            pt1 = (int(x_u), int(y_u))
            pt2 = (int(x_v), int(y_v))
            cv2.line(overlay, pt1, pt2, edge_color, int(edge_width))

    # 4. 混合背景和疊加層
    result = cv2.addWeighted(colored_bgr, 1 - network_alpha, overlay, network_alpha, 0)

    # 5. 保存
    cv2.imwrite(output_path, result)

    print(f"  ✓ 已保存疊加影像: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='將 MST 神經網路疊加到 Viridis 背景影像上',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必要參數
    parser.add_argument(
        '--network', '-n',
        required=True,
        help='MST 網路檔案路徑（.json 格式，包含節點座標）'
    )

    parser.add_argument(
        '--image', '-i',
        required=True,
        help='輸入影像路徑（綠色通道或彩色影像）'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='輸出影像路徑'
    )

    # 可選參數
    parser.add_argument(
        '--network-alpha',
        type=float,
        default=0.8,
        help='網路透明度 0-1（預設: 0.8，1 為完全不透明）'
    )

    parser.add_argument(
        '--edge-width',
        type=float,
        default=1.5,
        help='邊線寬度（預設: 1.5）'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    for file_path, name in [(args.network, '網路'), (args.image, '影像')]:
        if not Path(file_path).exists():
            print(f"✗ 錯誤: {name}檔案不存在: {file_path}")
            sys.exit(1)

    # 創建輸出目錄
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("神經網路疊加在 Viridis 影像上")
    print("=" * 60)

    try:
        # 步驟 1: 載入資料
        print(f"\n[1/3] 載入資料...")
        network = load_network(args.network)
        green_channel = extract_green_channel(args.image)
        print(f"  影像尺寸: {green_channel.shape}")

        # 步驟 2: 生成疊加影像
        print(f"\n[2/3] 生成疊加影像...")
        print(f"  網路透明度: {args.network_alpha}")
        print(f"  邊線寬度: {args.edge_width}")

        draw_network_simple(
            green_channel=green_channel,
            network=network,
            output_path=args.output,
            network_alpha=args.network_alpha,
            edge_width=args.edge_width
        )

        print("\n" + "=" * 60)
        print("✓ 疊加完成！")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 疊加失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
