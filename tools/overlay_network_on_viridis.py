"""
神經網路疊加在 Viridis 影像上

將 MST 重建的神經網路拓樸疊加到 Viridis 色彩映射的背景影像上

使用方式:
python overlay_network_on_viridis.py \\
    --network output/reconstruction/mst_forest.graphml \\
    --seeds output/seeds/seeds.json \\
    --image data/Original/S163-2_a.tif \\
    --output output/overlay.png
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import json
from typing import List, Dict, Tuple


def load_network(network_path: str) -> nx.Graph:
    """
    載入 GraphML 網路

    Args:
        network_path: GraphML 檔案路徑

    Returns:
        NetworkX Graph
    """
    print(f"  載入網路: {network_path}")
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


def load_seeds(seeds_path: str) -> Dict[str, dict]:
    """
    載入種子資料

    Args:
        seeds_path: seeds.json 檔案路徑

    Returns:
        種子字典 (seed_id -> seed data)
    """
    print(f"  載入種子: {seeds_path}")

    with open(seeds_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    seeds_data = data.get('seeds', [])
    if not seeds_data:
        raise ValueError(f"seeds.json 格式錯誤：找不到 'seeds' 欄位")

    # 建立種子映射
    seed_map = {}
    for seed in seeds_data:
        seed_id = seed['seed_id']
        seed_map[seed_id] = {
            'id': seed_id,
            'x': seed['position']['x'],
            'y': seed['position']['y'],
            'component_id': seed.get('component_id'),
            'seed_type': seed.get('type')
        }
        # 也用字串作為 key
        seed_map[str(seed_id)] = seed_map[seed_id]

    print(f"    種子數: {len(seeds_data)}")
    return seed_map


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


def get_component_colors(num_components: int) -> list:
    """生成分量顏色列表"""
    if num_components <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.hsv

    colors = [cmap(i / max(num_components, 1)) for i in range(num_components)]
    return colors


def draw_network_transparent(
    green_channel: np.ndarray,
    network: nx.Graph,
    seed_map: Dict[str, dict],
    output_path: str,
    network_alpha: float = 0.8,
    node_size: float = 30,
    edge_width: float = 2.0,
    background_alpha: float = 0.6,
    show_colorbar: bool = True,
    title: str = None,
    dpi: int = 150,
    figsize: tuple = None
):
    """
    繪製透明神經網路疊加在 Viridis 背景上

    Args:
        green_channel: 綠色通道灰階影像
        network: NetworkX Graph
        seed_map: 種子映射字典
        output_path: 輸出路徑
        network_alpha: 網路透明度 (0-1)
        node_size: 節點大小
        edge_width: 邊線寬度
        background_alpha: 背景透明度 (0-1)
        show_colorbar: 是否顯示 colorbar
        title: 圖表標題
        dpi: 輸出解析度
        figsize: 圖表尺寸
    """
    # 計算合適的圖表尺寸
    if figsize is None:
        height, width = green_channel.shape
        figsize = (width / 100, height / 100)
        max_size = 20
        if figsize[0] > max_size or figsize[1] > max_size:
            scale = max_size / max(figsize)
            figsize = (figsize[0] * scale, figsize[1] * scale)

    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 1. 繪製 Viridis 背景
    im = ax.imshow(green_channel, cmap='viridis', alpha=background_alpha, interpolation='nearest')

    # 2. 添加 colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, label='Green Channel Intensity',
                           pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

    # 3. 獲取連通分量
    components = list(nx.connected_components(network))
    print(f"  連通分量數: {len(components)}")
    colors = get_component_colors(len(components))

    # 4. 繪製每個分量的網路
    for comp_id, component_nodes in enumerate(components):
        color = colors[comp_id]
        subgraph = network.subgraph(component_nodes)

        # 繪製邊
        for u, v, data in subgraph.edges(data=True):
            seed_u = seed_map.get(u) or seed_map.get(str(u))
            seed_v = seed_map.get(v) or seed_map.get(str(v))

            if not seed_u or not seed_v:
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
                        ys = [pos[0] for pos in path]
                        xs = [pos[1] for pos in path]
                        ax.plot(xs, ys, color=color, linewidth=edge_width,
                               alpha=network_alpha, zorder=5)
                        continue
                except:
                    pass

            # 否則繪製直線
            ax.plot([seed_u['x'], seed_v['x']], [seed_u['y'], seed_v['y']],
                   color=color, linewidth=edge_width, alpha=network_alpha, zorder=5)

        # 繪製節點
        for node_id in component_nodes:
            seed = seed_map.get(node_id) or seed_map.get(str(node_id))
            if not seed:
                continue

            x, y = seed['x'], seed['y']
            degree = subgraph.degree(node_id)

            # 分支點用較大的標記
            if degree >= 3:
                marker = 's'  # 方塊
                size = node_size * 2
                linewidth = 2
            else:
                marker = 'o'  # 圓點
                size = node_size
                linewidth = 1

            ax.scatter(x, y, c=[color], marker=marker, s=size,
                      edgecolors='black', linewidths=linewidth,
                      alpha=network_alpha, zorder=10)

    # 5. 設定標題
    if title:
        ax.set_title(title, fontsize=14, weight='bold', pad=10)
    else:
        ax.set_title(f'MST Network Overlay ({len(components)} components)',
                    fontsize=14, weight='bold', pad=10)

    # 6. 設定軸標籤
    ax.set_xlabel('X coordinate (pixels)', fontsize=10)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=10)

    # 7. 調整佈局
    plt.tight_layout()

    # 8. 保存
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存疊加影像: {output_path}")


def draw_network_transparent_simple(
    green_channel: np.ndarray,
    network: nx.Graph,
    seed_map: Dict[str, dict],
    output_path: str,
    network_alpha: float = 0.8,
    node_size: int = 3,
    edge_width: float = 1.5,
    node_color: tuple = (255, 255, 0),  # BGR: 黃色
    edge_color: tuple = (0, 255, 255)   # BGR: 青色
):
    """
    簡化版：直接在影像上繪製網路（無邊框、軸等）

    Args:
        green_channel: 綠色通道灰階影像
        network: NetworkX Graph
        seed_map: 種子映射字典
        output_path: 輸出路徑
        network_alpha: 網路透明度 (0-1)
        node_size: 節點半徑（像素）
        edge_width: 邊線寬度（像素）
        node_color: 節點顏色 (B, G, R)
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
            seed_u = seed_map.get(u) or seed_map.get(str(u))
            seed_v = seed_map.get(v) or seed_map.get(str(v))

            if not seed_u or not seed_v:
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
            pt1 = (int(seed_u['x']), int(seed_u['y']))
            pt2 = (int(seed_v['x']), int(seed_v['y']))
            cv2.line(overlay, pt1, pt2, edge_color, int(edge_width))

    # 4. 繪製網路的節點
    # for component_nodes in components:
    #     subgraph = network.subgraph(component_nodes)

    #     for node_id in component_nodes:
    #         seed = seed_map.get(node_id) or seed_map.get(str(node_id))
    #         if not seed:
    #             continue

    #         center = (int(seed['x']), int(seed['y']))
    #         degree = subgraph.degree(node_id)

    #         # 分支點用較大的圓
    #         radius = node_size * 2 if degree >= 3 else node_size
    #         cv2.circle(overlay, center, radius, node_color, -1)
    #         cv2.circle(overlay, center, radius, (0, 0, 0), 1)  # 黑色邊框

    # 5. 混合背景和疊加層
    result = cv2.addWeighted(colored_bgr, 1 - network_alpha, overlay, network_alpha, 0)

    # 6. 保存
    cv2.imwrite(output_path, result)

    print(f"  ✓ 已保存疊加影像（簡化版）: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='將 MST 神經網路疊加到 Viridis 背景影像上',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必要參數
    parser.add_argument(
        '--network', '-n',
        required=True,
        help='MST 網路檔案路徑 (mst_forest.graphml)'
    )

    parser.add_argument(
        '--seeds', '-s',
        required=True,
        help='種子檔案路徑 (seeds.json)'
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
        '--background-alpha',
        type=float,
        default=0.6,
        help='背景透明度 0-1（預設: 0.6，僅完整模式）'
    )

    parser.add_argument(
        '--node-size',
        type=float,
        default=30,
        help='節點大小（預設: 30，完整模式；3，簡化模式）'
    )

    parser.add_argument(
        '--edge-width',
        type=float,
        default=2.0,
        help='邊線寬度（預設: 2.0，完整模式；1.5，簡化模式）'
    )

    parser.add_argument(
        '--title',
        default=None,
        help='圖表標題（可選，僅完整模式）'
    )

    parser.add_argument(
        '--colorbar',
        action='store_true',
        help='顯示 colorbar（預設: False，僅完整模式）'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='輸出解析度（預設: 150，僅完整模式）'
    )

    parser.add_argument(
        '--simple',
        action='store_true',
        help='簡化模式：輸出無邊框的影像（忽略 colorbar, title 等選項）'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    for file_path, name in [(args.network, '網路'), (args.seeds, '種子'), (args.image, '影像')]:
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
        seed_map = load_seeds(args.seeds)
        green_channel = extract_green_channel(args.image)
        print(f"  影像尺寸: {green_channel.shape}")

        # 步驟 2: 生成疊加影像
        print(f"\n[2/3] 生成疊加影像...")
        print(f"  網路透明度: {args.network_alpha}")
        print(f"  背景透明度: {args.background_alpha}")

        if args.simple:
            # 簡化模式
            print("  使用簡化模式（無邊框）")
            node_size = 3 if args.node_size == 30 else int(args.node_size)
            edge_width = 1.5 if args.edge_width == 2.0 else args.edge_width

            draw_network_transparent_simple(
                green_channel=green_channel,
                network=network,
                seed_map=seed_map,
                output_path=args.output,
                network_alpha=args.network_alpha,
                node_size=node_size,
                edge_width=edge_width
            )
        else:
            # 完整模式
            print("  使用完整模式（含邊框和軸）")
            draw_network_transparent(
                green_channel=green_channel,
                network=network,
                seed_map=seed_map,
                output_path=args.output,
                network_alpha=args.network_alpha,
                node_size=args.node_size,
                edge_width=args.edge_width,
                background_alpha=args.background_alpha,
                show_colorbar=args.colorbar,
                title=args.title,
                dpi=args.dpi
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
