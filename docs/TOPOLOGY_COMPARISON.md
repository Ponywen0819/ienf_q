# 拓扑比对工具使用指南

## 概述

`tools/compare_topologies.py` 是一个独立的拓扑比对工具，专注于计算两个拓扑图之间的平均 Hausdorff 距离，**不依赖于图像处理 Pipeline**。

### 主要特点

- ✅ **独立运行**：不需要 NeuralReconstructionPipeline
- ✅ **多格式支持**：Pickle、JSON、GraphML、GML
- ✅ **包含边路径点**：计算距离时包含节点和所有边上的路径点
- ✅ **批量处理**：支持批量比对整个目录
- ✅ **平均 Hausdorff 距离**：比最大距离更稳健

## 安装

工具已包含在项目中，无需额外安装。确保已安装项目依赖：

```bash
uv sync
```

## 使用方法

### 1. 单个拓扑对比对

比对两个拓扑文件：

```bash
uv run python tools/compare_topologies.py \
    --topology1 path/to/topology1.pkl \
    --topology2 path/to/topology2.pkl
```

**输出示例：**

```
================================================================================
比對結果
================================================================================
拓樸 1: topology1
  節點數: 322
  邊數: 248
  總點數: 5171

拓樸 2: topology2
  節點數: 1055
  邊數: 596
  總點數: 1604

平均 Hausdorff 距離: 25.9206 像素
================================================================================
```

### 2. 保存结果到文件

```bash
uv run python tools/compare_topologies.py \
    --topology1 topology1.pkl \
    --topology2 topology2.pkl \
    --output result.json
```

### 3. 批量比对

比对两个目录中的所有配对拓扑：

```bash
uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir output/predictions/ \
    --gt-dir output/ground_truth/ \
    --output results.csv
```

**批量模式规则：**
- 自动匹配相同文件名（不同扩展名也可以）
- 支持 `.pkl`, `.pickle`, `.json`, `.graphml`, `.gml` 扩展名
- 输出统计摘要（平均值、中位数、标准差等）

**批量输出示例：**

```
================================================================================
批次比對統計
================================================================================
總共比對: 150 對
成功: 148
失敗: 2

平均 Hausdorff 距離統計:
  平均值: 23.4567
  中位數: 21.2345
  標準差: 8.9012
  最小值: 5.6789
  最大值: 45.6789
================================================================================
```

### 4. 详细日志模式

```bash
uv run python tools/compare_topologies.py \
    --topology1 topology1.pkl \
    --topology2 topology2.pkl \
    --verbose
```

## 支持的文件格式

### 推荐格式

| 格式 | 扩展名 | 优点 | 缺点 |
|------|--------|------|------|
| **Pickle** | `.pkl`, `.pickle` | ✅ 最快<br>✅ 保留所有数据类型<br>✅ NetworkX 原生格式 | ❌ 不可读<br>❌ Python 专用 |
| **JSON** | `.json` | ✅ 可读<br>✅ 通用格式<br>✅ 易于调试 | ❌ 较大文件<br>❌ 略慢 |

### 其他格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| GraphML | `.graphml` | ⚠️ 路径属性可能丢失，不推荐用于包含复杂边属性的图 |
| GML | `.gml` | ⚠️ 简单文本格式，功能有限 |

## 保存拓扑文件

### 从代码中保存

```python
import networkx as nx
from tools.compare_topologies import TopologyLoader

# 创建或获取你的拓扑图
graph = nx.Graph()
# ... 添加节点和边 ...

# 保存
loader = TopologyLoader()

# 推荐：Pickle 格式
loader.save(graph, Path("output/my_topology.pkl"), format='pickle')

# 或 JSON 格式（可读）
loader.save(graph, Path("output/my_topology.json"), format='json')
```

### 边属性要求

拓扑图的边可以包含路径点：

```python
# 支持两种边属性名称
graph.add_edge(node1, node2, path=[(y1, x1), (y2, x2), ...])
# 或
graph.add_edge(node1, node2, **{'path-coordinates': [(y1, x1), (y2, x2), ...]})
```

## 实际使用案例

### 案例 1：比对 Pipeline 输出与 GT

```bash
# 1. 从 Pipeline 运行获得预测拓扑（假设 Pipeline 已修改为保存拓扑）
uv run python test_pipeline.py  # 生成 output/pred_topology.pkl

# 2. 从 GT 标注提取拓扑（假设已有提取脚本）
# 或者直接使用现有的 GT 拓扑文件

# 3. 比对
uv run python tools/compare_topologies.py \
    --topology1 output/pred_topology.pkl \
    --topology2 data/gt_topology.pkl \
    --output comparison_result.json
```

### 案例 2：评估不同参数配置

```bash
# 运行 Pipeline 生成不同配置的拓扑
uv run python test_pipeline.py --config config1.yaml  # -> output/config1_topology.pkl
uv run python test_pipeline.py --config config2.yaml  # -> output/config2_topology.pkl

# 批量比对所有配置
uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir output/configs/ \
    --gt-dir data/ground_truth/ \
    --output config_comparison.csv
```

### 案例 3：使用 S1585-2_a 样本测试

```python
# test_s1585_topology_comparison.py
from pathlib import Path
from PIL import Image
import numpy as np
from tools.compare_topologies import TopologyLoader, TopologyComparator
from tools.evaluate_dataset import TopologyExtractor

# 载入图像
data_dir = Path("data/S1585-2_a")
annotation = np.array(Image.open(data_dir / "annotation.png"))
label = np.array(Image.open(data_dir / "label.png"))

# 提取拓扑
extractor = TopologyExtractor()
graph_pred = extractor.extract_from_gt(annotation)
graph_gt = extractor.extract_from_gt(label)

# 保存拓扑
loader = TopologyLoader()
loader.save(graph_pred, Path("output/S1585-2_a_pred.pkl"), 'pickle')
loader.save(graph_gt, Path("output/S1585-2_a_gt.pkl"), 'pickle')

# 使用命令行工具比对
# uv run python tools/compare_topologies.py \
#     --topology1 output/S1585-2_a_pred.pkl \
#     --topology2 output/S1585-2_a_gt.pkl
```

## 输出格式

### JSON 输出格式

```json
{
  "label1": "topology1",
  "label2": "topology2",
  "num_nodes1": 322,
  "num_nodes2": 1055,
  "num_edges1": 248,
  "num_edges2": 596,
  "num_points1": 5171,
  "num_points2": 1604,
  "hausdorff_distance": 25.9206,
  "status": "success",
  "error": null
}
```

### CSV 输出格式（批量模式）

| sample_id | hausdorff_distance | num_nodes1 | num_nodes2 | num_edges1 | num_edges2 | num_points1 | num_points2 | status | error |
|-----------|-------------------|------------|------------|------------|------------|-------------|-------------|--------|-------|
| S1585-2_a | 25.9206 | 322 | 1055 | 248 | 596 | 5171 | 1604 | success | null |
| S163-2_a  | 18.4523 | 245 | 892 | 189 | 512 | 4321 | 1432 | success | null |

## 技术细节

### 点集提取

工具从拓扑图中提取：
1. **所有节点**：图的节点坐标 `(y, x)`
2. **所有边路径点**：边属性中的 `path` 或 `path-coordinates`
3. **去重**：自动移除重复点以提高效率

### 平均 Hausdorff 距离计算

```
d(A→B) = mean(min_distance(a, B) for a in A)
d(B→A) = mean(min_distance(b, A) for b in B)
avg_hausdorff(A, B) = (d(A→B) + d(B→A)) / 2
```

相比传统的最大 Hausdorff 距离：
- ✅ 对离群点更稳健
- ✅ 更能反映整体相似度
- ✅ 通常比最大距离小 30%-70%

## 故障排除

### 问题：找不到模块

```bash
# 确保在项目根目录运行
cd /path/to/ienf_q
uv run python tools/compare_topologies.py ...
```

### 问题：GraphML 格式失败

GraphML 格式对复杂边属性（如路径点列表）支持有限。**推荐使用 Pickle 或 JSON 格式**。

### 问题：批量模式找不到配对

确保文件名匹配（忽略扩展名）：
```
predictions/
  ├── sample1.pkl
  └── sample2.pkl

ground_truth/
  ├── sample1.json  # ✓ 会匹配
  └── sample3.pkl   # ✗ sample2 没有配对
```

### 问题：内存不足

对于非常大的拓扑（> 100,000 点），距离矩阵计算可能需要大量内存。考虑：
1. 分批处理
2. 使用更强大的机器
3. 实现采样策略（需要修改代码）

## 与 evaluate_dataset.py 的区别

| 特性 | compare_topologies.py | evaluate_dataset.py |
|------|----------------------|---------------------|
| **输入** | 拓扑文件 | 图像文件 |
| **依赖** | ❌ 无 Pipeline 依赖 | ✅ 需要 NeuralReconstructionPipeline |
| **用途** | 纯拓扑比对 | 完整评测（预处理 + 重建 + 评估） |
| **速度** | ⚡ 非常快 | 🐌 较慢（包含图像处理） |
| **灵活性** | ✅ 可用于任何拓扑图 | ❌ 仅用于特定数据格式 |

## 总结

`compare_topologies.py` 提供了一个**轻量、快速、独立**的拓扑比对解决方案，适合：

- ✅ 快速验证拓扑差异
- ✅ 比对不同算法生成的拓扑
- ✅ 评估参数调优效果
- ✅ 批量处理大量拓扑对
- ✅ 不需要重新运行图像处理 Pipeline

推荐使用 **Pickle** 或 **JSON** 格式存储拓扑，以获得最佳兼容性和性能。
