# 拓扑比对工具 - 快速参考

## 🚀 快速开始

### 提取数据集 GT 拓扑

```bash
# 从 data/ 文件夹批量提取所有 label.png 的拓扑
uv run python tools/extract_dataset_topologies.py
```

### 单个拓扑对比对

```bash
uv run python tools/compare_topologies.py \
    --topology1 topology1.pkl \
    --topology2 topology2.pkl
```

### 批量比对

```bash
uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir predictions/ \
    --gt-dir ground_truth/ \
    --output results.csv
```

---

## 📦 数据集拓扑提取

### 批量提取 GT 拓扑

```bash
# 默认: data/ -> output/topologies/
uv run python tools/extract_dataset_topologies.py

# 自定义路径
uv run python tools/extract_dataset_topologies.py \
    --data-dir data \
    --output-dir output/gt_topologies

# 详细模式
uv run python tools/extract_dataset_topologies.py --verbose
```

### 输出示例

```
提取摘要
总样本数: 17
成功: 16 (94.1%)
平均节点数: 357.8
平均边数: 291.6
输出目录: output/topologies
```

### 数据集结构要求

```
data/
├── S1585-2_a/
│   ├── image.png
│   ├── mask.png
│   ├── annotation.png
│   └── label.png          # 必须存在
└── ...
```

---

## 📁 文件格式

### 推荐格式

| 格式 | 命令 | 说明 |
|------|------|------|
| **Pickle** | `topology.pkl` | ⚡ 最快、最可靠 |
| **JSON** | `topology.json` | 📖 可读、易调试 |

### 从代码中保存

```python
from tools.compare_topologies import TopologyLoader

loader = TopologyLoader()

# Pickle 格式（推荐）
loader.save(graph, Path("output/topology.pkl"), 'pickle')

# JSON 格式（可读）
loader.save(graph, Path("output/topology.json"), 'json')
```

---

## 🔧 常用命令

### 保存结果

```bash
# JSON 格式
--output result.json

# CSV 格式（批量模式）
--output results.csv
```

### 详细日志

```bash
--verbose
```

### 完整示例

```bash
uv run python tools/compare_topologies.py \
    --topology1 output/pred.pkl \
    --topology2 data/gt.pkl \
    --output result.json \
    --verbose
```

---

## 📊 输出示例

### 单对比对

```
平均 Hausdorff 距離: 25.9206 像素

拓樸 1: topology1
  節點數: 322
  邊數: 248
  總點數: 5171

拓樸 2: topology2
  節點數: 1055
  邊數: 596
  總點數: 1604
```

### 批量比对

```
批次比對統計
總共比對: 150 對
成功: 148

平均 Hausdorff 距離統計:
  平均值: 23.4567
  中位數: 21.2345
  標準差: 8.9012
```

---

## 🛠️ 测试工具

### 单元测试

```bash
uv run python test_avg_hausdorff.py
```

### 集成测试

```bash
uv run python test_hausdorff_calculator.py
```

### 真实数据测试

```bash
uv run python test_s1585_hausdorff.py
```

### 拓扑比对工具测试

```bash
uv run python test_compare_tool.py
```

---

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| [DATASET_TOPOLOGY_EXTRACTION.md](docs/DATASET_TOPOLOGY_EXTRACTION.md) | 🔄 数据集拓扑提取指南 |
| [TOPOLOGY_COMPARISON.md](docs/TOPOLOGY_COMPARISON.md) | 📖 拓扑比对完整指南 |
| [CHANGES_AVG_HAUSDORFF.md](CHANGES_AVG_HAUSDORFF.md) | 🔧 技术改动说明 |
| [SUMMARY_TOPOLOGY_TOOLS.md](SUMMARY_TOPOLOGY_TOOLS.md) | 📊 开发总结 |

---

## ⚡ 性能提示

- ✅ 使用 **Pickle** 格式最快
- ✅ 批量处理节省时间
- ✅ 拓扑文件远小于图像文件
- ⚠️ 大拓扑（>100k 点）可能需要较多内存

---

## 🐛 故障排除

### 找不到模块
```bash
# 确保在项目根目录
cd /path/to/ienf_q
```

### GraphML 格式问题
```bash
# 改用 Pickle 或 JSON
loader.save(graph, path, 'pickle')  # 推荐
```

### 批量模式找不到配对
```bash
# 确保文件名匹配（忽略扩展名）
predictions/sample1.pkl  ←→  ground_truth/sample1.json  ✅
predictions/sample2.pkl  ←→  ground_truth/sample3.pkl  ❌
```

---

## 💡 使用技巧

### 1. 快速验证
```bash
# 直接比对拓扑文件，跳过图像处理
uv run python tools/compare_topologies.py \
    --topology1 pred.pkl \
    --topology2 gt.pkl
```

### 2. 参数调优
```bash
# 保存不同配置的拓扑，批量比对
uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir output/configs/ \
    --gt-dir data/gt/ \
    --output config_eval.csv
```

### 3. 调试分析
```bash
# 使用 JSON 格式便于查看
loader.save(graph, path, 'json')
cat topology.json | grep -A 5 "nodes"
```

---

## 🔄 完整工作流程

### 评测流程示例

```bash
# 步骤 1: 提取所有 GT 拓扑
uv run python tools/extract_dataset_topologies.py \
    --output-dir output/gt_topologies

# 步骤 2: 运行 Pipeline 生成预测拓扑
# (需要修改 Pipeline 保存拓扑)

# 步骤 3: 批量比对
uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir output/predictions \
    --gt-dir output/gt_topologies \
    --output evaluation_results.csv

# 步骤 4: 查看结果
cat evaluation_results.csv
```

### 单样本测试流程

```bash
# 步骤 1: 提取 GT 拓扑
uv run python tools/extract_dataset_topologies.py

# 步骤 2: 比对特定样本
uv run python tools/compare_topologies.py \
    --topology1 output/predictions/S1585-2_a_pred.pkl \
    --topology2 output/topologies/S1585-2_a_gt.pkl \
    --output result.json \
    --verbose
```

---

**版本**: 1.1
**更新**: 2026-02-09
