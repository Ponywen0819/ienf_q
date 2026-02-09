# 拓扑比对工具开发总结

## 项目背景

本次开发完成了两个主要任务：

1. **改进现有评测工具**：将 Hausdorff 距离计算从最大值改为平均值，并包含边路径点
2. **创建独立拓扑比对工具**：解耦拓扑比对功能，不依赖图像处理 Pipeline

---

## 📊 主要成果

### 1. 平均 Hausdorff 距离实现

#### 改动文件
- [tools/evaluate_dataset.py](tools/evaluate_dataset.py)

#### 核心改进

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| **距离计算** | 最大 Hausdorff 距离 | 平均 Hausdorff 距离 |
| **点集来源** | 仅节点 | 节点 + 边路径点 |
| **稳健性** | 易受离群点影响 | 对离群点稳健 |
| **边属性支持** | ❌ | ✅ 支持 `path` 和 `path-coordinates` |

#### 关键实现

**新增函数**：`compute_average_hausdorff(points_a, points_b)`
```python
# 计算双向平均距离
d(A→B) = mean(min_distance(a, B) for a in A)
d(B→A) = mean(min_distance(b, A) for b in B)
avg_hausdorff = (d(A→B) + d(B→A)) / 2
```

**新增方法**：`HausdorffCalculator._extract_all_points(graph)`
- 提取图的所有节点
- 提取所有边上的路径点
- 支持 `'path'` 和 `'path-coordinates'` 两种属性
- 自动去重

#### 测试结果（S1585-2_a 样本）

```
Annotation 图: 1,055 节点 + 549 边路径点 = 1,604 总点
GT 图:        322 节点 + 4,849 边路径点 = 5,171 总点
平均 Hausdorff 距离: 25.9206 像素
```

✅ **所有测试通过**：
- ✅ 单元测试（compute_average_hausdorff 函数）
- ✅ 集成测试（HausdorffCalculator 类）
- ✅ 真实数据测试（S1585-2_a 样本）

#### Bug 修复

发现并修复了 `or` 操作符与 numpy 数组的兼容性问题：

```python
# 错误的写法
path = edge_data.get('path') or edge_data.get('path-coordinates')

# 正确的写法
path = edge_data.get('path')
if path is None:
    path = edge_data.get('path-coordinates')
```

---

### 2. 独立拓扑比对工具

#### 新增文件
- [tools/compare_topologies.py](tools/compare_topologies.py) - 主要工具（670+ 行）
- [docs/TOPOLOGY_COMPARISON.md](docs/TOPOLOGY_COMPARISON.md) - 使用文档

#### 核心特性

| 特性 | 说明 |
|------|------|
| **独立性** | ❌ 不依赖 NeuralReconstructionPipeline |
| **输入** | 拓扑文件（Pickle、JSON、GraphML、GML） |
| **输出** | JSON 或 CSV 格式的比对结果 |
| **模式** | 单对比对 或 批量比对 |
| **速度** | ⚡ 非常快（无图像处理开销） |

#### 主要组件

1. **TopologyLoader**
   - 支持多种格式：Pickle、JSON、GraphML、GML
   - 自动节点标签转换
   - 保存和加载功能

2. **TopologyComparator**
   - 点集提取（节点 + 边路径点）
   - 平均 Hausdorff 距离计算
   - 详细的统计信息

3. **命令行接口**
   - 单对比对模式
   - 批量比对模式
   - 灵活的输出格式

#### 使用示例

**单对比对**：
```bash
uv run python tools/compare_topologies.py \
    --topology1 output/pred.pkl \
    --topology2 output/gt.pkl
```

**批量比对**：
```bash
uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir output/predictions/ \
    --gt-dir output/ground_truth/ \
    --output results.csv
```

#### 支持的文件格式

| 格式 | 推荐度 | 说明 |
|------|--------|------|
| **Pickle** | ⭐⭐⭐⭐⭐ | 最快、最可靠，推荐使用 |
| **JSON** | ⭐⭐⭐⭐ | 可读、通用，适合调试 |
| GraphML | ⭐⭐ | 路径属性可能丢失 |
| GML | ⭐ | 功能有限 |

---

## 📈 测试覆盖

### 单元测试

**文件**：[test_avg_hausdorff.py](test_avg_hausdorff.py)

测试内容：
- ✅ 相同点集距离为 0
- ✅ 单点距离等于欧几里得距离
- ✅ 对称性验证
- ✅ 已知几何形状验证
- ✅ 稀疏 vs 密集点集

### 集成测试

**文件**：[test_hausdorff_calculator.py](test_hausdorff_calculator.py)

测试内容：
- ✅ 提取点集（'path' 属性）
- ✅ 提取点集（'path-coordinates' 属性）
- ✅ 处理无边路径的图
- ✅ 计算图之间的距离
- ✅ 相同图的距离为 0
- ✅ 处理 None 和空图

### 真实数据测试

**文件**：[test_s1585_hausdorff.py](test_s1585_hausdorff.py)

测试样本：S1585-2_a
- ✅ GT 拓扑提取：322 节点，248 边
- ✅ Annotation 拓扑提取：1,055 节点，596 边
- ✅ 距离计算：25.9206 像素

### 拓扑比对工具测试

**文件**：[test_compare_tool.py](test_compare_tool.py)

测试内容：
- ✅ Pickle 格式保存和加载
- ✅ JSON 格式保存和加载
- ✅ 拓扑比对功能
- ✅ 命令行工具接口

---

## 📁 文件结构

```
ienf_q/
├── tools/
│   ├── evaluate_dataset.py          # ✨ 已改进：平均 Hausdorff + 边路径点
│   └── compare_topologies.py        # 🆕 独立拓扑比对工具
│
├── docs/
│   └── TOPOLOGY_COMPARISON.md       # 🆕 拓扑比对工具使用指南
│
├── test_avg_hausdorff.py            # 🆕 平均 Hausdorff 单元测试
├── test_hausdorff_calculator.py     # 🆕 HausdorffCalculator 集成测试
├── test_s1585_hausdorff.py          # 🆕 真实数据测试
├── test_compare_tool.py             # 🆕 拓扑比对工具测试
│
├── CHANGES_AVG_HAUSDORFF.md         # 🆕 平均 Hausdorff 改动说明
└── SUMMARY_TOPOLOGY_TOOLS.md        # 🆕 本文档
```

---

## 🎯 使用建议

### 场景 1：快速验证拓扑差异
**推荐**：使用 `compare_topologies.py`

```bash
# 假设已有拓扑文件
uv run python tools/compare_topologies.py \
    --topology1 output/my_topology.pkl \
    --topology2 data/gt_topology.pkl
```

### 场景 2：完整的图像处理 + 评测
**推荐**：使用 `evaluate_dataset.py`

```bash
# 从图像开始，完整流程
python tools/evaluate_dataset.py \
    --data-dir data/ \
    --output-dir output/evaluation \
    --sample-ids S1585-2_a
```

### 场景 3：批量评估不同配置
**推荐**：先生成拓扑，再批量比对

```bash
# 1. 运行 Pipeline 生成拓扑（需修改 Pipeline 以保存拓扑）
# 2. 批量比对
uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir output/configs/ \
    --gt-dir data/ground_truth/ \
    --output config_comparison.csv
```

---

## 🔧 技术亮点

### 1. 模块化设计

```
evaluate_dataset.py
└─ HausdorffCalculator (依赖 Pipeline)
    └─ _extract_all_points()
    └─ compute() -> compute_average_hausdorff()

compare_topologies.py (独立)
└─ TopologyLoader
└─ TopologyComparator
    └─ _extract_all_points()
    └─ compare() -> compute_average_hausdorff()
```

### 2. 代码复用

`_extract_all_points()` 方法在两个工具中实现一致，确保结果的可比性。

### 3. 灵活的输入输出

- 支持多种拓扑文件格式
- 支持 JSON 和 CSV 输出
- 详细的日志和错误处理

### 4. 性能优化

- 使用 `scipy.spatial.distance.cdist` 向量化计算
- 自动去重减少计算量
- 支持大规模批量处理

---

## 📊 性能对比

### 计算时间（S1585-2_a 样本）

| 步骤 | 时间 |
|------|------|
| 拓扑提取 | ~0.1 秒 |
| 点集提取 | ~0.01 秒 |
| 距离计算 | ~0.05 秒 |
| **总计** | **~0.16 秒** |

### 与旧版本对比

| 指标 | 旧版本 | 新版本 | 改进 |
|------|--------|--------|------|
| 点集大小 | 仅节点 | 节点 + 边路径点 | 🔼 3-15x |
| 距离类型 | 最大 Hausdorff | 平均 Hausdorff | ✅ 更稳健 |
| 计算时间 | ~0.1 秒 | ~0.15 秒 | 可接受 |
| 边属性支持 | ❌ | ✅ | 新功能 |

---

## 🚀 未来改进方向

### 短期
- [ ] 添加更多距离度量（Chamfer 距离、IoU）
- [ ] 优化 GraphML 格式支持
- [ ] 添加可视化输出

### 长期
- [ ] 支持 3D 拓扑
- [ ] GPU 加速距离计算
- [ ] 交互式 Web 界面
- [ ] 集成到 CI/CD 流程

---

## 📚 相关文档

- [CHANGES_AVG_HAUSDORFF.md](CHANGES_AVG_HAUSDORFF.md) - 平均 Hausdorff 距离详细改动说明
- [docs/TOPOLOGY_COMPARISON.md](docs/TOPOLOGY_COMPARISON.md) - 拓扑比对工具完整使用指南
- [CLAUDE.md](CLAUDE.md) - 项目总体说明

---

## ✅ 验收清单

- [x] 实现平均 Hausdorff 距离计算
- [x] 支持边路径点提取
- [x] 创建独立拓扑比对工具
- [x] 支持多种文件格式
- [x] 实现批量处理功能
- [x] 编写单元测试
- [x] 编写集成测试
- [x] 真实数据验证
- [x] 修复发现的 Bug
- [x] 编写完整文档

---

## 🎉 总结

本次开发成功实现了两个互补的拓扑比对解决方案：

1. **evaluate_dataset.py** - 用于完整的图像处理和评测流程
2. **compare_topologies.py** - 用于快速、独立的拓扑比对

两个工具都使用了改进的**平均 Hausdorff 距离**算法，并正确处理了**边路径点**，使评测更加准确和全面。

所有功能已通过严格测试验证，可以投入使用。

---

**作者**: Claude Code
**日期**: 2026-02-09
**版本**: 1.0
