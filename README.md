# 医疗影像 AI 病情识别系统

基于深度学习的医学影像（X 光 / CT）辅助诊断系统方案与工程骨架。

> ⚠️ **免责声明**：本项目为科研与教学用途的算法参考实现，**不是医疗器械**，不得直接用于临床诊断。
> 任何临床使用都必须经过注册审批（如 NMPA / FDA）、多中心验证与医生复核。

## 项目简介

本项目面向**胸部 X 光（CXR）**与**CT**影像，提供从数据处理、模型训练、评估到推理部署的完整流程骨架，
并配套完整的方案设计文档，可作为团队立项与研发的起点。

典型识别任务：

| 任务 | 影像类型 | 输出 |
| --- | --- | --- |
| 肺炎检测 | 胸部 X 光 | 正常 / 肺炎（二分类） |
| 肺结节检测 | 胸部 CT | 结节位置 + 良恶性 |
| 多病种分类 | 胸部 X 光 | 14 类胸部异常多标签 |
| 病灶分割 | CT / X 光 | 像素级病灶掩膜 |

## 目录结构

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                 # 配置
│   └── config.yaml
├── docs/                    # 方案设计文档
│   ├── 01-项目概述.md
│   ├── 02-需求分析.md
│   ├── 03-技术方案.md
│   ├── 04-模型设计.md
│   ├── 05-数据方案.md
│   ├── 06-系统架构.md
│   ├── 07-部署运维.md
│   ├── 08-评估与合规.md
│   └── 09-项目计划.md
├── src/
│   ├── data/                # 数据读取与预处理
│   ├── models/              # 模型定义
│   ├── train.py             # 训练入口
│   ├── evaluate.py          # 评估入口
│   └── inference.py         # 推理入口
├── scripts/                 # 辅助脚本
└── tests/                   # 测试
```

## 快速开始

```bash
# 1. 创建环境
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 准备数据（将影像按目录或 CSV 清单组织，见 docs/05-数据方案.md）
#    data/
#      train/  val/  test/

# 3. 训练
python -m src.train --config configs/config.yaml

# 4. 评估
python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best.pt

# 5. 单张推理
python -m src.inference --checkpoint checkpoints/best.pt --image path/to/xray.png
```

## 文档索引

- [项目概述](docs/01-项目概述.md)
- [需求分析](docs/02-需求分析.md)
- [技术方案](docs/03-技术方案.md)
- [模型设计](docs/04-模型设计.md)
- [数据方案](docs/05-数据方案.md)
- [系统架构](docs/06-系统架构.md)
- [部署运维](docs/07-部署运维.md)
- [评估与合规](docs/08-评估与合规.md)
- [项目计划](docs/09-项目计划.md)

## License

MIT（仅供学习研究）。见 [LICENSE](LICENSE)。
