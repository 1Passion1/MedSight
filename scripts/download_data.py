"""下载示例公开数据集的脚本。

注意：需遵守各数据集的许可与使用条款。以下是占位说明，
请根据实际数据源替换 URL 与处理逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path("data")

DATASETS = {
    "kaggle_pneumonia": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
    "nih_chestxray14": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
    "luna16": "https://luna16.grand-challenge.org/",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in DATASETS:
        print("用法: python scripts/download_data.py <dataset>")
        print("可用数据集:")
        for name, url in DATASETS.items():
            print(f"  {name:>20}: {url}")
        sys.exit(1)

    name = sys.argv[1]
    print(f"数据集 '{name}' 需从其官方页面手动下载并同意许可：")
    print(f"  {DATASETS[name]}")
    print(f"下载后请放置到 {DATA_DIR}/ 下并按 docs/05-数据方案.md 组织目录。")


if __name__ == "__main__":
    main()
