"""比较两个已经写入磁盘的 MiniGPT JSON 实验报告。"""
import argparse
import json
from pathlib import Path

from experiments.reporting import compare_reports


def main() -> None:
    parser = argparse.ArgumentParser(description='比较两次同配置实验是否复现')
    parser.add_argument('first', help='第一次实验 JSON 路径')
    parser.add_argument('second', help='第二次实验 JSON 路径')
    parser.add_argument('--atol', type=float, default=1e-8, help='绝对容差')
    args = parser.parse_args()
    first = json.loads(Path(args.first).read_text(encoding='utf-8'))
    second = json.loads(Path(args.second).read_text(encoding='utf-8'))
    report = compare_reports(first, second, atol=args.atol)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
