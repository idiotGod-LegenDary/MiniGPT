"""month 03 day 06 / Git m03d08：MiniGPT V1 发布证据核验。

仅用于最终项目验收与证据汇总，不创建模型、不训练、不修改已有功能代码。

从项目根目录执行：
    python -m tools.m03d08_release_audit \
        --test-log runs/m03d08_pytest.txt \
        --experiment runs/day07/comparison.json

运行后生成【新文件】：
    runs/m03d08_release_audit.json
    docs/m03d08_release_audit.md

约定：缺失的测试记录或实验记录必须明确标记“不可用”，不得虚构结论。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "requirements.txt",
    "model/mini_gpt.py",
    "model/attention.py",
    "training/data.py",
    "training/loop.py",
    "training/checkpoint.py",
    "training/adapter_io.py",
    "training/lora_integration.py",
    "experiments/reporting.py",
    "experiments/run_toy.py",
    "experiments/compare_full_lora.py",
    "tests/test_full_lora_comparison.py",
    "tests/test_lora_integration.py",
    "demos/demo_full_lora_comparison.py",
)


def inspect_repository(root: str | Path) -> dict[str, Any]:
    """TODO 1：检查 MiniGPT V1 所需的文件是否已整合进当前工作区。

    输入：项目根目录，str 或 Path。
    输出：严格包含以下字段的字典：
      root: str       # resolve() 后的绝对路径
      present: list[str] # REQUIRED_FILES 中实际存在的文件，按原顺序
      missing: list[str] # REQUIRED_FILES 中缺失的文件，按原顺序
      complete: bool  # 是否一个都未缺失

    要求：
      - root 不存在或不是目录时抛 FileNotFoundError。
      - 用 Path.is_file() 判断，不能只判断路径存在（目录不是文件）。
      - 不创建或修改项目文件，不执行 pytest，也不连接网络。
      - 本检查仅证明文件存在，不证明 Git 分支已合并或代码逻辑正确。
    """
    # TODO 1：独立完成。
    if not isinstance(root, (str, Path)):
        raise TypeError("root type must be str or Path")

    p = Path(root).resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"root not exist or not a content: {p}")

    present: list[str] = []
    missing: list[str] = []
    for name in REQUIRED_FILES:
        target = p / name
        if target.is_file():
            present.append(name)
        else:
            missing.append(name)

    return {
        "root": str(p),
        "present": present,
        "missing": missing,
        "complete": not missing,
    }


def parse_pytest_result(log_text: str) -> dict[str, Any]:
    """TODO 2：解析最近一次 pytest -q 输出的汇总行。

    输入：终端输出文本，例如：
      '.... [100%]\\n172 passed in 889.20s (0:14:49)\\n'
      '.....\\n1 failed, 20 passed, 1 skipped in 3.12s\\n'
    输出：严格包含以下字段：
      passed: int
      failed: int
      skipped: int
      duration_seconds: float | None
      successful: bool  # passed > 0 且 failed == 0

    规则：
      - 非字符串抛 TypeError；找不到 pytest 汇总行抛 ValueError。
      - 同时存在多段运行日志时，以【最后一个】含 passed/failed 的汇总行为准。
      - 兼容只有 '172 passed'、含 skipped、含 failed 的结果。
      - 不得把日志中间的 'collected 19 items' 或 '[100%]' 当作测试数。
      - duration_seconds 提取 'in 3.12s'；无耗时则为 None。
    """
    # TODO 2：独立完成，可用 re.findall / re.search。
    if not isinstance(log_text,str):
        raise TypeError('log_text type must be str')

    summary_lines = [line for line in log_text.splitlines()
                    if re.search(r"\b\d+\s+(?:passed|failed|skipped)\b",line)]

    if not summary_lines:
        raise ValueError("找不到 pytest 汇总行")

    summary = summary_lines[-1]  # 以最后一段为准

    def count(keyword: str) -> int:
        m = re.search(rf"(\d+)\s+{keyword}", summary)
        return int(m.group(1)) if m else 0

    passed = count("passed")
    failed = count("failed")
    skipped = count("skipped")

    m = re.search(r"\bin\s+([\d.]+)s", summary)
    duration_seconds = float(m.group(1)) if m else None

    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration_seconds": duration_seconds,
        "successful": passed > 0 and failed == 0,
    }

def _finite_number(value: Any, field: str) -> float:
    """已提供：检查来自 JSON 的实数，禁止 bool/NaN/Inf。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是非 bool 的实数")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} 必须是有限实数")
    return result


def extract_experiment_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    """TODO 3：从 Day 07 真正运行生成的 comparison.json 提取发布证据。

    输入：day07 实验报告的 Python dict（已经由 json.load 读取）。
    输出：严格包含以下键的字典（数值必须来自 report 而非手写常数）：
      full_baseline_val_loss: float
      full_final_val_loss: float
      lora_baseline_val_loss: float
      lora_final_val_loss: float
      full_trainable_params: int
      lora_trainable_params: int
      trainable_reduction_pct: float  # (1 - lora/full) * 100
      adapter_file_bytes: int
      initial_max_abs_diff: float
      reload_max_abs_diff: float
      merge_max_abs_diff: float
      base_unchanged: bool
      adapter_updated: bool
      numerical_checks_ok: bool

    读取路径：report['full']['baseline_val_loss'] 等；训练参数量位于
             report['full']['stats']['trainable_params'] 和 LoRA 对应位置；
             其余参照 report['adapter']。
    验证：
      - 缺字段抛 ValueError，不要让不可解释的 KeyError 泄漏给用户。
      - Loss、差异必须是有限非负实数；参数量和文件字节数必须为正整数。
      - base_unchanged / adapter_updated 必须为真正的 bool。
      - numerical_checks_ok 当且仅当三个输出差异均 <= 1e-5、
        base_unchanged 和 adapter_updated 都为 True。
      - 仅反映报告中的数据，不额外声称真实语言任务质量或 GPU 性能。
    """
    # TODO 3：独立完成，复用 _finite_number。
    def get(d, key, path):
        if not isinstance(d, Mapping) or key not in d:
            raise ValueError(f"报告缺少字段: {path}")
        return d[key]

    full = get(report, "full", "full")
    lora = get(report, "lora", "lora")
    adapter = get(report, "adapter", "adapter")

    # ---- 四个 loss：有限非负实数 ----
    full_baseline = _finite_number(get(full, "baseline_val_loss", "full.baseline_val_loss"), "full.baseline_val_loss")
    full_final    = _finite_number(get(full, "final_val_loss",    "full.final_val_loss"),    "full.final_val_loss")
    lora_baseline = _finite_number(get(lora, "baseline_val_loss", "lora.baseline_val_loss"), "lora.baseline_val_loss")
    lora_final    = _finite_number(get(lora, "final_val_loss",    "lora.final_val_loss"),    "lora.final_val_loss")
    for name, v in [("full.baseline_val_loss", full_baseline),
                    ("full.final_val_loss", full_final),
                    ("lora.baseline_val_loss", lora_baseline),
                    ("lora.final_val_loss", lora_final)]:
        if v < 0:
            raise ValueError(f"{name} 必须非负")

    # ---- 参数量：正整数 ----
    def positive_int(value, field):
        # 严格按契约只接受真正的 int：256.0、True 这类"数值上像整数"的输入
        # 不是合法参数量，不能被静默转换。
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} 必须是正整数")
        return value

    full_stats = get(full, "stats", "full.stats")
    lora_stats = get(lora, "stats", "lora.stats")
    full_params = positive_int(get(full_stats, "trainable_params", "full.stats.trainable_params"),
                               "full.stats.trainable_params")
    lora_params = positive_int(get(lora_stats, "trainable_params", "lora.stats.trainable_params"),
                               "lora.stats.trainable_params")

    # ---- 下降百分比：由数据算，不写常数 ----
    trainable_reduction_pct = (1 - lora_params / full_params) * 100

    # ---- adapter 部分 ----
    adapter_file_bytes = positive_int(get(adapter, "adapter_file_bytes", "adapter.adapter_file_bytes"),
                                  "adapter.adapter_file_bytes")
    initial_max_abs_diff = _finite_number(get(report, "initial_max_abs_diff", "initial_max_abs_diff"),
                                      "initial_max_abs_diff")
    reload_max_abs_diff  = _finite_number(get(adapter, "reload_max_abs_diff",  "adapter.reload_max_abs_diff"),
                                          "adapter.reload_max_abs_diff")
    merge_max_abs_diff   = _finite_number(get(adapter, "merge_max_abs_diff",   "adapter.merge_max_abs_diff"),
                                          "adapter.merge_max_abs_diff")
    for name, v in [("initial_max_abs_diff", initial_max_abs_diff),
                    ("reload_max_abs_diff", reload_max_abs_diff),
                    ("merge_max_abs_diff", merge_max_abs_diff)]:
        if v < 0:
            raise ValueError(f"{name} 必须非负")

    # ---- 两个 bool：必须是真正的 bool ----
    base_unchanged = get(adapter, "base_unchanged", "adapter.base_unchanged")
    adapter_updated = get(adapter, "adapter_updated", "adapter.adapter_updated")
    if not isinstance(base_unchanged, bool):
        raise ValueError("base_unchanged 必须是 bool")
    if not isinstance(adapter_updated, bool):
        raise ValueError("adapter_updated 必须是 bool")

    # ---- 综合判定 ----
    numerical_checks_ok = (
        initial_max_abs_diff <= 1e-5
        and reload_max_abs_diff <= 1e-5
        and merge_max_abs_diff <= 1e-5
        and base_unchanged is True
        and adapter_updated is True
    )

    return {
        "full_baseline_val_loss": full_baseline,
        "full_final_val_loss": full_final,
        "lora_baseline_val_loss": lora_baseline,
        "lora_final_val_loss": lora_final,
        "full_trainable_params": full_params,
        "lora_trainable_params": lora_params,
        "trainable_reduction_pct": trainable_reduction_pct,
        "adapter_file_bytes": adapter_file_bytes,
        "initial_max_abs_diff": initial_max_abs_diff,
        "reload_max_abs_diff": reload_max_abs_diff,
        "merge_max_abs_diff": merge_max_abs_diff,
        "base_unchanged": base_unchanged,
        "adapter_updated": adapter_updated,
        "numerical_checks_ok": numerical_checks_ok,
    }


def render_release_audit(audit: Mapping[str, Any]) -> str:
    """TODO 4：把审计结果转换成最终 Markdown 验收小结。

    输入：audit_project() 返回的结构：
      {'course_id', 'git_id', 'repository', 'tests', 'experiment', 'ready'}。
    输出：Markdown 字符串，必须包含：
      - 标题含 'MiniGPT V1'，并标明 git 标识 m03d08；
      - 文件是否齐全，若缺失则列出全部 missing；
      - 测试可用时写通过/失败数量，不可用时明确写 '测试日志缺失'；
      - 实验可用时写两路最终 val_loss、LoRA 训练参数量、恢复差异、
        Merge 差异；不可用时明确写 '实验报告缺失'；
      - 写明发布是否就绪；
      - 必须包含 '合成数据' 和 '不代表真实语言任务效果' 的局限说明。

    注意：不要把 unavailable 解释为测试通过；不要创建不存在的数字。
    """
    # TODO 4：独立完成，建议用 lines.append() 与 '\\n'.join(lines)。
    lines: list[str] = []

    git_id = audit.get("git_id", "m03d08")

    # 1. 标题：含 MiniGPT V1，标明 git 标识
    lines.append(f"# MiniGPT V1 发布验收小结（{git_id}）")
    lines.append("")

    # 2. 文件是否齐全
    repo = audit.get("repository", {})
    if repo.get("complete"):
        lines.append("## 文件检查")
        lines.append("- 必需文件齐全。")
    else:
        lines.append("## 文件检查")
        missing = repo.get("missing", [])
        lines.append(f"- 文件缺失（共 {len(missing)} 个）：")
        for name in missing:
            lines.append(f"  - `{name}`")
    lines.append("")

    # 3. 测试
    lines.append("## 测试结果")
    tests = audit.get("tests", {})
    if tests.get("available"):
        lines.append(f"- 通过：{tests.get('passed', 0)}")
        lines.append(f"- 失败：{tests.get('failed', 0)}")
        lines.append(f"- 跳过：{tests.get('skipped', 0)}")
        if tests.get("duration_seconds") is not None:
            lines.append(f"- 耗时：{tests['duration_seconds']} 秒")
        lines.append(f"- 是否全部成功：{'是' if tests.get('successful') else '否'}")
    else:
        # 注意：不可用时明确写"测试日志缺失"，不能写成"测试通过"
        reason = tests.get("reason", "测试日志缺失")
        lines.append(f"- 测试日志缺失（原因：{reason}）")
    lines.append("")

    # 4. 实验
    lines.append("## 实验结果")
    exp = audit.get("experiment", {})
    if exp.get("available"):
        lines.append(f"- Full 模型最终 val_loss：{exp.get('full_final_val_loss')}")
        lines.append(f"- LoRA 模型最终 val_loss：{exp.get('lora_final_val_loss')}")
        lines.append(f"- LoRA 可训练参数量：{exp.get('lora_trainable_params')}")
        lines.append(f"- 参数量下降比例：{exp.get('trainable_reduction_pct'):.2f}%")
        lines.append(f"- 重新加载后最大差异：{exp.get('reload_max_abs_diff')}")
        lines.append(f"- Merge 后最大差异：{exp.get('merge_max_abs_diff')}")
        lines.append(f"- 数值检查通过：{'是' if exp.get('numerical_checks_ok') else '否'}")
    else:
        reason = exp.get("reason", "实验报告缺失")
        lines.append(f"- 实验报告缺失（原因：{reason}）")
    lines.append("")

    # 5. 是否就绪
    lines.append("## 发布就绪")
    lines.append(f"- {'已就绪' if audit.get('ready') else '未就绪'}")
    lines.append("")

    # 6. 局限说明（题目要求必须出现这两个短语）
    lines.append("## 局限说明")
    lines.append("- 本实验基于合成数据，不代表真实语言任务效果。")

    return "\n".join(lines)


def audit_project(root: Path, test_log: Path, experiment: Path) -> dict[str, Any]:
    """已提供：串联四个 TODO，并明确标记缺失证据。"""
    repository = inspect_repository(root)
    if test_log.is_file():
        try:
            tests = {"available": True, **parse_pytest_result(test_log.read_text(encoding="utf-8"))}
        except (ValueError, TypeError, UnicodeError) as exc:
            tests = {"available": False, "reason": f"测试日志无法解析：{exc}"}
    else:
        tests = {"available": False, "reason": "测试日志缺失"}

    if experiment.is_file():
        try:
            raw_report = json.loads(experiment.read_text(encoding="utf-8"))
            evidence = {"available": True, **extract_experiment_evidence(raw_report)}
        except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
            evidence = {"available": False, "reason": f"实验报告无法解析：{exc}"}
    else:
        evidence = {"available": False, "reason": "实验报告缺失"}

    ready = bool(
        repository["complete"]
        and tests.get("available")
        and tests.get("successful")
        and evidence.get("available")
        and evidence.get("numerical_checks_ok")
    )
    return {
        "course_id": "month 03 day 06",
        "git_id": "m03d08",
        "repository": repository,
        "tests": tests,
        "experiment": evidence,
        "ready": ready,
    }


def main() -> None:
    """已提供：命令行入口，不要求修改。"""
    parser = argparse.ArgumentParser(description="MiniGPT V1 / m03d08 最终验收")
    parser.add_argument("--root", default=".", help="MiniGPT 项目根目录")
    parser.add_argument("--test-log", default="runs/m03d08_pytest.txt", help="本次全项目 pytest 日志")
    parser.add_argument("--experiment", default="runs/day07/comparison.json", help="实际实验报告")
    parser.add_argument("--output-json", default="runs/m03d08_release_audit.json", help="新生成的 JSON 审计报告")
    parser.add_argument("--output-md", default="docs/m03d08_release_audit.md", help="新生成的 Markdown 审计报告")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    def locate(value: str) -> Path:
        path = Path(value)
        return (root / path).resolve() if not path.is_absolute() else path

    result = audit_project(root, locate(args.test_log), locate(args.experiment))
    target_json = locate(args.output_json)
    target_md = locate(args.output_md)
    target_json.parent.mkdir(parents=True, exist_ok=True)
    target_md.parent.mkdir(parents=True, exist_ok=True)
    target_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    target_md.write_text(render_release_audit(result), encoding="utf-8")
    print("审计 JSON：", target_json)
    print("审计 Markdown：", target_md)
    print("发布证据就绪：", result["ready"])
    if not result["ready"]:
        print("注意：证据不齐时不得在 README 中写已完成正式发布。")


if __name__ == "__main__":
    main()
