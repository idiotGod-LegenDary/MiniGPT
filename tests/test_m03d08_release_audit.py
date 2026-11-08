"""month 03 day 06：MiniGPT V1 发布证据工具自动测试。

先完成 tools/m03d08_release_audit.py 的 4 个 TODO，再运行：
    python -m pytest tests/test_m03d08_release_audit.py -q
"""
from __future__ import annotations

import copy
from pathlib import Path
import pytest

from tools.m03d08_release_audit import (
    REQUIRED_FILES,
    inspect_repository,
    parse_pytest_result,
    extract_experiment_evidence,
    render_release_audit,
    audit_project,
)


def make_repo(root: Path):
    for name in REQUIRED_FILES:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("示例文件", encoding="utf-8")


def sample_comparison():
    return {
        "config": {"seed": 42},
        "initial_max_abs_diff": 0.0,
        "full": {
            "baseline_val_loss": 2.835182,
            "final_val_loss": 2.638760,
            "history": [],
            "stats": {"trainable_params": 5104, "total_params": 5104},
        },
        "lora": {
            "baseline_val_loss": 2.835182,
            "final_val_loss": 2.830074,
            "history": [],
            "stats": {"trainable_params": 256, "total_params": 5104},
        },
        "adapter": {
            "base_unchanged": True,
            "adapter_updated": True,
            "adapter_file_bytes": 3012,
            "reload_max_abs_diff": 0.0,
            "merge_max_abs_diff": 4.76837158203125e-7,
        },
    }


def test_files_all_found(tmp_path):
    make_repo(tmp_path)
    got = inspect_repository(tmp_path)
    assert set(got) == {"root", "present", "missing", "complete"}
    assert got["root"] == str(tmp_path.resolve())
    assert got["present"] == list(REQUIRED_FILES)
    assert got["missing"] == [] and got["complete"] is True


def test_files_missing_and_directory_is_not_file(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "training/loop.py").unlink()
    (tmp_path / "training/loop.py").mkdir()
    (tmp_path / "model/attention.py").unlink()
    got = inspect_repository(tmp_path)
    assert got["complete"] is False
    assert got["missing"] == ["model/attention.py", "training/loop.py"]


def test_nonexistent_root_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_repository(tmp_path / "nonexistent")


@pytest.mark.parametrize("log,expected", [
    (".... [100%]\n172 passed in 889.20s (0:14:49)", (172, 0, 0, 889.2, True)),
    ("collected 19 items\n19 passed in 6.37s", (19, 0, 0, 6.37, True)),
    ("1 failed, 20 passed, 2 skipped in 3.12s", (20, 1, 2, 3.12, False)),
    ("12 passed, 1 skipped", (12, 0, 1, None, True)),
    ("1 failed in 0.50s", (0, 1, 0, 0.5, False)),
])
def test_pytest_log_parse(log, expected):
    got = parse_pytest_result(log)
    assert set(got) == {"passed", "failed", "skipped", "duration_seconds", "successful"}
    assert tuple(got[k] for k in ("passed", "failed", "skipped", "duration_seconds", "successful")) == expected


def test_last_pytest_summary_used():
    got = parse_pytest_result("2 failed, 4 passed in 3.0s\n......\n172 passed in 889.20s")
    assert got["passed"] == 172 and got["failed"] == 0
    assert got["successful"] is True


def test_bad_logs_rejected():
    with pytest.raises(TypeError):
        parse_pytest_result(None)
    with pytest.raises(ValueError):
        parse_pytest_result("collected 19 items\n[100%]")


def test_extract_real_values_and_reduction():
    report = sample_comparison()
    old = copy.deepcopy(report)
    got = extract_experiment_evidence(report)
    assert set(got) == {
        "full_baseline_val_loss", "full_final_val_loss",
        "lora_baseline_val_loss", "lora_final_val_loss",
        "full_trainable_params", "lora_trainable_params", "trainable_reduction_pct",
        "adapter_file_bytes", "initial_max_abs_diff", "reload_max_abs_diff",
        "merge_max_abs_diff", "base_unchanged", "adapter_updated", "numerical_checks_ok",
    }
    assert got["full_trainable_params"] == 5104
    assert got["lora_trainable_params"] == 256
    assert got["trainable_reduction_pct"] == pytest.approx((1 - 256/5104)*100)
    assert got["full_final_val_loss"] == pytest.approx(2.638760)
    assert got["lora_final_val_loss"] == pytest.approx(2.830074)
    assert got["numerical_checks_ok"] is True
    assert report == old


@pytest.mark.parametrize("path,value", [
    (("adapter", "base_unchanged"), False),
    (("adapter", "adapter_updated"), False),
    (("adapter", "reload_max_abs_diff"), 0.02),
    (("adapter", "merge_max_abs_diff"), 0.02),
    (("initial_max_abs_diff",), 0.02),
])
def test_extract_flags_actual_failures(path, value):
    data = sample_comparison()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert extract_experiment_evidence(data)["numerical_checks_ok"] is False


@pytest.mark.parametrize("path,value", [
    (("full", "final_val_loss"), float("nan")),
    (("lora", "stats", "trainable_params"), -2),
    (("adapter", "adapter_file_bytes"), True),
    (("adapter", "base_unchanged"), 1),
])
def test_extract_invalid_values_rejected(path, value):
    data = sample_comparison()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        extract_experiment_evidence(data)


def test_missing_experiment_field_rejected():
    data = sample_comparison()
    del data["lora"]["baseline_val_loss"]
    with pytest.raises(ValueError):
        extract_experiment_evidence(data)


def test_markdown_contains_actual_data():
    audit = {
        "course_id": "month 03 day 06", "git_id": "m03d08",
        "repository": {"complete": True, "missing": []},
        "tests": {"available": True, "passed": 172, "failed": 0, "skipped": 0, "successful": True},
        "experiment": {"available": True, **extract_experiment_evidence(sample_comparison())},
        "ready": True,
    }
    md = render_release_audit(audit)
    assert isinstance(md, str)
    for expected in ("MiniGPT V1", "m03d08", "172", "2.63876", "2.830074", "256", "合成数据", "不代表真实语言任务效果"):
        assert expected in md


def test_markdown_no_fake_success_on_missing_evidence():
    audit = {
        "course_id": "month 03 day 06", "git_id": "m03d08",
        "repository": {"complete": False, "missing": ["training/loop.py"]},
        "tests": {"available": False, "reason": "测试日志缺失"},
        "experiment": {"available": False, "reason": "实验报告缺失"},
        "ready": False,
    }
    md = render_release_audit(audit)
    for expected in ("training/loop.py", "测试日志缺失", "实验报告缺失", "未就绪", "合成数据", "不代表真实语言任务效果"):
        assert expected in md


def test_full_audit_marks_missing_sources(tmp_path):
    make_repo(tmp_path)
    got = audit_project(tmp_path, tmp_path/"missing.txt", tmp_path/"no.json")
    assert got["ready"] is False
    assert got["tests"]["available"] is False
    assert got["experiment"]["available"] is False
