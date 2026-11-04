"""Week 04 Day 05：实验复现工具的功能契约测试（不提供 TODO 答案）。"""
import copy
import json
import math
from pathlib import Path

import pytest

from experiments.reporting import summarize_history, save_json_report, compare_reports
from experiments.run_toy import ExperimentConfig, _validate_config, run_experiment


def _history():
    return [
        {'step': 1, 'train_loss': 3.1},
        {'step': 2, 'train_loss': 2.6, 'val_loss': 2.9},
        {'step': 3, 'train_loss': 2.4},
        {'step': 4, 'train_loss': 2.2, 'val_loss': 2.8},
    ]


def _report():
    return {
        'config': {'seed': 42, 'num_steps': 4},
        'history': _history(),
        'baseline_val_loss': 3.5,
        'final_val_loss': 2.8,
    }


def test_summary_valid_values_and_purity():
    history = _history()
    before = copy.deepcopy(history)
    result = summarize_history(history)
    assert result == {
        'num_steps': 4,
        'first_train_loss': 3.1,
        'last_train_loss': 2.2,
        'best_val_loss': 2.8,
        'best_val_step': 4,
        'last_val_loss': 2.8,
    }
    assert history == before


def test_summary_no_validation_and_best_tie():
    row = [{'step': 1, 'train_loss': 3.0}, {'step': 2, 'train_loss': 2.0}]
    got = summarize_history(row)
    assert got['best_val_loss'] is None and got['best_val_step'] is None
    assert got['last_val_loss'] is None
    x = _history()
    x[1]['val_loss'] = 2.8
    assert summarize_history(x)['best_val_step'] == 2


@pytest.mark.parametrize('history', [
    [], [{'step': 2, 'train_loss': 2.0}],
    [{'step': 1, 'train_loss': float('nan')}],
    [{'step': 1, 'train_loss': -1.0}],
    [{'step': True, 'train_loss': 1.0}],
    [{'step': 1, 'train_loss': 2.0}, {'step': 1, 'train_loss': 1.0}],
    [{'step': 1, 'train_loss': 2.0, 'val_loss': float('inf')}],
])
def test_summary_rejects_bad_records(history):
    with pytest.raises((ValueError, TypeError, KeyError)):
        summarize_history(history)


def test_save_report_utf8_roundtrip_and_overwrite(tmp_path):
    record = _report()
    record['message'] = '中文实验报告'
    out = tmp_path / '报告' / 'run.json'
    path = save_json_report(out, record)
    assert isinstance(path, Path) and path == out.resolve()
    assert json.loads(path.read_text(encoding='utf-8')) == record
    assert '中文实验报告' in path.read_text(encoding='utf-8')
    save_json_report(out, {'second': True})
    assert json.loads(out.read_text(encoding='utf-8')) == {'second': True}
    assert list(out.parent.glob('*.tmp')) == []


def test_save_report_rejects_nan_and_bad_type(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        save_json_report(tmp_path / 'bad.json', {'nan': math.nan})
    with pytest.raises(TypeError):
        save_json_report(tmp_path / 'bad.json', ['not a dict'])


def test_compare_identical_and_changed_reports():
    first = _report()
    second = copy.deepcopy(first)
    assert compare_reports(first, second)['reproduced'] is True
    assert compare_reports(first, second)['max_train_loss_diff'] == 0
    second['history'][2]['train_loss'] += 0.1
    second['final_val_loss'] += 0.2
    out = compare_reports(first, second)
    assert out['reproduced'] is False
    assert out['max_train_loss_diff'] == pytest.approx(0.1)
    assert out['final_diff'] == pytest.approx(0.2)


def test_compare_rejects_mismatched_config_and_steps():
    first = _report()
    second = copy.deepcopy(first)
    second['config']['seed'] = 7
    with pytest.raises(ValueError):
        compare_reports(first, second)
    second = copy.deepcopy(first)
    second['history'][2]['step'] = 99
    with pytest.raises(ValueError):
        compare_reports(first, second)
    second = copy.deepcopy(first)
    del second['history'][1]['val_loss']
    with pytest.raises(ValueError):
        compare_reports(first, second)


def test_compare_rejects_invalid_tolerance():
    with pytest.raises(ValueError):
        compare_reports(_report(), _report(), atol=-1)
    with pytest.raises(ValueError):
        compare_reports(_report(), _report(), atol=float('nan'))
    with pytest.raises(ValueError):
        compare_reports(_report(), _report(), atol=True)


@pytest.mark.parametrize('fields', [
    {'num_steps': 0}, {'block_size': 8, 'train_tokens': 8},
    {'val_tokens': 2}, {'learning_rate': -0.1},
    {'vocab_size': 1}, {'dropout': 1.0},
])
def test_validate_bad_configs(fields):
    from dataclasses import replace
    with pytest.raises(ValueError):
        _validate_config(replace(ExperimentConfig(), **fields))


def test_real_minigpt_experiment_and_reproducibility_when_available():
    pytest.importorskip('model.mini_gpt')
    from dataclasses import replace
    config = replace(
        ExperimentConfig(),
        seed=123, num_steps=3, eval_every=1,
        train_tokens=48, val_tokens=32, block_size=4,
        batch_size=3, d_model=16, d_ff=32,
    )
    a = run_experiment(config)
    b = run_experiment(config)
    assert set(a) == {
        'config', 'environment', 'baseline_val_loss', 'final_val_loss',
        'history', 'summary', 'data_note',
    }
    assert a['config']['seed'] == 123
    assert a['environment']['device'] == 'cpu'
    assert len(a['history']) == 3
    assert a['summary']['num_steps'] == 3
    assert compare_reports(a, b)['reproduced'] is True
    assert '合成' in a['data_note']
