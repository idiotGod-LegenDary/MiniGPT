# Week 04 · Day 05：MiniGPT 可复现实验与项目整理

## 一、这次新增什么

只新增以下文件，直接合并到现有 MiniGPT 根目录即可：

- `experiments/__init__.py`：新包标记，无须修改。
- `experiments/reporting.py`：TODO 1~3，训练历史汇总、JSON 保存、复现对照。
- `experiments/run_toy.py`：TODO 4，真实调用 MiniGPT + Day01/02 数据与训练模块。
- `experiments/compare_saved.py`：已提供，两个 JSON 报告的一键复现对比。
- `tests/test_experiment_day05.py`：自动验收测试。
- `docs/week04_day05.md`：本说明。

**ZIP 完全不含** `training/data.py`、`training/loop.py`、
`training/checkpoint.py`、`training/adapter_io.py` 或 `README.md`，
不会覆盖前四日成果。

## 二、运行命令（Windows CMD）

在项目根目录运行：

```bat
python -m pytest tests/test_experiment_day05.py -q
python -m experiments.run_toy --seed 42 --steps 12 --output runs/day05_a.json
python -m experiments.run_toy --seed 42 --steps 12 --output runs/day05_b.json
python -m experiments.compare_saved runs/day05_a.json runs/day05_b.json
python -m pytest tests/ -q
```

两次运行必须使用相同配置和相同环境。生成的报告仅针对**合成数据上的
工程复现**，不能解读为真实 SFT 或通用语言性能验证。

## 三、必须实现的 TODO

1. `summarize_history`：严格验证 History，返回 6 个统计字段。
2. `save_json_report`：UTF-8 JSON + 同目录临时文件 + 原子替换。
3. `compare_reports`：严格匹配配置与步数；量化两次结果差异。
4. `run_experiment`：CPU 上构建原始 Token、先切分数据、构造
   MiniGPT、Baseline、训练、验证、汇总、报告；不可造假数据。

每个 TODO 的输入、输出、异常与边界均写在函数中文 docstring 中。

## 四、建议实验报告（自己填写真实运行值）

| 项目 | 实际结果 |
|---|---|
| Python / PyTorch 版本 | 待填写 |
| Git Commit ID | 待填写 |
| 模型配置 | 待填写 |
| Token 构造规则 | 周期性合成 Token，不含自然语言 |
| Baseline 验证 Loss | 待运行 |
| 最终验证 Loss | 待运行 |
| 训练首末 Loss | 待运行 |
| 两次同配置复现差异 | 待运行 |
| 全项目 pytest | 待运行 |

## 五、README 更新指引（不要直接覆盖原 README）

手动在原有 `README.md` 末尾增加“训练与实验复现”章节：

- 从 `model` 到 `training/data.py`、`training/loop.py` 的真实调用链。
- 上述一条训练命令 + 一条 pytest 命令。
- 一个指向 `docs/week04_day05.md` 的说明链接。
- 用实际测得的数字说明实验结果；可贴 JSON 文件片段。
- 明确声明当前只是 CPU 玩具规模实验，Adapter IO 尚未注入全部 MiniGPT 层；
  不宣称完成真实语料预训练、SFT、分布式训练或部署优化。

## 六、复现边界

使用固定随机种子只能控制一部分随机性，PyTorch 不承诺跨
硬件、版本、平台严格一致。本日复现限定同一运行环境中的
CPU 小模型、固定 Token 生成及数据采样逻辑。

## 七、提交

```bat
git status
git add experiments/ tests/test_experiment_day05.py docs/week04_day05.md
git commit -m "feat: add reproducible MiniGPT toy experiment"
git status
```

手动修改 README 时将其单独 `git add README.md`，不要让 ZIP 覆盖原文。
