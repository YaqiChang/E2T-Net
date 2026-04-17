# PLAN.md

## Scope
本轮只统一四处：

- `PLAN.md`
- `config.yml`
- `train.py`
- `scripts/experiments/run_jaad_pose_evidence_suite.sh`

不修改模型、数据集、loss 主体、forward、decoder、sampler 主逻辑。

## Config Source Of Truth
- 公共实验配置写入 `config.yml`
- `train.py` 从 `config.yml` 读取默认值
- CLI 显式参数覆盖 `config.yml`
- 实验脚本只传组间差异参数和运行时环境参数

## Shared Experiment Config
正式对比实验公共配置统一为：

- `data_dir=/media/meta/File/datasets/Intention/JAAD_dataset/PN_ego`
- `dataset=jaad`
- `out_dir=/media/meta/Data/CYQ/TIP/E2T-Net/output`
- `input=5`
- `output=5`
- `stride=5`
- `skip=1`
- `batch_size=256`
- `loader_workers=4`
- `pin_memory=True`
- `prefetch_factor=2`
- `loader_shuffle=True`
- `use_attribute=True`
- `use_image=False`
- `use_opticalflow=False`
- `auto_class_weights=False`
- `crossing_loss_type=ce`
- `label_smoothing=0.0`
- `n_epochs=30`
- `use_early_stopping=True`
- `early_stopping_patience=5`
- `early_stopping_min_delta=1e-4`

## Experiment Groups
- `MMDD_baseline`
- `MMDD_pose_direct_last`
- `MMDD_pose_accumulator`

组间差异只允许：

### baseline
- `--use_pose False`
- `--use_decision_accumulator False`

### pose_direct_last
- `--use_pose True`
- `--use_decision_accumulator False`
- `--belief_dim 64`
- `--belief_readout last`

### pose_accumulator
- `--use_pose True`
- `--use_decision_accumulator True`
- `--belief_dim 64`
- `--belief_readout last`

## Runtime Script Rules
- `PYTHON_BIN` 默认 `/home/meta/anaconda3/envs/3dhuman/bin/python`
- 只允许环境变量覆盖 `PYTHON_BIN` 和 `GPU_ID`
- `DATE_PREFIX="$(date +%m%d)"`
- 每组实验开始前打印实验名、开始时间、`CUDA_VISIBLE_DEVICES`、`PYTHON_BIN`
- 每组实验结束后打印实验名、结束时间
- 每组实验结束后释放显存再进入下一组

## Stop Rules
### Immediate stop
- `Arguments` 与预期不一致
- `input/output/stride/skip` 不是 `5/5/5/1`
- 组间开关与预期不一致
- 输出目录命名不一致
- 训练出现 `NaN` 或 `Inf`
- 直接报错退出

### Manual stop
- 长时间退化为单类预测
- `state_bal_acc` 或 `intent_bal_acc` 长期贴近 `0.5`
- `val_sc_pos` 或 `val_int_pos` 长期接近 `0.0` 或 `1.0`
- threshold 长期卡在极端值

### Cross experiment
- `baseline` 失败则不继续
- `pose_direct_last` 失败则不继续 `pose_accumulator`

