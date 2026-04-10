# 基本命令与指标
`ade`: 预测未来框中心轨迹的平均位移误差，越低越好  
`fde`: 最后一帧位移误差，越低越好  
`aiou`: 整段未来框的平均 IoU，越高越好  
`fiou`: 最后一帧 IoU，越高越好  
`state_acc`: step-level crossing 分类准确率，也就是未来每一步“过/不过”的准确率  
`intention_acc`: sequence-level intention 准确率，也就是整段里是否会发生 crossing 的准确率  
`f1_state`: step-level crossing 的 F1  
`f1_int`: sequence-level intention 的 F1  
`pre`: step-level precision  
`recall_sc`: step-level recall  
`pre_int`: sequence-level intention precision  
`recall_int`: sequence-level intention recall  

## Train
```sh
python train.py
```

当前训练逻辑说明：

- `model_best.pkl` 按验证集综合分数保存。
- 综合分数现在更偏向 `F1 + recall + balanced_accuracy`，不再过度偏向高 precision / 高 `f0.5`。
- `model_final.pkl` 只是最后一个 epoch 的权重，默认不作为正式评估模型。

## Eval
正式测试集评估：

```sh
python eval.py --checkpoint output/model_best.pkl --dtype test
```

说明：

- `test` 评估现在强制使用训练阶段保存在 `best_metrics.json` 里的验证集阈值。
- 不允许在 `test` 上重新扫阈值；这样可以避免把测试集用于调参。
- 如果需要做阈值诊断，只能在 `val` 上做。
- `eval.py` 现在默认会导出样本级基础结果表，不需要额外加参数。
- 默认导出目录是当前实验目录下面的 `res_analyze/`，例如：
  - checkpoint 在 `output/Lab0319/model_best.pkl`
  - 且 `artifact_dir=output/Lab0319`
  - 则结果默认保存到 `output/Lab0319/res_analyze/jaad_test_sample_predictions.csv`
- 如果需要覆盖默认目录或文件名，可以显式传：
  - `--sample_results_dir <dir>`
  - `--sample_results_name <file>`
- 如果只想做纯评估、不导出样本级结果，可以显式关闭：
  - `--save_sample_results False`

验证集诊断：

```sh
python eval.py --checkpoint output/model_best.pkl --dtype val --use_saved_thresholds False
```

## Workflow
1. 运行 `python train.py`
2. 查看 `output/best_metrics.json`，确认 `best_epoch` 和验证集指标
3. 使用 `python eval.py --checkpoint output/model_best.pkl --dtype test` 做一次正式测试集评估
4. 如需分析阈值敏感性，只在 `val` 上扫阈值，不在 `test` 上扫

## Notes
- 如果 `state` / `intent` 的 precision 很高但 recall 很低，优先排查阈值和 best checkpoint 选择逻辑，不要先改模型结构。
- 如果轨迹指标已经稳定而分类指标波动大，说明 baseline 更可能卡在分类决策层，而不是特征提取层。

## E2T Prototype Scaffold
为了并行推进“数据读取 → 证据编码 → 信念更新 → 意图输出 → 可视化分析”，当前仓库新增了一套独立原型文件。它们不接入现有 baseline，只提供最小接口和功能说明，便于后续逐步实现。

新增文件：

- `model/evidence_encoder.py`: 逐帧证据编码接口，预留轨迹/姿态/场景融合后的证据向量输出。
- `model/belief_updater.py`: 连续信念更新接口，预留 `leaky` 和 `gru` 两种模式。
- `model/intent_head.py`: 将信念状态映射为 crossing 二分类输出的接口。
- `model/e2t_net.py`: 串联证据编码、信念更新和意图头的总模型接口。
- `model/losses.py`: 原型实验损失函数接口，预留分类损失、时序平滑损失和早触发约束。
- `datasets/pie_intent_proto.py`: 原型数据集读取接口，独立于现有 baseline 的数据管线。
- `datasets/feature_transforms.py`: 特征归一化、拼接和时间差分等预处理接口。
- `scripts/train_e2t_proto.py`: 原型训练入口脚手架。
- `scripts/eval_e2t_proto.py`: 原型评估入口脚手架。
- `visualization/plot_belief_curve.py`: 信念曲线可视化脚手架。
- `visualization/plot_evidence_curve.py`: 证据曲线可视化脚手架。
- `configs/e2t_proto.yaml`: 原型实验专用配置文件。
- `utils/seed.py`: 原型随机种子接口。
- `utils/metrics.py`: 原型评估指标接口。
- `utils/logger.py`: 原型日志接口。

说明：

- 这些文件当前只有最基础的函数/类接口和说明，没有实现功能逻辑。
- 现有 baseline 文件如 `train.py`、`eval.py`、`model/network_image.py`、`datasets/pie.py` 没有因为这套原型而被改动结构。
- 后续实现建议优先顺序：
  1. `datasets/pie_intent_proto.py`
  2. `datasets/feature_transforms.py`
  3. `model/evidence_encoder.py`
  4. `model/belief_updater.py`
  5. `model/intent_head.py`
  6. `model/e2t_net.py`
  7. `scripts/train_e2t_proto.py`
  8. `scripts/eval_e2t_proto.py`


# 实验
## 现有baeline+trick复现结果

### Train best

和前一版比较强的 best（best_epoch=24 那版）相比，当前这个 best_metrics.json 的提升很明显：

```result
best_epoch: 24 -> 56
ade: 5.4118 -> 5.1258
fde: 9.1678 -> 8.6573
aiou: 0.8704 -> 0.8771
fiou: 0.7967 -> 0.8070
state_f1: 0.8073 -> 0.8870
intent_f1: 0.8389 -> 0.8966
state_recall: 0.6996 -> 0.8575
intent_recall: 0.7510 -> 0.8712
state_bal_acc: 0.8289 -> 0.8815
intent_bal_acc: 0.8504 -> 0.8897

```

结论：
训练稳定性明显更好。best_epoch=56 说明现在不再是前几轮就到顶，之前那个“best 太早出现”的问题基本缓解了。
轨迹分支更强了。ADE/FDE 更低，AIOU/FIOU 更高，这说明框预测本身在进步。
分类分支进步更大。尤其 state 和 intent 的 F1/recall/balanced_accuracy 都涨了很多，这比单纯提高 precision 更有价值。
阈值也更合理。现在是 state_threshold=0.75、intent_threshold=0.85，比你之前那种过高且偏保守的设置更像“模型真的学好了”，而不是纯靠抬高阈值保 precision。


### Eval result

#### eval on JAAD
```bash
Evaluating ...
Split: test
Task: 2D_bounding_box-intention
Learning rate: 0.001
Number of epochs: 100
Hidden layer size: 512

/media/meta/Data/CYQ/TIP/E2T-Net/output/Lab0319/model_best.pkl
Using saved validation thresholds: state=0.75, intent=0.85                                                     
Debug state confusion TP/FP/FN/TN: 21570 3198 7510 18927
Debug intent confusion TP/FP/FN/TN: 4407 646 1477 3711
| ade: 5.2237 | fde: 8.7841 | aiou: 0.8705 | fiou: 0.7955 | state_acc: 0.7909 | int_acc: 0.7927 | f1_int: 0.8059 | f1_state: 0.8011 | pre: 0.8709 | recall_sc: 0.7417 | Evaluating ...
Split: test
Task: 2D_bounding_box-intention
Learning rate: 0.001
Number of epochs: 100
Hidden layer size: 512

/media/meta/Data/CYQ/TIP/E2T-Net/output/Lab0319/model_best.pkl
Using saved validation thresholds: state=0.75, intent=0.85                                                     
Debug state confusion TP/FP/FN/TN: 21570 3198 7510 18927
Debug intent confusion TP/FP/FN/TN: 4407 646 1477 3711
| ade: 5.2237 | fde: 8.7841 | aiou: 0.8705 | fiou: 0.7955 | state_acc: 0.7909 | int_acc: 0.7927 | f1_int: 0.8059 | f1_state: 0.8011 | pre: 0.8709 | recall_sc: 0.7417 | bal_sc: 0.7986 | th_sc: 0.75 | pre_int: 0.8722 | recall_int: 0.7490 | bal_int: 0.8004 | th_int: 0.85 | t:678.4131
```

## Baseline 改动总结

下面总结的是目前为止基于现有 baseline 主线做过的改动，主要目标是先把训练稳定性、best checkpoint 选择逻辑和测试集评估规范理顺，再考虑改模型结构。

### 1. 训练与评估入口整理

- 新增了独立的 `eval.py`，不再沿用原来把评估 split 写死成 `val` 的旧逻辑。
- `eval.py` 现在按 `--dtype` 真正加载对应 split，正式测试时使用：
  - `python eval.py --checkpoint output/model_best.pkl --dtype test`
- `eval.py` 现在默认会把样本级基础结果导出到当前实验目录下的 `res_analyze/`，用于后续 `scripts/res_analyze/` 的离线分析。
- 导出目录优先跟随 `artifact_dir` / checkpoint 所在目录，而不是写死到仓库里的固定位置。
- `test.py` 保留，但不再建议作为正式测试入口。

### 2. 训练稳定性改动

在 `train.py` 中补充了几项稳定训练的措施：

- 增加 `weight_decay`
- 增加 `gradient clipping`
- 增加 `label smoothing`
- 增加 `ReduceLROnPlateau` 学习率调度
- 增加 `early stopping`
- 训练日志中增加了：
  - 当前学习率 `lr`
  - `no_improve`
  - best checkpoint score 的权重配置

这些改动的目标不是“人为拖后 best epoch”，而是减少前期偶然达到最优、后期迅速恶化的问题。

### 3. best checkpoint 选择逻辑调整

原先的综合分数偏向：

- 高 precision
- 高 `f0.5`

这会导致模型偏保守，容易出现：precision 很高,recall 偏低,验证集保存下来的 threshold 偏高.

目前在 `train.py` 中已经把综合评分更改为更偏向：

- `F1` `recall` `balanced_accuracy`

同时降低了 `precision` 的权重，并保留 `ADE` 作为轻度惩罚项。  
这样保存下来的 `model_best.pkl` 更接近“分类整体均衡且轨迹也不差”的 checkpoint，而不是单纯高 precision 的 checkpoint。

### 4. test 阈值来源固定

`test` 只能使用 `val` 上保存下来的阈值,不允许在 `test` 上重新扫 threshold

这样做的原因是：
在 `test` 上扫阈值只能作为诊断，不能作为正式结果,正式 test 结果必须避免把测试集用于调参

对应行为：

- `eval.py --dtype test` 会强制要求存在 `best_metrics.json` 里的验证集阈值
- 如果在 `test` 上显式关闭 `use_saved_thresholds`，现在会被拒绝

### 5. bool 参数解析修复

仓库里原先很多命令行参数使用 `type=bool`，这会导致如下问题：

- 传 `False` 也可能被解析成 `True`

目前已在 `train.py` 和 `eval.py` 中统一改成显式布尔解析，因此下面这类命令现在会按预期生效：

- `--use_saved_thresholds False`
- `--use_early_stopping False`
- `--use_balanced_sampler False`
- `--use_image False`

### 6. 当前 baseline 阶段性结论

到目前为止，这条 baseline 线上的结论是：

- 轨迹分支已经比较稳定，不是当前主要瓶颈
- 分类分支之前的主要问题更像是：
  - best checkpoint 选择偏保守
  - threshold 过高
  - precision / recall 不平衡
- 在修正训练稳定性、综合评分逻辑和评估规范后，验证集 best 已经从早期 `epoch 4/24` 逐步推进到更合理的中后期，例如 `epoch 56`
- 当前最重要的工作仍然是把 baseline 继续跑稳、把 test 结果固定下来，而不是立刻把精力放到复杂结构改动上

### 7. 为什么先不直接改 baseline 主体模型

目前的判断是：

- baseline 还有可通过训练逻辑和评估策略释放出来的性能
- 如果过早修改主模型结构，很容易把“训练策略问题”和“模型结构问题”混在一起
- 因此现在采用的是“两条线并行”：
  - baseline 主线继续稳定提升
  - E2T prototype 新线独立搭建，不污染已有主线文件

## idea测试顺序
### 可视化预&处理
- HRNet可以离线标注姿态,保存的关键点数据:[jaad_pose_annotations](../../../../File/datasets/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations.npz)

- 对jaad数据集结果观察,可以显示出姿态的提前倾向.
![alt text](inspect_change.jpg)
