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

## 现有复现结果



和前一版比较强的 best（best_epoch=24 那版）相比，当前这个 best_metrics.json 的提升很明显：

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
这几个结论最关键：

训练稳定性明显更好。best_epoch=56 说明现在不再是前几轮就到顶，之前那个“best 太早出现”的问题基本缓解了。
轨迹分支更强了。ADE/FDE 更低，AIOU/FIOU 更高，这说明框预测本身在进步。
分类分支进步更大。尤其 state 和 intent 的 F1/recall/balanced_accuracy 都涨了很多，这比单纯提高 precision 更有价值。
阈值也更合理。现在是 state_threshold=0.75、intent_threshold=0.85，比你之前那种过高且偏保守的设置更像“模型真的学好了”，而不是纯靠抬高阈值保 precision。