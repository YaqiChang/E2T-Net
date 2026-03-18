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
