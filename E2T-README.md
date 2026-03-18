ade: 预测未来框中心轨迹的平均位移误差，越低越好
fde: 最后一帧位移误差，越低越好
aiou: 整段未来框的平均 IoU，越高越好
fiou: 最后一帧 IoU，越高越好
state_acc: step-level crossing 分类准确率，也就是未来每一步“过/不过”的准确率
intention_acc: sequence-level intention 准确率，也就是整段里是否会发生 crossing 的准确率
f1_state: step-level crossing 的 F1
f1_int: sequence-level intention 的 F1
pre: step-level precision
recall_sc: step-level recall
pre_int: sequence-level intention precision
recall_int: sequence-level intention recall


# eval
```sh
python eval.py --checkpoint output/model_best.pkl --dtype test
```