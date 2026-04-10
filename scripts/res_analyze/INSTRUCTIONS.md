**实验标题**

**基于固定 checkpoint 的分类阈值敏感性、触发时距分层与数据分布差异分析**

**新增实现与使用**

当前分析代码已经按“单入口主流程 + 工具模块”落地在 `scripts/res_analyze/` 下。

文件结构：

- `run_analysis.py`
  统一主入口。根据 `--mode` 调用不同分析流程。
- `export_predictions.py`
  调用仓库根目录的 `eval.py`，导出样本级预测表。
- `io_utils.py`
  负责结果表读取、字段检查与 csv 保存。
- `metrics_utils.py`
  负责阈值扫描、confusion matrix 指标计算、time-to-trigger 分桶统计。
- `distribution_analysis.py`
  负责 val 与 eval/test 的分布差异统计。
- `plot_utils.py`
  负责阈值曲线与分桶曲线绘图。

配套的 `eval.py` 已增加可选导出参数，默认可以把结果导到 `scripts/res_analyze/results/<run_name>/` 下面。

常用命令：

1. 只导出固定 checkpoint 的样本级结果表

```bash
python scripts/res_analyze/run_analysis.py \
  --mode export \
  --checkpoint /media/meta/Data/CYQ/TIP/E2T-Net/output/Lab0319/model_best.pkl \
  --artifact_dir /media/meta/Data/CYQ/TIP/E2T-Net/output/Lab0319 \
  --split test \
  --results_dir /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319
```

2. 基于已导出的结果表做阈值扫描

```bash
python scripts/res_analyze/run_analysis.py \
  --mode threshold_scan \
  --predictions_csv /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319/jaad_test_sample_predictions.csv \
  --results_dir /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319
```

3. 基于已导出的结果表做触发时距分桶分析

```bash
python scripts/res_analyze/run_analysis.py \
  --mode time_bucket \
  --predictions_csv /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319/jaad_test_sample_predictions.csv \
  --results_dir /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319
```

4. 对 val 与 eval/test 的预测表做分布差异比较

```bash
python scripts/res_analyze/run_analysis.py \
  --mode distribution_compare \
  --val_predictions_csv /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319/jaad_val_sample_predictions.csv \
  --eval_predictions_csv /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319/jaad_test_sample_predictions.csv \
  --results_dir /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319
```

5. 导出并串行执行全部分析

```bash
python scripts/res_analyze/run_analysis.py \
  --mode all \
  --checkpoint /media/meta/Data/CYQ/TIP/E2T-Net/output/Lab0319/model_best.pkl \
  --artifact_dir /media/meta/Data/CYQ/TIP/E2T-Net/output/Lab0319 \
  --split test \
  --results_dir /media/meta/Data/CYQ/TIP/E2T-Net/scripts/res_analyze/results/Lab0319
```

---

**一、实验目的**

本实验用于分析当前模型在独立评测集上出现高精度、低召回现象的来源，明确问题究竟来自阈值设定、时间位置差异，还是验证集与评测集之间的数据分布偏移。

具体目标分为三项。

1. **阈值扫描分析**
   检查 state 与 intent 两个任务在不同阈值下的 precision、recall、F1、balanced accuracy 变化规律，判断当前阈值是否偏保守。

2. **按触发时距分层分析**
   按样本距离 crossing 或 intent trigger 的时间距离进行分桶，分析模型在远早期、临近触发和触发后阶段的识别能力差异，定位 recall 下降主要集中在哪一类时间区间。

3. **验证集与评测集分布差异分析**
   对比 val 与 eval 的类别比例、场景组成、样本来源和预测分数分布，判断当前性能下降是否与数据划分差异有关。

---

**二、实验总体原则**

1. **固定模型参数。**
   所有分析均在同一个 checkpoint 上完成，不允许在分析过程中重新训练或修改模型权重。

2. **固定推理流程。**
   所有样本使用同一份数据预处理、同一推理脚本、同一后处理方式，避免由于实现差异引入额外偏差。

3. **分析与训练解耦。**
   本实验只分析预测输出与数据属性之间的关系，不引入新的训练策略，不修改损失函数，不重新搜索超参数。

4. **先保存样本级结果，再做统计。**
   所有后续分析均建立在同一份样本级预测表之上，避免重复推理造成结果不一致。

5. **分别分析 state 与 intent。**
   两个任务不能合并讨论，每个任务必须独立画曲线、独立统计、独立解释。

---

**三、代码编写方法与逻辑**

### **1. 总体代码结构**

建议将分析代码拆成四个部分。

**第一部分，统一导出样本级预测结果。**
输入为固定 checkpoint 和指定数据集。
输出为一张样本级结果表。

每条样本至少保存以下字段。

* split
* video_id
* ped_id
* frame_id
* state_gt
* intent_gt
* state_score
* intent_score
* state_pred_at_saved_th
* intent_pred_at_saved_th
* trigger_frame 或 crossing_start_frame
* time_to_trigger

其中 `time_to_trigger` 建议统一定义为

$$
\Delta f = f_{\text{current}} - f_{\text{trigger}}
$$

若需要换算为时间，则可进一步定义

$$
\Delta t = \frac{\Delta f}{\text{fps}}
$$

这里 `f_current` 表示当前样本对应的帧编号，`f_trigger` 表示 crossing 或 intent trigger 的参考帧编号。

---

### **2. 阈值扫描模块**

该模块基于已保存的 `state_score` 与 `intent_score` 做离线扫描。

**实现逻辑如下。**

1. 读取样本级结果表。
2. 对 state 任务设定阈值序列，例如从 0.00 到 1.00，步长为 0.01。
3. 对每个阈值重新生成二值预测。
4. 计算 confusion matrix。
5. 基于 confusion matrix 计算 precision、recall、F1、balanced accuracy、accuracy。
6. 对 intent 任务重复同样过程。
7. 保存阈值与指标对应表，并绘制指标曲线。

**建议输出内容如下。**

* state 的 threshold 对 metric 曲线
* intent 的 threshold 对 metric 曲线
* 当前保存阈值在曲线中的位置标记
* 最优 F1 对应阈值
* 最优 balanced accuracy 对应阈值

---

### **3. 触发时距分层模块**

该模块用于分析模型在不同时间阶段的识别差异。

**实现逻辑如下。**

1. 读取样本级结果表。
2. 根据 `time_to_trigger` 将样本分桶。
3. 对每个桶分别统计 state 与 intent 的 TP、FP、FN、TN。
4. 计算每个桶的 precision、recall、F1。
5. 绘制随时间距离变化的指标曲线。

**分桶方式建议保持简单统一。**
例如按帧距离划分为以下几类。

* 远早期
* 中早期
* 临近触发
* 触发帧附近
* 触发后

若数据集帧率稳定，也可以直接按秒划分。

**这里的关键不是桶数多少，而是 val 与 eval 使用完全相同的分桶边界。**

---

### **4. 数据分布差异模块**

该模块用于比较 val 与 eval 的数据构成差异及预测分数偏移。

**实现逻辑如下。**

1. 分别读取 val 与 eval 的样本级结果表。
2. 统计 state 与 intent 的正负样本比例。
3. 统计每个 video_id 的样本数量分布。
4. 统计每个 ped_id 的样本数量分布。
5. 检查同一 ped_id 或同一视频是否跨 split 重复。
6. 比较 val 与 eval 在不同场景属性下的样本占比。若无显式场景标签，可先按视频来源替代。
7. 比较 val 与 eval 上正样本和负样本的 score 分布。

**建议输出内容如下。**

* val 与 eval 的类别比例表
* val 与 eval 的视频来源占比表
* val 与 eval 的 ped_id 重复检查结果
* state_score 与 intent_score 在 val 和 eval 上的分布图
* 正负样本 score 分布的重叠情况

---

**四、代码实现原则**

1. **先导出统一表，再开展一切分析。**
   不要为每个分析单独重新跑一遍推理。

2. **所有统计函数都基于同一份预测表。**
   这样不同实验之间的结果可直接对齐。

3. **字段命名保持一致。**
   state 与 intent 应使用平行命名，便于后续复用统计函数。

4. **分析代码只读输入，不修改原始预测结果。**
   阈值扫描时重新生成临时预测列，不覆盖原始 `score`。

5. **绘图与统计分离。**
   先保存 csv 结果表，再单独画图，便于后续复查。

6. **同一指标只采用一种定义。**
   尤其是 balanced accuracy、F1、precision、recall，必须在所有脚本中保持完全一致。

7. **区分验证阈值与分析阈值。**
   已保存阈值用于复现当前正式结果，扫描阈值用于观察趋势，这两类结果不能混写。

---

**五、结果分析原则**

### **1. 阈值扫描结果的分析原则**

1. **先看 precision 与 recall 的相对关系。**
   若 precision 明显高于 recall，说明模型在当前阈值下偏保守，漏检较多。

2. **再看 F1 与 balanced accuracy 的峰值位置。**
   若峰值出现在更低阈值处，说明当前保存阈值可能偏高。

3. **不要只看单点最优值。**
   还要看曲线在邻域内是否平稳。若曲线对阈值极其敏感，说明输出分数校准不稳定。

4. **state 与 intent 分开解释。**
   两者可能具有不同的最优阈值区间，不应强行统一。

---

### **2. 触发时距分层结果的分析原则**

1. **优先观察 recall 随时间的变化。**
   你的当前问题主要是漏检，因此 recall 是最关键指标。

2. **若远早期 recall 很低，临近触发 recall 才明显上升，说明模型更依赖后验显著证据。**

3. **若临近触发阶段仍然 recall 偏低，说明问题已不只是提前识别不足，还可能包含分类边界模糊或输出校准失衡。**

4. **若触发后 precision 快速上升而 recall 仍低，说明模型只愿在非常确定时输出正类。**

5. **分层结果必须结合样本数解释。**
   某一时间桶样本过少时，不能仅凭单个指标下结论。

---

### **3. val 与 eval 分布差异的分析原则**

1. **先看类别比例是否一致。**
   若正负样本比例差异明显，阈值迁移失效是正常现象。

2. **再看视频来源与个体来源是否一致。**
   若 eval 含有更多未见场景或未见个体，分类泛化下降是合理现象。

3. **最后看 score 分布是否发生整体偏移。**
   若 eval 上正样本分数整体更低，则说明不是特征完全无效，而是输出校准偏移。

4. **分布分析不能只给表格。**
   必须结合最终 FN 增多这一现象做解释，否则只能得到表面统计。

---

**六、最终解释口径**

本实验不用于证明新方法优于旧方法，而用于回答三个问题。

1. **当前高 precision、低 recall 是否主要由阈值偏高导致。**
2. **漏检主要集中在什么时间阶段。**
3. **当前验证集最优阈值为何不能稳定迁移到评测集。**

若三项分析完成后结论一致，例如都指向边界时刻识别不足与阈值迁移偏移，那么后续创新应优先围绕以下方向展开。

* 早期证据建模
* 连续意图状态建模
* 分类分数校准
* 边界样本损失设计

---

**七、建议的实验产出**

最终至少保留以下结果文件。

* 样本级预测结果表
* state 阈值扫描结果表
* intent 阈值扫描结果表
* 触发时距分桶统计表
* val 与 eval 分布对照表
* 对应图像文件

---

**可直接写在实验文档中的一句话版本**

> 在固定 checkpoint 不变的条件下，对 state 与 intent 两个分类任务分别开展阈值敏感性分析、触发时距分层分析以及验证集与评测集分布差异分析，以定位当前高精度低召回现象的主要来源，并为后续创新模块设计提供依据。
