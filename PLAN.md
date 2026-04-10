# E2T-Net 机制验证阶段代码计划

## 0. 目标与边界

本阶段只回答一个问题：**E2T 的核心机制是否有效**。

固定约束如下。

- 先不接 pose
- 先不换 SSM
- 先不做 diffusion decoder
- 先保留现有 JAAD 和 PIE 数据组织方式
- 先证明从共享表征加双头解码，改成 evidence accumulation 加条件解码后是否有效

本阶段的最小主链路定义为：

$$
x_{1:T} \rightarrow h_{1:T} \rightarrow e_{1:T} \rightarrow z_{1:T} \rightarrow \{p^{int},\ p^{state},\ \hat Y\}
$$

其中 $h_{1:T}$ 表示观测序列编码特征，$e_{1:T}$ 表示瞬时证据，$z_{1:T}$ 表示连续意图状态，$p^{int}$ 表示序列级意图概率，$p^{state}$ 表示未来 crossing state 预测，$\hat Y$ 表示未来轨迹。

---

## 1. 先修当前代码中的基础错误

### 1.1 必须先修的问题

当前 `train.py` 中训练集和验证集都被实例化为 `dtype='val'`，这会直接污染实验结论。当前 `lr` 参数被写成 `int`，默认值却是 `1e-5`，会导致学习率解析错误。当前 crossing 分支使用 `Linear + ReLU + Softmax` 再配 `BCELoss`，这一套不适合作为后续机制实验的稳定起点。当前 `utils.calculate_score` 的 TP、FP、FN 定义也不规范，后续指标必须改成标准二分类统计。 

### 1.2 修改文件

- `train.py`
- `utils.py`

### 1.3 修改内容

#### `train.py`

- 把训练集改成 `dtype='train'`
- 保留验证集 `dtype='val'`
- 把 `--lr` 类型从 `int` 改为 `float`
- 增加随机种子与日志保存路径检查
- 暂时关闭 image 和 optical flow 默认项，先跑机制基线

#### `utils.py`

- 重写 `calculate_score`
- 增加标准指标计算接口，至少包括 precision、recall、F1、accuracy、balanced accuracy
- 后续为阈值扫描和 time to trigger 分桶预留函数接口

### 1.4 完成标准

完成后应能在不改模型结构的情况下，重新得到可信的 baseline 训练和验证日志。

---

## 2. 本阶段建议保留的现有文件与职责

### 2.1 直接保留

#### `datasets/jaad.py`

本阶段可直接保留主体逻辑。它已经能输出 `speed`、`pos`、`future_speed`、`future_pos`、`future_cross`、`cross_label`、`ped_attribute`、`ped_behavior`、`scene_attribute` 等字段，并且已经删除了观测窗口内已经 crossing 的样本，适合做 anticipation 机制验证。

#### `datasets/pie.py`

先沿用现有组织方式，不在本阶段改标签结构。

#### `utils.py`

除指标函数外，其余轨迹度量和位置恢复函数可以继续使用。

### 2.2 作为旧基线保留

#### `model/network_image.py`

该文件不再继续作为主模型叠加修改，而保留为旧版 PTINet 基线实现。

---

## 3. 本阶段总的代码重构原则

### 3.1 不在旧文件上不断堆逻辑

当前 `model/network_image.py` 同时承担了模态编码、特征融合、双分支解码等职责，后续若继续堆 Evidence Head 和 Belief Cell，会很快失控。因此本阶段采用新文件独立实现，旧文件只用于 baseline 对照。

### 3.2 先拆成四类模块

本阶段建议把主模型拆成四层。

1. **Observation Encoder**，负责把原始观测变成时序特征
2. **Evidence Layer**，负责把时序特征变成瞬时证据
3. **Belief Layer**，负责把瞬时证据递推成连续意图状态
4. **Prediction Heads**，负责输出 intent、state 和 trajectory

### 3.3 先保持编码器简单

本阶段不换 backbone。观测编码器先沿用接近当前实现的轻量时序编码方式，只要求能输出逐时刻特征，不要求它成为创新点。创新点只放在 evidence 到 belief 的链条。

---

## 4. 需要新建的文件

## 4.1 `model/encoders.py`

### 目标

把当前散落在 `network_image.py` 里的多模态编码器整理成可复用模块，并统一输出逐时刻特征序列。

### 本阶段内容

- `TrajEncoder`
- `BehaviorEncoder`
- `SceneEncoder`
- 预留 `PoseEncoder` 接口但先不启用

### 输入输出约定

#### `TrajEncoder`

输入：

- `speed`，形状建议为 `[B, T_obs-1, 4]`
- `pos`，形状建议为 `[B, T_obs, 4]`

输出：

- `h_traj`，形状建议为 `[B, T_enc, D]`
- `c_traj`，可选，作为解码初值的全局上下文

#### `BehaviorEncoder`

输入：`ped_behavior`

输出：`h_behavior`

#### `SceneEncoder`

输入：`scene_attribute`

输出：`h_scene`

### 实现建议

第一版不再用 `LSTMVAE` 直接承担最终建模。可先用更简单的 `GRU` 或 `TemporalConv1D` 输出逐时刻特征，重点是把每时刻表示交给后续 Evidence Head。当前阶段不需要把 encoder 变成论文创新点。

---

## 4.2 `model/evidence.py`

### 目标

把多模态观测特征转成带语义的瞬时证据增量。

### 建议类

- `TemporalFusion`
- `EvidenceHead`

### 公式原型

$$
o_t = [h_t^{traj}; h_t^{behavior}; h_t^{scene}]
$$

$$
\phi_t = G_\theta(o_t)
$$

$$
u_t = W_u \phi_t + b_u
$$

$$
r_t = \sigma(W_r \phi_t + b_r)
$$

$$
e_t = r_t \odot \tanh(\nu_t)
$$

其中 $\nu_t$ 表示原始证据强度，$r_t$ 表示证据可靠度，$e_t$ 表示最终瞬时证据。

### 输出建议

第一版直接输出双通道证据：

$$
e_t = [e_t^{+}, e_t^{-}]
$$

分别对应 crossing 和 noncrossing 两个方向的支持量。

### 代码职责

- 接收多路时序特征
- 做逐时刻融合
- 生成双通道证据
- 同时返回可靠度，便于后续可视化

---

## 4.3 `model/belief.py`

### 目标

实现结构化意图状态递推，而不是黑盒 RNN hidden state。

### 建议类

- `LeakyAccumulator`
- `NeuralLCACell`

### 第一版建议

先实现两个版本，便于后续机制消融。

#### 版本 A，单通道 leaky

$$
z_t = \rho_t \odot z_{t-1} + e_t
$$

其中

$$
\rho_t = \sigma(g_\rho(\phi_t))
$$

#### 版本 B，双通道竞争型 Neural LCA

$$
a_t^{+} = \rho_t^{+} a_{t-1}^{+} - \gamma_t^{+} a_{t-1}^{-} + \beta_t^{+} e_t^{+}
$$

$$
a_t^{-} = \rho_t^{-} a_{t-1}^{-} - \gamma_t^{-} a_{t-1}^{+} + \beta_t^{-} e_t^{-}
$$

$$
z_t = a_t^{+} - a_t^{-}
$$

其中

- $\rho_t$ 表示泄露系数
- $\gamma_t$ 表示竞争强度
- $\beta_t$ 表示当前证据注入强度

这些量都由小型参数网络从当前融合特征中预测。

### 本阶段决定

正式主模型先采用 **版本 A**，同时保留版本 B 的接口。原因是第一阶段目标是先证明显式累积有效，复杂竞争项留到后续第二轮机制增强。

---

## 4.4 `model/heads.py`

### 目标

把下游输出头独立出来，避免主模型文件继续膨胀。

### 建议类

- `IntentHead`
- `StateHead`
- `TrajectoryHead`
- 可选 `TriggerHead` 占位类

### 各头职责

#### `IntentHead`

输入：最终连续状态 $z_T$ 或全序列聚合状态

输出：

$$
p^{int} = P(\tau \le T+H \mid x_{1:T})
$$

即未来窗口内是否 crossing 的概率。

#### `StateHead`

输入：$z_T$ 与历史运动上下文

输出：未来 `H` 步 crossing state logits。第一版直接输出 future sequence，不建 trigger。

#### `TrajectoryHead`

输入：历史运动上下文与 $z_T$

输出：future speed 或 future position。

第一版保留和现有代码一致的速度解码方式，但解码初值改为 belief conditioned context。

---

## 4.5 `model/e2t_net.py`

### 目标

作为新主模型入口，把编码器、证据层、信念层、输出头串起来。

### 建议结构

1. 调用各分支 encoder 得到逐时刻特征
2. 做逐时刻融合并输出 `e_1, ..., e_T`
3. 用 belief cell 递推出 `z_1, ..., z_T`
4. 用 `z_T` 读出 intent、state，并条件化 trajectory decoder
5. 返回所有训练所需中间量

### forward 建议返回内容

```python
{
    "traj_pred": ...,
    "state_logits": ...,
    "intent_logit": ...,
    "evidence_seq": ...,
    "belief_seq": ...,
    "reliability_seq": ...,
    "aux": {...}
}
```

这样训练和可视化会更清楚。

---

## 5. 需要修改的文件

## 5.1 `train.py`

### 目标

把训练逻辑从旧的双头输出改成面向 E2T 的多输出训练。

### 修改项

#### 1. 模型导入

把

```python
import model.network_image as network
```

改成

```python
import model.e2t_net as network
```

同时保留旧模型导入方式，便于 baseline 对照。

#### 2. 损失函数重构

从原来的

$$
\mathcal L = \mathcal L_{speed} + \mathcal L_{cross} + \mathcal L_{modal}
$$

改成

$$
\mathcal L = \mathcal L_{traj} + \lambda_1 \mathcal L_{state} + \lambda_2 \mathcal L_{intent} + \lambda_3 \mathcal L_{smooth}
$$

第一版先只保留四项。

- `L_traj` 对 future speed 或 future pos
- `L_state` 对 `future_cross`
- `L_intent` 对 `cross_label`
- `L_smooth` 对 belief 序列的时间平滑约束

#### 3. 指标输出

增加以下记录内容。

- state precision、recall、F1、balanced accuracy
- intent precision、recall、F1、balanced accuracy
- ADE、FDE、AIOU、FIOU
- belief 曲线日志
- evidence 可靠度日志

#### 4. 验证逻辑

删除当前由 future crossing 序列后处理得到 intention 的逻辑，改成直接读取 `IntentHead` 输出。

#### 5. 结果保存

建议额外保存每个 epoch 的 `belief_seq` 与 `intent_logit` 样本，便于后续画图。

---

## 5.2 `datasets/jaad.py`

### 本阶段修改原则

尽量少动主体，只补充机制验证需要的字段。

### 建议补充

- 增加 `sample_index`
- 增加 `time_to_trigger` 生成接口占位
- 增加 `future_cross_binary` 便于 state 监督
- 保证输出中的 `cross_label` 是标准 float 或 long 类型

### 本阶段不做

- 不接 pose
- 不重写图片处理逻辑
- 不处理伪标签

---

## 5.3 `model/vae.py`

### 本阶段建议

如果后续新建 `model/encoders.py` 后不再依赖 `LSTMVAE`，则该文件本阶段不再作为主模型依赖，只作为旧基线保留。

### 若仍想复用旧编码器

则至少需要把 `Encoder.forward` 和 `LSTMVAE.forward` 改成返回逐时刻 LSTM 输出序列。但从结构整洁性看，不建议继续把它作为主路径。

---

## 6. 模块之间的连接关系

本阶段主模型的建议数据流如下：

```text
speed, pos, ped_behavior, scene_attribute
    -> Encoders
    -> h_traj, h_behavior, h_scene
    -> TemporalFusion
    -> EvidenceHead
    -> evidence_seq, reliability_seq
    -> BeliefCell
    -> belief_seq
    -> IntentHead
    -> intent_logit
    -> StateHead
    -> state_logits
    -> TrajectoryHead
    -> traj_pred
```

### 输入形状建议

- `speed`：`[B, T_s, 4]`
- `pos`：`[B, T_p, 4]`
- `ped_behavior`：`[B, T_b, d_b]`
- `scene_attribute`：`[B, T_c, d_c]`

### 中间张量形状建议

- `h_*`：`[B, T, D]`
- `evidence_seq`：`[B, T, 2]`
- `belief_seq`：`[B, T, D_z]` 或 `[B, T, 1]`
- `intent_logit`：`[B, 1]`
- `state_logits`：`[B, H, 1]`
- `traj_pred`：`[B, H, 4]`

第一版建议 `D_z = 8` 或 `16`，不要过大。

---

## 7. 第一阶段机制验证的最小实验设计

## 7.1 实验目标

先证明显式 evidence accumulation 有效，而不是证明 backbone 更好，也不是证明 pose 有效。

## 7.2 最小四组对照

### 实验 A

**原始 PTINet baseline**

- 保留旧模型
- 仅修复训练错误与指标错误

### 实验 B

**Evidence only**

- 新 encoder
- 新 EvidenceHead
- 不做累积
- 对 `e_1, ..., e_T` 做平均池化后输出 intent 和 state

目的：验证显式证据建模本身是否比共享融合更好。

### 实验 C

**Evidence + Leaky Accumulator**

- 在实验 B 基础上加入单通道 belief 递推

目的：验证显式累积是否优于简单池化。

### 实验 D

**Evidence + Leaky Accumulator + Belief Conditioned Trajectory**

- 在实验 C 基础上让 trajectory decoder 条件于最终 belief 状态

目的：验证连续意图状态是否能改善轨迹预测。

---

## 8. 代码实施顺序

## Step 1

先修 `train.py` 和 `utils.py`，重新跑通旧 baseline。

## Step 2

新建 `model/encoders.py`，只支持 `pos`、`speed`、`ped_behavior`、`scene_attribute` 四类输入。

## Step 3

新建 `model/evidence.py`，输出双通道 evidence 与 reliability。

## Step 4

新建 `model/belief.py`，先实现单通道或低维 leaky belief cell。

## Step 5

新建 `model/heads.py`，把 intent、state、trajectory 头拆出来。

## Step 6

新建 `model/e2t_net.py`，串联整个前向。

## Step 7

修改 `train.py`，支持新返回格式和新损失。

## Step 8

跑实验 B、C、D，与实验 A 对照。

---

## 9. 每一步的验收标准

## 验收 1，基础可信性

- baseline 能正常训练
- train 和 val 正确拆分
- 指标计算正常

## 验收 2，模块可运行

- `evidence_seq` 和 `belief_seq` 的 shape 稳定
- 无 NaN
- `intent_logit` 与 `state_logits` 可正常反向传播

## 验收 3，机制有效性

至少满足以下之一。

- state F1 高于 baseline
- intent F1 高于 baseline
- belief conditioned trajectory 使 ADE 或 FDE 降低
- belief 曲线对正负样本呈现可分结构

---

## 10. 本阶段暂不做的内容

以下内容全部推迟到机制验证通过之后。

- pose 分支接入
- SSM 编码器替换
- trigger head
- hazard 建模
- diffusion trajectory decoder
- image 与 optical flow 的重新启用
- 伪标签与重标注
- consistency loss
- Neural LCA 双通道竞争正式版

---

## 11. 分支管理建议

### 分支一

`baseline_fix`

只修复训练和指标问题。

### 分支二

`e2t_v1_mechanism`

实现 `encoders + evidence + belief + heads + e2t_net`。

### 分支三

`e2t_v1_analysis`

增加可视化、阈值扫描、belief 曲线导出、时间分桶分析。

---

## 12. 本阶段交付物

本阶段结束时，代码侧应至少形成以下文件结构。

```text
model/
  network_image.py          # 旧 baseline 保留
  encoders.py               # 新增
  evidence.py               # 新增
  belief.py                 # 新增
  heads.py                  # 新增
  e2t_net.py                # 新增
train.py                    # 修改
utils.py                    # 修改
datasets/
  jaad.py                   # 少量修改
  pie.py                    # 视情况同步
```

并形成以下实验产物。

- baseline 日志
- 实验 B、C、D 日志
- state 和 intent 指标表
- ADE 和 FDE 表
- evidence 曲线图
- belief 曲线图

---

## 13. 当前阶段一句话任务定义

**先把 E2T 从“共享特征加双头解码”重构成“显式瞬时证据、连续意图状态、条件轨迹解码”的最小可运行机制模型，并在不引入 pose 和 SSM 的前提下验证机制有效。**
