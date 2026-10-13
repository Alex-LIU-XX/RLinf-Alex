# RLinf 中的 RL Token（RLT）实现详解：论文 × 代码逐层对照

> 分析对象：`RLinf` 仓库（分支 `dev/Alex`，基线 commit `fa1579c9`）
> 对照论文：*RL Token: Bootstrapping Online RL with Vision-Language-Action Models*（arXiv:2604.23073v2，Physical Intelligence）
> 论文精读另见同目录 [`RL-Token_讲解.md`](./RL-Token_讲解.md)

---

## 0. 结论先行

**有，而且是一套完整、可运行、已进 CI 的实现。**

RLinf 把论文的两阶段流程完整落地了：

| 论文阶段 | RLinf 中的名字 | 状态 |
|---|---|---|
| 阶段 A：RL Token 自编码瓶颈 + VLA SFT | **Stage 1** / `rlt` / `use_rlt` | ✅ 完整实现（含 stop-gradient、自回归重建、`L_ro + α·L_vla`） |
| 阶段 B：动作块级 off-policy actor-critic | **Stage 2** / `loss_type: rlt_ac`、`rlt_td3` | ✅ 完整实现（chunk 级 TD、双 Q、参考动作直通、BC 正则、reference dropout） |
| 人在回路 / 关键阶段只训一段 | `rlt_policy_switch` + `intervene_flag` | ✅ 真机为键盘+遥操作干预；仿真用"专家接管"替代人类 |
| 异步 rollout/learner | `train_async.py` + `AsyncRLTACFSDPPolicy` | ✅ 已实现 |

同时它也做了一些**论文里没有的工程化改动**（权重 ramp、干预感知的 BC target、仿真专家接管），也有**几处论文里的细节没有实现**（stride=2 子采样、UTD 常数、学到的交接预测、throughput 指标）。第 7 节有一张逐条对照表。

**一句话概括代码里的分工**：
冻结的 `openpi`（π0.5）同时干三件事——**感知**（产出 `z_rl` 作为 RL 状态）、**给起点**（产出 `ref_chunk` 参考动作）、**当锚点**（BC 正则把 actor 约束在它附近）；`rlt_mlp_policy` 这个 3 层小 MLP 只负责在关键阶段把动作块"磨"得又快又准。

---

## 1. 代码地图：论文组件 → 代码位置

| 论文概念 | 代码位置 | 说明 |
|---|---|---|
| RL Token encoder/decoder | `rlinf/models/embodiment/modules/rlt_token_transformer.py` | `RLTTokenEncoder` / `RLTTokenDecoder` / `RLTTokenTransformer` |
| RL Token 配置 | `rlinf/models/embodiment/openpi/rlt_config.py` | `OpenPiPytorchRLTConfig` / `build_rlt_config` |
| 接入 VLA（π0.5） | `rlinf/models/embodiment/openpi/pi0.py` | `_init_rlinf_runtime`、`_sft_forward_with_rlt_prefix`、`sft_forward` |
| 推理时抽 `z_rl` | `rlinf/models/embodiment/openpi/tasks/eval.py` | `Pi0Eval.extract_rlt_obs()` |
| Stage 1 入口 | `examples/sft/train_vla_sft.py` + `rlinf/workers/sft/fsdp_vla_sft_worker.py` | |
| Actor / Critic 网络 | `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` | `RLTMLPPolicy`（AC 主线） |
| TD3 变体网络 | `rlinf/models/embodiment/mlp_policy/rlt_td3_mlp_policy.py` | `RLTTD3MLPPolicy`（消融/对照） |
| 损失与训练循环（AC） | `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` | `RLTACLossMixin` / `RLTACFSDPPolicy` / `AsyncRLTACFSDPPolicy` |
| 损失与训练循环（TD3） | `rlinf/workers/actor/fsdp_rlt_td3_policy_worker.py` | `RLTTD3LossMixin` / `RLTTD3FSDPPolicy` |
| 动作路由（actor / ref / expert） | `rlinf/algorithms/rlt/route.py` | `RealworldRLTRoute` / `SimulatorRLTRoute` |
| 一次 rollout 的封装 | `rlinf/algorithms/rlt/rollout.py` | `predict_rlt_actions()` |
| 干预改写参考动作 | `rlinf/algorithms/rlt/transition.py` | `apply_rlt_interventions()`、`RLT_OBS_KEYS` |
| 仿真环境 | `rlinf/envs/sim/maniskill/maniskill_rlt_env.py` | `ManiskillRLTEnv`（env_type `maniskill_rlt`） |
| 真机键盘切换 | `rlinf/envs/real/wrappers/episode/policy_switch.py` | `KeyboardRLTPolicySwitchWrapper`（按 `b` 进入 actor 阶段） |
| 配置 | `examples/embodiment/config/{maniskill_rlt_stage2_ac_mlp,maniskill_rlt_stage2_td3_mlp,realworld_rlt_stage2_ac_mlp}.yaml`、`examples/sft/config/*_rlt_stage1_*.yaml`、`examples/embodiment/config/env/maniskill_rlt.yaml` | |
| 官方文档 | `docs/source-en/rst_source/examples/embodied/rlt.rst`（中文版 `source-zh/...`） | 842 行的使用指南 |
| 测试 | `tests/unit_tests/test_models.py`、`tests/unit_tests/test_data.py`、`tests/e2e_tests/{embodied,sft}/*rlt*.yaml` | |

---

## 2. 整体架构

```
【Stage 1：离线，跑在 SFT 框架里】

  LeRobot 演示数据（图像 + state + 动作）
        │
        ▼
  OpenPI π0.5  ──┬──► VLA flow-matching 损失 L_vla ─────────────┐
                 │                                              │
                 └──► prefix hidden states (B, M, 2048)         │  total_loss
                        （.detach()，梯度不回传）                 │  = L_ro
                              │                                  │  + α·L_vla
                              ▼                                  │
                     RLTTokenTransformer                         │
                     ├─ encoder → z_rl (1×2048) ─────────────────┤
                     └─ decoder → 重建 prefix，算 MSE L_ro ──────┘

  产物： checkpoints/global_step_X/actor （含 rlt_module.* 权重）
                                 │
                                 │ 整份冻结，作为 rollout.rlt_feature_model
                                 ▼
【Stage 2：在线 RL，跑在 embodied 框架里】

  env.step(chunk)  ⇄  要动作                                learner
        │                                                    │
        │  ┌─────────────────────────────────────────────────┘
        ▼  ▼
  冻结 VLA ──► extract_rlt_obs() ──► { z_rl, proprio, ref_chunk }
                                            │
                                            ▼
                                  RLTMLPPolicy（小 MLP，actor + critic）
                                            │
                                     routed action
                             （actor / VLA ref / expert 三选一）
                                            │
                             ┌──────────────┴───────────────┐
                             ▼                              ▼
                    发给环境执行整个 chunk            replay buffer
                                                    (z_rl, proprio, ref_chunk,
                                                     a, r, next_*)
                                                            │
                                                            ▼
                                          chunk 级 TD 更新 critic
                                          -q·Q + β·BC 更新 actor
                                                            │
                                            权重同步（weight_syncer）
                                                            │
                                            └──► 回填到 rollout 上的 actor 副本
```

---

## 3. 阶段一：RL Token 是怎么"造"出来的

### 3.1 训练入口与总损失

- 入口脚本：`examples/sft/train_vla_sft.py`，用 `SFTRunner` + `FSDPVlaSftWorker`。
- 配置：`examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml`（仿真）与 `realworld_rlt_stage1_sft_openpi_pi05.yaml`（真机）。
- 启动：`bash examples/sft/run_vla_sft.sh maniskill_rlt_stage1_sft_openpi_pi05`

真正拼损失的地方是 `pi0.py:824-847` 的 `Pi0.sft_forward`：

```python
if not self.rlt_cfg.use_rlt:                       # 普通 SFT
    return self._reduce_sft_loss(...)
per_element_loss, prefix_output, prefix_mask = self._sft_forward_with_rlt_prefix(...)
vla_loss = self._reduce_sft_loss(per_element_loss, use_action_chunk_loss)
rlt_loss, _ = self._rlt_forward(prefix_output, prefix_mask)
return {
    "loss":     rlt_loss + self.rlt_cfg.rlt_alpha * vla_loss,
    "vla_loss": vla_loss,
    "rlt_loss": rlt_loss,
}
```

对应论文的

$$\phi,\theta_{vla} = \arg\min \; \mathcal{L}_{ro}(\phi) + \alpha\,\mathcal{L}_{vla}(\theta_{vla})$$

`rlt_alpha` 就是 α（配置里设为 `1.0`）。`vla_loss` 是 π0.5 的 flow-matching 损失 `torch.square(v_t - u_t)`（`pi0.py:817-818`），不是动作的 token 交叉熵。

> 注意：`_sft_forward_with_rlt_prefix` 是**单独重写**了一遍前向（`pi0.py:776-822`），因为它必须把 `self.llm(...)` 输出的 **prefix hidden states** 留下来给 RLT 用；原版 `compute_loss()` 会把它丢掉。

### 3.2 从哪里读 VLA 表征？——prefix hidden states

```python
prefix_out, suffix_out = self.llm(
    [prefix_tokens, suffix_tokens], positions=..., mask=attn_mask, ...
)[0]
v_t = self.velocity_from_suffix(suffix_out[:, -self.action_horizon:])
loss = torch.square(v_t - u_t)
prefix_out, prefix_mask = self._select_rlt_prefix_embeddings(
    prefix_out.detach(), prefix_mask, observation.tokenized_prompt
)
return loss, prefix_out, prefix_mask
```

- 读的是 **π0.5 主干（PaliGemma/Gemma）最后一层的 prefix 位置 hidden states**，形状 `(B, M, 2048)`，`M = 图像 token + 语言 token`。
- 注意 `prefix_out.detach()`：**进入 RLT 分支的表征被切断了梯度**。
- `_select_rlt_prefix_embeddings`（`pi0.py:664-674`）里有个开关 `rlt_image_only`：为 `True` 时只保留图像 token、**丢掉语言 token**（论文脚注里提到的那种做法）；两个官方 Stage 1 配置都显式设成 `False`，也就是**语言 token 保留并参与重建**，与论文的 $z_{1:M}$ 一致。

### 3.3 Encoder：怎么得到 `z_rl`

`rlinf/models/embodiment/modules/rlt_token_transformer.py` 的 `RLTTokenEncoder`：

```python
self.input_proj     = Linear(input_dim, embed_dim) if input_dim != embed_dim else Identity()
self.rl_token_embed = nn.Parameter(sinusoidal_pe_init(1, embed_dim))       # 可学习的 <rl>
self.prefix_pos_enc = nn.Parameter(sinusoidal_pe_init(prefix_seq_len, embed_dim))
self.rl_token_pos_enc = nn.Parameter(sinusoidal_pe_init(1, embed_dim))
...
x = torch.cat([prefix_tokens, rl_tokens], dim=1)      # 序列尾部追加 <rl>
for layer in self.layers: x = layer(x, mask=mask)
return x[:, -1:]                                      # 取特殊位置的输出
```

一步步对照论文：

| 论文公式 | 代码 |
|---|---|
| $e_{rl} = e_\phi(\texttt{<rl>})$ | `rl_token_embed`（可学习参数）+ `rl_token_pos_enc` |
| $[z_{1:M}, e_{rl}]$ | `torch.cat([prefix_tokens, rl_tokens], dim=1)` |
| $z_{rl} = g_\phi([z_{1:M}, e_{rl}])_{M+1}$ | `return x[:, -1:]` |
| $z_{rl}$ 是 $1\times2048$ | `z_dim = embed_dim = 2048`；`encode_flat()` 把 `[B,1,2048]` 压成 `[B,2048]` |

Block 结构（`RLTSelfAttentionLayer`）：pre-LN 残差自注意力（`nn.MultiheadAttention`）+ GeGLU MLP；默认 **2 层、8 头、hidden 2048、mlp_ratio 4.0**。为了配合 FSDP，`RLTSelfAttentionLayer` 被加进了 `_no_split_modules`（`pi0.py:636-637`）。

### 3.4 Decoder：自回归重建（teacher forcing + causal mask）

`RLTTokenDecoder.forward`：

```python
frozen_targets  = target_embeddings.detach()          # ← stop-gradient
shifted_targets = frozen_targets[:, :-1]              # 右移一位
decoder_inputs  = torch.cat([rl_tokens, shifted_targets], dim=1)
causal_mask = torch.triu(ones(N, N, dtype=bool), diagonal=1)   # 上三角为 True = 屏蔽
...
return self.output_proj(x)                            # Linear(embed_dim, input_dim) = h_phi
```

对应论文

$$\mathcal{L}_{ro} = \mathbb{E}_{\mathcal{D}}\Big[\sum_{i=1}^{M}\big\| h_\phi\big(d_\phi([z_{rl}, \mathrm{sg}(z_{1:i-1})])\big)_i - \mathrm{sg}(z_i) \big\|_2^2\Big]$$

- **teacher forcing**：decoder 输入是 `[z_rl, sg(z_0), sg(z_1), ..., sg(z_{M-2})]`，第 i 个输出位置预测 $z_i$ —— 用 `shifted_targets` + causal mask 实现，位置 0 是 `z_rl`。
- **causal mask**：`torch.triu(..., diagonal=1)` 上三角为 True（`MultiheadAttention` 里 True 表示禁止），保证不会看到未来 target。单测 `test_decoder_causal_mask_blocks_future_teacher_targets`（`tests/unit_tests/test_models.py:157`）专门验证这一点。
- **重建目标**：就是 VLA 的 prefix hidden states，不是 logits、不是动作。`h_phi` 就是最后的 `output_proj`（`Linear(2048, 2048)`）。

### 3.5 stop-gradient 与"信息瓶颈"

代码里有**三处 detach**，这是理解整个设计的钥匙：

| 位置 | 代码 | 作用 |
|---|---|---|
| 传入 RLT 的 prefix | `pi0.py:819` `prefix_out.detach()` | $\mathcal{L}_{ro}$ 不回传到 VLA 主干 |
| encoder 输入 | `rlt_token_transformer.py:358` `frozen_prefix = prefix_embs.detach()` | 同上，双保险 |
| decoder target | `:243` `target_embeddings.detach()`、`:367` `target = prefix_embs.detach()` | 重建目标是常量 |

于是：**$\theta_{vla}$ 只被 $\alpha\mathcal{L}_{vla}$ 更新，$\phi$ 只被 $\mathcal{L}_{ro}$ 更新**。这与论文"关于 $\mathcal{L}_{ro}$ 把 VLA 视作冻结"的写法一致。

重建损失本体（`RLTTokenTransformer.loss`）：

```python
sq_error = torch.square(reconstructed - target)
if mask is not None:                      # 带 padding mask 的均值
    sq_error = sq_error * mask_expanded
    mse = sq_error.sum() / torch.clamp(mask_expanded.sum() * prefix_embs.shape[-1], min=1.0)
else:
    mse = sq_error.mean()
```

就是**平凡 MSE**，没有额外 head、没有 KL、没有 VQ/量化项。

**为什么这样能学到好表征？** 因为 decoder 只能看到 $z_{rl}$（1×2048）——它被迫成为 VLA 表征的**充分统计量**；2048 维的瓶颈又逼它丢掉与"生成动作"无关的噪声。单测 `test_reconstruct_detaches_targets_but_trains_encoder_and_decoder`（`test_models.py:214`）验证了 `prefix_embs.grad is None` 而 encoder/decoder 有梯度。

### 3.6 Stage 1 关键超参

| 项 | 值 | 出处 |
|---|---|---|
| `rlt_input_dim` / `rlt_embed_dim` | 2048 / 2048 | Stage 1 / Stage 2 YAML |
| `rlt_num_layers` / `rlt_num_heads` / `rlt_mlp_ratio` | 2 / 8 / 4.0 | 同上 |
| `rlt_prefix_seq_len` | 1024（dataclass 默认 768） | 超过会直接 `ValueError` |
| `rlt_alpha` | 1.0 | `rlt_config.py:31` 默认 1.0 |
| `rlt_image_only` | `False`（代码默认 `True`） | 官方配置显式覆盖 |
| `rlt_use_mask` | `True` | 用 padding mask |
| VLA 主干 | OpenPI π0.5（`pi05_base`，`paligemma_variant: gemma_2b`，`action_expert_variant: gemma_300m`） | 论文用 π0.6 |
| VLA 是否冻结 | **不冻结**（`train_expert_only: False`，`_freeze_after_load` 只在 `task in ("rl","dsrl")` 时冻结） | 联合训练 |

### 3.7 Stage 1 与论文的差异

1. **VLA 主干是 π0.5 而不是论文的 π0.6**，且 RLT 的 encoder 读的是 LLM prefix hidden states（2048 维），论文表述为"最后一层 token 嵌入"——实质相同。
2. `rlt_image_only` **代码默认 `True`（丢语言）**，论文是保留全部 token；靠配置覆盖。换配置时容易踩坑。
3. `rlt_encoder_type: "append_self_attention"` 这个键在**所有** Stage 1/2/e2e YAML 里都有，但**没有任何 Python 代码读它**，是死键（`build_rlt_config` 里没有它）。
4. `rl_token_embed` 是用正弦编码**初始化**的可学习参数，且额外有一个可学习的 `rl_token_pos_enc`；论文只说要"可学习 $e_{rl}$"，这属于实现选择。
5. MLP block 是 3 层 GeGLU，不是普通 2 层 MLP（论文未规定 block 内部）。

---

## 4. 阶段二：在线 RL 精修动作块

Stage 2 有两个 head 实现：

| 配置 | `loss_type` | `actor.model.model_type` | Worker | 定位 |
|---|---|---|---|---|
| AC 主线 | `rlt_ac` | `rlt_mlp_policy` | `RLTACFSDPPolicy` / `AsyncRLTACFSDPPolicy` | 论文的主方法 |
| TD3 变体 | `rlt_td3` | `rlt_td3_mlp_policy` | `RLTTD3FSDPPolicy` | 代码里自称 "ablation-style"，只有 ManiSkill 配置 |

### 4.1 状态怎么拼

冻结 VLA 的 `extract_rlt_obs()`（`openpi/tasks/eval.py:121-164`）一次返回三个东西：

```python
return {
    "z_rl":      z_rl,        # [B, 2048]，RL Token 读出
    "proprio":   proprio,     # [B, d_p]，本体感受
    "ref_chunk": ref_chunk,   # [B, H, action_dim]，VLA 参考动作块
}
```

然后（`rlt_mlp_policy.py:130-133`）：

```python
actor_state  = cat([ref_chunk, z_rl, proprio], dim=-1)   # 顺序：参考块在最前
critic_state = cat([z_rl, proprio], dim=-1)              # critic 不看参考块
```

对应论文 $x = (z_{rl}, s_p)$ 与 $\pi_\theta(a_{1:C}\mid x, \tilde a_{1:C})$。**注意**：`ref_chunk` 会被截断到 RL 的块长 —— `_get_ref_chunk()` 先 reshape 再 `[:, :chunk_len]`，即取 VLA 开环执行前缀（论文也是这么做的）。

#### `batch["curr_obs"]` 到底是什么（形状与含义）

⚠️ **它是一个 dict，不是单个 tensor**。三个 key 由 `rlinf/algorithms/rlt/transition.py:22` 固化成常量：

```python
RLT_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")
```

以仿真 ManiSkill 为例（`maniskill_rlt_stage2_ac_mlp.yaml:337-338`：`z_dim: 2048`、`proprio_dim: 9`；`:334-336`：`action_dim: 8`、`num_action_chunks: 10`）：

| key | 含义 | 形状 | 仿真实际值 |
|---|---|---|---|
| `z_rl` | **RL Token**——冻结 VLA 内部表征蒸馏出的紧凑向量 | `[B, z_dim]` | `[B, 2048]` |
| `proprio` | 本体感受（关节角 / 夹爪状态） | `[B, proprio_dim]` | `[B, 9]` |
| `ref_chunk` | **VLA 参考动作块**，**扁平**存储 | `[B, ref_num_action_chunks × action_dim]` | `[B, 80]` |

**语义**：一条 `curr_obs` = "**某个环境、在某个动作块边界的观测**"，即 actor 要输出这一块动作时"看到的东西"。`z_rl` 是核心——论文的关键设计就是"**actor-critic 不直接看图像**"，只看这个 2048 维向量 + 本体感受（5.4 节"不存原始图像"）。

#### `B` 是多少——最容易搞错的地方

**`B` 不等于环境数**，而是 **`actor.micro_batch_size`**（每卡）。追踪数据流：

1. **采集时**：`Trajectory.curr_obs[*]` 形状是 `[T, B_env, ...]`（`T` = 轨迹长度/块数，`B_env` = 并行环境数）；
2. **写 replay 时摊平**（`rlinf/data/storage/replay/buffer.py:765-769`）：

```python
flat["curr_obs"][key] = tensor.reshape(-1, *tensor.shape[2:])   # [T, B_env] → 压成一维
```

  于是每条记录 = **一个 (环境, 时间点) 的动作块转移**；

3. **采样时按索引取**（`_extract_chunk_from_flat_trajectory`，`:786-799`）→ 堆成 batch。

所以 $B = \texttt{actor.micro\_batch\_size}$。对照配置 `:322-323`：`micro_batch_size: 128`、`global_batch_size: 512`；若 world_size = 4，则每卡 `B = 512 / 4 = 128`。

进入模型后（`rlt_mlp_policy.py:120-133`）：

```text
actor_state  = [ref_chunk | z_rl | proprio]  = 80 + 2048 + 9 = [B, 2137]
critic_state = [z_rl | proprio]              = 2048 + 9      = [B, 2057]
```

⚠️ **`curr_obs` 在 CPU 上**：`TrajectoryStep.__post_init__`（`embodied_types.py:663-666`）会把 `forward_inputs` / `curr_obs` / `next_obs` 放到 CPU，worker 里需要手动搬 GPU。

⚠️ **同一个 `curr_obs` 被两处复用**：`forward_critic:238` 和 `forward_actor:305` 都取它，但 **critic 只用 `_critic_state` 挑出的 `[z_rl, proprio]`，actor 还要额外吃 `ref_chunk`**。

### 4.2 Actor 网络

`RLTMLPPolicy` 继承 `MLPPolicy`：

```python
self.backbone = nn.Sequential(
    layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
    layer_init(nn.Linear(256, 256)),     nn.Tanh(),
    layer_init(nn.Linear(256, 256)),     nn.Tanh(),
)
self.actor_mean = layer_init(nn.Linear(256, num_action_chunks * action_dim), std=0.01*sqrt(2))
```

前向（`rlt_mlp_policy.py:146-158`）：

```python
feat        = self.backbone(actor_state)
action_mean = self.actor_mean(feat)
action_std  = torch.full_like(action_mean, self.fixed_std)   # 固定标准差
probs       = Normal(action_mean, action_std)
action      = action_mean if deterministic else probs.rsample()
chunk_logprobs = probs.log_prob(action)
action      = torch.tanh(action)                             # 输出压到 [-1,1]
```

- **3 层 256 + tanh** 的 MLP + 线性输出头（论文写的是 2 层 256；TD3 变体才是 2 层）。
- **固定小标准差** `fixed_std: 0.002`（论文"固定小标准差"）✅。`MLPPolicy` 里虽然建了一个 `actor_logstd`，但 RLT 路径**从不使用**它（死参数，仍会进优化器）。
- `log_prob` 是在 **tanh 之前**算的（没有做 Jacobian 修正），而且**只用于记 metric**，不进损失。
- 输出维度 = `num_action_chunks × action_dim`（仿真 10×8=80；真机 10×7=70）。

### 4.3 Critic 与 chunk 级 TD

> **本节的关键前提**：RLT 的一条训练样本不是单步转移，而是 **chunk 级转移**。actor 一次输出整个动作块 $a_{1:C}$，环境连续执行 C 步，所以一条样本长这样：

```text
(curr_obs,  a_{1:C},  r_{1:C},  next_obs,  done)
             └ C 个动作   └ C 个奖励      └ 执行完 C 步后的状态
```

其中 `C = actor.model.num_action_chunks`（仿真 10、真机 10）。本节的"chunk 级 TD"就是围绕"一次跨越 C 步"展开的。

**先对比普通 TD 和 chunk 级 TD。** 普通单步 TD（TD3/SAC 的标准写法）是：

$$y = r_t + \gamma\,Q_{\psi'}(s_{t+1}, a_{t+1})$$

意思是"这一步的价值 = 这一步的即时奖励 + 打折后的下一步价值"。RLT 把它换成"跨越整个 chunk"：

$$\hat Q = \underbrace{\sum_{t'=1}^{C}\gamma^{t'-1}r_{t'}}_{\text{chunk 内 C 步奖励逐步打折}} \;+\; \underbrace{\gamma^{C}}_{\text{跨 C 步折扣}}\;\underbrace{\mathbb{E}_{a'\sim\pi_\theta}\big[Q_{\psi'}(x',a')\big]}_{\text{块末状态的价值}}$$

大白话：**"一整块动作值多少分" = "这 C 步马上拿到的分、按时间打折加起来" + "$\gamma^C$ 打折后的、块结束后那个状态的价值"。** 中间的 C 步状态和奖励被一次性吸收进这一个 target。

**为什么这一步很关键。** 假设任务有 200 步、奖励只在最后成功时给 1 分：单步 TD 要 backup 约 200 次才能把最后那 1 分传回起点；chunk 级 TD（C=10）一次 backup 跨 10 步，只需约 20 个决策点。**奖励传播路径缩短了 C 倍**——这就是下面说的"稀疏成功奖励只需跨过 $T/C$ 个决策点就能传回起点"，也是 RLT 能学会的根本原因之一。

**代码逐行拆解**（`fsdp_rlt_ac_policy_worker.py:233-299`）：

```python
# 1) chunk 内折扣回报： Σ γ^{t'} r_{t'}（:87-95）
def _discounted_chunk_rewards(self, rewards):
    rewards = rewards.reshape(rewards.shape[0], -1)    # [B, C]
    chunk_len = rewards.shape[-1]                       # C
    discounts = gamma ** arange(chunk_len)              # [γ^0, γ^1, ..., γ^(C-1)]
    return torch.sum(rewards * discounts, dim=-1, keepdim=True)

# 2) 跨 C 步 bootstrap
with torch.no_grad():                                   # 算 target 时只查值，不回传梯度
    next_actions, _, _ = self._next_actions_for_critic_target(next_obs)
    all_qf_next_target = self.target_model(             # 用 target network ψ'，不是在线 critic
        forward_type=SAC_Q, obs=next_obs, actions=next_actions)
    q_next = self._min_twin_q(all_qf_next_target)       # 双 Q 取 min（:79-81）
    reward_target      = self._discounted_chunk_rewards(rewards)
    reward_horizon     = int(rewards.reshape(rewards.shape[0], -1).shape[-1])  # = C
    bootstrap_discount = gamma ** reward_horizon        # = γ^C
    target_q_values    = reward_target + not_done * bootstrap_discount * q_next

critic_loss = F.mse_loss(all_data_q_values, target_q_values.expand_as(all_data_q_values))
```

逐一说明：

- **`_discounted_chunk_rewards`**：算公式里 $\sum\gamma^{t'-1}r_{t'}$。注意折扣指数**从 0 开始**（块内第一步不打折），与论文一致。
- **`torch.no_grad()`**：target 只用来当"老师"，不产生梯度。
- **`self.target_model`**：**target network** $\psi'$，TD3 那套"慢半拍的旧副本"，防止自举发散。
- **`_min_twin_q`**：两个 Q 头取 `torch.minimum(Q[...,0:1], Q[...,1:2])`，即 TD3 的 Twin 机制。
- **`bootstrap_discount = γ^C`**：跨了整个 chunk，所以打 $\gamma^C$ 折。
- **`not_done`**：把块内 C 个 `done` 压成一个布尔值——**只要这块里任意一步终止，整块就不 bootstrap**（`:246-248`）。因为 episode 中途结束，`next_obs` 就没有意义了。
- 代码另有 `bootstrap_type`：`"standard"`（用 `not_done`，默认）和 `"always"`（无视 done）。
- **`critic_loss`**：在线两个 Q 头对**当前** `(curr_obs, 动作块)` 的估计形状是 `[B, 2, 1]`，target 是 `[B, 1]`，用 `expand_as` 广播到两个头——**两个 Q 头一起拟合同一个 target**（取 min 只用在 target 上，和 TD3 一样）。

**代一个具体数字。** 取 C=4、γ=0.99，某条样本块内奖励 `[0,0,1,0]`（第 3 步成功）、块末价值 `q_next=5.0`、未终止：

- 折扣向量 `[1, 0.99, 0.9801, 0.9703]`
- `reward_target = 0·1 + 0·0.99 + 1·0.9801 + 0·0.9703 = 0.9801`
- `bootstrap_discount = 0.99^4 = 0.9606`
- **`target = 0.9801 + 0.9606 × 5.0 = 5.7831`**

然后让两个在线 Q 头对 `(curr_obs, a_{1:4})` 的估计去逼近 5.7831。

#### critic loss 的完整计算链

把 `forward_critic`（`fsdp_rlt_ac_policy_worker.py:233-299`）从头到尾串一遍：

```python
233:  @Worker.timer("forward_critic")
234:  def forward_critic(self, batch):
235:      use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
236:      bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
237:
238:      curr_obs = batch["curr_obs"]
239:      next_obs = batch["next_obs"]
240:      actions  = batch["actions"]
241:      rewards  = batch["rewards"]
242:      done_source = batch["terminations"]
243:      if use_simulator_transition_replay(self.cfg):
244:          done_source = batch["dones"]
245:      done_source = done_source.to(self.torch_dtype)
246:      not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
247:          dim=-1, keepdim=True)
248:
249:      # ── (1) target 计算：整段 no_grad ───────────────────────────
250:      with torch.no_grad():
251:          next_actions, _, _ = self._next_actions_for_critic_target(next_obs)
252:          all_qf_next_target = self.target_model(
253:              forward_type=ForwardType.SAC_Q, obs=next_obs, actions=next_actions)
254:          q_next = self._min_twin_q(all_qf_next_target)          # min(Q'₁, Q'₂)
255:
256:          reward_target = self._discounted_chunk_rewards(rewards)     # Σ γ^t' r
257:          reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
258:          bootstrap_discount = self.cfg.algorithm.gamma ** reward_horizon  # γ^C
259:          if bootstrap_type == "always":
260:              target_q_values = reward_target + bootstrap_discount * q_next
261:          elif bootstrap_type == "standard":
262:              target_q_values = reward_target + not_done * bootstrap_discount * q_next
263:
264:      # ── (2) 在线 Q 对当前 (s, a) 估值 ───────────────────────────
265:      all_data_q_values = self.model(
266:          forward_type=ForwardType.SAC_Q, obs=curr_obs, actions=actions)   # [B, 2]
267:
268:      # ── (3) MSE：两个头一起拟合同一个 target ────────────────────
269:      target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
270:      critic_loss = F.mse_loss(
271:          all_data_q_values, target_q_values.expand_as(all_data_q_values))
272:      return critic_loss, {"q_data": all_data_q_values.mean().item()}
```

（上面的行号是逻辑顺序的压缩呈现，实际代码里 `crossq` 分支穿插其间，见下方 ⚠️。）

**三个阶段的要点**：

**(1) target 阶段整段在 `torch.no_grad()` 里**——target 只当"老师"，不产生梯度。梯度只能从 `critic_loss` 流回**在线的** `self.model`。

**(2) 在线 Q 用 `self.model`，target Q 用 `self.target_model`**——注意 `:254` 用的是 `self.target_model`（target critic，$\psi'$），而 `:251` 的下一动作来自 `self.model`（在线 actor）。这一"半 target、半在线"的组合就是 4.3 节标 ⚠️ 的地方。

**(3) `expand_as` 让两个 Q 头共享同一个 target**：

```python
all_data_q_values    # [B, 2]     在线两个头
target_q_values      # [B, 1]     标量 target
target.expand_as(...) → [B, 2]    广播到两个头
F.mse_loss(...)      # 对标量：mean((Q1-y)² + (Q2-y)²) 的等价形式
```

**⚠️ 一个容易误解的点**：`F.mse_loss` 默认 `reduction="mean"`，会**对所有元素求平均**（两个头一起）。所以实际损失是

$$\mathcal{L}_Q = \frac{1}{2B}\sum_{b}\Big[\big(y_b - Q_{\psi_1}(s_b,a_b)\big)^2 + \big(y_b - Q_{\psi_2}(s_b,a_b)\big)^2\Big]$$

即与 TD3 论文的 $\sum_{i=1,2}(y-Q_{\psi_i})^2$ **只差一个常数因子 2**（不影响梯度方向，只影响有效学习率）。这和 `TD3_通俗讲解.md` 3.3 节那个"两个 Q 头都去拟合同一个 target"完全一致——**取 min 只发生在 target 上，不发生在损失里**。

**`done_source` 的选择**（`:242-244`）：仿真走 `batch["dones"]`（`use_simulator_transition_replay`），其余走 `batch["terminations"]`。两者语义不同：

| `bootstrap_type` | 行为 | 何时用 |
|---|---|---|
| `"standard"` | 终止也停 bootstrap（`not_done` 生效） | **RLT 三份配置都是这个** |
| `"always"` | 无条件 bootstrap（无视 done） | 很多 PPO/SAC 配置用 |

> 注意 `"always"` / `"standard"` 的实际语义在文档其它地方有更细的解释（`maniskill_ppo_openpi_pi05.yaml:53` 的注释）：`"standard"` 只在 truncation 时 bootstrap，`"always"` 在 truncation 或 termination 时都 bootstrap。RLT 用 `standard` 是因为**仿真里 episode 超时截断很常见**，不该把截断当成真终止。

**返回的两个东西**（`:299`）：

```python
return critic_loss, {"q_data": all_data_q_values.mean().item()}
```

- `critic_loss` 带梯度，交给训练循环 `.backward()`；
- `q_data` 只是 metric（`Q(s,a)` 的均值），**循环中的变量名是 `q_data`**，与 actor 那边的 `q_pi`（`Q(s,π(s))`）对应——这两个值的差（`q_data - q_pi`）在诊断上有意义：它反映"数据里的动作"和"策略当前动作"的估值差距。

#### target 权重的更新：`soft_update_target_model`

> **代码位置**：这个方法**不在 RLT 的文件里**，而是 **`rlinf/workers/actor/fsdp_sac_policy_worker.py:273-319`** 的 `EmbodiedSACFSDPPolicy.soft_update_target_model()`。RLT 通过继承拿到它（`fsdp_rlt_ac_policy_worker.py:672`）：

```python
class RLTACFSDPPolicy(RLTACLossMixin, RLTACReplayMixin, EmbodiedSACFSDPPolicy):
```

公式就是 **Polyak 软更新**（`TD3_通俗讲解.md` 6.4 节）：

$$\psi' \leftarrow (1-\tau)\,\psi' + \tau\,\psi \qquad (\tau = 0.005)$$

即"**每步只往在线网络挪 0.5%**"，让 target 像影子一样平滑跟随，没有硬更新的跳变台阶。

```python
273:  def soft_update_target_model(self, tau: Optional[float] = None):
280:      if tau is None:
281:          tau = self.cfg.algorithm.tau          # RLT: 0.005
283:      assert self.target_model_initialized
285:      with torch.no_grad():
286:          if not hasattr(self, "_target_shadow_f32"):        # 普通路径
288:              for (name1, online_param), (name2, target_param) in zip(
289:                  self.model.named_parameters(),
290:                  self.target_model.named_parameters()):
292:                  assert name1 == name2                      # 靠参数名一一对应
293:                  if "q_head" not in name1:
294:                      if self.target_update_type == "all":
295:                          target_param.data.mul_(1.0 - tau)
296:                          target_param.data.add_(online_param.data * tau)
297:                      else:
298:                          target_param.data.mul_(0.0)        # 非 q_head 参数：硬拷贝
299:                          target_param.data.add_(online_param.data)
300:                  else:
301:                      target_param.data.mul_(1.0 - tau)          # ★ q_head：EMA
302:                      target_param.data.add_(online_param.data * tau)
```

**四个关键细节**：

1. **靠参数名配對**（`:288-292`）：`zip(named_parameters())` + `assert name1 == name2`——两个模型的参数命名和顺序必须严格一致。
2. **`"q_head"` 特判**（`:293`）：只有 `q_head.*`（两个 Q 头）走 EMA；其余参数（actor 部分）默认**硬拷贝**，除非 `target_update_type == "all"`。**对 RLT 来说这部分意义不大**——AC 主线根本没有 target actor（4.3 节 ⚠️）。
3. **全程 `torch.no_grad()`**：target 永不参与反向传播，与上面"被梯度训练的只有 actor + 2 个 Q 头"一致。
4. **`requires_grad_(False)`**（`:117`）：target model 建出来时就把梯度关掉，这是更上一层的保证。

**bf16 精度保护（DSRL 路径）**：如果存在 `_target_shadow_f32` 属性，就走 `:303-319` 的另一条分支：

```python
269:  self._target_shadow_f32 = {}                              # 由 _init_target_shadow() 建立
270:  for name, param in self.target_model.named_parameters():
271:      self._target_shadow_f32[name] = param.data.float().clone()
...
315:  shadow = self._target_shadow_f32[name1]
316:  shadow.mul_(1.0 - tau).add_(online_param.data.float(), alpha=tau)  # 在 fp32 累积
317:  target_param.data.copy_(shadow.to(target_param.data.dtype))        # 再拷回 bf16
```

**为什么需要它**：`τ = 0.005` 意味着每次只挪 0.5%，而 **bf16 只有 8 位尾数（ULP ≈ 0.4%）**——更新幅度和舍入误差同量级，**直接存回 bf16 会把更新量舍掉一大半**（注释 `:265-267` 明说了这点）。用一个 float32 影子持有累积 EMA 状态（ULP ≈ 3.6e-8）就能避免精度损失，bf16 只当"呈现格式"。

**调用时机**（`fsdp_sac_policy_worker.py:664-669`，在每个 update step 末尾）：

```python
664:  # Soft update target network
665:  if (self.target_model_initialized
667:      and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0):
669:      self.soft_update_target_model()
```

RLT 配置 `target_update_freq: 1`（`maniskill_rlt_stage2_ac_mlp.yaml:60` 附近）→ **每一次 `update_one_epoch` 之后都软更新一次**。

**初始化时的一次硬拷贝**（`fsdp_sac_policy_worker.py:74`）：

```python
74:  self.soft_update_target_model(tau=1.0)     # τ=1 → ψ' ← ψ，等于整体复制
```

`tau=1.0` 让 EMA 退化成"直接赋值"，这样 target 一开始就和在线网络一致（否则是随机初始化，target 会给出无意义的值）。这也解释了 9.4 节表格里的 "`initialize_target=True`"。

**target model 是怎么建出来的**（`:114-118`）：

```python
114:  self.target_model = self._strategy.wrap_model(...)   # 同结构的独立副本
117:  self.target_model.requires_grad_(False)             # ★ 不参与梯度
118:  self.target_model_initialized = True
```

**📌 一张图看 target 的作用**：

```text
            每次 update step
   ┌──────────────────────────────────────────┐
   │ ① critic 更新（梯度）                     │
   │    y = Σγ^t'r + γ^C·min(Q'₁,Q'₂)  ← 用 target_model 算
   │    loss = MSE(在线 Q(s,a), y)     ← 更新在线 model
   │         ↓ backward                        │
   │    在线 Q1、Q2 各挪一步                    │
   ├──────────────────────────────────────────┤
   │ ② target 软更新（无梯度，Polyak）          │
   │    ψ'₁ ← 0.995·ψ'₁ + 0.005·ψ₁            │
   │    ψ'₂ ← 0.995·ψ'₂ + 0.005·ψ₂            │
   └──────────────────────────────────────────┘
```

**为什么要分这两步？** 因为如果直接用在线 Q 算 target，就是"**用自己刚更新的估计当自己的老师**"，会形成自我强化的发散循环（`TD3_通俗讲解.md` 6.1 节"考试不能自己批自己的卷子"）。target network 提供了一个**滞后的稳定锚**，每步只跟随 0.5%，时间常数约 $1/\tau = 200$ 步。

**Critic 网络结构。** `MultiQHead(hidden_dims=[256,256,256], num_q_heads=2)`（`rlinf/models/embodiment/modules/q_head.py:120`）是一个 `nn.ModuleList`，装 **2 个完全独立**的 `QHead`，每个输出 1 个标量，最后 `cat` 成 `[B, 2, 1]`。每个 `QHead`（`:22`）做的事是：

```python
x = torch.cat([state_features, action_features], dim=-1)  # [critic_state, flat 动作块]
q = self.net(x)        # [256,256,256,1]，LayerNorm + tanh，最后一层无激活
```

即输入是 `critic_state`（4.1 节的 `[z_rl, proprio]`，critic 不看参考块）拼上展平的动作块（C×action_dim），过 3 层 LayerNorm+tanh 的 MLP 输出一个标量。两个 Q 头要**独立初始化、独立训练**，它们的误差才不完全相关，取 min 才有意义。

**该节标记与论文对照：**

- ✅ **chunk 级 bootstrap**（论文最关键的设计之一）：稀疏成功奖励只需跨过 $T/C$ 个决策点就能传回起点。
- ✅ 双 Q 集成 + target 取 min；MSE 损失两个头一起拟合同一 target。
- ✅ target network：`tau=0.005` 的 Polyak 软更新，`target_update_freq: 1`。
- ⚠️ **AC 路径里 $a'$ 来自在线 actor `self.model`**（`self._next_actions_for_critic_target`，`fsdp_rlt_ac_policy_worker.py:227-231`），而不是 target actor；TD3 变体才用 `self.target_model` + 目标策略平滑噪声（`fsdp_rlt_td3_policy_worker.py:28-32`）。这是 AC 主线**不严格等价于 TD3** 的地方。

**一句话总结**：critic 是两个独立 Q 网络，输入 `[RL token + 本体感受]` 和整个动作块，输出"这一块动作值多少分"；训练目标 = 块内 C 步奖励折扣和 + $\gamma^C$×块末状态的 target 价值，两个 Q 头都去拟合、取 min 防高估。与普通 TD3 的唯一本质区别就是"一次 backup 跨 C 步"——奖励传播快 C 倍；以及下块动作取自在线 actor 而非 target actor 这个小瑕疵。

### 4.4 Actor 损失与 BC 正则

> **这一节回答的问题**：actor 网络拿到 `[ref, z_rl, proprio]` 之后发生了哪些前向？两个 Q 头为什么"算两个、只用一个"？BC 的目标从哪来、被什么改写？`bc_weight` / `q_weight` 是怎么调的？

**先说这一节在解决什么问题。** 4.3 节把 critic 训好了，现在要用 critic 去改 actor。但 RLT 的 actor 不能像普通 TD3 那样"无约束地往 Q 高处爬"——因为真机数据少、Q 曲面有很多 4.2/5.x 节说的假尖峰，纯追 Q 会立刻跑飞。所以 RLT 的 actor 目标是**"在 VLA 参考动作附近追 Q"**，这也是论文最核心的改写之一。

$$\mathcal{L}_\pi(\theta) = \mathbb{E}\Big[-\,Q_\psi(x,a_{1:C}) \;+\; \beta\,\big\|a_{1:C} - \tilde a_{1:C}\big\|_2^2\Big]$$

读法：**第一项让动作"更值钱"（追 Q），第二项让动作"别离 VLA 的建议太远"（BC 正则）。** 前者是 TD3 就有的，后者是 RLT 新增的。$\tilde a$ 就是 VLA 给的参考动作块。

**代码逐行拆解**（`fsdp_rlt_ac_policy_worker.py:301-368`）：

```python
pi, log_pi, _ = self.model(forward_type=SAC, obs=curr_obs,
                           apply_reference_dropout=True,
                           reference_dropout_prob=reference_dropout_prob)
all_qf_pi = self.model(forward_type=SAC_Q, obs=curr_obs, actions=pi, detach_encoder=True)
qf_pi = self._q1(all_qf_pi)                       # 用第 1 个 Q 头
ref_chunk = self._ref_chunk(curr_obs)
bc_loss, rlt_metrics = self._bc_metrics(pi, batch["actions"], ref_chunk,
                                        batch.get("intervene_flags", None))
bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
```

1. **第一次前向：拿当前动作 `pi`**。走 actor 前向（`rlt_mlp_policy.py:138 sac_forward`），从固定标准差 0.002 的高斯里 `rsample()` 采样再 `tanh`。注意 `apply_reference_dropout=True`——**训练 actor 时按 `reference_dropout_prob` 随机把参考块置零**（见 4.5）。
2. **第二次前向：让 critic 给 `pi` 打分**。`actions=pi` 送进 critic，得到 `all_qf_pi`，形状 `[B, 2, 1]`。`detach_encoder=True` 表示 critic 那一侧的 state 特征不回传梯度——**actor 的梯度只从 Q 对动作的偏导走**，不透过 critic 去改表征。
3. **`qf_pi = self._q1(all_qf_pi)`**：只用第 1 个 Q 头，不取 min（见下面的 ⚠️）。
4. **BC 项**：`_bc_metrics` 拿 `pi`、replay 里实际执行的 `batch["actions"]`、以及 `ref_chunk` 算 MSE。
5. **权重**：`_actor_objective_weights()` 返回一对系数 `(bc_weight, q_weight)`。
6. **合成损失**：`-q_weight * Q + bc_weight * BC`。取负号是因为要**最大化** Q，而优化器在做**最小化**。

**BC 项的三个细节**（`_bc_metrics`，`:97-146`）：

- **是平方 L2（MSE），不是 KL，也不是 log-likelihood**：`bc_error = mean(square(pi_chunk - bc_target), dim=-1)`（`:125`）。先在动作维上求均值得到"每个时间步的误差"，再对所有样本和时间步求总均值（`:126`）。
- **BC 的 target 会被人干预改写**（`:124`）：

```python
bc_target = torch.where(human_mask[..., None], action_chunk, bc_ref_chunk)
```

- 该 chunk 有 `intervene_flags` 时，BC 目标变成**实际执行的动作**（人类/专家的）；否则才是 VLA 的 `ref_chunk`。这等价于论文的"发生干预时把 buffer 里的参考块替换成人类动作"——即**人类接管过的地方，就直接模仿人类**。
- **单独统计两种误差**：`ref_error`（对 VLA 参考块的误差）和 `human_error`（对实际动作的误差）分别只在各自的 mask 内求均值，产出 `bc_ref_loss` / `bc_human_loss` / `human_mask_ratio` 三个 metric（`:128-145`）。注意它们**只是观察量**，真正进损失的是按 mask 拼起来的那个 `bc_loss`。
- **mask 的粒度是"时间步"**：`intervene_flags` 先 reshape 成 `[B, C, action_dim]` 再 `.any(dim=-1)`（`:120-121`），即**只要这一步有任意一个动作维被干预，整步就按人类动作算**。

**权重 `bc_weight` / `q_weight` 的命名映射与调度**（`_actor_objective_weights`，`:148-225`）：

- 论文的 $\beta$ → 代码的 `bc_weight`；代码额外多了一个 Q 项系数 `q_weight`（论文默认等价于 1）。
- **关掉调度时**（`actor_weight_schedule.enable: False`）直接用 `algorithm.bc_weight` / `algorithm.q_weight`。
- **打开调度时**（`enable: True`）分三段：

| 阶段 | 条件 | 取值 |
|---|---|---|
| warmup | `update_step < warmup_updates` | `warmup_bc_weight` / `warmup_q_weight` |
| ramp | 之后 `ramp_updates` 步内线性插值 | 从 warmup 值 → online 值 |
| online | ramp 结束 | `online_bc_weight` / `online_q_weight` |

**为什么要有这个调度**：训练刚开始时 critic 是随机噪声，**这时候信 Q 就是灾难**，所以要"重 BC、轻 Q"（仿真 warmup 是 `bc=7.0, q=0.05`）；等 critic 学准了，再逐步把话语权交给 Q（online 是 `bc=2.5, q=0.45`）。这与 TD3 的"延迟策略更新"、以及整篇文档反复强调的"actor 别追着噪声跑"是同一个思路，只是从"更新频率"换成了"损失权重"。

**各配置的实际取值：**

| 配置 | 固定权重 | 调度 |
|---|---|---|
| 仿真 `maniskill_rlt_stage2_ac_mlp.yaml:63-75` | `q=1.0, bc=1.0`（被调度覆盖） | ✅ warmup 20k（7.0 / 0.05）→ ramp 50k → online（2.5 / 0.45） |
| 真机 `realworld_rlt_stage2_ac_mlp.yaml:75-76` | `q=0.1, bc=5` | ❌ 没配 `actor_weight_schedule`，用固定值 |

真机之所以"重 BC 轻 Q"（5 : 0.1 = 50:1），是因为真机数据最贵、最怕跑飞；而仿真默认那对 `1.0 / 1.0` 实际会被调度覆盖掉，只在关闭调度时生效。

**没有熵项**：`forward_alpha` 直接 `raise NotImplementedError`（`:370-376`），配置一律 `alpha_type: fixed_alpha, initial_alpha: 0.0`、`backup_entropy: False`。代码里 `entropy = -log_pi.mean()` 只是**记 metric**，不进任何损失——所以它**不是**最大熵 SAC。

**⚠️ AC 路径硬编码用 Q1**（`_q1`，`:341`），配置里的 `actor_agg_q` 对它无效；只有 TD3 变体才会按 `min/q1/mean` 聚合。这一点和 TD3 的常见实现约定一致：**target 上取 min 降偏，actor 上只用 Q1 以免过度保守**（见 `TD3_通俗讲解.md` 第 7 节注 7）。

**一句话总结**：actor 目标 = "**在 VLA 参考动作附近追 Q**"——`-q_weight·Q1(在线actor采样的动作) + bc_weight·MSE(动作, 参考块或人类动作)`；梯度只从 Q 对动作的偏导流回 actor（critic 特征被 detach），没有熵项，权重按 warmup→ramp→online 从"重 BC"切换到"重 Q"。

---

### 4.4.1 调用链：`self.model(...)` 到底跳到哪

4.4 节那两行 `self.model(forward_type=...)` 是**分派调用**，每次都走同一条三跳路径：

```text
① fsdp_rlt_ac_policy_worker.py:309   self.model(forward_type=SAC, obs=curr_obs, ...)
        ↓  nn.Module.__call__ → forward()
② mlp_policy.py:137  MLPPolicy.forward()        ← 按 forward_type 分派 + preprocess_env_obs
        ↓  forward_type == ForwardType.SAC
③ rlt_mlp_policy.py:138  RLTMLPPolicy.sac_forward()   ← 真正算动作
```

**第 ② 跳是统一入口**（`mlp_policy.py:137-160`），一张表对应一种前向：

| `ForwardType`（`base_policy.py:19-29`） | 落到哪个方法 |
|---|---|
| `SFT` | `sft_forward` |
| **`SAC`** | **`sac_forward`** |
| **`SAC_Q`** | **`sac_q_forward`** |
| `CROSSQ` / `CROSSQ_Q` | `crossq_forward` / `crossq_q_forward` |
| `DEFAULT` | `default_forward` |

同一跳里还会 `obs = self.preprocess_env_obs(obs)`。**RLT 覆写了这个函数**（`rlt_mlp_policy.py:81-86`）：基类版本只保留 `states` 一个 key（`mlp_policy.py:122-124`），RLT 版本遍历所有 key 逐个 `.to(device)`——因为 RLT 的 obs 是 `z_rl` / `proprio` / `ref_chunk` 三个字段，不能只留 `states`。

**第 ③ 跳是唯一 RLT 特化的一跳**：

```python
138:  def sac_forward(self, obs, apply_reference_dropout=False,
139:                  reference_dropout_prob=0.0, deterministic=False, **kwargs):
146:      actor_state = self._actor_state(obs, apply_reference_dropout=..., ...)
151:      feat = self.backbone(actor_state)              # 父类的 3 层 256 + tanh
152:      action_mean = self.actor_mean(feat)            # [B, 80]
153:      action_std  = torch.full_like(action_mean, self.fixed_std)   # 0.002
154:      probs = Normal(action_mean, action_std)
155:      action = action_mean if deterministic else probs.rsample()
156:      chunk_logprobs = probs.log_prob(action)        # tanh 之前，无 Jacobian 修正
157:      action = torch.tanh(action)                    # 压到 [-1, 1]
158:      return action, chunk_logprobs, None
```

返回的三元组正好对应上层解包 `pi, log_pi, _ = self.model(...)`：`_` 是恒为 `None` 的占位。

> ⚠️ **基类里也有一份 `MLPPolicy.sac_forward`**（`mlp_policy.py:179-200`），它用的是 `obs["states"]`——那是普通 MLP SAC 路径，**RLT 不走它**（RLT 的 obs 里根本没有 `states` 这个 key）。读代码时注意区分：`feat = self.backbone(obs["states"])` 属于父类，被 `RLTMLPPolicy.sac_forward` 覆盖掉了。

**对照：worker 里三次前向分别落在哪**

| 调用处 | `forward_type` | 落到哪 |
|---|---|---|
| `forward_critic:254` | `SAC_Q` | `sac_q_forward` → `q_head`（target 网络） |
| `forward_actor:309` | `SAC` | `sac_forward` → actor 出动作 |
| `forward_actor:320` | `SAC_Q` | `sac_q_forward` → 给 `pi` 打分 |

📌 **两个 `sac_forward` 的角色**：训练走 `forward_actor` 传 `apply_reference_dropout=True`；rollout/eval 走 `predict_action_batch`（`rlt_mlp_policy.py:206-222`），它直接调 `self.sac_forward(obs, deterministic=(mode=="eval"))`，**不传 dropout 参数**——这就是 4.5 节"训练 50% / 推理始终提供"的代码依据。

---

### 4.4.2 `_actor_state`：actor 的输入装配线

```python
120:  def _actor_state(self, obs, *, apply_reference_dropout=False,
126:                   reference_dropout_prob=0.0) -> torch.Tensor:
127:      ref_chunk = self._get_ref_chunk(obs)
128:      if apply_reference_dropout:
129:          ref_chunk = self._maybe_drop_reference(ref_chunk, reference_dropout_prob)
130:      return torch.cat([ref_chunk, self._get_z(obs), self._get_proprio(obs)], dim=-1)
```

**`*` 是 keyword-only 分隔符**：后两个参数必须用关键字传，防止 bool 和 float 位置写反——那种 bug 极难查。默认值 `False` / `0.0` 意味着**默认不做 dropout**。

**第一步：`_get_ref_chunk`**（`:100-105`）做"reshape → 截断 → 压平"：

```python
ref_chunk = self._flatten_batch(obs["ref_chunk"]).reshape(B, -1, self.step_action_dim)  # [B,80]→[B,10,8]
ref_chunk = ref_chunk[:, : self.chunk_len]        # 截断到 C=10
return ref_chunk.reshape(ref_chunk.shape[0], -1)  # → [B, 80]
```

`[:, :chunk_len]` 在 `ref_num_action_chunks > num_action_chunks` 时承担"**只取 VLA 开环执行的前 C 步**"的语义。

**第二步：`_maybe_drop_reference`**（`:107-118`）逐样本、**整块置零**（详见 4.5）。

**第三步：拼接**——`dim=-1` 上首尾相接：

| 片段 | 形状 | 维数 |
|---|---|---|
| `ref_chunk` | `[B, 80]` | 10 × 8 |
| `z_rl` | `[B, 2048]` | RL Token |
| `proprio` | `[B, 9]` | 本体感受 |
| **`actor_state`** | **`[B, 2137]`** | — |

**⚠️ 拼接顺序必须和 `__init__` 算 `obs_dim` 的顺序一致**，否则权重对不上。而且三个 head 的顺序各不相同（见第 8 节那条坑）：**AC actor 是 `[ref, z, proprio]`，AC critic 是 `[z, proprio]`，TD3 是 `[z, proprio, ref]`**。

**对照 `_critic_state`**（`:132-133`）：`torch.cat([z_rl, proprio], dim=-1)` → **`[B, 2057]`，不含参考块**。理由：参考块是"给策略的建议"，不是环境属性；critic 要评估的是动作本身在这个状态下值多少分（$Q_\psi(x,a)$），不是"有提示时值多少分"。

---

### 4.4.3 `sac_q_forward`：critic 打分与双 Q 输出形状

```python
160:  def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
161:      del shared_feature
162:      critic_state = self._critic_state(obs)              # [B, 2057]
163:      if detach_encoder:
164:          critic_state = critic_state.detach()
165:      return self.q_head(critic_state, self._flatten_batch(actions))
```

- **`del shared_feature`**：声明"这个参数不用"（RLT 是纯 MLP，没有和 actor 共享的视觉 backbone），既给读者消除疑问，也避免 linter 报未使用参数。
- **`_flatten_batch(actions)`**（`:88-92`）：幂等的"保证二维"工具，`[B, C, A]` 或 `[B, D]` 都能接。
- **`self.q_head`** 是子模块（不是方法），调用走 `nn.Module.__call__` → `MultiQHead.forward`（`q_head.py:150`）→ 循环两个独立 `QHead.forward`（`:99`）：

```python
x = torch.cat([state_features, action_features], dim=-1)   # [B, 2057+80] = [B, 2137]
q = self.net(x)    # [256,256,256,1]，LayerNorm + tanh，最后一层无激活
```

> 巧合：critic 单头输入 = 2057 + 80 = **2137**，和 actor 输入维度数值相同。**含义完全不同**——actor 是把参考块当"条件"，critic 是把动作当"被评估对象"，拼接顺序也不一样。

**输出形状**：

```python
q_vs = [Q1(...), Q2(...)]           # 各自 [B, 1]
return torch.cat(q_vs, dim=-1)      # → [B, 2]
```

**⚠️ 文档多处的 `[B, 2, 1]` 是近似写法，实际是 `[B, 2]`**（`output_dim=1` 那一维在 `cat` 时被吃掉）。好在消费端统一用 `[..., 0:1]` 这种写法，**对 `[B,2]` 和 `[B,2,1]` 都成立**：

```python
# fsdp_rlt_ac_policy_worker.py:79-85
def _min_twin_q(self, all_q_values):  return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])
def _q1(self, all_q_values):          return all_q_values[..., 0:1]
```

**`detach_encoder` 的两种用法**（这是理解 actor 梯度的关键）：

| 调用处 | `detach_encoder` | 意图 |
|---|---|---|
| `forward_critic:281` | 不传（`False`） | 训练 critic 自己，梯度正常回传 |
| `forward_actor:320` | **`True`** | 只让梯度从 Q 沿"**动作**"这一路回传到 actor |

对 $Q_\psi(x,\pi_\theta(x))$ 求 $\theta$ 的梯度有两项：$\frac{\partial Q}{\partial a}\frac{\partial a}{\partial\theta}$（想要的）和 $\frac{\partial Q}{\partial x}\frac{\partial x}{\partial\theta}$（**改观测骗高 Q，不想要**）。`detach` 直接把第二项掐掉，**保证 actor 只能通过改动作来降低损失**。

> 在 RLT 里 `z_rl` / `proprio` 都是叶子输入张量（来自冻结 VLA），第二项本来就等于 0。这个开关是**给通用 SAC 框架里那些有可训练 encoder 的策略**准备的防御性设计。

**📌 双 Q 分工：为什么"算两个、只用一个"**

这是最容易困惑的地方——答案是**两个 Q 服务两个不同的环节**：

| 用途 | 用几个头 | 代码 |
|---|---|---|
| **算 critic 的 TD target** | **两个，取 min** | `forward_critic:259` → `_min_twin_q` |
| **actor 的梯度** | **只要第 1 个** | `forward_actor:341` → `_q1` |

- **双 Q 的设计目标是给 critic 用的**：`y = r + γ·min(Q'₁, Q'₂)` 里的取 min 是为了**主动引入向下的（悲观）偏差去抵消过估计**，这要求两个网络独立初始化、独立训练（否则误差相关，取 min ≈ 取任意一个，白搭）。两个头**都**要用 `F.mse_loss(all_data_q_values, target.expand_as(...))` 拟合同一个 target，所以两个都不能省。
- **actor 只用 Q1 是 TD3 的明确约定**：取 min 的目的是给 **target 降偏**；如果 actor 也追 min，相当于在优化时又加一层悲观，**策略会过度保守、学不动**。论文的 actor 梯度就是 $\nabla_a Q_{\psi_1}$。
- **为什么固定选 Q1 而不是 Q2？** 两个头地位对等，选哪个都行，但**必须固定**——来回换会让 actor 的梯度方向抖动。`_q1` 硬编码 `[..., 0:1]` 就是这个"固定"。

**直觉**：双 Q 像两个评审，职责是**互相制衡**，不是"给选手更多分"。**取 min 用来防骗，不用来防冒进**——用在 target 上防骗，用在 actor 上就会变成防冒进。

> ⚠️ AC 路径**硬编码用 Q1**，配置里的 `actor_agg_q`（`q1`/`min`/`mean`）对它**无效**；只有 TD3 变体才会按配置聚合。

**metric 采集段**（`:336-342`）：

```python
num_q_values = all_qf_pi.shape[-1]                    # 用 shape 反推头数，不写死 2
metrics = {f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
           for q_id in range(num_q_values)}           # q_value_0 / q_value_1
qf_pi = self._q1(all_qf_pi)                           # 进损失的那个（保留梯度！）
metrics["q_pi"] = qf_pi.mean().item()
```

- **两个头分开记**是调试双 Q 的关键：趋势一致 = 正常；**两值差距拉大 = 训练不稳**；单头暴涨 = 典型高估/发散信号。
- `[..., q_id]` 里 `...` 跳过所有前置维度，对 `[B,2]` 和 `[B,2,1]` 都免疫。
- **`.item()` 是硬性要求**：`metrics` 要跨进程 all-reduce、写日志、序列化，留着张量会**持有整张计算图的引用**，每个 update step 攒一份，显存很快爆掉。
- **`qf_pi` 绝不能 `.item()`/`.detach()`**——它要进损失 `actor_loss = -q_weight * qf_pi.mean() + ...`。

---

### 4.4.4 BC 正则的四个输入：`ref_chunk` 与 `actions` 各从哪来

```python
344:  ref_chunk = self._ref_chunk(curr_obs)
345:  bc_loss, rlt_metrics = self._bc_metrics(
346:      pi=pi,                              # 在线 actor 刚采样的动作，带梯度
347:      actions=batch["actions"],           # 采集时真正执行的动作
348:      ref_chunk=ref_chunk,                # VLA 参考块（未 dropout）
349:      intervene_flags=batch.get("intervene_flags", None),
350:  )
351:  metrics.update(rlt_metrics)
```

#### `ref_chunk` 的完整来源链路

```text
┌─ 冻结 π0.5 VLA ─────────────────────────────────────────────┐
│ extract_rlt_obs()  rlinf/models/embodiment/openpi/tasks/eval.py:121-164
│   ├─ z_rl      = _encode_rlt_flat(prefix)        [B, 2048]
│   ├─ proprio   = env_obs["states"]               [B, 9]
│   └─ ref_chunk = VLA 采样出的动作块（_sample_actions_from_prefix_cache）
│                                                   [B, 80]  ★ 源头
│   注意：z_rl 与 ref_chunk 共用同一个 prefix cache，几乎不额外花钱
└─────────────────────────────────────────────────────────────┘
                        ↓  打包进 rlt_obs（RLT_OBS_KEYS = ("z_rl","proprio","ref_chunk")）
┌─ 采集 & 路由 ───────────────────────────────────────────────┐
│ route.py / transition.py                                    │
│   有干预时：curr_obs["ref_chunk"] ← 人类实际动作  ★ 改写      │
│   （tests/unit_tests/test_data.py:1822-1824 断言了这个行为）  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─ Replay Buffer ─────────────────────────────────────────────┐
│ [T, B_env, ...] --reshape(-1, *shape[2:])--> [T×B_env, ...] │
│ sample → batch["curr_obs"]["ref_chunk"]           [B, 80]   │
└─────────────────────────────────────────────────────────────┘
                        ↓  训练时分成两种加工
        ┌───────────────┴───────────────┐
        ↓                               ↓
① _get_ref_chunk（模型侧）        ② _ref_chunk（worker 侧）
   + 50% dropout                    无 dropout，只 reshape + 截断
        ↓                               ↓
   actor 的输入条件                   BC 的目标
```

**⚠️ 这里是最容易看糊的地方**：两个函数**几乎同名、返回形状也一样，但 dropout 行为相反**：

| | ① `_get_ref_chunk` | ② `_ref_chunk` |
|---|---|---|
| 文件 | `rlt_mlp_policy.py:100-105` | `fsdp_rlt_ac_policy_worker.py:64-69` |
| dropout | ✅ 训练时 50% 概率整块置零 | ❌ **永远完整** |
| 用途 | actor 网络的输入条件 | BC 损失的目标 |

**为什么 BC 目标不能 dropout？** 否则"让 pi 去靠近 ref"就失去了目标（总不能让 pi 去靠近一个零向量）。

#### `batch["actions"]` 的来源

它是 **rollout 时真正送进环境执行的**
那块动作**，经**三路路由**选出（`rlinf/algorithms/rlt/rollout.py::predict_rlt_actions`）：

| 来源 | 何时用 |
|---|---|
| **actor** | 常态：RL 策略自己出的动作 |
| **VLA ref** | `rlt_policy_switch` 触发时（"回到老师那里"） |
| **专家 / 人类** | 干预请求时（人接管或专家示范） |

链路：`route_output.actions` → `TrajectoryStep.actions`（`embodied_types.py:610-611`，注释写明 "Executed training actions, float `[B, C, A]` or flattened `[B, D]`"）→ 写 replay 时 `reshape(-1, *shape[2:])` 摊平（`buffer.py:763`）→ 采样取索引 → `batch["actions"]` `[B, 80]`。

**它和 `curr_obs` 用同一套摊平逻辑**，所以 `flat["actions"][i]` 与 `flat["curr_obs"]["ref_chunk"][i]` **一一对应**——这正是 `_bc_metrics` 能按行做 MSE 的基础。

**⚠️ replay 里躺着好几份"动作"，别搞混**：

| 字段 | 是什么 | 谁在用 |
|---|---|---|
| **`actions`** | **实际执行的动作块**（路由后） | BC 目标、critic 的 $Q(s,a)$ |
| `intervene_actions` | 人类/专家给的动作（`EnvTransition`） | 采集时经路由写进 `actions` |
| `ref_chunk`（在 obs 里） | VLA 的参考块 | actor 输入、BC 目标 |
| `pi`（不存，现算） | 当前 actor 采样出的动作 | 被训练的一方 |

**`intervene_flags` 的兜底**：`TrajectoryStep.__post_init__`（`embodied_types.py:668-669`）在 `actions` 存在且 `intervene_flags is None` 时自动补一个全 False 的同形张量；而 worker 这边仍用 `.get(..., None)` 兜第二次——**`None` 是合法输入**，`_bc_metrics` 会解释成"没有任何干预"。

`_bc_metrics` 内部核心两行（`:124-126`）：

```python
bc_target = torch.where(human_mask[..., None], action_chunk, bc_ref_chunk)
bc_error  = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
bc_loss   = torch.mean(bc_error)
```

即 **逐步选目标：被干预过的步用人类动作，没被干预的步用 VLA 参考块**，然后算平方 L2。两次求均值（先动作维、再全体）保证每个动作维权重相同。

注意论文那句"发生干预时把 buffer 里的参考块替换成人类动作"在代码里有**两个实现**，互为双保险：**采集时替换**（`route.py`/`transition.py`）+ **训练时按 flag 改写**（`_bc_metrics:124` 的 `torch.where`）。

**返回值分工**：`bc_loss` 带梯度进损失；`rlt_metrics` 是纯诊断字典（`bc_loss` / `bc_ref_loss` / `bc_human_loss` / `human_mask_ratio` / `policy_mask_ratio`），**全部已 `.detach().item()`**。两个分项在各自 mask 内单独求均值（分母 `torch.clamp(..., min=1.0)` 防除零），用来分辨"是偏离 VLA 太远"还是"没学会学人类"。

---

### 4.4.5 损失合成与诊断指标

```python
353:  entropy = -log_pi.mean()
354:  bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
355:  actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
356:  metrics.update(weight_metrics)
357:  metrics["action_ref_abs_mean"] = (
358:      (self._flatten_chunk(pi) - self._flatten_chunk(ref_chunk)).abs().mean().detach().item())
364:  metrics["weighted_q"]  = (q_weight * qf_pi.mean()).detach().item()
365:  metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
366:  metrics["reference_dropout_prob"] = reference_dropout_prob
```

**① `entropy = -log_pi.mean()`——只是记录，不进损失。**

`log_pi` 来自 `sac_forward` 的 `chunk_logprobs`（`rlt_mlp_policy.py:156`，**tanh 之前算的**、无 Jacobian 修正），上层先 `sum(dim=-1)` 把 `[B, 80]` 压成 `[B, 1]`（C×A 个独立高斯的联合 log 概率 = 各自之和）。`-log_pi` 是熵的蒙特卡洛估计。它作为返回值第二项交出去（`:368`），**不进任何损失**——因为 RLT 关掉了熵项。

**② `actor_loss` 的梯度流向**：

```text
actor_loss
 ├─ -q_weight * qf_pi.mean()
 │     └─ qf_pi ← Q1(critic_state, pi)      detach_encoder=True 已断开状态侧
 │          └─ pi ← rsample() ← actor_mean ← backbone ← θ    ✅ 唯一梯度路径
 └─ + bc_weight * bc_loss
       └─ bc_loss ← (pi - bc_target)²       bc_target 是常量（buffer 数据）
            └─ pi                                                ✅ 同样回到 θ
```

**两项都只从 `pi` 这一条路回传梯度**——Q 项的状态侧被 detach，BC 项的目标是无梯度的 replay 数据。这就是"actor 只能靠改动作降低损失，不能靠改观测作弊"的保证。

**③ 各诊断指标的意义**：

| metric | 含义 / 怎么读 |
|---|---|
| `action_ref_abs_mean` | `pi` 离 VLA 建议的平均绝对距离。**太小** = actor 在复读 VLA、RL 没起作用；**太大** = 跑太远有假尖峰风险。注意这里用的是**未 dropout** 的 `ref_chunk`，量到的是真实偏离量 |
| `weighted_q` / `weighted_bc` | 两项**加权后**的实际贡献。只记系数或只记原始值都判断不出"谁在主导"；`weighted_bc ≫ weighted_q` = 模仿期，`weighted_q` 追上 = RL 开始主导 |
| `reference_dropout_prob` | 记录本次实际用的概率。**因为代码默认是 `0.0`**（`:306-308` 的 `.get(..., 0.0)`），新写的配置忘了这一项就等于关掉了 dropout——记进日志才能一眼看出这个 run 到底开没开 |
| `q_value_0` / `q_value_1` | 两个 Q 头的均值，监控双头是否分裂/发散 |
| `q_pi` | actor 实际会用到的那份打分（= Q1） |

**📌 哪些带梯度、哪些不带**：

| 变量 | 处理 | 原因 |
|---|---|---|
| `actor_loss` / `qf_pi` / `bc_loss` | **保留梯度** | 要 `.backward()` |
| `entropy` | 保留（但没进损失） | 上层只是取值 |
| `metrics` 里**所有**值 | **`.detach().item()`** | 脱离计算图，防显存泄漏 |

最后一行是硬性要求：留在计算图上的张量每个 step 攒一份，很快 OOM。

---

### 4.4.6 `_actor_objective_weights`：BC/Q 权重调度器

`fsdp_rlt_ac_policy_worker.py:148-225`。它**不改网络、不算损失**，只决定"当前这一步两项各占多大权重"。

**两种模式**：

```python
150:  schedule_cfg = self.cfg.algorithm.get("actor_weight_schedule", {})
151:  schedule_enabled = bool(schedule_cfg.get("enable", False))
152:  if not schedule_enabled:                       # 关掉调度：一整轮用固定值
153:      bc_weight = float(self.cfg.algorithm.get("bc_weight", 1.0))
154:      q_weight  = float(self.cfg.algorithm.get("q_weight", 1.0))
```

**打开调度后走三段式**（`:167-216`）：

```text
阶段     条件                              取值
------------------------------------------------------------------------
warmup   update_step < warmup_updates       warmup_bc_weight / warmup_q_weight
ramp     之后 ramp_updates 步内              在 warmup 值与 online 值之间线性插值
online   ramp 结束                          online_bc_weight / online_q_weight
```

插值公式（`:199-212`）：

```python
ramp_progress = min(1.0, max(0.0,
    float(int(self.update_step) - weight_warmup_updates + 1) / float(ramp_updates)))
bc_weight = warmup_bc_weight + ramp_progress * (online_bc_weight - warmup_bc_weight)
q_weight  = warmup_q_weight  + ramp_progress * (online_q_weight  - warmup_q_weight)
```

**仿真配置的实际数字**（`maniskill_rlt_stage2_ac_mlp.yaml:68-75`：`warmup_updates: 20000`、`ramp_updates: 50000`）：

```text
阶段     条件                         bc_weight     q_weight     bc:q
----------------------------------------------------------------------
warmup   step < 20000                 7.0           0.05         140 : 1
ramp     20000 <= step < 70000        7.0 → 2.5     0.05 → 0.45  线性过渡
online   step >= 70000                2.5           0.45         5.6 : 1
```

两条曲线**同时**变化：BC 从 7.0 降到 2.5，Q 从 0.05 升到 0.45——**Q 的权重涨了 9 倍**。

**各配置对照**：

| 配置 | 是否开调度 | 固定值 / 调度 |
|---|---|---|
| `maniskill_rlt_stage2_ac_mlp.yaml:63-75` | ✅ 开 | `1.0/1.0` 被覆盖；7.0/0.05 → 2.5/0.45 |
| `maniskill_rlt_stage2_td3_mlp.yaml:63-76` | ✅ 开 | 7.0/0.05 → 2.8/0.50 |
| `realworld_rlt_stage2_ac_mlp.yaml:75-76` | ❌ 没配 → 关 | **固定 `bc=5, q=0.1`（50:1）** |

**为什么要这个调度**：训练刚开始时 **critic 是个随机函数**，它的"价值高地"全是噪声（TD3 通俗讲解 4.1 节"新手教练瞎指挥"）。这时候让 actor 追 Q 就是**追着噪声跑**。所以前期把 `q_weight` 压到 0.05，让 actor 老实模仿 VLA + 人类示范；中期 critic 学准了逐步交权；后期 `q_weight` 提到 0.45，让 RL 真正去磨最后几毫米精度。

这和 TD3 的 **Delayed policy update** 同源，只是从"更新频率"换成了"损失权重"。真机不开调度是因为 run 短、不确定性大，手工调三段曲线难复现，索性用一组固定值**一直停在偏 warmup 的状态**，保守优先。

**`weight_metrics` 的内容**（`:218-224`）：`bc_weight`、`q_weight`、`actor_weight_schedule_enabled`（1.0/0.0）、`actor_weight_in_warmup`、`actor_weight_ramp_progress`（0→1）。

关闭调度时也返回**同样的 key**（只是 enabled=0、progress=1），**保证日志 schema 一致**，画曲线不用区分两种情况。`actor_weight_ramp_progress` 还是个天然进度条——长时间卡住说明 `update_step` 没在涨（训练卡死了）。

---

### 4.5 Reference action dropout（防"复读机"）

`rlt_mlp_policy.py:107-118`：

```python
def _maybe_drop_reference(self, ref_chunk, reference_dropout_prob):
    keep_prob = 1.0 - reference_dropout_prob
    keep_mask = torch.rand((B, 1), device=...) < keep_prob
    return ref_chunk * keep_mask.to(ref_chunk.dtype)     # 整条参考块置零
```

- 逐样本、**整块置零**，只在训练路径 `sac_forward(apply_reference_dropout=True)` 时生效；`predict_action_batch`（rollout/eval）**永远传参考块**。✅ 与论文一致（训练 50% / 推理始终提供）。
- ⚠️ **代码默认 `reference_dropout_prob = 0.0`**（`fsdp_rlt_ac_policy_worker.py:306-308` 的 `.get(..., 0.0)`），0.5 只是三个官方 YAML 里的值。自己新写的配置忘了写这一项就**等于关掉了 dropout**。

### 4.6 训练节奏：UTD 与 critic:actor

> **UTD 是什么**：Update-To-Data ratio（更新/数据比），即"**每条新 transition 被用来做多少次梯度更新**"。
>
> $$\text{UTD} = \frac{\text{梯度更新次数}}{\text{新采集的 transition 条数}}$$
>
> - **UTD = 1**：采一条数据、做一次更新（SAC/TD3 论文的常规设置）；
> - **UTD = 5**：采一条数据、做 5 次更新。
>
> **为什么能大于 1？** 因为 off-policy 算法能用 replay buffer 里的旧数据——既然旧数据也能训，就不必"一条数据只学一次"。**为什么真机尤其在意？** 采样太贵：把 UTD 拉高等于**用算力换样本**，这正是 RLT 能在"几十分钟真机练习"里学会的关键杠杆之一。
>
> 打比方：UTD 就像复习遍数——同样上完一节课（1 条数据），有人复习 1 遍（UTD=1），有人把同一份笔记做 5 遍（UTD=5）。后者考得更好，但更费时间（算力）。

**三个参数各管什么**（`fsdp_rlt_ac_policy_worker.py:731-797`）：

| 参数 | 作用 | 仿真值 | 真机值 |
|---|---|---|---|
| `algorithm.update_epoch` | 每个"训练周期"做多少次更新 | 5 | 8 |
| `rlt_schedule.train_every_transitions` | 攒够多少条新 transition 算一个周期 | 5 | 未配 |
| `rlt_schedule.max_updates_per_train_step` | 单次最多跑多少次更新（限流） | 400 | 未配 |
| `rlt_schedule.warmup_min_size` | replay 至少多大才开始训 | 10000 | — |
| `rlt_schedule.warmup_post_collect_updates` | 开训后再补多少"纯 warmup"更新 | 30000 | — |
| `algorithm.critic_actor_ratio` | 每 N 次 critic 更新做 1 次 actor 更新 | 4 | 4 |

**核心公式**（`:787-794`）：

```python
online_cycles         = online_transitions // train_every_transitions
desired_total_updates = warmup_required_updates + online_cycles * update_epoch
pending_updates       = max(desired_total_updates - update_step, 0)
updates_to_run        = min(pending_updates, max_updates)      # max_updates = 400
```

读法：目标是"**每攒够 `train_every_transitions` 条新数据，就做 `update_epoch` 次更新**"，于是

$$\text{UTD} = \frac{\texttt{update\_epoch}}{\texttt{train\_every\_transitions}} = \frac{5}{5} = 1 \quad(\text{仿真})$$

> 📌 `update_epoch` 这个名字有点误导：它不是"把一个 batch 过几遍"（那通常叫 `n_epochs`），而是**每个数据周期的更新次数**——所以要和 `train_every_transitions` 一起才决定 UTD。

**两个安全阀**：

- `warmup_min_size` + `warmup_post_collect_updates`：buffer 太小的时候学不到东西，先攒够 10000 条、再白跑 30000 次更新当预热（`rlt/ready_for_online` 这个 metric 标记转折点，`:801-803`）；
- `max_updates_per_train_step: 400`：`desired_total_updates` 是**累积式**的（用 `update_step` 做差分），采样快、训练慢的话欠账会越积越多。400 是"单次最多补 400 次"，防止某一轮训练卡死，剩下的欠账留到下次（`pending_update_budget`）。

**`critic_actor_ratio`：第二个节奏控制**（`:865-870`）：

```python
for _ in range(updates_to_run):
    update_actor = int(self.update_step) % int(self.critic_actor_ratio) == 0
    metrics_data = self.update_one_epoch(train_actor=update_actor)
    self.update_step += 1
    critic_updates_run += 1                     # critic 每次都更新
    actor_updates_run  += int(update_actor)     # actor 只在整除时更新
```

| 配置 | `critic_actor_ratio` | 实际比例 |
|---|---|---|
| 论文 | — | 1 actor : 2 critic |
| `maniskill_rlt_stage2_td3_mlp.yaml:97` | 2 | 1:2 ✅ 与论文一致 |
| `maniskill_rlt_stage2_ac_mlp.yaml:96` | **4** | **1:4** ⚠️ |
| `realworld_rlt_stage2_ac_mlp.yaml:104` | **4** | **1:4** ⚠️ |

这就是 TD3 的 **Delayed policy update**（`TD3_通俗讲解.md` 第 4 节）——critic 先学准，actor 才不会追着噪声跑。AC 主线拉到 4:1 是**更保守**的选择，和 4.4 节"重 BC、轻 Q"的取向一致。

**📌 UTD 与 critic:actor 是两个正交的概念**：

| | UTD | critic:actor |
|---|---|---|
| 回答的问题 | "**每条数据**学几次" | "critic 和 actor **谁更新得多**" |
| 管什么 | 学习总量 / 样本效率 | 两者的**相对**节奏 |
| 对应参数 | `update_epoch` / `train_every_transitions` | `critic_actor_ratio` |

**与论文的差异**：论文明确写 UTD=5、每 1 次 actor 配 2 次 critic。RLinf 里：

- **UTD 不是常数 5**，而是由 `update_epoch / train_every_transitions` 动态算出来的（仿真 5/5 = **约 1 update/transition**）；
- **critic:actor 在 AC 配置是 4:1**，只有 TD3 配置是 2:1（与论文一致）。

---

### 4.6.1 一次 update 的内部：两个 loss 怎么被消费

4.3 节给了 `critic_loss`、4.4 节给了 `actor_loss`。**这两个 loss 从不相加**——它们各走各的优化器，先后独立更新。

> 代码位置：**`rlinf/workers/actor/fsdp_sac_policy_worker.py:560-671`** 的 `EmbodiedSACFSDPPolicy.update_one_epoch()`。RLT 通过 `RLTACFSDPPolicy(RLTACLossMixin, RLTACReplayMixin, EmbodiedSACFSDPPolicy)` 继承直接用，**自己没写训练循环**；`RLTACLossMixin` 只负责提供 `forward_critic` / `forward_actor` 这两个"算 loss"的方法。

**完整流程**：

```text
update_one_epoch(train_actor=True)                     :560
  │
  ├─ ① 采样 + 切 micro-batch                          :565-579
  │     global_batch = next(buffer_dataloader_iter)
  │     split_dict_to_chunk(global_batch, global_bs // micro_bs)
  │
  ├─ ② critic 分支（每次都做）                          :581-605
  │     qf_optimizer.zero_grad()
  │     for 每个 micro-batch:
  │         critic_loss, critic_metrics = forward_critic(batch)
  │         critic_loss = critic_loss / gradient_accumulation   # 梯度累积缩放
  │         critic_loss.backward()
  │     qf_grad_norm = model.clip_grad_norm_(critic_optim.clip_grad)
  │     qf_optimizer.step()                            # ★ 更新 critic
  │     qf_lr_scheduler.step()
  │
  ├─ ③ actor 分支（每 ratio 次才做）                     :607-663
  │     if update_step % critic_actor_ratio == 0 and train_actor:
  │         optimizer.zero_grad()
  │         for 每个 micro-batch:
  │             actor_loss, entropy, q_metrics = forward_actor(batch)
  │             actor_loss = actor_loss / gradient_accumulation
  │             actor_loss.backward()
  │         actor_grad_norm = model.clip_grad_norm_(optim.clip_grad)
  │         optimizer.step()                           # ★ 更新 actor
  │         lr_scheduler.step()
  │
  └─ ④ target 软更新                                   :664-669
        soft_update_target_model()                      # Polyak，见 4.3 节
```

#### (a) 两个优化器：参数按"名字"分组

`fsdp_sac_policy_worker.py:120-137`：

```python
120:  self.use_dsrl = _openpi_is_dsrl(self.cfg)
122:  if use_dsrl:
125:      param_filters = {"critic": ["critic_image_encoder", "critic_state_encoder", "q_head"]}
128:  else:
128:      param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}   # ← RLT 走这条
129:  filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
130:  optimizers = self.build_optimizers(model=self.model,
132:                                     main_optim_config=self.cfg.actor.optim,
133:                                     param_filters=param_filters,
134:                                     filtered_optim_config=filtered_optim_config)
136:  self.optimizer    = optimizers[0]      # ★ actor 优化器（主优化器）
137:  self.qf_optimizer = optimizers[1]      # ★ critic 优化器
```

分组逻辑在 `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py:668-716`：

```python
668:  for name, param in model.named_parameters():
669:      if not param.requires_grad:       # target_model 是 requires_grad=False，天然排除
670:          continue
673:      for key, filters in param_filters.items():
675:              if f in name:             # ★ 名字里含 "q_head" → 归 critic
676:                  filtered_params_dict[key].append(param)
685:      main_optim_params.append(param)   # ★ 其余的（actor 部分）→ 归 actor
```

对 `RLTMLPPolicy`，实际分组是：

| 参数名 | 归属 | 优化器 | 学习率来源 |
|---|---|---|---|
| `backbone.*`（3 层 256 MLP） | **actor** | `self.optimizer` | `actor.optim.lr` |
| `actor_mean.*` | **actor** | `self.optimizer` | `actor.optim.lr` |
| `actor_logstd`（死参数，4.2 节） | actor | `self.optimizer` | `actor.optim.lr` |
| **`q_head.qs.0.*`** | **critic** | `qf_optimizer` | `actor.critic_optim.lr` |
| **`q_head.qs.1.*`** | **critic** | `qf_optimizer` | `actor.critic_optim.lr` |
| `target_model.*` | — | **不进任何优化器** | `requires_grad=False` |

**为什么必须分开**（三个理由）：

1. **梯度不能互相污染**：用一个优化器的话，`step()` 会同时动两边——而 actor 只在 1/4 的步数里更新，混在一起语义就乱了；
2. **学习率可独立配置**：`actor.optim` 和 `actor.critic_optim` 是两套配置（RLT 里都是 1e-4，但可以调成不同）；
3. **符合 off-policy 的语义**：critic 拟合值函数、actor 追策略梯度，本来就是两个不同的目标。

#### (b) 梯度累积

```python
561:  global_batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
568:  train_micro_batch_list = split_dict_to_chunk(
569:      global_batch, global_batch_size_per_rank // self.cfg.actor.micro_batch_size)
...
586:  critic_loss = critic_loss / self.gradient_accumulation   # 缩放
587:  critic_loss.backward()                                   # 累积到 .grad
597:  self.qf_optimizer.step()                                 # 累积完才更新一次
```

`gradient_accumulation` 在 `:854-858` 算出：

```python
854:  self.gradient_accumulation = (self.cfg.actor.global_batch_size
856:                                // self.cfg.actor.micro_batch_size
857:                                // self._world_size)
```

RLT 配置 `global_batch_size: 512`、`micro_batch_size: 128`：若 `world_size=4`，则每卡 global batch = 128、**micro-batch 数 = 1**（即 `gradient_accumulation = 1`）。代码写成通用形式是为了支持更大的 global batch。

**每个 micro-batch 的 loss 都除以 `gradient_accumulation`，所以梯度累加等价于对 micro-batch 取平均。**

#### (c) 两个 loss 的对照

| | `critic_loss` | `actor_loss` |
|---|---|---|
| 来自 | `forward_critic`（`fsdp_rlt_ac_policy_worker.py:233-299`） | `forward_actor`（同文件 `:301-368`） |
| 公式 | `MSE(Q_online, y)`，`y = Σγ^t'r + γ^C·min(Q'₁,Q'₂)` | `−q_weight·Q1 + bc_weight·BC` |
| 更新谁 | `q_head.qs.0/1` | `backbone` + `actor_mean` |
| 优化器 | `qf_optimizer` | `optimizer` |
| 频率 | **每次都做** | 每 `critic_actor_ratio` 次做 1 次（AC=4，TD3=2） |
| 裁剪阈值 | `actor.critic_optim.clip_grad` | `actor.optim.clip_grad` |
| LR 调度器 | `qf_lr_scheduler` | `lr_scheduler` |

**两者共用的**：同一批数据（`train_micro_batch_list`）、同一个 `self.model`（一个模块里装两套参数）、同一个 `self.update_step`（决定 actor 是否更新）。

#### (d) 两层判断的一致性

RLT 那边（`:864-870`）：

```python
864:  for _ in range(updates_to_run):
865:      update_actor = int(self.update_step) % int(self.critic_actor_ratio) == 0
866:      metrics_data = self.update_one_epoch(train_actor=update_actor)
868:      self.update_step += 1
```

而 `update_one_epoch` 内部**又检查一遍**（`:607`）：

```python
607:  if self.update_step % self.critic_actor_ratio == 0 and train_actor:
```

两个条件完全相同，且 `update_step` 是**调用之后**才自增，所以行为一致——第二层是防御性冗余。`train_actor` 参数的意义是：调度器已经判定"这次不更新 actor"时，外面可以显式传 `False`。

#### (e) RLT 特有的两点

1. **`entropy` 只是返回值，不进任何优化器**：`:613` 接住它，`:659` 只记 metric。因为 RLT 的 `alpha_type: fixed_alpha` → `self.alpha_optimizer is None` → `:632` 那段（含 `forward_alpha` 调用）整段跳过。
2. **`forward_alpha` 会抛异常**（`fsdp_rlt_ac_policy_worker.py:370-376`），所以即使误配了 `alpha_type` 也会**立刻报错**，不会静默出错。

#### (f) ⚠️ 一个值得注意的行为：两次裁剪互相影响

按代码顺序推一遍（第 N 次 update，N ≥ 2）：

```text
:581  qf_optimizer.zero_grad()      → 只清零 critic 参数
:587  critic_loss.backward()        → 写入 critic 梯度
:593  self.model.clip_grad_norm_()  → ⚠️ 计算**整个 model** 的梯度范数
       └─ 此刻 actor 参数上还有**上一次 actor 更新留下的陈旧梯度**
:597  qf_optimizer.step()           → 更新 critic（step 不清零梯度）
...
:608  optimizer.zero_grad()         → 这时才清零 actor
:615  actor_loss.backward()         → 写入 actor 梯度
:623  self.model.clip_grad_norm_()  → ⚠️ 此刻 critic 梯度也还在
```

根因是 `clip_grad_norm_` 的实现（`rlinf/hybrid_engines/fsdp/strategy/fsdp2.py:172-181`）：

```python
172:  grad_norm = get_grad_norm(model.parameters(), dp_group=..., norm_type=...)
177:  clip_grad_by_total_norm_(model.parameters(), max_grad_norm=self.cfg.optim.clip_grad,
180:                           total_norm=grad_norm)
```

而 `get_grad_norm`（`rlinf/hybrid_engines/fsdp/utils.py:815-819`）只跳过 `p.grad is None` 的参数：

```python
815:  grads_for_norm = [p.grad.detach() for p in parameters if p.grad is not None]
```

**后果**：

| 影响 | 说明 |
|---|---|
| `critic/grad_norm` metric 被**高估** | 混进了 actor 的陈旧梯度 |
| critic 梯度被**裁得更狠** | `clip_coeff = max_norm / (总范数)`，分母偏大 → 系数偏小 |
| actor 的裁剪同理 | 分母里含 critic 梯度 |
| ✅ **但不会污染更新本身** | 各自的 `step()` 前都 `zero_grad()` 过自己的参数——被误缩放的陈旧梯度在用到之前就清零了 |

**结论**：**数值上安全，但裁剪强度与配置意图不完全一致**（两个 `clip_grad` 实际作用在"critic + actor 联合范数"上）。RLT 里两个 `clip_grad` 恰好都是 `10.0`（`maniskill_rlt_stage2_ac_mlp.yaml:347,352`），所以后果有限；若把两者调成不同的值，意图与实际的偏离会更明显。

> 这是**读代码推导**的结论，未跑测试验证。想确认的话可以直接对比 `critic/grad_norm`（`:603` 已记进 metric）在第 1 次与第 2 次 update 的值——第 1 次没有陈旧梯度，第 2 次有。

#### (g) 完整数据流图

```text
         buffer_dataloader
                │  global_batch (512)
                ▼
      split → micro_batch_list [128 × N]
                │
        ┌───────┴────────┐
        ▼                ▼
   forward_critic   forward_actor          ← RLTACLossMixin 提供
        │                │
   critic_loss      actor_loss
  (MSE, 稀疏奖励)   (−q_weight·Q1 + bc_weight·BC)
        │                │
   /grad_accum      /grad_accum
        │                │
   backward()       backward()
        │                │
   ┌────┴────┐      ┌────┴─────┐
   │ q_head  │      │ backbone │          ← 梯度只落在各自的参数上
   │  .qs.0  │      │actor_mean│
   │  .qs.1  │      │          │
   └────┬────┘      └────┬─────┘
        │                │
  clip_grad_norm    clip_grad_norm
        │                │
 qf_optimizer.step  optimizer.step
  (critic 更新)      (actor 更新, 1/4 频率)
        │                │
        └───────┬────────┘
                ▼
      soft_update_target_model()          ← Polyak，每步
        ψ' ← 0.995·ψ' + 0.005·ψ
                ▼
          metrics 记录
   sac/critic_loss, sac/actor_loss,
   critic/grad_norm, actor/grad_norm, ...
```

---

## 5. 数据流与编排

### 5.1 一次 rollout 的完整链路

`rlinf/algorithms/rlt/rollout.py::predict_rlt_actions()` 是阶段二每次决策的入口：

```python
with torch.no_grad():
    rlt_obs = feature_model.extract_rlt_obs(env_obs)      # 冻结 VLA → z_rl / proprio / ref_chunk
    actions, result = policy_model.predict_action_batch(env_obs=rlt_obs, mode=mode, return_obs=True)
    route_output = rlt_route.route(RLTRouteContext(       # actor / ref / expert 三选一
        env_obs=env_obs, rlt_obs=rlt_obs, student_actions=actions, result=result,
        mode=mode, rlt_switch_flags=..., intervene_requested=..., expert_model=..., version=...,
    ))
    actions, result = route_output.actions, route_output.result
    _append_rlt_transition_obs(...)                        # 写入 rlt_transition_* （下一状态特征）
```

用文字展开（对应论文 Algorithm 1）：

1. 每个 **动作块边界**：冻结 VLA 前向一次 → `z_rl`、`proprio`、`ref_chunk`；顺带算一次 VLA 参考块（同一个 prefix cache，不重复前向）。
2. 小 MLP actor 以前述状态为输入，输出 `C×action_dim` 的动作块。
3. `route` 决定这一整块用 **actor** / **VLA ref** / **专家**。
4. 环境执行整块（`chunk_step`），拿到 chunk 级 reward/done 与信息（`rlt_switch_flags`、`intervene_flag`）。
5. 组装 transition，写入 replay buffer（按 `record_transition` 标记过滤）。

### 5.2 三路动作路由

`rlinf/algorithms/rlt/route.py` 有两个实现，按环境类型自动选择（`build_rlt_route`）：

**(a) `RealworldRLTRoute`**（真机）——只看"切换标志"：

```python
routed_actions = torch.where(rlt_switch_flags, actions, ref_actions[:, :C, :A])
result["forward_inputs"]["record_transition"] = rlt_switch_flags[:, :1]
result["forward_inputs"]["actor_switch"] = ...
```

按 `b` 之前用 VLA `ref_chunk`，按 `b` 之后用 actor。**这个标志只决定"谁执行"，不影响是否记录 transition**（真机路径下 `record_transition` 等于 actor 标志）。

**(b) `SimulatorRLTRoute`**（仿真）——关键阶段 + 预热门控 + 专家接管：

```python
ready_for_online = self._ready_for_online(ctx.version)     # version >= warmup_post_collect_updates
critical_phase   = _last_info_bool(ctx.rlt_switch_flags, ...)
actor_switch     = critical_phase & ready_for_online
expert_takeover  = requested_expert_takeover & ready_for_online & (mode == "train") & (expert_model is not None)

routed_actions = torch.where(actor_switch, actions, base_actions)      # base = VLA ref
if expert_takeover.any():
    expert_actions = predict_expert_actions(...)                       # 更强的 SFT checkpoint
    routed_actions = torch.where(expert_takeover, expert_actions, routed_actions)
    forward_inputs["ref_chunk"][:, :C] = torch.where(expert_takeover, expert_actions, ...)  # 改写参考块
    intervene_flags[expert_takeover] = True
```

四个输出标志很关键，也是监控指标来源：
- `record_transition` ← `critical_phase`（只有关键阶段的样本进 buffer）
- `actor_switch` ← `actor_switch & ~expert_takeover`（这一步真的是 actor 在控制吗）
- `intervention_requested` ← 环境是否请求了专家接管
- `intervene_flags` ← 实际是否用了专家动作（进 Trajectory，供 BC target 使用）

### 5.3 人干预如何写进 replay

两条独立的通路：

1. **`TrajectoryStep.from_parts`**（`embodied_types.py`）：环境返回 `intervene_actions` / `intervene_flags` 时，`apply_interventions()` 用 `expert_actions * flags + model_actions * (~flags)` 替换实际执行动作，并同步 `forward_inputs["action"]`、展开 `intervene_flags`。
2. **`apply_rlt_interventions`**（`rlinf/algorithms/rlt/transition.py`）：把 `curr_obs["ref_chunk"]` 里对应的块**替换成干预动作**：

```python
ref_actions[:, : flags.shape[1]] = torch.where(flags, actions, ref_actions[:, : flags.shape[1]])
obs["ref_chunk"] = ref_actions.reshape_as(ref_chunk)
```

这正是论文 Algorithm 1 第 8 行"若发生干预，把 buffer 中的参考块替换成 $a^{human}$"。

> 细节：**只替换 `curr_obs` 的 `ref_chunk`**；`next_obs["ref_chunk"]` 仍是冻结 VLA 新采出来的一条参考（因为它是下一个状态的参考，不该被上一步的干预污染）。另外仿真路由在 `route.py:230-237` 又会**原地改写一遍** `forward_inputs["ref_chunk"]`，两处效果一致、互为兜底。

### 5.4 Replay buffer 里存了什么

`RLT_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")`（`transition.py:22`）。一条 transition 形如：

```
curr_obs  = {z_rl, proprio, ref_chunk}      # 来自本 chunk 边界的 forward_inputs
action    = 实际下发给环境的动作块（actor / VLA / 专家）
intervene_flags = 逐维展开的干预标志
rewards   = [B, C] 的 chunk 内逐步奖励
next_obs  = {z_rl, proprio, ref_chunk}      # rlt_transition_* 前缀，来自 next forward_inputs
dones/terminations/truncations
```

- **不存原始图像**：RL 状态就是 `z_rl + proprio`，这也是论文"actor-critic 不直接看图像"的落地。
- 仿真路径会把 rollout 轨迹**逐时间步拆成单样本 Trajectory**（`_transition_replay_trajectories`，`fsdp_rlt_ac_policy_worker.py:462-578`）再塞进 `TrajectoryReplayBuffer`，并按 `record_transition` 过滤。
- 代码里还有一条"把带干预的样本另存进 `demo_buffer`"的分支（`fsdp_rlt_ac_policy_worker.py:630-637`），但 RLT 的几个官方配置都**没有配 `algorithm.demo_buffer`**，所以 `demo_buffer is None`、这条分支目前是空转的。
- 真机路径直接整条轨迹入库（不做 `record_transition` 过滤）。

**⚠️ 论文的 stride 子采样没有实现**：论文用 stride=2 在同一动作块的中间步也造样本（1 秒 ≈ 25 个 RL 样本）。RLinf 的轨迹时间维是**按动作块**累加的（`AccumulatorOutput._append_chunk` 每个 chunk 追加一个 `TrajectoryStep`），即**一个执行块 ≈ 一条 transition**（有效 stride = C）。数据量主要靠**环境并行数**（仿真 64 训练 env / 256 评估 env）补，而不是靠时间维子采样。

### 5.5 Warmup 与训练调度

论文的 warmup 是"先让 VLA 跑 $N_{warm}$ 步填 buffer"。RLinf 把它拆成了两层：

1. **数据层 `rlt_schedule`**（`fsdp_rlt_ac_policy_worker.py:731-827`）：
   - `warmup_min_size: 10000`：replay buffer 攒够这么多样本才开始训练；
   - `warmup_post_collect_updates: 30000`：先"空转"这么多 update 才认为 `ready_for_online`；
   - `train_every_transitions: 5` + `update_epoch: 5`：之后每新增 5 条 transition 做 5 次更新；
   - `max_updates_per_train_step: 400`：单次 collect 后的更新上限（防止 learner 追不上 rollout）。
2. **行为层 `SimulatorRLTRoute._ready_for_online`**：`version >= warmup_post_collect_updates` 之前，**路由一直执行 VLA `ref_chunk`**——这一段就是论文的"用 VLA 填 buffer"。

两个容易误解的点：

- **warmup 的"步数"统计的是 learner 的 `update_step`，不是环境步数**。这个 `version` 通过权重同步从 learner 传到 rollout worker（`get_rollout_sync_version` 返回 `update_step`），再作为 `ctx.version` 进路由。
- **仿真在 warmup 期间其实也在往 buffer 写数据**：`record_transition` 只由 `critical_phase` 决定，所以只要进了关键阶段，即使当前还是 VLA 在控制，样本照样入库（动作=ref，参考=ref）。真机路径则完全没有 `record_transition` 过滤，**按 `b` 之前的 VLA 片段也会整条入库**（`_ingest_rollout_trajectories` 的 else 分支直接 `add_trajectories(recv_list)`）。

监控：`train/rlt/ready_for_online`、`updates_to_run`、`skip_reason`、`pending_update_budget`、`train/replay/transition_count`、`train/replay/actor_switch_rate` 等（见 `rlt.rst` 的 Monitoring 一节）。

### 5.6 关键阶段（critical phase）怎么定

论文：**人类操作员决定何时把控制权交给 RL**，测试时再训 VLA 预测交接时刻。RLinf 两种落地：

**(a) 真机 —— 键盘**：`KeyboardRLTPolicySwitchWrapper`（`rlinf/envs/real/wrappers/episode/policy_switch.py`），按 `b` 之后 `rlt_switch_flags = True`，并且每个 episode 开始自动复位。配套 `keyboard_reward_wrapper: rlt_policy_switch`。

**(b) 仿真 —— 规则自动门控**：`ManiskillRLTEnv._rlt_auto_enter_actor`（`maniskill_rlt_env.py:641+`），用任务信息判断：

```yaml
rlt_policy_switch:
  task_mode: full_task        # VLA 负责关键阶段前，actor 负责关键阶段
  trigger_mode: auto          # 从任务信息自动算关键阶段
  latch_until_done: True      # 一旦进入就不再退出
  auto_gate:
    require_grasp: True       # 已抓住 peg 才进入
    require_not_success: True
    near_hole_x_min: -0.16    # 离孔足够近
    near_hole_yz_margin: 1.5
```

也就是说：**论文的"人决定交接点"→ RLinf 真机用键盘（仍是人），仿真用规则门控（`peg_head_hole_x` / `require_grasp` 等几何量）**。论文那个"微调 VLA 预测交接时刻"的自动切换**没有实现**。

### 5.7 人干预的仿真替代品：专家接管

仿真里没有人类，RLinf 用 `expert_takeover` 顶上——一个**单独加载的、训练更充分的 SFT 专家 checkpoint**（`rollout.expert_model`，`use_rlt: False`）：

- 触发模式 `stalled_progress`：只在关键阶段内、且"连续 3 个块没有可测量的插入进展"时才接管；
- 进展用 `peg_head_hole_x`（x 方向前进）、`abs_y/abs_z`（对齐）、以及组合分数 `x - progress_yz_weight * yz_dist` 判定；
- 接管时的动作用 `intervene_flags` 标出，并**改写 `ref_chunk`**，于是 BC target 变成专家动作（见 4.4/5.3）；
- **评估时强制关闭**（`env.eval.rlt_policy_switch.expert_takeover.enable: False`），保证测的是学到的 actor。

这是个很聪明的适配：把论文"人在关键阶段接管"的数据分布，用规则+专家模型造出来了。

### 5.7.1 真机实操：人类怎么接管机械臂

> 仿真那节讲的是"专家的替代品"，这一节讲**真人上手时到底按什么、动什么**。

真机接管有**两个独立的人机交互通道**，别混在一起：

| 通道 | 硬件 | 作用 |
|---|---|---|
| **① 遥操作接管** | SpaceMouse / GELLO / PICO / 手套 | **临时覆盖动作**（人来带机械臂） |
| **② 键盘/pedal 切阶段** | 键盘 `b` 键 | **切换 VLA 与 Stage2 actor 的控制权** |

#### (a) 配置：声明 `teleop` 设备

在 `env.train` / `env.eval` 里声明。RLT 真机用的是 `spacemouse`（`realworld_rlt_stage2_ac_mlp.yaml:119` 和 `:139`）：

```yaml
env:
  train:
    teleop: spacemouse                        # ← 通道①
    keyboard_reward_wrapper: rlt_policy_switch  # ← 通道②
```

可选设备在 `rlinf/robotics/parts/teleop/` 里注册（`@TeleopDevice.register(...)`）：

| 名字 | 类 | 形态 |
|---|---|---|
| `spacemouse` | `spacemouse.py:31` | 6-DoF 小鼠标 + 两个按键 |
| `gello` / `gello_joint` | `gello.py:34` / `gello_joint.py:37` | 主臂（关节空间），可左右各一个 |
| `pico` | `pico.py:50` | VR 手柄 |
| `glove` | `glove.py:29` | 手套（灵巧手） |
| `so101_leader` | `so101_leader.py:39` | SO-101 主臂 |

多设备可以列成数组，各驱动一个部位（双臂各一只 GELLO，`realworld_collect_data_gello_joint_dual_franka.yaml:66-68`）：

```yaml
teleop:
  - {gello_joint: {drives: left}}
  - {gello_joint: {drives: right}}
# 同时要填 left_gello_port / right_gello_port
```

`teleop: none` = 不接设备，控制权完全在策略手上（评估配置常用）。

> ⚠️ **设备必须和环境的动作空间匹配**：`examples/embodiment/config/env/realworld_gim_arm_peg_insertion.yaml:14` 的注释提醒过——那个环境要**关节**指令，只能用 `gello_joint`，SpaceMouse（笛卡尔）不行。

#### (b) 接管怎么发生：`TeleopIntervention` 包装器

核心在 `rlinf/envs/real/wrappers/teleop/intervention.py:91-158`，是一个标准 `gym.Wrapper`，每次 `step()` 决定用谁的动作：

```python
131:  def step(self, action):                      # action = 策略给的
134:      sample = self.device.read(self, action)  # 读设备：你在动吗？按了吗？
136:      if sample.action is None:                applied, overridden = action, False
137:      elif sample.active:                      # ★ 你正在操作
139:          self._last_active = time.monotonic()
140:          applied, overridden = sample.action, True
141:      elif self.intervening:                   # 松手后的 hold 窗口内仍归你
143:          applied, overridden = sample.action, True
144:      elif sample.apply_when_inactive:         # 只应用设备的"状态部分"（如夹爪开合）
146:          applied, overridden = sample.action, False
147:      else:                                    applied, overridden = action, False
150:      obs, reward, terminated, truncated, info = self.env.step(applied)
152:      if overridden:
153:          info["intervene_action"] = applied    # ★ 记下"人类动作"
155:          info["intervene_flag"] = np.ones(1)   # ★ 记下"这一步被干预"
```

三个关键机制：

- **判据是"你有没有动"，不是按某个键**。以 SpaceMouse 为例（`spacemouse.py:119-129`）：`moved = norm(twist) > MOVEMENT_EPSILON`，推一下 puck 或按任一按键 → `driving=True` → 控制权归你。
- **松手后有 hold 窗口**（`TeleopDevice.timeout`，默认 `0.5` 秒，`intervention.py:52`）：

```python
@property
def intervening(self):
    return time.monotonic() - self._last_active < self.device.timeout
```

  这样短暂松手不会立刻把控制权交给策略，避免来回抖动。
- **`apply_when_inactive`**：某些设备的动作含"必须一直生效的状态部分"（如灵巧手的保持姿态），即使你没在驱动也要送进环境。SpaceMouse 的夹爪就走这条（`spacemouse.py:111-117`）。
- **`on_action_chunk_begin()`**（`:160-162`）：每个动作块开始时通知设备，让"只在当前块内保持"的东西松手。

#### (c) 接管数据如何变成训练信号

这是 RLT 最关键的一点——**你的接管不是"打断"，而是直接变成一条模仿样本**。链路（接 5.3 节）：

```text
info["intervene_action"] / info["intervene_flag"]
    ↓  TrajectoryStep.from_parts → apply_interventions()
按 flag 合并成最终执行的 actions（expert_actions * flags + model_actions * ~flags）
    ↓  apply_rlt_interventions（transition.py）
curr_obs["ref_chunk"] ← 被替换成人类的动作
    ↓  存进 replay
BC 损失里：bc_target = where(human_mask, actions, ref_chunk)   （worker :124）
```

一句话：**你接管过的那几步，BC 的目标从"VLA 建议"换成"你做的动作"**——模型直接向你学习。所以真机跑 RLT 时你的角色是：**在旁边盯着，看到策略要插偏了就用 SpaceMouse 掰一把**。这一掰同时贡献两样东西：把这一把拉回正轨（避免失败）+ 一条高质量的模仿样本。

#### (d) 通道②：键盘切换 VLA / Stage2

`keyboard_reward_wrapper: rlt_policy_switch` 会挂上 `KeyboardRLTPolicySwitchWrapper`（`rlinf/envs/real/wrappers/episode/policy_switch.py:22-56`）：

```python
22:  class KeyboardRLTPolicySwitchWrapper(KeyboardSession):
23:      """Press ``b`` to enter the RLT critical phase."""
45:          if key == "b":
47:                  event = "enter_actor"
48:                  self._rlt_switch_flags = True
50:                  self.log("Pedal 'b' pressed; switching RLT rollout to Stage2 actor.")
```

- **按 `b`**（也可以接成脚踏板 pedal）→ 从 VLA 切到 **Stage2 actor**；flag 写进 `info["rlt_switch_flags"]`，传给路由 `route()` 决定这一段用谁的动作；
- 每个 episode 开始时 `begin_episode()` 复位为 `False`（回到 VLA 起点）；
- 按键有 **0.2 秒防抖**（`session.py:36`），按住不会疯狂触发。

**为什么要这个？** RLT 的设计是"**只在关键阶段（critical phase）训一小段**"——如插销的最后几毫米。前半段交给 VLA，快到关键阶段时你按一下 `b` 交给 RL 策略。这样 RL 只需学好那一小段，样本效率才可能做到"几十分钟"。

#### (e) 完整操作流程

```text
1. 硬件：机械臂 + SpaceMouse + 键盘
2. 配置：teleop: spacemouse / keyboard_reward_wrapper: rlt_policy_switch
3. 启动：bash examples/embodiment/run_realworld_async.sh realworld_rlt_stage2_ac_mlp
4. 每个 episode：
   ├─ 起始阶段：VLA 在跑（Stage1）
   ├─ 机械臂接近目标时按 `b` → 切到 Stage2 actor
   ├─ 盯着看：
   │    ├─ 插得好 → 什么都不做，让它自己学
   │    └─ 要跑偏 → 推 SpaceMouse 接管一把，松手 0.5s 后自动交还
   └─ 结束/成功 → 数据自动进 replay，边跑边训
```

#### (f) 实操注意点

1. **hold 窗口别设太长**：默认 0.5 秒。设长了策略会"失联"，且人类动作占比过高。
2. **接管要"有价值"**：看 `human_mask_ratio`（4.4 节）——太低说明没帮上忙，太高说明策略太差或你在过度干预，BC 会主导、RL 学不动。
3. **接管是学习信号，不只是救场**：`bc_human_loss` 专门统计"模型学你学得像不像"。
4. **仿真里没有人类**：用 `expert_takeover` 替代（见上一节）。
5. **评估时该关掉**：`env.eval` 里如果不需要人工干预，用 `teleop: none`，否则测出来的不是策略的真实水平。

### 5.8 同步 vs 异步

| 模式 | 出厂配置 | 入口 | Runner | Worker |
|---|---|---|---|---|
| 同步 | `maniskill_rlt_stage2_{ac,td3}_mlp.yaml` | `examples/embodiment/train_embodied_agent.py` | `EmbodiedRunner` | `RLTACFSDPPolicy` |
| 异步 | `realworld_rlt_stage2_ac_mlp.yaml`（`run_realworld_async.sh`） | `examples/embodiment/train_async.py` | `AsyncEmbodiedRunner` | `AsyncRLTACFSDPPolicy` |

异步版复用了同一套 `_ingest_rollout_trajectories` / `_rlt_updates_to_run`，只在训练循环里 `await asyncio.sleep(0)` 让出控制权并支持 `_drain_received_trajectories` 批量收数据。论文要求的"rollout 与 learning 异步"✅。

**注意：仓库里出厂的配置是"仿真同步、真机异步"**——
`maniskill_rlt_stage2_{ac,td3}_mlp.yaml` 走 `EmbodiedRunner.run`（每个 global step 内 `env.interact` + `rollout.generate` → `actor.recv_rollout_trajectories` → `actor.run_training`，顺序执行），
`realworld_rlt_stage2_ac_mlp.yaml` 走 `AsyncEmbodiedRunner.run`（三个长驻异步循环 + `_recv_queue` 缓冲）。所以论文的"异步"只在真机配置里体现。

注意：`runner.use_training_pipeline=True` 对 `rlt_ac` / `rlt_td3` 会直接报错（`train_embodied_agent.py:59-72`）。

### 5.9 仿真任务与奖励

- 环境：`env_type: maniskill_rlt` → `PegInsertionSideWideClearance-v1`（`pd_joint_delta_pos`，8 维动作，9 维 qpos，`sim_freq: 100 / control_freq: 10`）。
- 观测：`wrap_obs_mode: rlt_openpi_joint`，`num_images_in_input: 2`（主视图 + 腕部），指令固定为 `insert the peg in the hole`。
- 奖励：`reward_mode: only_success` 的**稀疏二值**奖励 ✅，但**是机器判定的**，不是论文里的人类标注：仿真是 `reward = info["success"] * 1.0`（`maniskill_env.py:214-215`）；真机是 Franka 任务的几何判定（到达 `target_ee_pose` 给 1.0，`use_dense_reward: False` 时关掉 `exp(-500·d²)` 的稠密项）。
- 真机环境：`env/realworld_peg_insertion`（Franka），`cluster.num_nodes: 2` 且带一个 `franka` 硬件节点组；干预靠 `teleop: spacemouse` 写入 `info["intervene_action"]/["intervene_flag"]`。
- 论文里 50 Hz / 14 维动作；这里是 **10 Hz / 8 维**，`C=10` 恰好是 1 秒。

---

## 6. 怎么跑起来

```bash
# 0) 环境（RLT 用 openpi 运行时）
bash requirements/install.sh embodied --model openpi --env maniskill_libero
source .venv/bin/activate

# 1) 数据：ManiSkill 关节控制 LeRobot 数据集（官方已放出）
export HF_LEROBOT_HOME=/path/to/lerobot_root
huggingface-cli download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
    --repo-type dataset --local-dir ${HF_LEROBOT_HOME}/maniskill_peginsertionside_joint

python toolkits/lerobot/calculate_norm_stats.py \
    --config-name pi05_rlt_maniskill_joint --repo-id maniskill_peginsertionside_joint

# 2) Stage 1：联合训练 π0.5 + RLT token transformer
#    改 examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml 里的
#    data.train_data_paths / actor.model.model_path / openpi_data.norm_stats_path
bash examples/sft/run_vla_sft.sh maniskill_rlt_stage1_sft_openpi_pi05
#    产物： logs/<run>/checkpoints/global_step_<step>/actor

# 3) Stage 2：冻结 Stage 1，训小 actor-critic
#    把 rollout.rlt_feature_model.model_path 指向上面那个 actor 目录
bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_ac_mlp   # 或 ..._td3_mlp
#    真机：bash examples/embodiment/run_realworld_async.sh realworld_rlt_stage2_ac_mlp
```

三个必须对齐的东西（`rlt.rst` 反复强调）：Stage 1 的 `repo_id` / `config_name` / `norm_stats.json` 与 Stage 2 的 `rollout.rlt_feature_model` 必须一致，否则 VLA 参考动作的尺度会漂。另外 Stage 1 的 checkpoint 要指向 `rollout.rlt_feature_model.model_path`（不是 `rollout.model` / `actor.model`），因为它把 `rlt_module.*` 权重存在完整 wrapper 里。

---

## 7. 论文特性逐条对照表

| # | 论文设计 | RLinf 实现 | 位置 | 状态 |
|---|---|---|---|---|
| 1 | 可学习 `<rl>` token 追加到 VLA 表征后 | `rl_token_embed` + `rl_token_pos_enc` | `rlt_token_transformer.py:131,135` | ✅ |
| 2 | 轻量 encoder transformer 取特殊位输出 | 2 层 self-attn，`x[:, -1:]` | `:107-187` | ✅ |
| 3 | $z_{rl}$ 维度 1×2048 | `rlt_embed_dim: 2048` | `rlt_config.py:33` | ✅ |
| 4 | decoder 自回归重建 VLA 表征 | `RLTTokenDecoder` + causal mask | `:190-296` | ✅ |
| 5 | stop-gradient 重建目标 | 三处 `.detach()` | `:243,358,367`；`pi0.py:819` | ✅ |
| 6 | $\mathcal{L}_{ro}$ 为 L2/MSE | `torch.square(...)` 带 mask 归约 | `:369-379` | ✅ |
| 7 | $\mathcal{L}_{ro} + \alpha\mathcal{L}_{vla}$ | `rlt_loss + rlt_alpha * vla_loss` | `pi0.py:843-847` | ✅ |
| 8 | VLA 与 RLT 模块随后全部冻结 | Stage 2 单独加载为 `rlt_feature_model`，不参与训练 | `huggingface_worker.py:167-174` | ✅ |
| 9 | 状态 $x=(z_{rl}, s_p)$ | `cat([z_rl, proprio])` 给 critic | `rlt_mlp_policy.py:132-133` | ✅ |
| 10 | critic 对整块动作估值 $Q(x, a_{1:C})$ | `flat_action_dim = C × action_dim` | `:57,165` | ✅ |
| 11 | chunk 级 bootstrap $\gamma^C$ | `_discounted_chunk_rewards` + `gamma**reward_horizon` | worker `:87-95,270-278` | ✅ |
| 12 | 双 Q 集成 + target 取 min | `_min_twin_q` + `MultiQHead(num_q_heads=2)` | worker `:79-81` | ✅ |
| 13 | target network | Polyak `tau=0.005` | `fsdp_sac_policy_worker.py:273-319` | ✅ |
| 14 | actor 条件于 VLA 参考动作块 | `cat([ref_chunk, z_rl, proprio])` | `rlt_mlp_policy.py:130` | ✅ |
| 15 | 高斯策略、固定小标准差 | `Normal(mu, fixed_std=0.002)` | `:152-154` | ✅ |
| 16 | BC / KL 正则 $\beta\lVert a-\tilde a\rVert^2$ | 平方 L2（MSE），权重 `bc_weight` | worker `:355` | ✅（L2 而非 KL） |
| 17 | reference dropout 50%，仅训练 | 逐样本整块置零，推理不丢 | `:107-118`，worker `:309-314` | ✅（配置 0.5，**代码默认 0.0**） |
| 18 | VLA warmup 填 buffer | `warmup_min_size` + `warmup_post_collect_updates` + 路由走 ref | worker `:731-827`，`route.py:154-176` | ✅ |
| 19 | 干预时把 buffer 参考块换成人类动作 | `apply_rlt_interventions` / `bc_target = where(human_mask, a, ref)` | `transition.py:60-85`，worker `:124` | ✅ |
| 20 | 只训关键阶段 | `rlt_policy_switch` + `record_transition` 过滤 | `route.py:240`，`maniskill_rlt_env.py` | ✅ |
| 21 | 人类决定交接时刻 | 真机键盘 `b`；仿真规则自动门控 | `policy_switch.py`，`maniskill_rlt_env.py:641+` | ✅（方式不同） |
| 22 | 人在环路：人类标注二值奖励 | 奖励由环境几何判定；人类只负责交接与遥操作干预 | `maniskill_env.py:214-215`，`real/franka/base.py:484-490` | ⚠️ 未用人工奖励 |
| 23 | 人在环路：人类遥操作干预 | 真机 spacemouse 干预 + `intervene_action/flag`；仿真用专家接管替代 | `teleop/intervention.py`，`expert_takeover` | ✅（仿真为替代品） |
| 24 | 测试时 VLA 预测交接时刻 | —— | —— | ❌ 未实现 |
| 25 | 动作块 stride=2 子采样增广 | 一个执行块 ≈ 一条 transition | `embodied_trajectory.py:_append_chunk` | ❌ 未实现 |
| 26 | UTD = 5，critic:actor = 2:1 | `update_epoch / train_every_transitions`；4:1（AC） / 2:1（TD3） | 配置 + worker `:865` | ⚠️ 不同 |
| 27 | rollout 与 learning 异步 | 同步 + 异步双路径（仿真出厂用同步，真机用异步） | `train_async.py`，`async_embodied_runner.py` | ⚠️ 部分 |
| 28 | throughput（每 10 分钟成功次数）指标 | 用 `env/success_once`、`env/episode_len` | `metric_utils.py` | ⚠️ 未提供 |
| 29 | 基础模型 π0.6（SigLIP 400M + Gemma 4B + 860M action expert） | π0.5（`gemma_2b` + `gemma_300m`） | `openpi/config_name: pi05_*` | ⚠️ 模型不同 |
| 30 | 控制频率 50 Hz，单步 14 维，C=10 | 仿真 10 Hz，8 维，C=10（=1 秒） | `env/maniskill_rlt.yaml` | ⚠️ 频率不同 |
| 31 | actor/critic 为 2 层 MLP hidden 256 | AC：3 层 256 tanh；TD3：2 层 256 ReLU | `mlp_policy.py:91-98` | ⚠️ 层数略异 |
| 32 | 参考块长 H=50，执行前缀 C=10 | 仿真 H=C=10；真机 H=20 → 取前 C=10 | 配置 | ⚠️ 不同 |
| 33 | ——（论文未提） | `actor_weight_schedule`：BC/Q 权重 warmup→ramp | worker `:148-225` | ➕ 扩展 |
| 34 | ——（论文未提） | 仿真"停滞进展 → 专家接管"触发器 | `maniskill_rlt_env.py:374+` | ➕ 扩展 |
| 35 | ——（论文未提） | 同步 / 异步双路径可切换 | `train_async.py` | ➕ 扩展 |

图例：✅ 与论文一致 ｜ ⚠️ 实现方式/超参不同 ｜ ❌ 未实现 ｜ ➕ 论文之外的扩展

---

## 8. 读代码时容易踩的坑

1. **两个 `model_type` 别搞混**：`rollout.rlt_feature_model` 是**冻结的 Stage 1 VLA**（含 `rlt_module.*`）；`actor.model` 是 **Stage 2 那个小 MLP**（`rlt_mlp_policy`）。`rollout.model` 只是同步到 rollout 上的 actor 副本。
2. **`z_rl` 的拼接顺序不统一**：AC actor 是 `[ref, z, proprio]`，AC critic 是 `[z, proprio]`，TD3 是 `[z, proprio, ref]`。改网络时要照抄对应的顺序。
3. **`reference_dropout_prob` 代码默认 0**，`rlt_image_only` 代码默认 `True`（丢语言）。两者都靠 YAML 覆盖才有论文行为。
4. **AC 的 Q 用 Q1**，配置项 `actor_agg_q` 在 AC 路径不生效；`agg_q: min` 也只是注释性的（RLT 的 target 计算里是硬编码 `_min_twin_q`）。
5. **`forward_alpha` 会抛异常**：RLT 走的是固定 α=0 的无熵版本，别把 `alpha_type` 设成别的。
6. **`rlt_encoder_type` 是死键**，写它没有任何作用。
7. **ManiSkill 的 `bootstrap_type: standard`** 会在 done 时去掉 bootstrap 项；`use_simulator_transition_replay()` 为真时 done 取 `batch["dones"]` 而不是 `terminations`。
8. **`norm_stats.json` 必须 Stage 1 / Stage 2 共用同一份**，否则 VLA 参考动作尺度漂移，BC 正则会把你往错误的方向拉。
9. **干预只改写 `curr_obs["ref_chunk"]`**，`next_obs["ref_chunk"]` 保持 VLA 原样；BC target 的 `human_mask` 是**按动作块**取的（`reshape(-1, C, A).any(dim=-1)`，对 action 维取 any），也就是说一个块里只要有任一维被干预，整块都改用执行动作当 target。
10. **真机奖励不是人类标的**，是 Franka 到目标位姿的几何判定；想复刻论文的"人工奖励"需要自己加 wrapper（`use_dense_reward` 只控制稠密/稀疏，不引入人工）。
11. **仿真 warmup 期间也在采数据**（关键阶段内的 VLA 样本会入库），真机则把切换前所有 VLA 片段都入库；不要以为 buffer 里全是 actor 的数据。

---

## 9. 测试与验证

- 单元测试：
  - `tests/unit_tests/test_models.py`：`test_decoder_causal_mask_blocks_future_teacher_targets`（因果掩码不漏未来）、`test_reconstruct_output_shape_matches_prefix_embeddings`、`test_reconstruct_detaches_targets_but_trains_encoder_and_decoder`（stop-gradient 语义）。
  - `tests/unit_tests/test_data.py`：`test_rlt_output_matches_main_transition_and_intervention_behavior`（transition/干预语义）、`test_rollout_runs_terminal_inference_for_rlt_without_a_value_head`、`test_rollout_keeps_next_forward_inputs_only_for_rlt`。
- 端到端：
  - `tests/e2e_tests/sft/maniskill_rlt_stage1_sft_openpi_pi05.yaml`
  - `tests/e2e_tests/embodied/maniskill_rlt_stage2_ac_mlp.yaml`（把 `update_epoch→1`、`warmup_min_size→16`、`warmup_post_collect_updates→2`、`max_epochs→2` 缩小到能在 CI 里跑完）
- 复现实验曲线：`docs/source-en/rst_source/examples/embodied/rlt.rst` 末尾给出的 peg_insertion 训练结果图。

---

## 10. 关键代码索引（速查）

```
阶段一
  examples/sft/train_vla_sft.py
  examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml
  rlinf/workers/sft/fsdp_vla_sft_worker.py
  rlinf/models/embodiment/openpi/pi0.py            # 568-609 init, 664-696 select/encode,
                                                   # 776-822 SFT+prefix, 824-847 loss 汇总
  rlinf/models/embodiment/openpi/rlt_config.py
  rlinf/models/embodiment/modules/rlt_token_transformer.py

阶段二 · 网络
  rlinf/models/embodiment/mlp_policy/mlp_policy.py          # 共享 backbone / MultiQHead
  rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py      # AC actor/critic
  rlinf/models/embodiment/mlp_policy/rlt_td3_mlp_policy.py  # TD3 变体
  rlinf/models/embodiment/modules/q_head.py

阶段二 · 损失与训练循环
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py   # 87-95 chunk 折扣, 233-299 critic,
                                                     # 301-368 actor, 462-578 replay 拆分,
                                                     # 731-827 调度, 892-1008 async
  rlinf/workers/actor/fsdp_rlt_td3_policy_worker.py
  rlinf/workers/actor/fsdp_sac_policy_worker.py       # 基类：Polyak / update_one_epoch

阶段二 · 数据与编排
  rlinf/algorithms/rlt/{__init__,rollout,route,transition,expert}.py
  rlinf/data/schema/embodied_types.py                 # TrajectoryStep / 干预处理
  rlinf/data/schema/embodied_trajectory.py            # 轨迹累加（每个 chunk 一个 step）
  rlinf/workers/rollout/hf/huggingface_worker.py      # rlt_feature_model 装配、next_rlt_obs
  rlinf/workers/env/env_worker.py                     # chunk_step / rlt_switch_flags

环境
  rlinf/envs/sim/maniskill/maniskill_rlt_env.py       # env_type maniskill_rlt，关键阶段门控
  rlinf/envs/real/wrappers/episode/policy_switch.py   # 真机键盘切换
  rlinf/envs/real/wrappers/teleop/intervention.py     # 真机遥操作干预

配置 / 文档 / 测试
  examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml
  examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml
  examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml
  examples/embodiment/config/env/maniskill_rlt.yaml
  docs/source-en/rst_source/examples/embodied/rlt.rst
  tests/unit_tests/{test_models.py,test_data.py}
  tests/e2e_tests/{sft,embodied}/*rlt*.yaml
```

---

*本文档基于 `dev/Alex` 分支（commit `fa1579c9`）逐文件阅读整理；行号以该 commit 为准。论文侧概念可对照同目录 [`RL-Token_讲解.md`](./RL-Token_讲解.md)。*
