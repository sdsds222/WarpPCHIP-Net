# WarpPCHIP-Net
WarpPCHIP-Net integrates RNN memory, convolutional extraction, and PCHIP warping for O(N) long-sequence modeling via serial token processing. Core innovations: non-uniform PCHIP memory (sparse distant for compression, dense recent for detail) and learnable sampling grid. Ideal for language modeling and time-series forecasting.

Here is the direct, "as-is" English translation of the technical summary we developed.
WarpPCHIP-Net (RNN-H Hybrid Architecture) Technical Points and Detailed Steps
I. High-Level Concept
WarpPCHIP-Net is an O(N) (Linear Complexity) long-sequence modeling architecture.
Its core idea is to establish a "Dual Memory" System:
 * Working Memory: A standard RNN (like GRU or LSTM) that is responsible for processing the "here and now" input x_k and generating a "current intent" vector h_k.
 * Long-Term Archive: A continuous function f_H(t) built on PCHIP interpolation. This "archive" is built upon all historical RNN states H = [h_1, ..., h_(k-1)]. It is a lossless, differentiable, and searchable "historical semantic trajectory database."
At each timestep, the model uses the "Working Memory" (h_k) as a Query to retrieve the information it needs from the "Long-Term Archive" (f_H(t)), then fuses both to make a final prediction.
II. Core Components
 * RNN (Controller): A standard RNN unit. Its job is not to remember all of history, but to:
   * Generate h_k (current intent).
   * Generate H_history (as the "raw material" for the long-term archive).
 * PCHIP Continuous Archive (f_H(t)): A mathematical function. It accepts a timestamp t (e.g., t=50.5) and can estimate what the h vector (semantic concept) was at that "moment".
 * Global Map Module: A fixed-policy sampler. It uses a fixed "far-sparse, near-dense" L-point grid to sample from f_H(t), generating a stable, low-resolution "historical macro-summary" c_global,k.
 * Dynamic Warped Sampler: A dynamic-policy sampler. It is driven by a "Decision Network (MLP)" which looks at h_k and c_global,k and then actively decides which W "critical points" in f_H(t) to sample at high resolution to fetch the most relevant details h_warped,k.
III. Detailed Forward Pass Steps (at Timestep k)
Assume the model is processing the k-th token and has already stored k-1 historical h vectors.
Stage 1: Generate "Current Intent" (Working Memory)
 * Input:
   * x_k (embedding vector of the current token, dimension d)
   * h_(k-1) (RNN state from the previous step, dimension hidden_dim)
 * Computation: h_k = RNN(h_(k-1), x_k)
 * Output: h_k (dimension hidden_dim)
   * Interpretation: h_k is our "working memory" and "query vector." It represents the state "me, having just seen x_k" and contains the intent of "what information I need now."
Stage 2: Build "Long-Term Archive" (PCHIP Memory)
 * Input:
   * H_history = [h_1, h_2, ..., h_(k-1)] (all past RNN states, dimension (k-1) x hidden_dim)
   * T = [1, 2, ..., k-1] (corresponding timestamps)
 * Computation: The PCHIP algorithm constructs d independent continuous curves over T and H_history.
 * Output: f_H(t) (a continuous function)
   * Interpretation: This function is now our "historical database." We can query it with any time point t (in the range [1, k-1], including decimals), and it will return an estimated h vector.
Stage 3: Scan "Global Map" (Macro Context)
 * Input:
   * f_H(t) (our "historical database")
   * L (a fixed thumbnail length, e.g., L=64)
 * Computation:
   a.  Generate a fixed L-point non-uniform grid t_grid_L (e.g., using logarithmic spacing, making it dense near k-1 and sparse near 1).
   b.  Sample these L points on f_H(t): H_thumb,k = f_H(t_grid_L) (result dimension L x hidden_dim)
   c.  Pass H_thumb,k into a 1D Convolutional Stack (CNN) and a Squeeze-and-Excitation (SE) module.
   d.  Apply Global Average Pooling to the CNN's output.
 * Output: c_global,k (a vector, dimension hidden_dim)
   * Interpretation: c_global,k is a "macro-summary" or "blurry map" of the entire historical semantic trajectory.
Stage 4: Perform "Dynamic Decision" (Generate Query Coordinates)
 * Input:
   * h_k ("current intent" from Stage 1)
   * c_global,k ("macro map" from Stage 3)
 * Computation:
   a.  decision_input = concat(h_k, c_global,k) (concatenate the two vectors, dimension 2 * hidden_dim)
   b.  theta_k = Decision_MLP(decision_input)
 * Output: theta_k (a vector, dimension W, e.g., W=128)
   * Interpretation: theta_k is a "position parameter" vector, representing where the Decision Network thinks the W points of high-value information are approximately located.
Stage 5: Generate "Warped Sampling Grid" (Precise Positioning)
 * Input:
   * theta_k (position parameters from Stage 4)
   * k-1 (the total length of the current history)
 * Computation:
   a.  t_unscaled = sigmoid(theta_k) (normalize the values of theta_k to the [0, 1] range)
   b.  t_positions = t_unscaled * (k-1) (scale the coordinates to the historical range of [0, k-1])
   c.  t_warped = sort(t_positions) (sort these W coordinate points to ensure they are monotonically increasing)
 * Output: t_warped (a W-dimensional vector)
   * Interpretation: This is the set of W timestamps that the model has actively decided to "precisely look at" in the history. It can contain multiple, discontinuous "dense clusters."
Stage 6: Execute "High-Resolution Sampling" (Retrieve Details)
 * Input:
   * f_H(t) (our "historical database")
   * t_warped (the W precise coordinates from Stage 5)
 * Computation:
   a.  h_warped,k = f_H(t_warped) (Sample the function f_H(t) at these W points. All hidden_dim dimensions use the same t_warped coordinates).
 * Output: h_warped,k (a W x hidden_dim tensor)
   * Interpretation: This is the "high-resolution semantic movie" that the model actively retrieved from the "long-term archive." It represents the "historical influence selected by the current word h_k."
Stage 7: Fuse and Predict (Final Output)
 * Input:
   * h_k ("current intent" from Stage 1)
   * h_warped,k ("retrieved details" from Stage 6)
 * Computation:
   a.  h_warped_summary = GlobalAveragePool(h_warped,k) (or use 1D CNN + Pooling to compress W x hidden_dim into 1 x hidden_dim)
   b.  final_input = concat(h_k, h_warped_summary) (fuse the "current intent" with the "historical summary it retrieved")
   c.  logits = Prediction_MLP(final_input)
 * Output: logits (the prediction for the next word)
   * Interpretation: The model makes its final decision based on both "its current state" and "the information it actively selected from its history."
Stage 8: Advance (Prepare for Next Step)
 * Action:
   a.  The model outputs logits.
   b.  The system advances to timestep k+1.
   c.  h_k (the current state) now becomes h_(k-1) (the previous state).
   d.  h_k is added to the H_history archive, ready to be used in the next loop's Stage 2 to build the new f_H(t).


# Thinking part:

### Architecture Technical Summary: Hybrid Sequence Model with Continuous Interpolation and State Compression

This architecture is a hybrid sequence model designed to achieve dynamic, sparse access to long-range dependencies under O(N) linear computational complexity.

It processes local context via a stateful sequential controller (i.e., W-Head) and handles non-local, sparse global context via a stateless dynamic read head (i.e., R-Head).

#### I. Core Components (Static Definition)

1. **Stateful Controller (RNN_ctrl):**
   * **Definition:** A recurrent neural network (e.g., GRU or LSTM) serving as the system's backbone.
   * **Responsibilities:** 1. Sequentially process the input sequence e_k. 2. Maintain a hidden state h_k that encodes dense sequential history. 3. Act as the "decision center" for all sub-modules.

2. **Continuous Memory Modules:**
   * **E (Embedding Matrix):** An N x d_model discrete embedding matrix.
   * **H (Heatmap Vector):** An N x 1 discrete heat (access) vector.
   * **f_E(t) (Content Interpolator):** A differentiable interpolation function based on E (e.g., PCHIP). Input a continuous scalar t (e.g., t=4.2), output a d_model-dimensional interpolated vector e_read.
   * **f_H(t) (Heat Interpolator):** A differentiable interpolation function based on H. Input t, output a scalar h_read.

3. **O(1) State Caches:**
   * **C_vec (Content Cache):** A d_model-dimensional vector. Recursively compresses all content e_read read by R-Head via EMA (Exponential Moving Average).
   * **J_vec (Action Cache):** A d_action-dimensional vector. Recursively compresses all actions j_t executed by R-Head via EMA.

4. **Auxiliary Processors (MLPs):**
   * **ActionEncoder (MLP):**
     * **Input:** [t_R, D] (R-Head's absolute position t_R and relative distance D = k - t_R).
     * **Output:** j_t (d_action-dimensional action embedding) for updating J_vec.
   * **JumpController (MLP):**
     * **Input:** concat(h_k, C_vec_old, J_vec_old). (C_vec_old and J_vec_old represent the caches from the previous step).
     * **Output:** t_R (next continuous jump position).
   * **OutputPredictor (MLP):**
     * **Input:** concat(h_k, e_read, h_read, D, ...).
     * **Output:** Logits (log probabilities over the vocabulary).

#### II. Computation Graph (Dynamic Flow)

This is the detailed forward propagation (Forward Pass) when the model processes the **k-th timestep**. Assume M R-Head jumps (for simplicity, the following describes the case for M=1).

##### Step 1: Sequential State Update

1. The controller RNN_ctrl receives its hidden state h_(k-1) from the previous timestep and the current position's word embedding e_k (directly read from E[k]).
2. RNN_ctrl computes and outputs its current timestep's hidden state h_k.
   * h_k = RNN_ctrl(h_(k-1), e_k)
   * h_k now encodes all dense sequential context up to k.

##### Step 2: R-Head Policy Execution

1. JumpController is activated to decide R-Head's target position.
2. It collects all available historical context:
   * h_k (from Step 1)
   * C_vec_old (from R-Head content cache at step k-1)
   * J_vec_old (from R-Head action cache at step k-1)
3. JumpController outputs a continuous scalar t_R:
   * policy_input = concat(h_k, C_vec_old, J_vec_old)
   * t_R = sigmoid(MLP_jump(policy_input)) * k
   * (Sigmoid is used to constrain t_R to the left of W-Head, i.e., in the [0, k] range).

##### Step 3: Differentiable Memory Access

1. R-Head executes the jump to target t_R (e.g., t_R = 4.2).
2. The model reads data from continuous memory via interpolators f_E and f_H:
   * `e_read = f_E(t_R)` (interpolated from E[4] and E[5] to obtain a d_model-dimensional vector)
   * `h_read = f_H(t_R)` (interpolated from H[4] and H[5] to obtain a scalar)

##### Step 4: Predictive Output Generation

1. OutputPredictor collects all available contextual information to make the final prediction:
   * h_k (dense sequential context)
   * e_read (sparse content read back by R-Head)
   * h_read (heat read back by R-Head)
   * D = k - t_R (relative distance)
   * C_vec_old (R-Head's historical content)
   * J_vec_old (R-Head's historical actions)
2. OutputPredictor computes Logits for predicting the (k+1)-th word:
   * predict_input = concat(h_k, e_read, h_read, D, C_vec_old, J_vec_old)
   * Logits = MLP_predict(predict_input)

##### Step 5: State and Memory Update

After Logits are used to compute the loss (Loss), the model updates its state and memory for the next step (k+1). These three updates can be performed in parallel:

1. **Update Content Cache (C_vec):**
   * C_vec_new <- beta * C_vec_old + (1 - beta) * e_read
   * (beta is the EMA forgetting factor, e.g., 0.8). This is an O(d) operation.

2. **Update Action Cache (J_vec):**
   * j_t = ActionEncoder([t_R, D]) (MLP, O(d^2) operation)
   * J_vec_new <- alpha * J_vec_old + (1 - alpha) * j_t
   * (alpha is the EMA forgetting factor, e.g., 0.9). This is an O(d) operation.

3. **Update Heatmap (H):**
   * This is a differentiable write operation.
   * The model assigns a heat increment delta_h to t_R (e.g., delta_h = 1).
   * delta_h is linearly distributed to the adjacent integer indices of t_R.
   * For example, for t_R = 4.2:
     * H[4] <- H[4] + (1 - 0.2) * delta_h
     * H[5] <- H[5] + (0.2) * delta_h

##### Step 6: Advance

1. W-Head index k increments to k+1.
2. h_(k-1) <- h_k.
3. C_vec_old <- C_vec_new.
4. J_vec_old <- J_vec_new.
5. Return to **Step 1**.

# WarpPCHIP-Net

WarpPCHIP Net 融合RNN记忆、卷积提取及PCHIP扭曲采样，实现O(N)长序列建模与串行token处理。其核心创新：非均匀PCHIP内存（远期稀疏压缩、近期密集细节）和学可学习采样网格（端到端自由多点聚焦）。适用于语言建模和时间序列预测。

一、概念
WarpPCHIP-Net 是一种 O(N)（线性复杂度）的长序列建模架构。
其核心思想是建立一个**“双记忆”系统**：
 * 工作记忆 (Working Memory): 一个标准的 RNN（如 GRU 或 LSTM），它负责处理“此时此刻”的输入 x_k，并生成一个“当前意图”向量 h_k。
 * 长期档案 (Long-Term Archive): 一个基于 PCHIP 插值构建的连续函数 f_H(t)。这个“档案”是建立在所有历史 RNN 状态 H = [h_1, ..., h_(k-1)] 之上的，它是一个无损的、可微分的、可搜索的“历史语义轨迹数据库”。
在每个时间步，模型使用“工作记忆”(h_k) 作为查询（Query），去“长期档案”(f_H(t)) 中检索它需要的信息，然后将两者融合以做出最终预测。
二、核心组件
 * RNN (控制器): 一个标准的 RNN 单元。它的工作不是记住所有历史，而是：
   * 生成 h_k（当前意图）。
   * 生成 H_history（作为长期档案的“原材料”）。
 * PCHIP 连续档案 (f_H(t)): 一个数学函数。它接收一个时间戳 t (例如 t=50.5)，并能估算出在那个“时刻”的 h 向量（语义概念）是什么样子的。
 * 全局地图模块 (Global Map Module): 一个固定策略的采样器。它使用一个固定的“远稀疏、近密集”的 L 点网格，从 f_H(t) 中采样，生成一个稳定的、低分辨率的“历史宏观摘要” c_global,k。
 * 动态扭曲采样器 (Dynamic Warped Sampler): 一个动态策略的采样器。它由一个“决策网络 (MLP)”驱动，该网络会查看 h_k 和 c_global,k，然后主动决定应该去 f_H(t) 的哪 W 个“关键点”进行高分辨率采样，以获取最相关的细节 h_warped,k。
三、详细的前向传播步骤 (在时间步 k)
假设模型正在处理第 k 个 token，并且已经存储了 k-1 个历史 h 向量。
阶段 1：生成“当前意图” (工作记忆)
 * 输入：
   * x_k (当前 token 的嵌入向量, 维度 d)
   * h_(k-1) (上一步的 RNN 状态, 维度 hidden_dim)
 * 计算： h_k = RNN(h_(k-1), x_k)
 * 输出： h_k (维度 hidden_dim)
   * 解读： h_k 是我们的“工作记忆”和“查询向量”。它代表了“刚刚看到了 x_k 之后的我”这个状态，并包含了“我现在需要什么信息”的意图。
阶段 2：构建“长期档案” (PCHIP 内存)
 * 输入：
   * H_history = [h_1, h_2, ..., h_(k-1)] (所有过去的 RNN 状态，维度 (k-1) x hidden_dim)
   * T = [1, 2, ..., k-1] (对应的时间戳)
 * 计算： PCHIP 算法在 T 和 H_history 上构建 d 个独立的连续曲线。
 * 输出： f_H(t) (一个连续函数)
   * 解读： 这个函数现在是我们的“历史数据库”。我们可以用任意时间点 t (在 [1, k-1] 范围内，包括小数) 来查询它，它会返回一个估算的 h 向量。
阶段 3：扫描“全局地图” (宏观上下文)
 * 输入：
   * f_H(t) (我们的“历史数据库”)
   * L (一个固定的缩略图长度, e.g., L=64)
 * 计算：
   a.  生成一个固定的 L 点非均匀网格 t_grid_L (例如，使用对数间隔，使其在 k-1 附近密集，在 1 附近稀疏)。
   b.  在 f_H(t) 上采样这 L 个点：H_thumb,k = f_H(t_grid_L) (结果维度 L x hidden_dim)
   c.  将 H_thumb,k 传入一个 1D 卷积栈 (CNN) 和挤压-激励 (SE) 模块。
   d.  对 CNN 的输出进行全局平均池化 (Global Average Pooling)。
 * 输出： c_global,k (一个向量, 维度 hidden_dim)
   * 解读： c_global,k 是对整个历史语义轨迹的“宏观总结”或“模糊地图”。
阶段 4：执行“动态决策” (生成查询坐标)
 * 输入：
   * h_k (来自阶段 1 的“当前意图”)
   * c_global,k (来自阶段 3 的“宏观地图”)
 * 计算：
   a.  decision_input = concat(h_k, c_global,k) (拼接两个向量，维度 2 * hidden_dim)
   b.  theta_k = Decision_MLP(decision_input)
 * 输出： theta_k (一个向量, 维度 W, e.g., W=128)
   * 解读： theta_k 是一个“位置参数”向量，它代表了决策网络“认为” W 个高价值信息点大致在哪里。
阶段 5：生成“扭曲采样网格” (精确定位)
 * 输入：
   * theta_k (来自阶段 4 的位置参数)
   * k-1 (当前历史的总长度)
 * 计算：
   a.  t_unscaled = sigmoid(theta_k) (将 theta_k 的值归一化到 [0, 1] 范围)
   b.  t_positions = t_unscaled * (k-1) (将坐标缩放到 [0, k-1] 的历史范围)
   c.  t_warped = sort(t_positions) (对这 W 个坐标点进行排序，确保它们单调递增)
 * 输出： t_warped (一个 W 维向量)
   * 解读： 这是模型最终主动决定要去历史中“精确查看”的 W 个时间戳。它可以包含多个不连续的“密集簇”。
阶段 6：执行“高分辨率采样” (检索细节)
 * 输入：
   * f_H(t) (我们的“历史数据库”)
   * t_warped (来自阶段 5 的 W 个精确坐标)
 * 计算：
   a.  h_warped,k = f_H(t_warped) (在 f_H(t) 上采样这 W 个点。所有 hidden_dim 维度都使用相同的 t_warped 坐标)。
 * 输出： h_warped,k (一个 W x hidden_dim 的张量)
   * 解读： 这是模型从“长期档案”中主动检索回来的“高分辨率语义电影”。它代表了“当前词 h_k 所选择的历史影响”。
阶段 7：融合与预测 (最终输出)
 * 输入：
   * h_k (来自阶段 1 的“当前意图”)
   * h_warped,k (来自阶段 6 的“检索到的细节”)
 * 计算：
   a.  h_warped_summary = GlobalAveragePool(h_warped,k) (或者用 1D CNN + Pooling 将 W x hidden_dim 压缩成 1 x hidden_dim)
   b.  final_input = concat(h_k, h_warped_summary) (融合“当前意图”和“它检索到的历史摘要”)
   c.  logits = Prediction_MLP(final_input)
 * 输出： logits (对下一个词的预测)
   * 解读： 模型基于“它现在的状态”和“它主动从历史中选择的信息”共同做出最终决定。
阶段 8：推进 (为下一步做准备)
 * 动作：
   a.  模型输出 logits。
   b.  系统推进到时间步 k+1。
   c.  h_k (当前状态) 现在变成了 h_(k-1) (上一步状态)。
   d.  h_k 被添加进 H_history 档案库，准备在下一个循环的阶段 2 中被用于构建新的 

# 设想部分 Thinking part:


### 架构技术总结：连续插值与状态压缩的混合序列模型

本架构是一个混合序列模型，其设计目标是在 O(N) 的线性计算复杂度下，实现对长距离依赖的动态、稀疏访问。

它通过一个有状态的顺序控制器 （即 W-Head）来处理局部上下文，并辅以一个无状态的动态读取头（即 R-Head）来处理非局部的、稀疏的全局上下文。

#### 一、 核心组件（静态定义）

1.  **有状态控制器 (Stateful Controller - RNN_ctrl):**
    * **定义:** 一个循环神经网络（如 GRU 或 LSTM），作为系统的主干。
    * **职责:** 1. 顺序处理输入序列 e_k。 2. 维护一个编码了稠密顺序历史的隐藏状态 h_k。 3. 作为所有子模块的“决策中心”。

2.  **连续内存模块 (Continuous Memory Modules):**
    * **E (Embedding Matrix):** N x d_model 的离散嵌入矩阵。
    * **H (Heatmap Vector):** N x 1 的离散热度（访问）向量。
    * **f_E(t) (内容插值器):** 一个基于 E 的可微分插值函数（例如 PCHIP）。输入一个连续标量 t (例如 t=4.2)，输出一个 d_model 维的插值向量 e_read。
    * **f_H(t) (热度插值器):** 一个基于 H 的可微分插值函数。输入 t，输出一个标量 h_read。

3.  **O(1) 状态缓存 (State Caches):**
    * **C_vec (内容缓存):** d_model 维向量。通过 EMA（指数移动平均）递归压缩 R-Head *读取*过的所有内容 e_read。
    * **J_vec (动作缓存):** d_action 维向量。通过 EMA 递归压缩 R-Head *执行*过的所有动作 j_t。

4.  **辅助处理器 (Sub-Processors - MLPs):**
    * **ActionEncoder (MLP):**
        * **输入:** [t_R, D] (R-Head 的绝对位置 t_R 和相对距离 D = k - t_R)。
        * **输出:** j_t (d_action 维的动作嵌入)，用于更新 J_vec。
    * **JumpController (MLP):**
        * **输入:** concat(h_k, C_vec_old, J_vec_old)。(C_vec_old 和 J_vec_old 代表上一轮的缓存)。
        * **输出:** t_R (下一个连续跳跃位置)。
    * **OutputPredictor (MLP):**
        * **输入:** concat(h_k, e_read, h_read, D, ...)。
        * **输出:** Logits (词汇表的对数概率)。

#### 二、 计算图（动态流程）

这是模型在处理**第 k 个时间步**时的详细前向传播（Forward Pass）步骤。假设我们有 M 次 R-Head 跳跃（为简化，下面描述 M=1 的情况）。

##### 步骤 1: 顺序状态更新 (Sequential State Update)

1.  控制器 RNN_ctrl 接收其上一时刻的隐藏状态 h_(k-1)和当前位置的词嵌入 e_k（从 E[k] 直接读取）。
2.  RNN_ctrl 计算并输出其当前时刻的隐藏状态 h_k。
    * h_k = RNN_ctrl(h_(k-1), e_k)
    * h_k 现在编码了到 k 为止的所有稠密顺序上下文。

##### 步骤 2: R-Head 策略执行 (Read Head Policy Execution)

1.  JumpController 被激活，用于决定 R-Head 的目标位置。
2.  它收集当前所有可用的历史上下文：
    * h_k (来自步骤 1)
    * C_vec_old (来自 k-1 步的 R-Head 内容缓存)
    * J_vec_old (来自 k-1 步的 R-Head 动作缓存)
3.  JumpController 输出一个连续标量 t_R：
    * policy_input = concat(h_k, C_vec_old, J_vec_old)
    * t_R = sigmoid(MLP_jump(policy_input)) * k
    * (使用 sigmoid 将 t_R 约束在 W-Head 的左侧，即 [0, k] 范围内)。

##### 步骤 3: 可微分内存访问 (Differentiable Memory Access)

1.  R-Head 执行跳跃，目标为 t_R（例如 t_R = 4.2）。
2.  模型通过插值器 f_E 和 f_H 从连续内存中读取数据：
    * `e_read = f_E(t_R)` (从 E[4] 和 E[5] 插值得到 d_model 维向量)
    * `h_read = f_H(t_R)` (从 H[4] 和 H[5] 插值得到标量)

##### 步骤 4: 最终输出生成 (Predictive Output Generation)

1.  OutputPredictor 收集所有可用的上下文信息，以做出最终预测：
    * h_k (稠密顺序上下文)
    * e_read (R-Head 读回的稀疏内容)
    * h_read (R-Head 读回的热度)
    * D = k - t_R (相对距离)
    * C_vec_old (R-Head 的历史内容)
    * J_vec_old (R-Head 的历史动作)
2.  OutputPredictor 计算 Logits，用于预测第 k+1 个词：
    * predict_input = concat(h_k, e_read, h_read, D, C_vec_old, J_vec_old)
    * Logits = MLP_predict(predict_input)

##### 步骤 5: 状态与内存更新 (State and Memory Update)

在 Logits 被用于计算损失（Loss）之后，模型为下一步（k+1）更新其状态和内存。这三项更新可以并行执行：

1.  **更新内容缓存 (C_vec):**
    * C_vec_new <- beta * C_vec_old + (1 - beta) * e_read
    * (beta 是 EMA 遗忘因子, 例如 0.8)。这是一个 O(d) 的操作。

2.  **更新动作缓存 (J_vec):**
    * j_t = ActionEncoder([t_R, D]) (MLP, O(d^2) 操作)
    * J_vec_new <- alpha * J_vec_old + (1 - alpha) * j_t
    * (alpha 是 EMA 遗忘因子, 例如 0.9)。这是一个 O(d) 的操作。

3.  **更新热度地图 (H):**
    * 这是一个可微分写入操作。
    * 模型为 t_R 分配一个热度增量 delta_h (例如 delta_h = 1)。
    * delta_h 被按线性比例分配给 t_R 相邻的整数索引。
    * 例如，对于 t_R = 4.2：
        * H[4] <- H[4] + (1 - 0.2) * delta_h
        * H[5] <- H[5] + (0.2) * delta_h

##### 步骤 6: 推进 (Advance)

1.  W-Head 索引 k 增加到 k+1。
2.  h_(k-1) <- h_k。
3.  C_vec_old <- C_vec_new。
4.  J_vec_old <- J_vec_new。
5.  返回**步骤 1**。


