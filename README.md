# WarpPCHIP-Net
WarpPCHIP-Net integrates RNN memory, convolutional extraction, and PCHIP warping for O(N) long-sequence modeling via serial token processing. Core innovations: non-uniform PCHIP memory (sparse distant for compression, dense recent for detail) and learnable sampling grid. Ideal for language modeling and time-series forecasting.

## Technical Details, Concepts, and Steps

The following explains the architecture's core concepts and technical details, including the newly added PCHIP non-uniform resolution optimization, and clarifies the magnification mechanism, PCHIP multi-dimensional modeling, convolutional scanning, and sampling-point adjustment. We describe the model's construction and runtime flow step by step, covering concepts and logic only, without any specific code. We emphasize that magnifying key details is achieved by adjusting the positions of sampling points; PCHIP models all dimensions of all historical tokens; the convolutional kernels scan all dimensions on the thumbnail simultaneously; and sampling-point adjustments are synchronized, with one shared scheme across all dimensions.

In addition, we upgrade the warping mechanism to a fully learnable sampling grid: the decision network directly outputs a length W parameter vector (for example, offsets or positions), allowing the network to freely plan the distribution of sampling points rather than relying on a fixed center. This enables end-to-end learning of optimal sampling locations, densifying or sparsifying any historical regions as needed, even magnifying multiple discontinuous areas at once, to better capture complex long-range dependencies. Constraints such as sigmoid normalization to the range [0, k] and soft sorting ensure monotonic increase and ordering of sampling points, avoiding overlap or disorder.

### I. Core Component Concepts

1. RNN Controller: The sequential-memory core that reads inputs step by step and maintains a fixed-size hidden state h_k. This state captures current intent, such as recognizing local patterns in the sequence. The hidden state updates recursively at each time step based on h_{k-1} and the current input x_k, in a form like
   h_k = RNN(h_{k-1}, x_k),
   where the RNN can be a GRU or LSTM for improved stability.

2. PCHIP Continuous Memory: Converts the historical embedding sequence E (embeddings of all tokens before the current token k, an N x d tensor, with d dimensions) into a continuous curve f_E(t). Modeling details: PCHIP is applied simultaneously to all d dimensions of all historical tokens, with each dimension's curve built independently to ensure continuity and differentiable sampling for the multi-dimensional history. Specifically, for each dimension j (1 to d), we use historical points (t_i, E_{i,j}) to build a piecewise cubic Hermite polynomial; shape-preserving properties prevent spurious oscillations. New optimization: the base resolution is non-uniform. The distant past is sparse (larger spacing between sampling points to compress far-past information and reduce redundancy and noise), while the recent past is dense (smaller spacing to retain recent detail). This is achieved by applying a nonlinear transform to the time axis t (for example, logarithmic scaling t_scaled = log(t + 1) * scale_factor). PCHIP curves are then constructed on these non-uniform points. Benefits: higher density near the present boosts local accuracy; far-past compression reduces wasted computation, while retaining overall O(N) complexity.

3. Convolutional Thumbnail: A submodule that scans historical macro-patterns to produce a fixed-size map vector c_global,k. Details: from the PCHIP curve we sample a fixed-length thumbnail H_thumb,k of length L (an L x d tensor). Sampling points follow the prior non-uniform scheme (sparse far past, dense recent past) distributed over [0, k], and the thumbnail is dynamically redrawn as k grows to track the latest changes. We then apply a 1D convolutional stack: kernels (size for example 3 or 5) simultaneously scan each dimension of the historical thumbnail, sliding along the L time points while extracting patterns across the d dimensions in parallel, capturing cross-dimensional temporal relations and local correlations (for example, edge detection or periodic motifs). Next, apply a Squeeze-and-Excitation (SE) mechanism: global average pool the thumbnail to get a d-dimensional vector; pass it through a small MLP (two fully connected layers) to compute per-dimension scaling factors (sigmoid); multiply them back dimension-wise to recalibrate features, suppressing noise and amplifying key patterns. Finally, global pooling (average or max) summarizes to a fixed vector c_global,k, representing a macro location and pattern summary of the entire history.

4. Decision Network: Fuses all information to compute warping parameters theta_k. The input is the concatenation of the intent h_k, the map c_global,k, and the current input x_k (total dimension 3 * hidden_dim), processed by an MLP (for example, 2 to 3 fully connected layers, ReLU activations). It directly outputs a length W parameter vector (for example, position offsets delta_t or absolute positions) used to generate the warped grid t_warped. The network has full freedom to plan: through training, it learns optimal distributions, creating dense clusters at any positions (thus magnifying multiple discontinuous key details).

### II. Model Construction Steps

1. Define Inputs and Outputs: Specify embedding dimension d, hidden-state size hidden_dim, fixed thumbnail length L (for example, 64, to keep computation constant), warped-grid length W (for example, 128, as a fixed number of sampling points), and the output dimension (for example, vocabulary size). Ensure all components support multi-dimensional d and non-uniform PCHIP.

2. Build the RNN Module: Configure a standard RNN (for example, GRU) to update the hidden state h_k. The RNN input layer projects x_k (dimension d) to hidden_dim, while h_{k-1} is passed directly; the output layer remains hidden_dim. Optionally add a residual connection (for example, h_k = h_k + h_{k-1}) to alleviate vanishing gradients.

3. Implement PCHIP Memory: Create the continuous curve representation. Steps: collect embeddings E of all historical tokens (N x d); apply a nonlinear transform to the time axis t (from 0 to k-1) to make the far past sparse and the recent past dense (for example, t_scaled[i] = log(i + 1) / log(k) * k, for normalization); for each dimension j, independently but synchronously build the PCHIP curve f_E^j(t): compute Hermite coefficients for each segment, ensuring monotonicity-preserving and differentiability (gradients propagate via linear combinations). The result is a multi-dimensional continuous function that supports vector sampling at arbitrary t (query each dimension in parallel at sampling time).

4. Add the Convolutional Stack: Design a sequence of 1D convolution layers (for example, 2 to 3 Conv1D with kernel size 3, stride 1, padding to keep length L). Emphasis: kernels scan all dimensions of H_thumb,k simultaneously; while sliding along the time axis, they process the d channels to extract spatiotemporal motifs (for example, repeated sequence patterns). Combine with SE: global-average-pool the thumbnail to a d-vector; an MLP (FC to hidden_dim/16 then back to d, sigmoid) outputs d scaling factors; multiply them back dimension-wise. Optional batch normalization can further stabilize training.

5. Set Up the Decision Network: Use an MLP with the concatenated input vector (dimension 3 * hidden_dim); hidden layers can taper (for example, from 2 * hidden_dim down to hidden_dim); output a length W vector theta_k (position parameters). Emphasize free planning: the output directly drives warping, for example
   t_warped = sort(sigmoid(theta_k) * k),
   ensuring points lie in [0, k] and are ordered; or treat theta_k as gaps with
   t_warped = cumsum(softplus(theta_k))
   to accumulate intervals. Constraints are implemented via differentiable operations to prevent collapse.

6. Integrate Warped Sampling: Define an initial uniform grid t_uniform (length W, evenly spaced over [0, 1]). Warping details: nonlinearly adjust sampling-point positions based on theta_k, for example
   t_warped[i] = cumsum(exp(theta_k[i])) * (k / sum(exp(theta_k))),
   enabling arbitrary density distributions (the network learns theta_k to create small intervals at key locations for magnification). Produce an ordered t_warped (length W, within [0, k]). Sample from the PCHIP continuous curve (modeled across all dimensions and benefiting from the non-uniform base resolution) uniformly across dimensions: all d dimensions share the same t_warped scheme, synchronously querying each dimension's curve to obtain high-resolution detail vectors h_warped,k (W x d), with learned high-density positions at multiple locations.

7. Final Prediction Layer: Another MLP fuses the intent h_k and details h_warped,k (flatten h_warped,k or use a 1D conv plus pooling to hidden_dim). Concatenate and pass through two fully connected layers to compute output logits. Optional attention can weight the warped points during fusion.

### III. Forward Pass Flow (Per Time Step)

At each time step k, the model executes the following five phases in a loop, ensuring end-to-end differentiability.

Phase 1: Generate Sequential Intent

* Receive the previous hidden state h_{k-1} and current input embedding x_k (dimension d).
* Update via the RNN to obtain the new intent vector h_k = RNN(h_{k-1}, x_k), summarizing local context and short-term dependencies.

Phase 2: Produce the Global Map

* From the PCHIP curve (with non-uniform resolution and modeling across all dimensions), sample a fixed-length thumbnail H_thumb,k (L x d), with sampling points spread over [0, k] and dynamically updated as k grows to reflect the latest history.
* Use the 1D convolutional stack to scan the thumbnail; kernels simultaneously scan each dimension, sliding over the time axis while processing all d dimensions to extract multi-scale patterns.
* Apply the SE mechanism: global-average-pool to compute a mean feature vector (d-dimensional); a small MLP outputs d scaling factors (sigmoid); adjust the thumbnail dimension-wise to amplify key features and suppress noise.
* Global pooling (average) yields the map vector c_global,k, representing a macro summary of the history.

Phase 3: Decision and Localization

* Concatenate the intent vector h_k, map vector c_global,k, and current input x_k (total dimension 3 * hidden_dim).
* Pass through the decision MLP to compute parameters theta_k (length W), with the network freely planning the sampling-point distribution for subsequent warping.

Phase 4: Execute Scaling and High-Resolution Sampling

* Use theta_k to warp the initial uniform grid t_uniform, nonlinearly adjusting sampling-point positions. Based on theta_k, compute gaps or offsets to realize arbitrary densities (for example, small values create dense points to magnify key details; large values create sparse regions). Generate an ordered t_warped (length W, within [0, k]) via cumulative sums or soft sorting.
* Sample uniformly from the PCHIP continuous curve across dimensions (all-dimension modeling with non-uniform resolution): all d dimensions share the same t_warped scheme, synchronously querying each dimension's curve to obtain high-resolution detail vectors h_warped,k (W x d), where the model-learned positions can be densely clustered at multiple sites.

Phase 5: Final Prediction and Progression

* Fuse the intent h_k and details h_warped,k (flatten or pool h_warped,k), and compute outputs (for example, next-token logits) via the prediction MLP.
* Advance to the next time step k+1; update the history E by appending x_k.

# WarpPCHIP-Net
WarpPCHIP Net 融合RNN记忆、卷积提取及PCHIP扭曲采样，实现O(N)长序列建模与串行token处理。其核心创新：非均匀PCHIP内存（远期稀疏压缩、近期密集细节）和学可学习采样网格（端到端自由多点聚焦）。适用于语言建模和时间序列预测。

## 技术细节、概念和步骤

以下详细讲解架构的核心概念和技术细节，包括新加入的 PCHIP 非均匀分辨率优化，以及对放大机制、PCHIP 多维建模、卷积扫描和采样点调整的澄清。我们分步骤描述模型的构建和运行流程，只交代概念和逻辑，不涉及具体代码实现。重点强调放大关键细节是通过调整采样点位置实现的；PCHIP 针对所有历史 token 的所有维度建模；卷积核在缩略图的每个维度上同时扫描；采样点调整是同步的，所有维度共用一套方案。

此外，我们将扭曲机制优化为完全学可学习的采样网格：决策网络直接输出一个长度 W 的参数向量（例如偏移或位置向量），让网络全权自由规划采样点的分布，而非基于固定中心。这允许模型端到端学习最优采样点位置，能任意密集或稀疏任意历史位置，甚至同时放大多个不连续区域，提高对复杂长依赖的捕捉灵活性。采样点通过约束（如 sigmoid 归一化到 [0, k] 范围并软排序）确保单调递增和有序，避免重叠或混乱。

### I. 核心组件概念

1. **RNN 控制器**：系统的顺序记忆核心，负责逐步读取输入并维护固定大小的隐藏状态 h_k。这个状态捕捉当前意图，例如识别序列中的局部模式。隐藏状态在每个时间步基于上一步 h_{k-1} 和当前输入 x_k 递归更新，公式类似 h_k = RNN(h_{k-1}, x_k)，其中 RNN 可以是 GRU 或 LSTM 以提升稳定性。

2. **PCHIP 连续内存**：将历史嵌入序列 E（所有当前扫描词 k 之前的 token 的嵌入，一个 N x d 张量，d 是维度）转换为连续曲线 f_E(t)。建模细节：针对所有历史 token 的所有 d 个维度同时进行 PCHIP 建模，每个维度独立构造曲线，确保多维历史的连续性和可微分采样。具体过程：对于每个维度 j (1 到 d)，使用历史点 (t_i, E_{i,j}) 构建分段三次埃尔米特多项式，保形性确保曲线不引入额外振荡。新优化：初始分辨率非均匀——远期历史稀疏（采样点间隔大，压缩远期信息以减少冗余和噪声），近期历史密集（采样点间隔小，保留近期细节）。这通过对时间轴 t 应用非线性变换（如对数尺度 t_scaled = log(t + 1) * scale_factor）实现，变换后基于这些非均匀点构建 PCHIP 曲线。好处：近期高密度提升局部精度，远期压缩减少计算浪费，同时保持整体 O(N) 复杂度。

3. **卷积缩略图**：子模块扫描历史宏观模式，生成固定大小的地图向量 c_global,k。细节：从 PCHIP 曲线采样固定长度 L 的缩略图 H_thumb,k（一个 L x d 张量），采样点按照之前远期稀疏、近期密集的调整非均匀分布在 [0, k] 上，随着 k 增长动态局部重绘以捕捉最新变化。然后用一维卷积栈处理：卷积核（大小例如 3 或 5）在历史缩略图的每个维度上同时进行特征扫描——核沿着 L 个时间点滑动，并在 d 个维度上并行提取模式，捕捉跨维度的时序关系和局部相关性（如边缘检测或周期模式）。接着应用挤压-激励机制（SE）：对缩略图进行全局平均池化，得到一个 d 维向量；通过小型 MLP（两层全连接）计算每个维度的缩放因子（sigmoid 激活）；逐维度乘回缩略图以重标定特征，压制噪声并放大关键模式。最后全局池化（平均或最大）汇总成固定向量 c_global,k，代表整个历史的宏观位置信息和模式总结。

4. **决策网络**：融合所有信息，计算扭曲参数 theta_k。输入为意图 h_k、地图 c_global,k 和当前输入 x_k 的拼接（总维度 3 * hidden_dim），通过多层感知机（MLP，例如 2-3 层全连接，ReLU 激活）处理。直接输出一个长度 W 的参数向量（例如位置偏移 delta_t 或直接位置），用于生成扭曲网格 t_warped。网络全权自由规划：模型通过训练学习最优分布，能在任意位置创建密集簇（放大多个不连续关键细节）。

### II. 模型构建步骤

1. **定义输入和输出**：指定嵌入维度 d、隐藏状态大小 hidden_dim、固定缩略图长度 L（例如 64，以保持常量计算）、扭曲网格长度 W（例如 128，作为固定采样点数）和输出维度（例如词汇表大小）。确保所有组件兼容多维 d 和非均匀 PCHIP。

2. **构建 RNN 模块**：设置标准 RNN（如 GRU），用于更新隐藏状态 h_k。RNN 输入层将 x_k（维度 d）投影到 hidden_dim，上一步 h_{k-1} 直接传入；输出层保持 hidden_dim 大小。添加残差连接（如 h_k = h_k + h_{k-1}）可选，以缓解梯度消失。

3. **实现 PCHIP 内存**：创建连续曲线表示。细节步骤：收集所有历史 token 的嵌入 E（N x d）；对时间轴 t (0 到 k-1) 应用非线性变换，使远期稀疏、近期密集（例如 t_scaled[i] = log(i + 1) / log(k) * k，以归一化）；针对每个维度 j 独立但同步构建 PCHIP 曲线 f_E^j(t)：计算每个分段的埃尔米特系数，确保保单调性和可微分（梯度通过线性组合传播）；结果是一个多维连续函数，支持任意 t 的向量采样（采样时对每个维度并行查询）。

4. **添加卷积栈**：设计一维卷积层序列（例如 2-3 层 Conv1D，核大小 3，stride 1，padding 以保持 L）。强调：卷积核在缩略图 H_thumb,k 的每个维度上同时扫描，核滑动时处理所有 d 维度（通道数 d），提取时空模式（如序列中的重复 motif）。结合挤压-激励：全局平均池化缩略图得到 d 维向量；MLP（FC 到 hidden_dim/16 再回 d，sigmoid）计算 d 个缩放因子；逐维度乘回特征。添加批归一化可选，以稳定训练。

5. **设置决策网络**：使用多层感知机（MLP），输入拼接向量（维度 3 * hidden_dim），隐藏层大小递减（例如 hidden_dim * 2 到 hidden_dim），输出长度 W 的向量 theta_k（位置参数）。强调自由规划：输出向量直接用于扭曲，例如 t_warped = sort(sigmoid(theta_k) * k)，确保点在 [0, k] 内有序；或 theta_k 作为偏移，t_warped = cumsum(softplus(theta_k)) 以累积间隔。约束通过可微操作实现，防止塌缩。

6. **整合扭曲采样**：定义初始均匀网格 t_uniform（长度 W，均匀在 [0, 1]）。扭曲细节：基于 theta_k（W 维向量）非线性调整采样点位置，例如 t_warped[i] = cumsum(exp(theta_k[i])) * (k / sum(exp(theta_k)))，实现任意密度分布（网络学习 theta_k 以在关键处创建小间隔，实现放大）；生成扭曲网格 t_warped（长度 W，有序）；从 PCHIP 曲线采样，得到 h_warped,k（W x d）。强调：调整采样点是同步的，所有 d 维度共用一套采样方案（同一 t_warped 网格应用于所有维度），确保一致性和效率；自由规划允许多个密集簇，同时放大不连续细节。

7. **最终预测层**：另一个 MLP 融合意图 h_k 和细节 h_warped,k（先对 h_warped,k 展平或一维卷积池化到 hidden_dim），输入拼接后通过 2 层全连接计算输出 logits。添加注意力可选，以加权融合 warped 点。

### III. 前向传播流程步骤（每个时间步）

模型在每个时间步 k 执行以下五个阶段的循环，确保端到端可微分：

#### 阶段 1：生成顺序意图

- 接收上一步隐藏状态 h_{k-1} 和当前输入嵌入 x_k（维度 d）。
- 通过 RNN 更新计算新意图向量 h_k = RNN(h_{k-1}, x_k)，总结局部上下文和短期依赖。

#### 阶段 2：生成全局地图

- 从 PCHIP 曲线（非均匀分辨率，所有维度建模）采样固定长度 L 的历史缩略图 H_thumb,k（L x d），采样点均匀在 [0, k]，随着 k 增长动态更新以反映最新历史。
- 使用一维卷积栈扫描缩略图，卷积核在每个维度上同时进行特征扫描，滑动覆盖时间轴并处理所有 d 维度，提取多尺度模式。
- 应用挤压-激励机制：全局平均池化计算平均特征向量（d 维）；小型 MLP 输出 d 个缩放因子（sigmoid）；逐维度调整缩略图以放大关键特征、压制噪声。
- 全局池化（平均）得到地图向量 c_global,k，代表历史宏观总结。

#### 阶段 3：决策与定位

- 拼接意图向量 h_k、地图向量 c_global,k 和当前输入 x_k（总维度 3 * hidden_dim）。
- 通过决策 MLP 计算参数 theta_k（W 维向量），网络自由规划采样点分布，用于后续扭曲。

#### 阶段 4：执行缩放与高分辨率采样

- 使用决策参数 theta_k 扭曲初始均匀网格 t_uniform，非线性调整采样点位置：基于 theta_k 计算间隔或偏移，实现任意密度（例如小 theta_k 值创建密集点，放大关键细节；大值创建稀疏）；通过累积和或软排序生成有序 t_warped（长度 W，在 [0, k] 内）。
- 从 PCHIP 连续曲线（所有维度建模，受益于非均匀分辨率）统一采样：所有 d 维度共用同一 t_warped 方案，同步查询每个维度曲线，得到高分辨率细节向量 h_warped,k（W x d），在模型学习的多个位置信息密度高。
- 

#### 阶段 5：最终预测与推进

- 融合意图 h_k 和细节 h_warped,k（展平或池化 h_warped,k），通过预测 MLP 计算输出（例如下一个词的 logits）。
- 推进到下一时间步 k+1，开始新循环，历史 E 更新为 E + x_k。

# 设想部分 Thinking part:


### 架构技术总结：连续插值与状态压缩的混合序列模型

本架构是一个混合序列模型 (Hybrid Sequence Model)，其设计目标是在 O(N) 的线性计算复杂度下，实现对长距离依赖的动态、稀疏访问。

它通过一个**有状态的顺序控制器 (Stateful Sequential Controller)**（即 W-Head）来处理局部上下文，并辅以一个**无状态的动态读取头 (Stateless Dynamic Read Head)**（即 R-Head）来处理非局部的、稀疏的全局上下文。

#### 一、 核心组件（静态定义）

1.  **有状态控制器 (Stateful Controller - RNN_ctrl):**
    * **定义:** 一个循环神经网络（如 GRU 或 LSTM），作为系统的**主干（Backbone）**。
    * **职责:** 1. 顺序处理输入序列 e_k。 2. 维护一个编码了稠密顺序历史的隐藏状态 h_k。 3. 作为所有子模块的“决策中心”。

2.  **连续内存模块 (Continuous Memory Modules):**
    * **E (Embedding Matrix):** N x d_model 的离散嵌入矩阵。
    * **H (Heatmap Vector):** N x 1 的离散热度（访问）向量。
    * **f_E(t) (内容插值器):** 一个基于 E 的**可微分插值函数**（例如 PCHIP）。输入一个连续标量 t (例如 t=4.2)，输出一个 d_model 维的插值向量 e_read。
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

1.  控制器 RNN_ctrl 接收其**上一时刻的隐藏状态 h_(k-1)** 和**当前位置的词嵌入 e_k**（从 E[k] 直接读取）。
2.  RNN_ctrl 计算并输出其**当前时刻的隐藏状态 h_k**。
    * `h_k = RNN_ctrl(h_(k-1), e_k)`
    * h_k 现在编码了到 k 为止的**所有稠密顺序上下文**。

##### 步骤 2: R-Head 策略执行 (Read Head Policy Execution)

1.  JumpController 被激活，用于决定 R-Head 的目标位置。
2.  它收集当前所有可用的历史上下文：
    * h_k (来自步骤 1)
    * C_vec_old (来自 k-1 步的 R-Head 内容缓存)
    * J_vec_old (来自 k-1 步的 R-Head 动作缓存)
3.  JumpController 输出一个连续标量 t_R：
    * `policy_input = concat(h_k, C_vec_old, J_vec_old)`
    * `t_R = sigmoid(MLP_jump(policy_input)) * k`
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
    * `predict_input = concat(h_k, e_read, h_read, D, C_vec_old, J_vec_old)`
    * `Logits = MLP_predict(predict_input)`

##### 步骤 5: 状态与内存更新 (State and Memory Update)

在 Logits 被用于计算损失（Loss）之后，模型为**下一步（k+1）**更新其状态和内存。这三项更新可以**并行执行**：

1.  **更新内容缓存 (C_vec):**
    * `C_vec_new <- beta * C_vec_old + (1 - beta) * e_read`
    * (beta 是 EMA 遗忘因子, 例如 0.8)。这是一个 O(d) 的操作。

2.  **更新动作缓存 (J_vec):**
    * `j_t = ActionEncoder([t_R, D])` (MLP, O(d^2) 操作)
    * `J_vec_new <- alpha * J_vec_old + (1 - alpha) * j_t`
    * (alpha 是 EMA 遗忘因子, 例如 0.9)。这是一个 O(d) 的操作。

3.  **更新热度地图 (H):**
    * 这是一个**可微分写入 (Differentiable Write)** 或 **“溅射” (Splatting)** 操作。
    * 模型为 t_R 分配一个热度增量 delta_h (例如 delta_h = 1)。
    * delta_h 被按线性比例分配给 t_R 相邻的整数索引。
    * 例如，对于 t_R = 4.2：
        * `H[4] <- H[4] + (1 - 0.2) * delta_h`
        * `H[5] <- H[5] + (0.2) * delta_h`

##### 步骤 6: 推进 (Advance)

1.  W-Head 索引 k 增加到 k+1。
2.  `h_(k-1) <- h_k`。
3.  `C_vec_old <- C_vec_new`。
4.  `J_vec_old <- J_vec_new`。
5.  返回**步骤 1**。


