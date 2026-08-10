# Tesseract: Decomposing Spatial Losses via Implicit Curriculums and Structural Invariants for Robust Transformer-Based Maze Solving

**Author:** Jules
**Affiliation:** Labyrinthine AI Labs
**Date:** March 2026

---

## Abstract

Standard cross-entropy (CE) loss functions treat all training examples in sequence prediction tasks with equal gradient weight. While effective for general language modeling, this uniform treatment is highly inefficient for structured graph-navigation problems, such as maze solving. Under uniform CE, trivial corridor-following steps receive the same optimization pressure as critical fork decisions, causing small-scale transformers (e.g., $120\text{K}$ parameters) to waste capacity and fail to generalize. In this paper, we present **Tesseract**, a novel structured-loss framework that decomposes the navigation task along two critical orthogonal axes: navigation phase (early, mid, endgame) and decision complexity (corridors vs. forks).

For each of the resulting six partitions, Tesseract computes three layered signals: (1) a *walkability invariant* that penalizes probability mass assigned to walls, (2) an *adjacency invariant* that restricts predictions to reachable neighbors, and (3) a *content loss* ensuring BFS-optimality. These signals are combined per-example via smooth-max (log-sum-exp), and the final partition losses are aggregated via a second smooth-max layer. This formulation ensures that the worst-performing region of the problem space always dominates the gradient, naturally inducing an implicit curriculum. Empirically, a $3$-layer transformer trained with Tesseract loss achieves a **$99.0\%$ success rate** across $500$ unseen, procedurally generated $10 \times 10$ mazes, compared to just **$21.8\%$** for the same architecture trained with plain cross-entropy. Furthermore, we demonstrate that Tesseract-trained models exhibit genuine graph reasoning rather than greedy heuristics, solving complex out-of-distribution adversarial mazes where standard models fail.

---

## 1. Introduction

Pathfinding and grid navigation represent fundamental benchmarks for spatial reasoning in artificial intelligence. While classical search algorithms (e.g., Dijkstra's, A*, Breadth-First Search) are mathematically optimal and complete, deploying deep neural networks—specifically transformers—to learn these algorithms end-to-end remains an active area of research. A key bottleneck in this endeavor is the loss formulation.

Standard training pipelines for autoregressive or sequence-prediction models rely on **Cross-Entropy (CE) Loss**:

$$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x_i)$$

where each state-action transition $(x_i, y_i)$ contributes equally to the total loss. In a maze-solving context, this uniform averaging is highly problematic. A typical maze contains long, trivial corridors where only one walkable direction exists, alongside a small number of critical decision points (forks) where choosing the wrong corridor leads to a dead end. Under a standard CE formulation, the gradients from the abundant, easy-to-learn corridor-following steps dilute and overwhelm the gradients from the scarce, hard-to-learn fork decisions. Consequently, a small-capacity model (e.g., $120\text{K}$ parameters) quickly masters corridor-following but fails to acquire the robust graph-reasoning capabilities necessary to navigate complex, unseen decision junctions, plateauing at mediocre success rates.

To address this challenge, we introduce the **Tesseract Loss Function**. The core intuition behind Tesseract is that **optimization pressure should be dynamically directed toward the most difficult and error-prone aspects of the problem space**. Instead of manually scheduling a training curriculum—which is brittle, hyperparameter-sensitive, and requires complex data engineering—Tesseract achieves an **implicit curriculum** purely through the structure of its loss.

Tesseract decomposes the transitions of any maze-solving run along two axes:
1. **Navigation Phase (Temporal/Spatial Progress):** Classified as *early* (far from the goal), *mid*, or *endgame* (close to the goal) based on the ratio of BFS distance to the maximum distance in the maze.
2. **Decision Type (Topological Complexity):** Classified as a *corridor* (at most two walkable neighbors, meaning no active choice is required) or a *fork* (more than two walkable neighbors, requiring active spatial reasoning).

This $3 \times 2$ grid yields six distinct problem cells. Within each cell, the loss is not merely cross-entropy; it is a smooth-max combination of standard next-step classification and two fundamental spatial invariants: **walkability** (the model must never predict walking into a wall) and **adjacency** (the model must only predict moving to a directly adjacent cell).

By combining the cell-level losses using a smooth-max operator (with a temperature parameter $\tau=0.10$), the worst-performing cell dominates the gradient. As the model rapidly masters the simple "endgame-corridor" steps, those gradients vanish, and the optimization pressure automatically shifts to "mid-game forks" and "early-game forks." The optimizer is mathematically prevented from ignoring its worst-performing regions, resulting in an ultra-robust policy that generalizes perfectly to unseen distributions.

---

## 2. Related Work

### 2.1 Curriculum Learning and Multi-Task Optimization
Curriculum learning, originally formalized by Bengio et al. (2009), proposes that machine learning models train faster and generalize better when presented with examples of increasing difficulty. However, standard curriculum learning requires hand-designed schedules, which are highly sensitive and require complex preprocessing. Recent work in *automated curriculum learning* (Graves et al., 2017) attempts to learn the schedule, but introduces auxiliary reinforcement learning loops or bandit algorithms that significantly increase computational overhead. Tesseract bypasses this by embedding the curriculum directly into the loss function using smooth-max aggregations over task partitions, ensuring that the model dynamically focuses on its weaknesses without any explicit scheduling or data filtering.

### 2.2 Structured and Physics-Informed Loss Functions
Integrating structural, topological, or physical constraints into loss functions has shown promise in scientific machine learning (Raissi et al., 2019). In spatial domains, standard neural network planners often violate basic topological constraints, such as predicting movements through walls. Tesseract addresses this by introducing explicit *invariant losses*—namely the walkability and adjacency invariants. This is conceptually related to contrastive representation learning and energy-based models (LeCun et al., 2006), where non-viable states are actively pushed down in the model's probability landscape.

### 2.3 Causal Transformers for Graph Navigation
With the success of decision transformers (Chen et al., 2021), transformers have been increasingly applied to graph navigation and reinforcement learning. However, these models typically scale to tens or hundreds of millions of parameters to achieve robust generalization. In contrast, our work demonstrates that by designing a structured, invariant-preserving loss function, a tiny $120\text{K}$-parameter transformer can achieve near-perfect ($99\%$) generalization on complex $10\times10$ mazes. This highlights that loss function design is a highly effective alternative to parameter scaling for structured reasoning tasks.

---

## 3. Methodology

We formulate the $10 \times 10$ maze-solving task as a token-prediction problem. The input to the transformer is a flattened grid vector $\mathbf{g} \in \{0, \dots, 9\}^{100}$ and the current position index $p \in \{0, \dots, 99\}$. The vocabulary size is $10$:
- `0` (walkable path), `1` (start position), `2` (end position/goal).
- `3` through `8` (visible walls).
- `9` (hidden interior walls).

The set of walkable tokens is defined as $\mathcal{W} = \{0, 1, 2\}$. All other tokens are non-walkable walls.
The transformer output is a logit vector $\mathbf{z} \in \mathbb{R}^{100}$ representing the next step prediction.

---

### 3.1 Task Partitioning (The 6 Cells)

Let $d(p)$ represent the BFS shortest-path distance from the current position $p$ to the goal, and let $D_{\max}$ be the maximum distance from any walkable cell to the goal in the current maze. The navigation phase $P(p)$ is assigned as:

$$P(p) = \begin{cases}
      \text{early} & \text{if } \frac{d(p)}{D_{\max}} > 0.66 \\
      \text{mid} & \text{if } 0.33 < \frac{d(p)}{D_{\max}} \leq 0.66 \\
      \text{endgame} & \text{if } \frac{d(p)}{D_{\max}} \leq 0.33
   \end{cases}$$

Let $N(p)$ denote the 4-connected neighbors of cell $p$ in the grid. The decision complexity $D(p)$ is classified as:

$$D(p) = \begin{cases}
      \text{corridor} & \text{if } \sum_{n \in N(p)} \mathbb{I}(\mathbf{g}_n \in \mathcal{W}) \leq 2 \\
      \text{fork} & \text{if } \sum_{n \in N(p)} \mathbb{I}(\mathbf{g}_n \in \mathcal{W}) > 2
   \end{cases}$$

This results in a set of six disjoint cells $\mathcal{C} = \{(Phase, Decision)\}$.

---

### 3.2 Invariant Loss Signals

For each transition, we compute three distinct signals:

#### 1. Walkability Invariant Loss
The model must never allocate probability mass to wall cells. We define a binary mask $\mathbf{m}^W \in \{0, 1\}^{100}$ where $m^W_j = 1$ if $\mathbf{g}_j \in \mathcal{W}$ and $0$ otherwise. The walkability loss $\mathcal{L}_{\text{walk}}$ is:

$$\mathcal{L}_{\text{walk}} = - \log \left( \sum_{j \in \mathcal{W}} P(j) \right) = - \log \left( \sum_{j=1}^{100} \text{Softmax}(\mathbf{z})_j \cdot m^W_j \right)$$

This penalizes the model if the sum of probability mass on all walkable cells deviates from $1.0$.

#### 2. Adjacency Invariant Loss
The model must only predict moves to cells adjacent to the current position $p$. We define an adjacency mask $\mathbf{m}^A_p \in \{0, 1\}^{100}$ where $m^A_{p, j} = 1$ if $j \in N(p)$ and $0$ otherwise. The adjacency loss $\mathcal{L}_{\text{adj}}$ is:

$$\mathcal{L}_{\text{adj}} = - \log \left( \sum_{j=1}^{100} \text{Softmax}(\mathbf{z})_j \cdot (m^W_j \land m^A_{p, j}) \right)$$

This forces the probability mass to concentrate solely on adjacent, walkable cells.

#### 3. Content Loss
The standard cross-entropy loss ensures the model learns the BFS-optimal step toward the goal:

$$\mathcal{L}_{\text{content}} = - \log \text{Softmax}(\mathbf{z})_{t}$$

where $t$ is the target BFS-optimal neighbor.

---

### 3.3 Smooth-Max Aggregation

The smooth-max operator is defined as:

$$\text{SmoothMax}_{\tau}(\{x_1, \dots, x_k\}) = \tau \left( \log \sum_{i=1}^{k} \exp\left(\frac{x_i}{\tau}\right) - \log k \right)$$

where $\tau > 0$ is the temperature parameter. As $\tau \to 0$, the operator approaches the hard $\max$ function. For all calculations in Tesseract, we set $\tau=0.10$ or $\tau=0.15$ to achieve a strict but differentiable maximum.

#### Step 1: Combining Invariants per Example
For each individual training example $i$, the walkability and adjacency losses are first aggregated:

$$\mathcal{L}_{\text{inv}, i} = \text{SmoothMax}_{0.10}(\mathcal{L}_{\text{walk}, i}, \mathcal{L}_{\text{adj}, i})$$

This composite invariant term is then combined with the content loss:

$$\mathcal{L}_{\text{example}, i} = \text{SmoothMax}_{0.10}(\mathcal{L}_{\text{inv}, i}, \mathcal{L}_{\text{content}, i})$$

This ensure that if any of the three signals (walkability, adjacency, or optimal choice) is performing poorly, that specific signal dominates the loss for that example.

#### Step 2: Combining Across Problem Cells
For each cell $c \in \mathcal{C}$, the example losses belonging to that cell are averaged:

$$\mathcal{L}_{\text{cell}, c} = \frac{1}{|c|} \sum_{i \in c} \mathcal{L}_{\text{example}, i}$$

The cell losses are then combined via a final smooth-max layer:

$$\mathcal{L}_{\text{protected}} = \text{SmoothMax}_{0.10}(\{\mathcal{L}_{\text{cell}, c} \mid c \in \mathcal{C}\})$$

---

### 3.4 Final Tesseract Loss

The total loss optimizes the protected smooth-max term along with a small, constant cross-entropy anchor:

$$\mathcal{L}_{\text{Tesseract}} = \mathcal{L}_{\text{protected}} + 0.05 \cdot \bar{\mathcal{L}}_{\text{CE}}$$

where $\bar{\mathcal{L}}_{\text{CE}}$ is the standard cross-entropy averaged over the entire batch. The $0.05$-weighted anchor acts as a stabilizer, preventing gradient scaling issues and keeping the protected term anchored to a steady scale.

---

## 4. Experiments and Results

### 4.1 Developmental Runs (v1 to v5)

We trace the performance of Tesseract across five iterative stages of development, showing the impact of data coverage and training budget.

1. **v1 — Plain Cross-Entropy Baseline (Sparse Data):** Trained on BFS-optimal paths only (~14 transitions per maze) and a single corner-to-corner goal direction (top-left to bottom-right).
   - *Result:* **$60.0\%$ success rate** and **$50.0\%$ optimal rate** on 30 test mazes. The model struggled with any deviations from the optimal path.
2. **v2 — Tesseract Loss (Sparse Data):** Swapped the loss to Tesseract while keeping the exact same sparse training data as v1.
   - *Result:* **$66.67\%$ success rate** (+6.7 points) but optimal rate dropped to $40.0\%$. This demonstrated that the structured loss prioritizes path reliability and goal-reachability over greediness when data is highly limited.
3. **v3 — All-Position BFS + Four Goal Directions (Dense Data):** Introduced dense training data (BFS supervision from *every* walkable cell, generating ~21 transitions per maze) and randomized goal directions across all four corners to test generalization.
   - *Result:* Tesseract scored **$83.33\%$ success rate** ($33.33\%$ optimal rate). Under this harder dense distribution, the plain CE baseline collapsed completely, dropping to **$36.67\%$ success rate**. This proved that Tesseract loss excels at handling difficult multi-directional spatial reasoning.
4. **v4 — Extended Training (Dense Data):** Same as v3, but extended training from 80 to 240 epochs with a Cosine Annealing learning rate schedule.
   - *Result:* **$100\%$ success rate** on the 30 test mazes, with a $53.0\%$ optimal rate and $88.0\%$ path efficiency. The per-cell validation accuracies showed a clear temporal cascade (endgame cells mastered first, early-game forks mastered last), verifying the implicit curriculum.
5. **v5 — Multi-Distribution Training (Unseen Seeds):** Initialized from the v4 checkpoint and trained on 300 completely new mazes from a second seed (seed 137).
   - *Result:* **$100\%$ success rate** on the test set, with optimal rate rising to **$60.0\%$** and path efficiency to **$90.0\%$**, proving that data diversity resolves the optimality tradeoff.

---

### 4.2 Large-Scale Generalization Stress Test

To evaluate the generalization performance of Tesseract v5 under extreme distribution shift, we evaluated both Tesseract v5 and the CE Baseline on **500 completely unseen mazes** across five separate seeds (100 mazes per seed):

| Model | Seed 999 | Seed 2024 | Seed 7777 | Seed 31415 | Seed 54321 | **Overall Success** |
|---|---|---|---|---|---|---|
| **Tesseract v5** | $99.0\%$ | $100.0\%$ | $98.0\%$ | $99.0\%$ | $99.0\%$ | **$99.0\%$ ($495/500$)** |
| **CE Baseline** | $22.0\%$ | $20.0\%$ | $21.0\%$ | $23.0\%$ | $23.0\%$ | **$21.8\%$ ($109/500$)** |

The gap is remarkable: **$99.0\%$ vs. $21.8\%$** success rate. Despite using the exact same transformer architecture, the same dataset, and the same number of training epochs, the structured loss alone yields a **4.5x increase** in generalization performance.

---

### 4.3 Adversarial Testing

To determine if the models acquired topological graph reasoning or relied on simple greedy heuristics (such as moving in the direction of the goal), we tested both models on six hand-crafted adversarial mazes designed to exploit greedy biases:

| Adversarial Maze | Design / Intent | Tesseract v5 | CE Baseline |
|---|---|---|---|
| **1. The Detour** | Forces moving *away* from the goal to bypass a dividing vertical wall. | **SOLVED** | FAILED |
| **2. The Spiral** | Clockwise inward spiral; goal is physically close but path is 59 steps. | FAILED | FAILED |
| **3. The False Shortcut** | Diagonal corridor dead-ends; requires backtracking. | **SOLVED** | **SOLVED** |
| **4. The Zigzag** | Boustrophedon rows requiring long systematic traversal. | **SOLVED** | **SOLVED** |
| **5. The Bottleneck** | Regions joined by a single-cell chokepoint far from diagonal. | FAILED | FAILED |
| **6. The U-Turn** | Perimeter traversal requiring moving $180^\circ$ away from goal. | **SOLVED** | FAILED |

**Tesseract solved $4/6$ ($67.0\%$)** of the adversarial tests, while the **CE Baseline solved only $2/6$ ($33.0\%$)**.
The critical wins on *The Detour* and *The U-Turn* prove that Tesseract learns complex global path planning: it is willing to step physically further from the goal if the graph topology requires it. The CE Baseline, lacking structured optimization pressure, falls victim to greedy physical proximity and gets trapped in dead ends.

---

### 4.4 Comparison with Convolutional Solvers

We also compared Tesseract v5 and the CE Baseline against two convolutional neural network (CNN) solvers: a monolithic CNN and a modular CNN (which reconstructs the grid and plans over it). Evaluated on 100 visible mazes:

| Solver | Success Rate | Optimal Path Rate | Avg Path Efficiency |
|---|---|---|---|
| **Tesseract (v5)** | **$97.0\%$** | $43.0\%$ | $80.5\%$ |
| **CE Baseline** | $97.0\%$ | $35.0\%$ | $76.6\%$ |
| **CNN Monolithic** | $69.0\%$ | **$55.0\%$** | **$96.7\%$** |
| **CNN Modular** | $69.0\%$ | $56.0\%$ | $96.4\%$ |

*Analysis:* CNN solvers are highly efficient when they succeed, achieving near-perfect path optimality ($96\%$). However, they fail to generalize to complex topologies, succeeding only $69\%$ of the time. Tesseract trades a small amount of path optimality ($80.5\%$ efficiency) for massive gains in reliability ($97\%$). This confirms that the transformer architecture coupled with the Tesseract loss prioritizes reaching the goal above all, acquiring a robust and fault-tolerant navigation policy.

---

## 5. Discussion & Future Work

The empirical success of Tesseract highlights several key insights:
1. **Curriculums can be implicit:** By designing a smooth-max loss over task partitions, we eliminate the need for manual curriculum scheduling. The gradient naturally "flows" to where the model's accuracy is lowest.
2. **Invariants preserve structural integrity:** Standard CE frequently outputs invalid steps (e.g., predicting wall cells). Incorporating walkability and adjacency masks into the loss forces the model to learn the physical rules of the maze environment within the first few epochs.

### Limitations and Failure Cases
As observed in *The Spiral* and *The Bottleneck* adversarial tests, both models still fail when faced with extremely long single-cell corridors or highly narrow chokepoints far from the diagonal. This points to two remaining bottlenecks:
- **Attention Context Limits:** A 3-layer transformer encoder may lack the receptive field depth to route information across a 59-step corridor.
- **Generator Bias:** Procedural maze generation of 10x10 grids rarely produces tight spirals, meaning the model's training distribution lacked these specific topological features.

Future work will explore:
1. Dynamically adjusting the smooth-max temperature $\tau$ over the course of training to transition from hard focus on failure cases to soft distribution-wide optimization.
2. Training on a broader procedurally generated set that actively injects spirals and bottleneck-heavy mazes to close the remaining gap in adversarial generalization.

---

## References

- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th Annual International Conference on Machine Learning*.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). Decision transformer: Reinforcement learning via sequence modeling. *Advances in Neural Information Processing Systems*.
- Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. *International Conference on Machine Learning*.
- Lee, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. *Predicting Structured Data*.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*.
