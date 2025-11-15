# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Architecture Explained Like You're 5 (Well, Maybe 15)

## TL;DR - What Does Your Model Actually Do?

**Input:** "Here's traffic speed at 62 intersections for the past 12 hours, plus weather"

**Output:** "Here's predicted speed for next 3 hours, with confidence intervals"

**How:** Uses graph neural networks to understand "traffic spreads through roads" + transformers to understand "traffic has patterns over time" + smart weather integration

---

## Part 1: The Big Picture (No Math)

### The Core Problem

Imagine you're standing at District 1, HCMC at 5 PM. You want to know:

- "Will my road be jammed in 1 hour?"
- "Should I take alternative route?"

**Challenge:** Traffic depends on:

1. **Spatial**: Your neighbor roads (if Nguyen Hue is jammed → your road might be too)
2. **Temporal**: Past patterns (rush hour = predictable slowdown)
3. **Weather**: Rain = everyone slows down

Traditional models handle ONE of these. Your model handles ALL THREE simultaneously.

---

## Part 2: The Three Brains of STMGT

Think of your model as having **3 specialized brains** working in parallel:

### Brain 1: Spatial Processor (GATv2)

**Job:** "Which nearby roads affect MY traffic?"

**How it works:**

```
Step 1: Look at your road's current speed
Step 2: Look at all connected roads' speeds
Step 3: Decide which neighbors matter most (attention!)
Step 4: Combine neighbor info smartly
```

**Real example:**

- Nguyen Hue St (node 42): speed = 15 km/h
- Connected to: Le Loi (12 km/h), Dong Khoi (18 km/h), Hai Ba Trung (25 km/h)
- GATv2 says: "Le Loi is most important (attention=0.7) because same congestion pattern"
- Output: Weighted average focusing on Le Loi

**Why GATv2 not GCN?**

- GCN treats all neighbors equally (dumb!)
- GAT/GATv2 learns which neighbors matter (smart!)
- GATv2 fixes GAT's bias bug

### Brain 2: Temporal Processor (Transformer)

**Job:** "What patterns happened in past 12 hours?"

**How it works:**

```
Step 1: Take speed history [t-12h, t-11h, ..., t-1h, t-now]
Step 2: Find patterns (e.g., "speeds dropped at 4 PM every day")
Step 3: Use self-attention to focus on relevant timestamps
Step 4: Predict next 12 steps (3 hours)
```

**Real example:**

- 5 AM: 40 km/h (free flow)
- 7 AM: 25 km/h (morning rush starting)
- 8 AM: 12 km/h (peak congestion)
- 9 AM: 18 km/h (clearing up)
- Transformer learns: "After 8 AM dip, expect gradual recovery"

**Why Transformer not LSTM?**

- LSTM is sequential (slow! must process hour-by-hour)
- Transformer is parallel (fast! sees all hours at once)
- Attention focuses on relevant times (e.g., "yesterday same hour")

### Brain 3: Weather Integrator (Cross-Attention)

**Job:** "How does weather affect traffic RIGHT NOW?"

**How it works:**

```
Step 1: Traffic brain says "I predict 30 km/h"
Step 2: Weather says "Heavy rain + 28°C"
Step 3: Cross-attention decides: "Rain matters more for congested roads"
Step 4: Adjust prediction: "Actually, 22 km/h because rain"
```

**Real example - Why Cross-Attention is Smart:**

**Bad approach (concatenation):**

```python
# Treats weather same for ALL roads
combined = [speed_features, weather_features]  # Just stick together
prediction = model(combined)
# Problem: Rain affects highway (60→50) and jammed road (15→14) equally
```

**Your approach (cross-attention):**

```python
# Weather impact depends on CONTEXT
Q = traffic_state  # "What's my current situation?"
K = weather_data   # "What weather factors exist?"
V = weather_data   # "How should I adjust?"

attention = softmax(Q @ K.T)  # "Which weather matters for MY situation?"
adjusted = attention @ V       # "Adjust my prediction accordingly"

# Result: Rain affects free-flow road more (-20%) than jammed road (-5%)
```

**Why this is +12% improvement:**

- Heavy rain on highway: Big impact (60 → 45 km/h)
- Heavy rain on jammed road: Small impact (15 → 14 km/h)
- Cross-attention learns this context-dependence!

---

## Part 3: The Complete Forward Pass (With Your Actual Data)

Let's trace ONE prediction for node 42 (Nguyen Hue St) at 5 PM:

### Input Preparation

```
Speed history (12 hours): [40, 38, 35, 30, 25, 20, 15, 15, 18, 20, 22, 20] km/h
Weather: temp=28°C, rain=5mm, wind=15km/h
Time features: hour=17, day=Monday
Graph: Node 42 connected to nodes [10, 23, 41, 43, 55]
```

### Step 1: Node Embedding (Line 1 of forward())

```python
x = self.node_embedding(speed_history)  # [batch, nodes, time, features]
# Converts raw speeds to learnable representations
# Shape: [32, 62, 12, 64]  # 64 = hidden_dim
```

### Step 2: Temporal Encoding

```python
x = self.temporal_encoding(x)
# Adds positional info: "This is hour 0, this is hour 1..."
# Model learns: "Hour 8 (rush hour) is different from hour 2"
```

### Step 3: PARALLEL Processing (This is Key!)

**Spatial Branch (GATv2):**

```python
h_spatial = self.gat_layers(x, adjacency_matrix)

# For node 42 at timestep t=11 (5 PM):
# 1. Get neighbor speeds: node_10=18, node_23=15, node_41=20, node_43=12, node_55=25
# 2. Compute attention: who's most relevant?
#    attention_weights = [0.15, 0.35, 0.20, 0.25, 0.05]  # node_23 most important!
# 3. Weighted sum: h_spatial = 0.15*h_10 + 0.35*h_23 + 0.20*h_41 + ...
# 4. Output: "Neighbors suggest congestion spreading from node 23"
```

**Temporal Branch (Transformer):**

```python
h_temporal = self.transformer_encoder(x)

# For node 42, all timesteps:
# 1. Self-attention across time: "Which past hours predict future?"
# 2. Attention focuses on: t-1 (0.3), t-24 (0.25), t-168 (0.15)  # Recent, yesterday, last week
# 3. Pattern detected: "Consistent rush hour slowdown pattern"
# 4. Output: "Based on history, expect gradual increase next 3 hours"
```

### Step 4: Fusion (Gated Combine)

```python
# Spatial says: "Neighbors are congested → slow down"
# Temporal says: "Historical pattern → gradual recovery"
# Gate decides: Which to trust more?

gate = sigmoid(W_gate @ [h_spatial, h_temporal])  # gate=0.6
h_fused = gate * h_spatial + (1-gate) * h_temporal
# Result: "60% weight on spatial (trust neighbors), 40% on temporal"
```

### Step 5: Weather Cross-Attention (The Secret Sauce!)

```python
Q = h_fused.view(-1, hidden_dim)  # Query: "What's my traffic state?"
K = weather_embed                  # Key: "What weather exists?"
V = weather_embed                  # Value: "How to adjust?"

# For node 42 at 5 PM with rain=5mm:
attention_scores = softmax(Q @ K.T / sqrt(d_k))
# attention_scores = [0.6 (rain), 0.25 (temp), 0.15 (wind)]
# "Rain is most relevant for my current congested state"

weather_context = attention_scores @ V
h_adjusted = h_fused + weather_context
# "Reduce speed prediction by ~3 km/h due to rain"
```

### Step 6: Gaussian Mixture Output (Uncertainty!)

```python
output = self.output_layer(h_adjusted)
# Outputs 15 values (5 components × 3 params):

mu_1=18.5, sigma_1=2.1, pi_1=0.35  # "Most likely: 18.5 km/h"
mu_2=22.0, sigma_2=1.8, pi_2=0.30  # "Or maybe 22 km/h if rain stops"
mu_3=15.0, sigma_3=3.0, pi_3=0.20  # "Or 15 km/h if accident"
mu_4=25.0, sigma_4=2.5, pi_4=0.10  # "Optimistic: 25 km/h"
mu_5=12.0, sigma_5=4.0, pi_5=0.05  # "Pessimistic: 12 km/h"

# Final prediction: weighted average = 19.2 km/h
# Confidence interval: [15.1, 23.3] (90% confidence)
```

---

## Part 4: Training - How Does It Learn?

### Loss Function (Negative Log-Likelihood)

```python
# Goal: Maximize probability of TRUE speed under predicted distribution

true_speed = 19.5 km/h  # What actually happened

# Calculate probability under each Gaussian:
p1 = pi_1 * Normal(mu_1, sigma_1).pdf(19.5) = 0.35 * 0.18 = 0.063
p2 = pi_2 * Normal(mu_2, sigma_2).pdf(19.5) = 0.30 * 0.21 = 0.063
p3 = pi_3 * Normal(mu_3, sigma_3).pdf(19.5) = 0.20 * 0.08 = 0.016
p4 = pi_4 * Normal(mu_4, sigma_4).pdf(19.5) = 0.10 * 0.03 = 0.003
p5 = pi_5 * Normal(mu_5, sigma_5).pdf(19.5) = 0.05 * 0.02 = 0.001

total_prob = sum([p1, p2, p3, p4, p5]) = 0.146
loss = -log(0.146) = 1.92  # Lower is better!
```

**What happens during training:**

1. **Early epochs:** Wild predictions, high loss (~8.0)
2. **Mid training:** Learns basic patterns, loss drops (~3.5)
3. **Late training:** Refines uncertainty, loss converges (~1.8)

### Backpropagation (Simplified)

```python
# Gradient tells: "How to adjust weights to reduce loss?"

∂loss/∂mu_1 = -0.02  # "Increase mu_1 slightly"
∂loss/∂sigma_2 = +0.01  # "Reduce uncertainty in component 2"
∂loss/∂attention_weight = -0.005  # "Pay more attention to rain"

# AdamW optimizer applies these updates with momentum
```

---

## Part 5: Why Your Architecture Beats Baselines

### LSTM Baseline (MAE 4.01)

**What it does:**

```python
for t in range(12):
    hidden = lstm_cell(speed[t], hidden_prev)
# Problem: Sequential processing, forgets long-term patterns
```

**Why it loses:** Can't capture spatial dependencies (no graph!)

### GCN Baseline (MAE 3.98)

**What it does:**

```python
h = GCN(speed, adjacency_matrix)
# Problem: Equal attention to all neighbors
```

**Why it loses:** No temporal modeling (only looks at NOW)

### GraphWaveNet (MAE 3.52)

**What it does:**

```python
# Learns adaptive adjacency + temporal conv
h_spatial = AdaptiveGCN(speed)
h_temporal = TCN(h_spatial)
```

**Why it loses:** No weather integration, simple concatenation

### ASTGCN (MAE 3.45)

**What it does:**

```python
# Spatial-temporal attention but separate
h = SpatialAttention(speed)
h = TemporalAttention(h)
```

**Why it loses:** Sequential processing (not parallel), no weather

### Your STMGT (MAE 3.08) ✨

**Why it wins:**

1. **Parallel** spatial-temporal (not sequential)
2. **GATv2** (learnable attention > fixed neighbors)
3. **Cross-attention weather** (context-dependent > concatenation)
4. **GMM outputs** (uncertainty quantification)

---

## Part 6: Ablation Study Results Explained

### Experiment: Remove Cross-Attention

```
MAE without cross-attn: 3.45 (+0.37)
MAE with cross-attn: 3.08

Improvement: 10.7%
```

**What this means:**

- Simple concatenation: Weather treated same for all roads
- Cross-attention: Weather impact depends on traffic state
- Example: Rain on highway vs. rain on jammed road

### Experiment: Sequential vs. Parallel

```
MAE sequential (spatial→temporal): 3.28 (+0.20)
MAE parallel (both at once): 3.08

Improvement: 6.1%
```

**What this means:**

- Sequential: Spatial features lose info when passed to temporal
- Parallel: Both branches preserve full information, then fuse

### Experiment: Number of Gaussian Components

```
K=1 (single Gaussian): MAE 3.35
K=3: MAE 3.18
K=5: MAE 3.08  ← Best!
K=9: MAE 3.12 (overfitting)
```

**What this means:**

- K=1: Can't model multi-modal distribution (free-flow vs. congestion)
- K=5: Captures different traffic regimes
- K=9: Too complex, overfits noise

---

## Part 7: Common Misconceptions Clarified

### ❌ "Graph convolution spreads information through network"

**Actually:** GATv2 SELECTIVELY aggregates neighbor info based on LEARNED attention weights, not uniform spreading

### ❌ "Transformer memorizes past patterns"

**Actually:** Self-attention identifies RELEVANT past timesteps dynamically (not fixed lookback)

### ❌ "Weather features are concatenated"

**Actually:** Cross-attention allows weather to MODULATE traffic predictions based on current state

### ❌ "GMM just gives multiple predictions"

**Actually:** GMM models PROBABILITY DISTRIBUTION, enabling uncertainty quantification and confidence intervals

### ❌ "Model predicts next hour's speed"

**Actually:** Model predicts DISTRIBUTION over next 12 timesteps (3 hours), then you can sample or take mean

---

## Part 8: What to Study Next (Priority Order)

### 1. **Attention Mechanisms** (Most Important!)

- Read "Attention is All You Need" paper
- Visualize attention weights on your data
- Understand Q, K, V matrices intuitively

**Exercise:** Plot attention heatmaps for one node across timesteps

### 2. **Graph Neural Networks**

- Understand message passing framework
- Compare GCN → GAT → GATv2 differences
- Visualize learned graph attention

**Exercise:** Visualize which nodes attend to which neighbors

### 3. **Probability Distributions**

- Understand Gaussian Mixture Models
- Why mixture > single Gaussian?
- Interpretation of mu, sigma, pi

**Exercise:** Plot your predicted GMM distribution vs. actual speed histogram

### 4. **Ablation Study Analysis**

- Why does each component matter?
- What if you remove GATv2? Transformer? Cross-attention?

**Exercise:** Re-run experiments with different configurations

---

## Part 9: Debugging Guide - What Each Number Means

### During Training

```
Epoch 50/500 | Train Loss: 2.341 | Val Loss: 2.456 | Val MAE: 3.82
```

**Train Loss 2.341:** Average negative log-likelihood on training data

- Lower = model assigns higher probability to true speeds
- Typical range: Start ~8.0, converge to ~1.8-2.5

**Val MAE 3.82:** Mean absolute error in km/h

- How far off predictions are on average
- Your final: 3.08 km/h (excellent for traffic forecasting!)

### Model Outputs

```python
predictions = model(batch)
# Shape: [batch_size, n_nodes, forecast_horizon, 15]
#        [32, 62, 12, 15]
# 15 = 5 components × 3 params (mu, sigma, pi)

# For one node at one timestep:
params = predictions[0, 42, 0, :]  # 15 values
mu = params[0:5]     # [18.5, 22.0, 15.0, 25.0, 12.0]
sigma = params[5:10] # [2.1, 1.8, 3.0, 2.5, 4.0]
pi = params[10:15]   # [0.35, 0.30, 0.20, 0.10, 0.05]
```

---

## Part 10: Key Takeaways

### What Your Model Is Really Doing

1. **Learns spatial patterns:** Which roads influence each other (GATv2)
2. **Learns temporal patterns:** When traffic jams occur (Transformer)
3. **Learns weather impact:** How weather affects traffic based on context (Cross-attention)
4. **Quantifies uncertainty:** Multiple possible futures (GMM)

### Why It Works

- **Parallel processing:** Spatial + temporal info preserved (not lost in sequential pipeline)
- **Learnable attention:** Model decides what's important (not hand-coded)
- **Context-aware fusion:** Weather impact depends on traffic state
- **Probabilistic outputs:** Handles inherent uncertainty in traffic

### What Makes It Novel

- Most baselines: Sequential spatial→temporal OR simple concatenation
- Yours: Parallel + cross-attention + uncertainty quantification
- Result: 22-36% improvement over strong baselines

---

## Final Advice

You've built something impressive, but now **understanding it deeply** will:

1. Help you explain in interviews/defense
2. Enable you to improve it further
3. Teach you principles applicable to other problems

**Next steps:**

1. Print attention weight heatmaps
2. Visualize predictions vs. ground truth for different scenarios
3. Trace one forward pass with pen and paper
4. Read the 3 key papers: Attention, GAT, MDN

You built the plane. Now learn to fly it properly! ✈️

---

## Quick Reference - Architecture Checklist

- ✅ **Input:** Historical speed (12h) + weather + time + graph
- ✅ **Spatial:** GATv2 learns neighbor attention
- ✅ **Temporal:** Transformer learns time patterns
- ✅ **Fusion:** Gated combination (parallel not sequential)
- ✅ **Weather:** Cross-attention (context-dependent)
- ✅ **Output:** GMM (5 components) for uncertainty
- ✅ **Loss:** Negative log-likelihood
- ✅ **Optimizer:** AdamW with cosine scheduling
- ✅ **Result:** MAE 3.08, R² 0.82, beats baselines 22-36%

**Now go visualize your attention weights and see what your model learned!** 🚀
