# 🧠 DelfNet - Deep Self-Organizing Neural Network

> Making compatible improvements to the existing Transformer architecture to construct a neural network structure that can express the time dimension t, depth, and self-organizing characteristics to a certain extent. Through research in multiple experimental directions, I hope to improve the training and inference efficiency and accuracy of existing DNNs, while also providing some new insights for future neural network architecture design.


## ✏️ Note
I will write an article in this repository in my spare time and in a personal capacity (currently just a draft，english version is coming later), titled "The fundamental idea of deep self-organizing neural networks". I don't have GPU hardware resources to conduct large-scale experiments. If anyone is interested in collaborating to conduct research based on the ideas in this article, complete model training and testing, and produce any results, please contact me: 188997452@qq.com, or discuss via GitHub issues.

- The method is completely open. You can discuss it with me, or verify/use/modify/innovate on your own;

- The only requirement: *Cite the source of this project and explain it in your work; give this project a star; share your research direction, experimental results, or anything else you'd like to share*.

---

## 🤝 Seeking Collaboration

Potential co-authors are welcome to contact and discuss, or verify independently and inform me, to jointly advance the following research directions of DelfNet:

- Weight training methods, adding parameter constraints, optimizing objective functions, selection and design of activation functions, etc.
- Training and inference efficiency optimization, including software implementation, hardware acceleration, etc.
- Others

Collaboration methods:
- 🔬 Code implementation, experimental verification
- 💡 Discussion and inspiration methods
- ✍️ Optimizing paper writing
- 🎨 Creating illustrations
- 🔧 Others
---

## ⚠️ Existing Problems in DNNs

### 📉 Training Problems

| Problem | Description |
|----------------------|------|
| 🔴 Low training efficiency | No direct linkage between neural network layers |
| 🔴 Limited gradient propagation | The dy of each layer only acts on the weights of the current layer |

### 🏗️ Structural Problems

| Problem | Description |
|----------------------|------|
| 🔴 Network too deep | Large models should have sufficiently large parameter counts, but they don't necessarily need so many "independent" parameter layers; just like the deep structure of the human brain might not have as many as L=80 layers? |
| 🔴 Lack of local information interaction | Neurons don't form local networks, neurons within the same layer have no connections, and local neurons don't engage in sufficient information sharing and competition (competition should lead to different effects like activation and inhibition). For example, SOM (Self-organizing map) and Hopfield networks emphasize information exchange within a group of neurons |
| 🔴 ANN models don't model time t | Neurons can only perform one instantaneous processing and then output information to other neurons, without the opportunity for local reprocessing; of course, perhaps existing models implicitly simulate the change of time t through more layers. So why don't we explicitly model this time dimension t? |
| 🔴 Fixed weight parameters | After training is completed, weight parameters are fixed and cannot reflect the biological ability to adapt to the environment - such as rapid response, rapid association, etc. This may be related to the theory of the brain's system 1 and system 2, where system 1 is fast response and system 2 is slow response. Mainstream models now don't distinguish between fast and slow, which is very inefficient (wasting computing power) in many cases |
| 🔴 Sparse model training is difficult | When sparse training uses only some weight parameters, the model's performance becomes random, some good, some bad. This may be related to the lack of linked updates between front and back layers. (Reference: lottery ticket hypothesis) |

### 🚀 Inference Problems

| Problem | Description |
|----------------------|------|
| 🔴 Bandwidth bottleneck | Decoding requires loading complete weights every time, which is too inefficient; this problem severely limits the application efficiency of current DNN inference scenarios |
| 🔴 Storage bottleneck | Model parameters are all independent and difficult to compress, leading to high storage costs; this problem is particularly prominent on some Edge devices |

---

## 💡 Sources of Inspiration (For Inspiration Only)

| Source | Description |
|----------------------|------|
| 🔵 SOM, Hopfield | Emphasize information exchange within neurons |
| 🔵 RNN | Can be seen as a disguised neuron self-influence structure |
| 🔵 GNN | GNN emphasizes information exchange between nodes |
| 🔵 Transformer Circuits | Neural network circuit research, sparse paths |
| 🔵 Biological neurons | After activation or membrane potential recovery, the neuron itself changes, used to quickly adapt to the environment |
| 🔵 Human training effect | Training makes responses to things faster, possibly activating fast channels in the neural network (?) |

---

## 🎯 Core Idea

Model the connections between neurons within a layer to achieve self-organizing characteristics. Through incremental parameter expression, achieve cross-block linked parameter updates.
**DelfNet cannot solve all the above problems, but hopes to provide a new perspective. See the paper for details**

---

## 🔧 Core Scheme

### 📐 Cluster-level Structure

```
Cluster0 { Block0, Block1, …, Block4 }
Cluster1 { Block5, Block6, …, Block9 }
…
```
A Cluster represents a group of neurons, which can be seen as an independent neural network functional area (or viewed as a layer of neural network with independently updated neurons), where neurons within the cluster exchange information. The Block concept is retained to correspond with the Transformer Block concept for easier understanding. For example, Block0, Block1, … in Cluster0 represent the update changes of this cluster of neurons over time t. When a Block is a Cluster by itself, it degenerates to the original transformer structure.

### 🔑 Key Detail: Delta Weight Incremental Neural Network

Use **△ weight** to express the weight changes of subsequent Blocks relative to previous Blocks within a cluster:

```
w0
w1 = w0 + △1
w2 = w1 + △2 = w0 + △1 + △2
W3 = w2 + △3 = w0 + △1 + △2 + △3
…
```

### ⭐ Core Advantages:
- Total parameter scale remains unchanged;
- When updating variables of subsequent Blocks, dy directly acts on variables of previous Blocks, creating a **linkage effect**
- If △ is very sparse, or has regular changes, it may be possible to achieve effective compression of parameters, greatly alleviating the decoding bandwidth bound problem and model weight storage problem.

---

## 🧪 Model Hypotheses (To Be Verified)

| Hypothesis | Description |
|----------------------|------|
| 🟢 △ changes smoothly | The change of △ should not be random, jumping, or completely free. The adjacent △ changes within a Cluster should be smooth |
| 🟢 △ sparsity | △ may be sparse. The linked update between Blocks gives △ the characteristic of being more easily controlled to be sparse. △ represents inhibition or enhancement of input signals |
| 🟢 △ generation method | △ may be influenced by offline training (*opportunity to model different characteristics of each neuron: some neurons complete calculation in one step with fast response; while others need more time steps for delayed response*). It may also be influenced by the output of previous Blocks (Data-Dependent), calculated online; this generation method both retains weight parameter freedom and doesn't need offline storage, opening new opportunities for decoding inference scenarios - **partial parameters generated online, alleviating bandwidth bound problem** |

Note: The above hypotheses may affect model performance and need to be verified in experiments. These hypotheses may need to be verified through multiple experimental stages, and not all will necessarily be effective or work simultaneously.

---

## 🗺️ Future Research Directions

> From near to far

| Phase | Goal | Description |
|----------------------|------|------|
| **Phase 1** | 🔨 Basic transformation | Transform CNN/Transformer into DelfNet form and complete training. Solve problems encountered in the process |
| **Phase 2** | 📊 Constraint research | Study the smoothness constraints, sparsity constraints and other hypotheses between △; achieve better scaling law experimental results with the same parameters (optimal) |
| **Phase 3** | ⚡ Sparse compression | △ is highly sparse and can be compressed by specific sparse compression encoding. During inference, load w and compressed △ in advance, solve partial bandwidth problems, greatly improve decoding performance |
| **Phase 4** | 🚀 Compute-storage fusion | Assume △ is generated by data, use compute to replace storage, alleviate the defect of fixed parameters, simulate the adaptive ability of biological neurons to the environment (input) |

---

## 📧 Contact

If interested, welcome to contact!
- Dr. Dabing, Email: 188997452@qq.com

---

<p align="center">
  <i>DelfNet — Exploring New Possibilities for Neural Networks</i>
</p>