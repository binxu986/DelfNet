# 🧠 DelfNet - Deep Self-Organizing Neural Network

> A compatibility improvement on existing Transformer architecture, constructing a neural network structure that can express the time dimension t, depth, and self-organizing characteristics to some extent. Through research in multiple experimental directions, we hope to improve the training and inference efficiency and accuracy of existing DNNs, and also hope to provide some new ideas for future neural network architecture design.


## ✏️ Note
I will write an article in my spare time, in my personal capacity, in this repository (currently just a draft), titled "The fundamental idea of deep self-organizing neural networks". I don't have GPU hardware conditions to do large-scale experiments. If anyone is interested in collaborating, researching based on the ideas in this article, completing model training and testing, and achieving any results, please contact me: 188997452@qq.com, or discuss through GitHub issues.

- The method is completely public. You can discuss with me, or verify/use/modify/innovate on your own;

- The only requirement: *Cite the source of this project and explain it in any of your work; give this project a star; share your research direction, experimental results, or anything else you want to share*.

---

## 🤝 Seeking Collaboration

Potential Co-authors are welcome to contact and discuss, or inform me after independent verification, to jointly advance the following research directions of DelfNet:

- Weight training methods, adding parameter constraints, optimizing objective functions, selection and design of activation functions, etc.
- Training and inference efficiency optimization, including software implementation, hardware acceleration, etc.
- Others

Collaboration methods:
- 🔬 Code implementation, experimental verification
- 💡 Discussing heuristic methods
- ✍️ Optimizing paper writing
- 🎨 Drawing figures
- 🔧 Others
---

## ⚠️ Existing Problems in DNN

### 📉 Training Problems

| Problem | Description |
|----------------------|------|
| 🔴 Low training efficiency | No direct coupling between neural network layers |
| 🔴 Limited gradient propagation | The dy of each layer only acts on the current layer's weights |

### 🏗️ Structural Problems

| Problem | Description |
|----------------------|------|
| 🔴 Network is too deep | Large models should have sufficient parameter counts, but don't necessarily need so many "independent" parameter layers; just as the deep structure of the human brain may not have L=80 layers? |
| 🔴 Lack of local information interaction | Neurons do not form local networks, neurons within the same layer have no connections, and local neurons do not fully share information and compete (competition should lead to different effects such as activation and inhibition). For example, SOM (Self-organizing map) and Hopfield networks emphasize information exchange within a group of neurons |
| 🔴 ANN models do not model time t | Neurons can only perform one instantaneous processing and then output information to other neurons, without the opportunity for local reprocessing; of course, perhaps existing models implicitly simulate changes in time t through more layers. So why don't we explicitly model this time dimension t? |
| 🔴 Weight parameters are fixed | After training, weight parameters are fixed and cannot reflect the biological ability to adapt to the environment - such as fast reaction, quick association, etc. This may be related to the theory of brain system1 and system2, where system1 is fast reaction and system2 is slow reaction. Current mainstream models do not distinguish between fast and slow, which is very inefficient (wasting computing power) in many cases |
| 🔴 Sparse model training is difficult | During sparse training, only some weight parameters are used, leading to randomness in model effectiveness, some good, some bad. This may be related to the lack of joint updates in front and back layer training. (Refer to lottery ticket hypothesis) |

### 🚀 Inference Problems

| Problem | Description |
|----------------------|------|
| 🔴 Bandwidth bottleneck | Decoding requires loading complete weights every time, which is too inefficient; this problem seriously limits the application efficiency of current DNN inference scenarios |
| 🔴 Storage bottleneck | Model parameters are all independent and difficult to compress, leading to high storage costs; this problem is particularly prominent on some Edge devices |

---

## 💡 Sources of Inspiration (For inspirational thinking only)

| Source | Description |
|----------------------|------|
| 🔵 SOM, Hopfield | Emphasize information exchange within neurons |
| 🔵 RNN | Can be seen as a disguised neuron self-influence structure |
| 🔵 GNN | GNN emphasizes information exchange between nodes |
| 🔵 Transformer Circuits | Neural network circuit research, sparse paths |
| 🔵 Biological neurons | After activation or membrane potential recovery, the neuron itself changes, used for rapid adaptation to the environment |
| 🔵 Human training effect | Through training, reaction to things becomes faster, possibly activating fast channels in the neural network (?) |

---



## 🎯 Core Idea

Model the connections between neurons within a layer to achieve self-organizing characteristics. Through incremental parameter expression, achieve cross-block parameter joint updates.
**DelfNet cannot solve all the above problems, but hopes to provide a new perspective. Details can be found in the paper body**

---



## 🔧 Core Solution

### 📐 Cluster-level Structure


```
Cluster0 { Block0, Block1, …, Block9 }
Cluster1 { Block10, Block11, …, Block19 }
…
```
Cluster represents a group of neurons, which can be seen as an independent neural network functional region (or viewed as a layer of neural network with a batch of independently updating neurons), where neurons within the cluster exchange information. The concept of Block continues to be used to correspond to the Transformer Block concept for easy understanding. For example, Block0, Block1, ... in Cluster0 express the update changes of this cluster of neurons over time t. When one Block is one Cluster, it degenerates to the original transformer structure. By modeling the changes of neuron clusters over time t, we have the opportunity to perform "temporal encoding" on neuron states, which is not explicitly expressed in current neural network structures (currently only possibilities are opened, but how to do it specifically is still an open question).

![Fig1](./figs/neuron_cluster_network.png "Neuron Cluster Internal Network Diagram")







### 🔑 Key Detail: Delta Weight Incremental Neural Network

Use **△ weight** to express the weight changes of subsequent Blocks relative to previous Blocks within a cluster:

```
w0
w1 = w0 + △1
w2 = w1 + △2 = w0 + △1 + △2
W3 = w2 + △3 = w0 + △1 + △2 + △3
…
```

Core advantages:
- The expression is fully compatible with the original network weight parameters, and the total parameter count remains unchanged;
- When updating subsequent Block variables, dy directly acts on previous Block variables, producing a **coupling effect**
- If △ is very sparse or has regular changes, it may be possible to achieve effective compression of parameters to greatly alleviate the decoding bandwidth bound problem and model weight storage problem.



### 🕐 Temporal Encoding

Because of the changes in time step $\Delta_i$, we can record the changes of neuron clusters over time t (hidden states over time). By recording these hidden states change sequences, we can encode the changes of neuron clusters over time t. (Currently just an idea, specific implementation still needs research, for example, simple convolution operations can be performed on hidden states, outputting time-encoded hidden states of the same size as the original, as input to the next block, which can maintain the original computational structure unchanged, etc.)

![Fig2.1](./figs/temporal_encoding.png "Fig2.1. Temporal Encoding Diagram")
Temporal Encoding Diagram



---

## 🧪 Model Hypotheses (To be verified)

| Hypothesis | Description |
|----------------------|------|
| 🟢 △ changes smoothly | The change of △ should not be random, jumping, or completely free. The changes of adjacent △ within a Cluster should be smooth |
| 🟢 Temporal encoding | We first assume that neuron parameters are unchanged after offline training, and the activation order of neurons forms temporal encoding of input data |
| 🟢 △ sparsity | △ may be sparse. The joint updates between Blocks make △ easier to control as sparse. △ represents suppression or enhancement of input signals |
| 🟢 △ generation method | △ may be influenced by offline training (*temporal encoding mentioned above: different characteristics of each neuron: some neurons complete calculation in one step with fast response; while others need more time steps for delayed response*). It may also be influenced by the output of previous Blocks (Data-Dependent), obtained through online calculation: this generation method both preserves weight parameter freedom and does not need offline saving, opening new opportunities for decoding inference scenarios - **partial parameters generated online, alleviating bandwidth bound problem** |

Note: The above hypotheses may affect model performance and need to be verified in experiments. These hypotheses may need to be verified through multiple experimental stages, not necessarily all effective, and may not all take effect simultaneously.

---

## 🗺️ Future Research Directions

> From near to far

| Stage | Goal | Description |
|----------------------|------|------|
| **Phase 1** | 🔨 Basic transformation | Transform CNN/Transformer into DelfNet form, complete training. Solve problems encountered in the process |
| **Phase 2** | 📊 Constraint research | Study smooth constraints and sparse constraints between △, and how to utilize "temporal encoding" characteristics; achieve better scaling law experimental results with equal parameters |
| **Phase 3** | ⚡ Sparse compression | △ is highly sparse and can be encoded by specific sparse compression. During inference, pre-load w and compressed △ to solve partial bandwidth problems and greatly improve decoding performance |
| **Phase 4** | 🚀 Compute-storage fusion | Assume △ is generated by data, use computation instead of storage, alleviate the defect of fixed parameters, simulate the adaptive ability of biological neurons to the environment (input) |

---

## 📧 Contact

If interested, welcome to contact!
- Dr. Dabing, Email: 188997452@qq.com

---

<p align="center">
  <i>DelfNet — Exploring New Possibilities for Neural Networks</i>
</p>
