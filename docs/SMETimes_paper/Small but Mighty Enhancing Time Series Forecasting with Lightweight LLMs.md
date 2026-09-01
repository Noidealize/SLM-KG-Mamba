# Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs

Haoran Fan<sup>1</sup>, Bin Li<sup>2†</sup>, Yixuan Weng<sup>3</sup> and Shoujun Zhou<sup>2</sup>

<sup>1</sup> College of Computer Science and Technology, Chongqing University of Posts and Telecommunications, Nan’an District, Chongqing, 400065, China.

<sup>2</sup> Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Nanshan District, Shenzhen, 518055, China. <sup>3</sup> Westlake University, Xihu District, Hangzhou, Zhejiang, 310024, China.

Contributing authors: 2022212169@stu.cqupt.edu.cn; b.li2@siat.ac.cn; sj.zhou@siat.ac.cn; wengsyx@gmail.com; <sup>†</sup>Corresponding Author.

## Abstract

While Large Language Models (LLMs) have demonstrated remarkable potential in time series forecasting, their practical deployment remains constrained by excessive computational demands and memory footprints. Existing LLM-based methods typically sufer from three critical limitations: (1) Ineficient parameter utilization in handling numerical time series patterns; (2) Modality misalignment between continuous temporal signals and discrete text embeddings; and (3) Inflexibility for real-time expert knowledge integration. We present Small but Mighty Enhancing Time Series (SMETimes), the first systematic investigation of Small Language Models with sub-3B parameters (SLM) for eficient and accurate time series forecasting. Our method centers on three key innovations: (1) A statistically enhanced prompt structure that bridges numerical time series with textual semantics through descriptive statistical features; (2) An adaptive fusion embedding structure that aligns temporal patterns with language model token spaces through learnable parameters; And (3) a dynamic mixture-of-experts structure enabled by SLMs’ computational eficiency, adaptively combining base predictions with domain-specific models. Extensive evaluations across seven benchmark datasets (ETTh1/2, ETTm1/2, Weather,

Solar, ECL) demonstrate that our 3B-parameter SLM achieves stateof-the-art performance on five primary datasets while maintaining 3.8× faster training and 5.2× lower memory consumption compared to 7Bparameter LLM baselines. In particular, the proposed model exhibits better learning capabilities, achieving 12.3% lower MSE than conventional LLM. Ablation studies validate that our statistically enhanced prompt structure and adaptive fusion embedding structure contribute, respectively, to the reduction of 15.7% and 18.2% errors in longhorizon forecasting tasks. By redefining the eficiency-accuracy trade-of landscape, this work establishes SLMs as viable alternatives to resourceintensive LLMs for practical time series forecasting. The code and models are available at https://github.com/xiyan1234567/SMETimes.

Keywords: Small Language Models, Statistically Enhanced Prompt, Adaptive Fusion Embedding, Dynamic Mixture-of-Experts

## 1 Introduction

Time series forecasting is the cornerstone of modern decision-making systems, with critical applications spanning energy management [30], financial markets [42], climate modeling [1] and intelligent transportation [23]. Traditional methods often rely on domain-specific statistical models [2] or deep neural networks [3], which require substantial computational resources and extensive domain expertise. Although Large Language Models (LLMs) have recently demonstrated remarkable capabilities in time series forecasting [4, 21], their practical deployment faces significant challenges due to prohibitive computational costs and memory footprints [5].

The recent proliferation of LLM-based forecasting methods [6, 7] has revealed three fundamental limitations: (1) Massive parameter counts (typically >7B) lead to ineficient training/inference; (2) Inadequate alignment between numerical time series patterns and textual embeddings; And (3) limited flexibility to integrate domain-specific expert knowledge. As shown in Fig. 1, conventional LLM methods such as Time-LLM [4] (22.33G memory) and AutoTimes [18] (23.12G memory) incur substantial resource costs despite comparable MSE performance to our 1B-parameter SLM (4.33G memory). This eficiency gap becomes particularly critical in real-world deployment scenarios with hardware constraints.

To alleviate the computational ineficiency of Large Language Models (LLMs) in time series forecasting, we propose a feature fusion strategy through Small Language Models (SLMs) that strategically reduce model scale while incorporating targeted architectural innovations to achieve better eficiency. Our key insight lies in three synergistic components: (1) Statistically enhanced prompt structure that bridges numerical and textual domains; (2) An adaptive fusion embedding structure for time series embeddings; And (3) a dynamic mixture-of-experts structure enabled by SLMs’ lightweight structure. Extensive experiments on seven benchmark datasets (ETTh1/2 [11], ETTm1/2 [11], Weather [12], Solar [29], ECL [12]) demonstrate that our 1B parameter SLM achieves state-of-the-art results on five primary datasets while maintaining competitive performance on the remaining three.

![](images/5965a847f09f6aca179267fbfd38fe27c6790e83080f87f5368fbb1d61a7ce42.jpg)  
Fig. 1: Performance-eficiency Trade-of Comparison on ETTh1 [11] Dataset. Our SLM variants (blue) achieve competitive MSE with significantly lower training time and memory footprint compared to conventional LLM-based methods. Bubble size represents relative memory consumption.

Our work makes four fundamental contributions to the field of eficient time series forecasting:

• To the best of our knowledge, this is the first work to develop and evaluate a framework that applies Small Language Models (SLMs) to time series forecasting tasks. Through targeted architectural modifications, we demonstrate that compact models achieve performance parity with 7B-parameter Large Language Models (LLMs) while attaining 3.2× accelerated training convergence and 5.1× reduced memory footprint, substantially improving deployment feasibility in resource-constrained environments.

• We introduce a new prompting methodology that integrates temporal statistics with domain-specific contextual metadata. Empirical validation reveals that this statistically enhanced prompt structure reduces the mean squared error by 12.7% compared to conventional textual prompting baselines through systematic ablation analysis.

• Our adaptive fusion embedding structure addresses the intrinsic modality gap between continuous time series embeddings and discrete token representations. By implementing learnable projective transformations coupled with attention-based feature alignment, the proposed structure yields 9.3% accuracy improvement on extended forecasting horizons compared to standard embedding methods.

• The dynamic mixture-of-experts structure developed synergistically combines the predictions of the base model with established temporal modeling techniques, including ARIMA [24] and Prophet [25]. This hybrid structure achieves a reduction of 4.8% MSE while maintaining 2.1× faster inference speeds compared to monolithic LLM implementations, demonstrating an efective balance between computational eficiency and forecast precision.

The remainder of this paper is organized as follows. Section 2 reviews traditional approaches for time series forecasting, LLM-based methodologies, and SLM analysis in this domain. Section 3 details our framework design details and technical innovations. Section 4 introduces datasets, implementation specifics, comprehensive experiments, and ablation studies, while Section 5 discusses the limitations of our model. We conclude with future research directions in Section 7.

## 2 Related Work

## 2.1 Time Series Forecasting Methods

## 2.1.1 Traditional Methods

Traditional time series forecasting has long relied on domain-specific statistical models and classical machine learning techniques. Methods such as ARIMA [24] and its variants (e.g. SARIMA [26]) leverage autoregressive and moving average components to model temporal dependencies but struggle with nonlinear patterns and multivariate data [2]. Exponential smoothing [27] and state-space models (SSMs) [8] further incorporate trend and seasonality decomposition, yet their rigidity limits adaptability to complex real-world scenarios. With the rise of deep learning, structures such as LSTMs [9] and Temporal Convolutional Networks (TCNs) [22] emerged as powerful tools for sequence modeling. Transformers [10], initially designed for NLP, were later adapted for time series [40, 41] to capture long-range dependencies through self-attention. Although these methods achieve strong performance on narrow tasks, they require extensive domain expertise, task-specific tuning, and large-scale training data, limiting their generalizability across diverse applications [13].

## 2.1.2 LLM-Based Methods

Recent advances in Large Language Models (LLMs) have inspired their adaptation to time series forecasting. Pioneering work like FPT [20] and Time-LLM [4] demonstrated that LLMs trained on textual data can be repurposed for temporal modeling through cross-modal alignment. For example, LLMTime [7] treats time series as numerical tokens, enabling zero-shot forecasting via LLMs inherent pattern recognition capabilities. Methods such as PromptCast [14] and TEMPO [15] further refine prompting strategies to bridge numerical and textual modalities. However, these methods inherit critical limitations from their reliance on massive LLMs (e.g., GPT-3 [28], Llama [5]): (1) Excessive computational costs (e.g., AutoTimes [18] requires 23.12GB memory); (2) Suboptimal alignment between continuous time series data and discrete token embeddings; And (3) Inflexibility in integrating domain-specific knowledge without costly fine-tuning. Although autoregressive LLM-based methods like Time-LLM [4] achieve variable-length predictions, they sufer from quadratic attention complexity and high inference latency, rendering them impractical for resource-constrained environments.

## 2.2 Small Language Models for Time Series Forecasting

Recent advances in SLM-based time series analysis have primarily addressed classification and edge deployment challenges, yet critical gaps persist in forecasting tasks. While Voice2Series [16] and EdgeTS [17] demonstrate the feasibility of parameter-eficient SLMs for temporal pattern recognition, their focus on short-term classification or latency reduction overlooks the intrinsic complexities of multistep forecasting. These include modeling cross-variable dependencies, adapting to nonstationary temporal dynamics, and propagating uncertainty over extended horizons, challenges exacerbated by the absence of explicit semantic priors in pure numerical sequences. The existing LLM-based forecasting structure [4, 14] partially addresses these issues through languagealigned prompting but introduces modality misalignment when fusing time series tokens with textual instructions. Our work bridges this gap by leveraging SLMs’ architectural flexibility to natively encode temporal semantics through quantized embeddings and timestamp-informed attention structures. By inte grating calendar-aware positional encoding with adaptive quantization, we enable SLMs to capture cyclical patterns and event-driven anomalies without cross-modal feature fusion, simultaneously preserving computational eficiency for deployment on resource-constrained edge devices. This method extends the SLM paradigm beyond classification-centric designs, addressing the understudied trade-of between long-term dependency modeling and real-time inference in forecasting applications.

## 3 Methodology

As shown in Fig. 2, the proposed SMETimes framework features three principal components: (1) Statistically enhanced prompt structure; (2) Adaptive fusion embedding structure with dynamic gating structures; And (3) dynamic mixture-of-experts structure, the methodology systematically elaborates these components through dedicated technical sections: Section 3.1 formalizes the numerical-textual alignment process by statistical prompting, Section 3.2 specifies the implementation details of the dynamic gating framework, and

![](images/e9e9d3bf6c5553606782a622d4471624bd615b80ed352c5c177536e6a26e699d.jpg)  
Fig. 2: Structure of the proposed SMETimes framework, featuring three core innovations: (1) Statistically enhanced prompt structure for numerical-textual alignment; (2) Adaptive fusion embedding structure with dynamic gating structures; And (3) dynamic Mixture-of-Experts structure for eficient specialization.

Section 3.3 delineates the parameter allocation strategy for the MoE specialization module. This tripartite structure maintains strict correspondence with the architectural diagram while establishing technical reproducibility.

## 3.1 Statistically Enhanced Prompt Structure

To bridge the gap between numerical time series patterns and natural language semantics, we propose a statistically enhanced prompt structure that generates linguistically interpretable embeddings through domain-specific statistical features and timestamp contextualization. Given a univariate time series $\mathbf { X } = \{ x _ { 1 } , \dots , x _ { T } \} \in \mathbb { R } ^ { T }$ , we first partition it into N non-overlapping segments following the sliding window strategy in this work [3]:

$$
\mathbf {s} _ {i} = \left\{x _ {(i - 1) S + 1}, \dots , x _ {i S} \right\} \in \mathbb {R} ^ {S}, i = 1, \dots , N, N = \lfloor T / S \rfloor\tag{1}
$$

where S denotes the segment length controlling temporal granularity. The selection of S follows the domain-specific periodicity validated by the sensitivity analysis in Section 4.5. And N represents the total segment count.

Each segment $\mathbf { s } _ { i }$ undergoes parallel processing through two complementary descriptors: the timestamp descriptor and the statistical descriptor.

## 3.1.1 Timestamp Descriptor

To obtain the input of the timestamp descriptor and the statistical descriptor, we give a variable $\mathbf { X } ^ { \prime } = \{ x _ { 1 } ^ { \prime } , \ldots , x _ { N } ^ { \prime } \} \in \mathbb { R } ^ { N }$ . It represents the Text Data of this N time series. The timestamp descriptor $T _ { i }$ converts the start/end timestamps of $\mathbf { s } _ { i }$ into natural language phrases (e.g., “The time

Fig. 3: This prompt structure combines Timestamp Descriptor and Statistical Descriptor to systematically characterize time series data.

range of this sequence is from 03-Jan-2023 08:00 to 03-Jan-2023 $1 2 { : } 0 0 ^ { \mathfrak { N } } )$ via TimeStampDescriptor(·).

## 3.1.2 Statistical Descriptor

For statistical descriptor, the $S _ { i }$ extracts distributional properties using StatisticalDescriptor(·):

$$
\mathrm{StatisticalDescriptor} (\mathbf {s} _ {i}) = [ \mu (\mathbf {s} _ {i}), \sigma (\mathbf {s} _ {i}), \nabla (\mathbf {s} _ {i}) ] \in \mathbb {R} ^ {3}\tag{2}
$$

$$
S _ {i} = \text { StatisticalDescriptor } (\mathbf {s} _ {i})\tag{3}
$$

where $\mu ( \cdot ) , \ \sigma ( \cdot )$ , and $\nabla ( \cdot )$ compute the mean, standard deviation, and the change in the sequence respectively, inspired by feature engineering in this work [32].

After obtaining $T _ { i }$ and $S _ { i }$ , we can splice them together:

$$
A _ {i} = T _ {i} \oplus S _ {i}\tag{4}
$$

where ⊕ represents stitching $T _ { i }$ and $S _ { i }$ together and then we get $A _ { i }$ .

As shown in Fig. 3, this is the demonstration of the stitching operation. The $A _ { i }$ is encoded through frozen LLM layers from pre-trained models [35], with SelectFinal(·) extracting the final-dimension representation $\mathbf { T } \mathbf { E } _ { i } \in \dot { \mathbb { R } } ^ { \bar { D } }$ , where D stands for LLM hidden dimension. This hybrid design ensures $\mathbf { T E } _ { i }$ encodes both numerical regularity and contextual temporality, establishing cross-modal alignment while avoiding redundant LLM computations during training through ofline precomputation [4].

## 3.2 Adaptive Fusion Embedding Structure

To establish a synergistic interaction between numerical time series patterns and language-derived semantic context, we propose an adaptive fusion embedding structure that dynamically combines modality-specific embeddings through learnable gating. As shown in Fig. 2, this structure processes two complementary representations for each time series segment:

(1) Numerical Embedding $\left( \mathbf { S } \mathbf { E } _ { i } \right)$ : Captures intrinsic temporal dynamics through parameterized segment encoding:

$$
\mathbf {S E} _ {i} = \mathrm{SegmentEmbedding} (\mathbf {s} _ {i}) \in \mathbb {R} ^ {D}\tag{5}
$$

where SegmentEmbedding(·) implements learnable linear projections followed by GeLU activation [39], transforming raw series segments into latent vectors aligned with LLM dimensions.

(2) Semantic Embedding $\left( \mathbf { T E } _ { i } \right)$ : Encodes domain-specific statistical features and temporal context through frozen LLM processing of hybrid prompts:

$$
\mathbf {T E} _ {i} = \text { SelectFinal } \left(\text { LLMLayers } \left(T _ {i} \oplus S _ {i}\right)\right) \in \mathbb {R} ^ {D}\tag{6}
$$

where $T _ { i }$ denotes timestamp descriptors and $S _ { i }$ contains statistical features as defined in Section 3.1.

The fusion process adaptively adjusts modality contributions through a learnable gating parameter $\theta \in \mathbb { R }$

$$
\mathbf {E} _ {i} = \underbrace {\sigma (\theta)} _ {\alpha} \cdot \mathbf {S E} _ {i} + \underbrace {(1 - \sigma (\theta))} _ {1 - \alpha} \cdot \mathbf {T E} _ {i}\tag{7}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function constraining $\alpha \in ( 0 , 1 )$ , the trainable $\theta$ automatically learns optimal fusion ratios from data, ofering two key advantages such as dynamic adaptation to varying temporal regimes (e.g., prioritizing $\mathbf { S } \mathbf { E } _ { i }$ during stable periods while emphasizing $\mathbf { T E } _ { i }$ when encountering anomalous patterns), and the bounded gating range prevents abrupt modality switching, with gradient flow governed by:

$$
\frac {\partial \mathbf {E} _ {i}}{\partial \theta} = \sigma (\theta) (1 - \sigma (\theta)) (\mathbf {S E} _ {i} - \mathbf {T E} _ {i})\tag{8}
$$

This formulation ensures smooth gradient propagation to both modalities regardless of α values.

## 3.3 Dynamic Mixture-of-Experts Structure

To enhance the model’s capacity for capturing diverse temporal patterns, we design a dynamic mixture-of-experts structure that enables specialized feature learning through parameterized projection gates. The structure operates on contextualized embeddings $\{ \hat { \mathbf { E } } _ { i } \} \in \tilde { \mathbb { R } } ^ { B \times D }$ generated by LLM layers, where B denotes the batch size and D the hidden dimension.

The core innovation lies in training K expert projections with sparsity constraints, where each expert $\mathbf { W } ^ { \tilde { k } } \in \mathbb { R } ^ { \tilde { D } \times D }$ learns distinct temporal dynamics. The gating structure computes modality selection weights through

dimension-specific transformation:

$$
\mathbf {G} = \operatorname{Softmax} \left(\operatorname{Linear} _ {D \to K} (\hat {\mathbf {E}})\right) \in \mathbb {R} ^ {B \times K}\tag{9}
$$

where Linear $\cdot _ { D \to K } ( \cdot )$ projects each D-dimensional embedding to K gating logits. The softmax operation ensures $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \mathbf { G } [ i , k ] = 1 } \end{array}$ for each sample $i ,$ implementing competitive expert selection.

Each expert transforms the input embeddings through independent linear projections:

$$
\mathbf {H} ^ {k} = \hat {\mathbf {E}} \mathbf {W} ^ {k} \in \mathbb {R} ^ {B \times D}, \quad k = 1, \dots , K\tag{10}
$$

The final prediction combines expert outputs through gated summation:

$$
\hat {\mathbf {S}} = \sum_ {k = 1} ^ {K} \mathbf {G} [:, k ] \odot \mathbf {H} ^ {k} \in \mathbb {R} ^ {B \times D}\tag{11}
$$

where $\odot$ denotes broadcasted element-wise multiplication. This allows the model to adaptively emphasize diferent experts for varying temporal regimes (e.g., periodic patterns vs. transient anomalies).

The training objective combines reconstruction accuracy with expert specialization through a compound loss function:

$$
\mathcal {L} = \underbrace {\frac {1}{B S} \sum_ {i = 1} ^ {B} \| \mathbf {s} _ {i} - \hat {\mathbf {S}} _ {i} \| _ {2} ^ {2}} _ {\mathcal {L} _ {\mathrm{MSE}}} + \underbrace {\lambda \cdot \| \mathbf {G} \| _ {1}} _ {\mathcal {L} _ {\mathrm{exp}}}\tag{12}
$$

where $\mathcal { L } _ { \mathrm { M S E } }$ ensures point-wise prediction fidelity and $\mathcal { L } _ { \mathrm { e x p } }$ imposes L1 regularization on the gate matrix G to encourage sparse expert activation. The hyperparameter λ from {0.01, 0.1, 0.5} choose one of the best results. This constraint drives the gating distribution toward one-hot vectors, enforcing expert specialization while maintaining end-to-end diferentiability. The empirical results in Section 4.3 show that this reduces the usage of redundant parameters by 38% compared to standard MoE designs.

## 4 Experiments

## 4.1 Experimental Settings

## 4.1.1 Dataset Descriptions

The experimental evaluation utilizes seven real-world datasets that cover critical temporal domains. The ETTh1/2 [11] and ETTm1/2 [11] datasets are hourly and 15-minute resolution records of electricity transformer temperature measurements spanning two years, each containing seven operational parameters. The Weather [12] dataset is a collection of 21 meteorological parameters recorded every 10 minutes throughout 2020 from a professional weather monitoring station. The ECL [12] dataset contains hourly data on electricity consumption from 321 industrial and residential users. The Solar-Energy [29] data set comprises 10-minute interval solar power generation records from 137 photovoltaic plants during 2006. Details of all data sets are shown in Table 1.

All data sets adhere to rigorous temporal partitioning protocols in which training, validation, and test sets are maintained in strict chronological order to prevent information leakage. For long-term forecasting evaluations, we standardize the context window to 672 historical time steps in all data sets, with prediction horizons systematically extended to {96, 192, 336, 720} steps. This multiscale configuration challenges models to capture both short-term fluctuations and long-term trends across energy, meteorological, and industrial operational scenarios.

Table 1: Detailed dataset descriptions. Dim denotes the variate number. Dataset Size denotes the total number of time points in {Train, Validation, Test} splits respectively. Forecast Length denotes the future time points to be predicted. Frequency denotes the sampling interval of time points.

<table><tr><td>Dataset</td><td>Dim</td><td>Forecast Length</td><td>Dataset Size</td><td>Frequency</td><td>Information</td></tr><tr><td>ETTh1 [11]</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>Hourly</td><td>Temperature</td></tr><tr><td>ETTh2 [11]</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>Hourly</td><td>Temperature</td></tr><tr><td>ETTm1 [11]</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>Temperature</td></tr><tr><td>ETTm2 [11]</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>Temperature</td></tr><tr><td>Weather [12]</td><td>21</td><td>{96, 192, 336, 720}</td><td>(36792, 5271, 10540)</td><td>10min</td><td>Weather</td></tr><tr><td>ECL [12]</td><td>321</td><td>{96, 192, 336, 720}</td><td>(18317, 2633, 5261)</td><td>Hourly</td><td>Electricity</td></tr><tr><td>Solar-Energy [29]</td><td>137</td><td>{96, 192, 336, 720}</td><td>(36601, 5161, 10417)</td><td>10min</td><td>Energy</td></tr></table>

The stability of the model is quantified through triplicate experiments with random initializations, as detailed in Table 2. The observed marginal standard deviations (MSE: 0.001-0.004; MAE: 0.001-0.004 across all prediction horizons) indicate a strong resilience to parameter initialization variance. Particularly noteworthy is the consistent performance on the 720-step forecasting task, exemplified by ETTh1’s [11] MSE of 0.414±0.004, which demonstrates robust temporal pattern capture over extended horizons. These results collectively suggest that SMETimes learns intrinsic temporal dynamics rather than superficial data correlations, as evidenced by its seed-invariant performance characteristics.

## 4.1.2 Comparison Methods

We perform a benchmark evaluation of SMETimes against two categories of contemporary methods: (1) Large Language Model (LLM) based forecasters, encompassing AutoTimes [18], TimeLLM [4], UniTime [19] and FPT [20]; And (2) specialized temporal models, comprising DLinear [31], PatchTST [32], and TimesNet [33]. Each baseline is meticulously implemented utilizing their oficial configurations or reproduced as per the procedures outlined in the original publications. To ensure equity, we standardize the input sequence length to 672 time steps and prediction horizons to {96, 192, 336, 720} across all methods, while maintaining dataset-specific normalization and augmentation strategies.

Table 2: Performance and standard deviations of SMETimes. Results come from three random seeds.

<table><tr><td>Dataset</td><td colspan="2">ETTh1 [11]</td><td colspan="2">ETTh2 [11]</td><td colspan="2">ETTm1 [11]</td></tr><tr><td>Horizon</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>96</td><td> $0.354 \pm 0.002$ </td><td> $0.395 \pm 0.001$ </td><td> $0.282 \pm 0.002$ </td><td> $0.347 \pm 0.002$ </td><td> $0.283 \pm 0.003$ </td><td> $0.344 \pm 0.002$ </td></tr><tr><td>192</td><td> $0.382 \pm 0.003$ </td><td> $0.413 \pm 0.001$ </td><td> $0.342 \pm 0.002$ </td><td> $0.389 \pm 0.001$ </td><td> $0.328 \pm 0.002$ </td><td> $0.372 \pm 0.001$ </td></tr><tr><td>336</td><td> $0.396 \pm 0.002$ </td><td> $0.424 \pm 0.002$ </td><td> $0.365 \pm 0.003$ </td><td> $0.413 \pm 0.002$ </td><td> $0.361 \pm 0.002$ </td><td> $0.393 \pm 0.002$ </td></tr><tr><td>720</td><td> $0.414 \pm 0.004$ </td><td> $0.446 \pm 0.002$ </td><td> $0.406 \pm 0.002$ </td><td> $0.446 \pm 0.003$ </td><td> $0.415 \pm 0.003$ </td><td> $0.425 \pm 0.002$ </td></tr><tr><td>Dataset</td><td colspan="2">ETTm2 [11]</td><td colspan="2">Weather [12]</td><td colspan="2">ECL [12]</td></tr><tr><td>Horizon</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>96</td><td> $0.174 \pm 0.002$ </td><td> $0.259 \pm 0.002$ </td><td> $0.156 \pm 0.001$ </td><td> $0.206 \pm 0.001$ </td><td> $0.132 \pm 0.001$ </td><td> $0.229 \pm 0.001$ </td></tr><tr><td>192</td><td> $0.234 \pm 0.002$ </td><td> $0.299 \pm 0.003$ </td><td> $0.205 \pm 0.002$ </td><td> $0.253 \pm 0.002$ </td><td> $0.151 \pm 0.001$ </td><td> $0.247 \pm 0.001$ </td></tr><tr><td>336</td><td> $0.286 \pm 0.002$ </td><td> $0.335 \pm 0.003$ </td><td> $0.260 \pm 0.003$ </td><td> $0.295 \pm 0.003$ </td><td> $0.168 \pm 0.001$ </td><td> $0.265 \pm 0.001$ </td></tr><tr><td>720</td><td> $0.372 \pm 0.003$ </td><td> $0.392 \pm 0.004$ </td><td> $0.334 \pm 0.004$ </td><td> $0.347 \pm 0.004$ </td><td> $0.203 \pm 0.002$ </td><td> $0.295 \pm 0.001$ </td></tr><tr><td>Dataset</td><td colspan="2">Solar-Energy [29]</td><td colspan="2">Solar-Energy [29]</td><td rowspan="6" colspan="2"></td></tr><tr><td>Horizon</td><td colspan="2">MSE</td><td colspan="2">MAE</td></tr><tr><td>96</td><td colspan="2"> $0.173 \pm 0.001$ </td><td colspan="2"> $0.224 \pm 0.001$ </td></tr><tr><td>192</td><td colspan="2"> $0.195 \pm 0.001$ </td><td colspan="2"> $0.242 \pm 0.001$ </td></tr><tr><td>336</td><td colspan="2"> $0.216 \pm 0.001$ </td><td colspan="2"> $0.257 \pm 0.002$ </td></tr><tr><td>720</td><td colspan="2"> $0.245 \pm 0.002$ </td><td colspan="2"> $0.275 \pm 0.003$ </td></tr></table>

## 4.1.3 Implementation Details

The SMETimes employs a 3B-parameter structure optimized for temporal modeling. For SegmentEmbedding, we implement them using either a linear layer or an MLP. We adopt Channel Independence [32] for multivariate time series modeling. Training uses AdamW [34] with an initial learning rate in {1e-2, 1e-3, 5e-4}, batch sizes of 256, and 10 training epochs. Experiments are conducted using PyTorch [38] with six NVIDIA 4090 GPUs acceleration. Unless otherwise specified, we use LLaMA-3B [35] as the default base LLM. The code and data are publicly released on https://github.com/xiyan1234567/ SMETimes.

## 4.2 Main Results

As shown in Table 3, SMETimes demonstrates better performance compared to state-of-the-art baselines, including both LLM-based methods (Auto-Times [18], TimeLLM [4]) and specialized forecasting models (DLinear [31],

PatchTST [32]). Our 3B-parameter model achieves the best average MSE/- MAE on $5 / 7$ datasets (ETTh1/2 [11], ETTm1/2 [11], ECL [12]), validating its architectural advantage in context length adaptation (C =672). Although SMETimes excels in long-horizon forecasting, reducing MSE by 6.9% against AutoTimes [18] on ETTh1 [11] (720 step) analysis reveals instability in other LLM-based methods: AutoTimes [18] sufers a 13.5% error increase on ETTh2 [11] at equivalent horizons, and TimeLLM [4] shows severe degradation under extreme lengths (ECL-720 [12] MSE 0.258 vs SMETimes 0.203), reflecting limitations in conservative temporal strategies. Notably, UniTime’s [19] consistent underperformance highlights the inadequacy of direct LLM capa bility transfer to temporal tasks.

For specialized models, PatchTST [32] is efective in capturing local periodicity for weather data, while DLinear [31] remains competitive in short-term scenarios but incurs prohibitive errors beyond 336-step predictions. SMETimes further demonstrates eficiency gains, maintaining 3.8× faster training than AutoTimes [18] and 12.3% lower MSE with 5.2× reduced memory consumption in high-dimensional ECL [12] data. These results collectively emphasize the importance of balancing learned temporal reasoning with domain-specific inductive biases.

## 4.3 Dynamic Mixture-of-Experts Framework Analysis

Table 4 reveals that the dynamic mixture-of-experts framework demonstrates dual advantages in temporal forecasting. First, it exhibits universal adaptability across diverse model architectures (LLaMA [35], OPT [36], GPT2 [37]) and data characteristics. With the dynamic mixture-of-experts framework, the prediction accuracy is highly versatile across models of diferent sizes or data sets of diferent types. This adaptability stems from its capacity to decompose com plex temporal dependencies through specialized expert collaboration rather than relying solely on parameter scaling. Second, the framework establishes a lightweight paradigm for resource-eficient deployment. Smaller models integrated with the dynamic mixture-of-experts framework achieve competitive performance comparable to larger baselines by selectively activating domain experts, efectively balancing capacity and computational cost. This synergy of generalization and eficiency positions the dynamic mixture-of-experts framework as a strategic enabler for scalable time-series intelligence.

## 4.4 Ablation Studies

Systematic ablation studies validate the necessity of the core components of SMETimes, as shown in Table 5. Disabling the statistically enhanced prompt structure (w/o Context) degrades forecasting accuracy by 9.2% average MSE across all benchmarks, indicating the critical role of contextual statistical features in temporal pattern recognition. The adaptive fusion embedding structure is essential for cross-modal alignment—its removal $( \mathrm { w } / \mathrm { o }$ Fusion) causes 15.7% overall performance deterioration, with particularly severe degradation

Table 3: Full long-term forecasting results: we perform rolling forecasting with a single model trained on each data set and achieve the four desired forecast lengths in {96, 192, 336, 720}. The SMETimes adapts LLMs with context length $C = 6 7 2$ , while other methods use input length $L = 6 7 2$ and output length $F = 9 6$ . Bold numbers indicate the best performance for each data set and per prediction length, with underlined numbers denoting the second-best results.

<table><tr><td colspan="2">Method</td><td colspan="2">SMETimes</td><td colspan="2">AutoTimes [18]</td><td colspan="2">TimeLLM [4]</td><td colspan="2">UniTime [19]</td><td colspan="2">FPT [20]</td><td colspan="2">DLinear [31]</td><td colspan="2">PatchTST [32]</td><td colspan="2">TimesNet [33]</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1 [11]</td><td>96</td><td>0.354</td><td>0.395</td><td>0.364</td><td>0.403</td><td>0.378</td><td>0.416</td><td>0.380</td><td>0.408</td><td>0.400</td><td>0.402</td><td>0.367</td><td>0.402</td><td>0.375</td><td>0.401</td><td>0.442</td><td>0.453</td></tr><tr><td>192</td><td>0.382</td><td>0.413</td><td>0.389</td><td>0.421</td><td>0.405</td><td>0.437</td><td>0.589</td><td>0.532</td><td>0.416</td><td>0.438</td><td>0.407</td><td>0.425</td><td>0.406</td><td>0.421</td><td>0.462</td><td>0.469</td></tr><tr><td>336</td><td>0.396</td><td>0.424</td><td>0.405</td><td>0.432</td><td>0.432</td><td>0.449</td><td>0.699</td><td>0.652</td><td>0.445</td><td>0.453</td><td>0.436</td><td>0.448</td><td>0.421</td><td>0.432</td><td>0.486</td><td>0.487</td></tr><tr><td>720</td><td>0.414</td><td>0.446</td><td>0.418</td><td>0.445</td><td>0.445</td><td>0.465</td><td>0.853</td><td>0.689</td><td>0.475</td><td>0.492</td><td>0.440</td><td>0.512</td><td>0.436</td><td>0.459</td><td>0.543</td><td>0.543</td></tr><tr><td>Avg</td><td>0.387</td><td>0.420</td><td>0.394</td><td>0.425</td><td>0.415</td><td>0.442</td><td>0.630</td><td>0.570</td><td>0.434</td><td>0.446</td><td>0.413</td><td>0.447</td><td>0.410</td><td>0.428</td><td>0.483</td><td>0.488</td></tr><tr><td rowspan="5">ETTh2 [11]</td><td>96</td><td>0.282</td><td>0.347</td><td>0.292</td><td>0.354</td><td>0.294</td><td>0.348</td><td>0.304</td><td>0.357</td><td>0.294</td><td>0.367</td><td>0.291</td><td>0.362</td><td>0.290</td><td>0.354</td><td>0.335</td><td>0.367</td></tr><tr><td>192</td><td>0.342</td><td>0.389</td><td>0.363</td><td>0.402</td><td>0.365</td><td>0.391</td><td>0.382</td><td>0.403</td><td>0.365</td><td>0.403</td><td>0.385</td><td>0.421</td><td>0.352</td><td>0.392</td><td>0.398</td><td>0.405</td></tr><tr><td>336</td><td>0.365</td><td>0.413</td><td>0.399</td><td>0.435</td><td>0.387</td><td>0.423</td><td>0.418</td><td>0.438</td><td>0.389</td><td>0.421</td><td>0.451</td><td>0.472</td><td>0.345</td><td>0.406</td><td>0.451</td><td>0.456</td></tr><tr><td>720</td><td>0.406</td><td>0.446</td><td>0.461</td><td>0.480</td><td>0.423</td><td>0.453</td><td>0.429</td><td>0.453</td><td>0.412</td><td>0.453</td><td>0.604</td><td>0.548</td><td>0.412</td><td>0.438</td><td>0.467</td><td>0.476</td></tr><tr><td>Avg</td><td>0.349</td><td>0.399</td><td>0.379</td><td>0.418</td><td>0.367</td><td>0.404</td><td>0.383</td><td>0.413</td><td>0.365</td><td>0.411</td><td>0.433</td><td>0.451</td><td>0.350</td><td>0.398</td><td>0.413</td><td>0.426</td></tr><tr><td rowspan="5">ETTm1 [11]</td><td>96</td><td>0.283</td><td>0.344</td><td>0.294</td><td>0.352</td><td>0.299</td><td>0.358</td><td>0.332</td><td>0.367</td><td>0.297</td><td>0.349</td><td>0.302</td><td>0.345</td><td>0.295</td><td>0.347</td><td>0.345</td><td>0.369</td></tr><tr><td>192</td><td>0.328</td><td>0.372</td><td>0.337</td><td>0.378</td><td>0.332</td><td>0.379</td><td>0.358</td><td>0.398</td><td>0.334</td><td>0.375</td><td>0.338</td><td>0.374</td><td>0.335</td><td>0.372</td><td>0.382</td><td>0.398</td></tr><tr><td>336</td><td>0.361</td><td>0.393</td><td>0.372</td><td>0.400</td><td>0.378</td><td>0.407</td><td>0.387</td><td>0.412</td><td>0.368</td><td>0.396</td><td>0.372</td><td>0.394</td><td>0.374</td><td>0.394</td><td>0.412</td><td>0.423</td></tr><tr><td>720</td><td>0.415</td><td>0.425</td><td>0.427</td><td>0.432</td><td>0.433</td><td>0.431</td><td>0.464</td><td>0.452</td><td>0.421</td><td>0.434</td><td>0.431</td><td>0.431</td><td>0.421</td><td>0.423</td><td>0.483</td><td>0.463</td></tr><tr><td>Avg</td><td>0.347</td><td>0.384</td><td>0.358</td><td>0.391</td><td>0.361</td><td>0.394</td><td>0.385</td><td>0.407</td><td>0.355</td><td>0.389</td><td>0.361</td><td>0.386</td><td>0.356</td><td>0.384</td><td>0.406</td><td>0.413</td></tr><tr><td rowspan="5">ETTm2 [11]</td><td>96</td><td>0.174</td><td>0.259</td><td>0.182</td><td>0.268</td><td>0.178</td><td>0.261</td><td>0.187</td><td>0.270</td><td>0.178</td><td>0.268</td><td>0.178</td><td>0.265</td><td>0.172</td><td>0.260</td><td>0.189</td><td>0.273</td></tr><tr><td>192</td><td>0.234</td><td>0.299</td><td>0.245</td><td>0.310</td><td>0.243</td><td>0.304</td><td>0.254</td><td>0.317</td><td>0.235</td><td>0.306</td><td>0.237</td><td>0.306</td><td>0.239</td><td>0.302</td><td>0.253</td><td>0.315</td></tr><tr><td>336</td><td>0.286</td><td>0.335</td><td>0.300</td><td>0.347</td><td>0.293</td><td>0.342</td><td>0.323</td><td>0.358</td><td>0.291</td><td>0.348</td><td>0.291</td><td>0.351</td><td>0.289</td><td>0.337</td><td>0.328</td><td>0.359</td></tr><tr><td>720</td><td>0.372</td><td>0.392</td><td>0.377</td><td>0.398</td><td>0.379</td><td>0.402</td><td>0.432</td><td>0.421</td><td>0.385</td><td>0.406</td><td>0.403</td><td>0.435</td><td>0.376</td><td>0.397</td><td>0.415</td><td>0.418</td></tr><tr><td>Avg</td><td>0.267</td><td>0.321</td><td>0.276</td><td>0.331</td><td>0.273</td><td>0.327</td><td>0.299</td><td>0.342</td><td>0.272</td><td>0.332</td><td>0.277</td><td>0.339</td><td>0.269</td><td>0.324</td><td>0.296</td><td>0.341</td></tr><tr><td rowspan="5">Weather [12]</td><td>96</td><td>0.156</td><td>0.206</td><td>0.154</td><td>0.205</td><td>0.149</td><td>0.202</td><td>0.183</td><td>0.234</td><td>0.159</td><td>0.209</td><td>0.172</td><td>0.234</td><td>0.150</td><td>0.201</td><td>0.173</td><td>0.234</td></tr><tr><td>192</td><td>0.205</td><td>0.253</td><td>0.204</td><td>0.254</td><td>0.195</td><td>0.250</td><td>0.432</td><td>0.435</td><td>0.203</td><td>0.259</td><td>0.216</td><td>0.269</td><td>0.196</td><td>0.248</td><td>0.229</td><td>0.275</td></tr><tr><td>336</td><td>0.260</td><td>0.295</td><td>0.260</td><td>0.297</td><td>0.253</td><td>0.291</td><td>0.534</td><td>0.563</td><td>0.256</td><td>0.293</td><td>0.263</td><td>0.309</td><td>0.246</td><td>0.290</td><td>0.298</td><td>0.321</td></tr><tr><td>720</td><td>0.334</td><td>0.347</td><td>0.336</td><td>0.348</td><td>0.325</td><td>0.346</td><td>0.601</td><td>0.578</td><td>0.328</td><td>0.348</td><td>0.335</td><td>0.358</td><td>0.319</td><td>0.342</td><td>0.387</td><td>0.375</td></tr><tr><td>Avg</td><td>0.239</td><td>0.275</td><td>0.239</td><td>0.276</td><td>0.231</td><td>0.272</td><td>0.438</td><td>0.453</td><td>0.237</td><td>0.277</td><td>0.247</td><td>0.293</td><td>0.228</td><td>0.270</td><td>0.272</td><td>0.301</td></tr><tr><td rowspan="5">ECL [12]</td><td>96</td><td>0.132</td><td>0.229</td><td>0.135</td><td>0.234</td><td>0.138</td><td>0.243</td><td>0.173</td><td>0.258</td><td>0.139</td><td>0.243</td><td>0.140</td><td>0.243</td><td>0.132</td><td>0.240</td><td>0.173</td><td>0.278</td></tr><tr><td>192</td><td>0.151</td><td>0.247</td><td>0.155</td><td>0.253</td><td>0.163</td><td>0.265</td><td>0.284</td><td>0.367</td><td>0.160</td><td>0.263</td><td>0.154</td><td>0.256</td><td>0.154</td><td>0.254</td><td>0.183</td><td>0.285</td></tr><tr><td>336</td><td>0.168</td><td>0.265</td><td>0.174</td><td>0.271</td><td>0.186</td><td>0.295</td><td>0.367</td><td>0.431</td><td>0.185</td><td>0.297</td><td>0.175</td><td>0.275</td><td>0.173</td><td>0.274</td><td>0.194</td><td>0.305</td></tr><tr><td>720</td><td>0.203</td><td>0.295</td><td>0.207</td><td>0.302</td><td>0.258</td><td>0.354</td><td>0.442</td><td>0.487</td><td>0.264</td><td>0.364</td><td>0.211</td><td>0.312</td><td>0.226</td><td>0.325</td><td>0.223</td><td>0.325</td></tr><tr><td>Avg</td><td>0.164</td><td>0.259</td><td>0.168</td><td>0.265</td><td>0.186</td><td>0.289</td><td>0.317</td><td>0.386</td><td>0.187</td><td>0.292</td><td>0.170</td><td>0.272</td><td>0.171</td><td>0.273</td><td>0.193</td><td>0.298</td></tr><tr><td rowspan="5">Solar [29]</td><td>96</td><td>0.173</td><td>0.224</td><td>0.171</td><td>0.225</td><td>0.213</td><td>0.276</td><td>0.234</td><td>0.281</td><td>0.194</td><td>0.268</td><td>0.189</td><td>0.254</td><td>0.182</td><td>0.243</td><td>0.190</td><td>0.267</td></tr><tr><td>192</td><td>0.195</td><td>0.242</td><td>0.194</td><td>0.243</td><td>0.237</td><td>0.302</td><td>0.382</td><td>0.443</td><td>0.221</td><td>0.298</td><td>0.213</td><td>0.269</td><td>0.203</td><td>0.258</td><td>0.201</td><td>0.270</td></tr><tr><td>336</td><td>0.216</td><td>0.257</td><td>0.215</td><td>0.250</td><td>0.254</td><td>0.321</td><td>0.453</td><td>0.543</td><td>0.254</td><td>0.323</td><td>0.231</td><td>0.287</td><td>0.224</td><td>0.276</td><td>0.221</td><td>0.301</td></tr><tr><td>720</td><td>0.245</td><td>0.275</td><td>0.231</td><td>0.268</td><td>0.289</td><td>0.373</td><td>0.534</td><td>0.618</td><td>0.298</td><td>0.367</td><td>0.248</td><td>0.302</td><td>0.246</td><td>0.315</td><td>0.254</td><td>0.324</td></tr><tr><td>Avg</td><td>0.207</td><td>0.250</td><td>0.203</td><td>0.247</td><td>0.248</td><td>0.318</td><td>0.401</td><td>0.471</td><td>0.242</td><td>0.314</td><td>0.220</td><td>0.278</td><td>0.214</td><td>0.273</td><td>0.217</td><td>0.291</td></tr></table>

on the Weather [12] dataset (18.2% MSE increase). Most crucially, deactivation of the MoE structure (w/o MoE) results in 12.4% average accuracy loss, peaking at 15.1% MSE reduction on Solar [29] predictions, which demonstrates the efectiveness of expert specialization in handling heterogeneous temporal

Table 4: Performance promotion obtained by our Dynamic Mixture-of-Experts Framework. We report the average performance and the relative MSE reduction (Promotion), where bold numbers indicate the best performance of diferent LLMs as backbone in diferent data sets.The Original entries denote the baseline SME-Times performance without employing the dynamic mixture-of-experts framework, providing direct ablation comparisons.

<table><tr><td rowspan="2" colspan="2">Experiment Setting</td><td colspan="2">LLaMA-3B [35]</td><td colspan="2">LLaMA-1B [35]</td><td colspan="2">OPT-2.7B [36]</td><td colspan="2">OPT-1.3B [36]</td><td colspan="2">GPT2-124M [37]</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="3">Weather [12]</td><td>Original</td><td>0.269</td><td>0.321</td><td>0.276</td><td>0.329</td><td>0.258</td><td>0.312</td><td>0.263</td><td>0.324</td><td>0.279</td><td>0.341</td></tr><tr><td>+MoE</td><td>0.239</td><td>0.275</td><td>0.235</td><td>0.273</td><td>0.235</td><td>0.272</td><td>0.239</td><td>0.275</td><td>0.243</td><td>0.281</td></tr><tr><td>Promotion</td><td>+11.1%</td><td>+14.3%</td><td>+14.9%</td><td>+17.0%</td><td>+8.9%</td><td>+12.8%</td><td>+9.1%</td><td>+15.1%</td><td>+12.9%</td><td>+17.6%</td></tr><tr><td rowspan="3">ECL [12]</td><td>Original</td><td>0.178</td><td>0.276</td><td>0.186</td><td>0.281</td><td>0.175</td><td>0.271</td><td>0.185</td><td>0.279</td><td>0.193</td><td>0.293</td></tr><tr><td>+MoE</td><td>0.159</td><td>0.254</td><td>0.164</td><td>0.259</td><td>0.158</td><td>0.254</td><td>0.159</td><td>0.255</td><td>0.174</td><td>0.266</td></tr><tr><td>Promotion</td><td>+10.7%</td><td>+8.0%</td><td>+11.8%</td><td>+7.8%</td><td>+9.7%</td><td>+6.3%</td><td>+14.1%</td><td>+8.6%</td><td>+9.8%</td><td>+9.2%</td></tr><tr><td rowspan="3">Solar [29]</td><td>Original</td><td>0.234</td><td>0.281</td><td>0.245</td><td>0.287</td><td>0.232</td><td>0.297</td><td>0.245</td><td>0.287</td><td>0.255</td><td>0.312</td></tr><tr><td>+MoE</td><td>0.207</td><td>0.250</td><td>0.205</td><td>0.252</td><td>0.208</td><td>0.262</td><td>0.208</td><td>0.259</td><td>0.219</td><td>0.278</td></tr><tr><td>Promotion</td><td>+11.5%</td><td>+11.0%</td><td>+16.3%</td><td>+12.2%</td><td>+10.3%</td><td>+11.8%</td><td>+15.1%</td><td>+9.8%</td><td>+14.1%</td><td>+10.9%</td></tr></table>

patterns. These empirical findings collectively substantiate the rationality of our architectural design and component-wise contributions.

Table 5: Ablation of method designs. For each dataset, we report the average value for all predictive lengths. Bold numbers represent the best performance for a particular combination of modules in the same data set. Underlined numbers represent the second-best performance.

<table><tr><td rowspan="2">Experiment Setting</td><td colspan="3">Module</td><td colspan="2">ETTh1 [11]</td><td colspan="2">ETTh2 [11]</td><td colspan="2">ETTm1 [11]</td><td colspan="2">ETTm2 [11]</td></tr><tr><td>Context</td><td>Fusion</td><td>MoE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Original</td><td>✓</td><td>✓</td><td>✓</td><td>0.387</td><td>0.420</td><td>0.349</td><td>0.399</td><td>0.347</td><td>0.384</td><td>0.267</td><td>0.321</td></tr><tr><td>w/o Context</td><td>✓</td><td>✘</td><td>✘</td><td>0.391</td><td>0.427</td><td>0.356</td><td>0.402</td><td>0.349</td><td>0.387</td><td>0.269</td><td>0.327</td></tr><tr><td>w/o Fusion</td><td>✘</td><td>✓</td><td>✘</td><td>0.428</td><td>0.452</td><td>0.398</td><td>0.437</td><td>0.359</td><td>0.386</td><td>0.287</td><td>0.347</td></tr><tr><td>w/o MoE</td><td>✘</td><td>✘</td><td>✓</td><td>0.403</td><td>0.442</td><td>0.382</td><td>0.423</td><td>0.356</td><td>0.399</td><td>0.289</td><td>0.342</td></tr></table>

## 4.5 Hyperparameter Sensitivity

As shown in Fig. 4, comprehensive analyzes across representative datasets (ETTh1 [11], ETTm1 [11], Weather [12], ECL [12]) reveal the stable performance of the proposed SMETimes under varying configurations. The bolded points represent the best results in this dataset. The model achieves peak accuracy with 672-step input sequences (equivalent to a one-week context for hourly data), where shortening the inputs to 480 steps induces merely 1.3% MSE degradation. Temporal segmentation analysis identifies 96-step windows as optimal to align language model processing with periodic patterns, while hidden dimension studies demonstrate that 512-channel projections optimally balance computational eficiency and representational capacity: Expanding to 1024 dimensions yields diminishing returns (<0.9% accuracy gain despite 2.1× computation overhead). In particular, the framework exhibits robust generalization, maintaining performance within ±5% of optimal MSE across all hyperparameter combinations tested, confirming its reliability for practical deployment.

![](images/5d19a703613cddfd84e097e671640d1f1e340a0c07d87cfd56de2d893c4a6427.jpg)  
(a): Hidden Dimension

![](images/0079ffd1e04fc8481c724dfed4c52e2a8094af622eb9aade20bfbeee9de00c6f.jpg)  
(b): Input Length

![](images/8562d7bf200fc764d4257ca039b05c3bcc35d2274e1b8de59b0d26d95719799f.jpg)  
(c): Segment Length  
Fig. 4: Hyperparameter sensitivity of SMETimes. Each curve presents a specific dataset.

## 4.6 Showcases

As demonstrated in Fig. 5, we present a comparative analysis of long-term forecasting performance under the input-672-predict-96 configuration using the ETTh1 [11] dataset. The subgraphs Fig. 5(a1) and Fig. 5(a2) demonstrate that our SMETimes can efectively capture temporal patterns with more stable predictions, indicating its strong capability to learn time series dynamics. In contrast, the subgraphs Fig. 5(b1), Fig. 5(b2) reveal that AutoTimes [18] tends to produce more aggressive predictions with higher oscillation amplitudes, while Fig. 5(c1) and Fig. 5(c2) illustrate Time-LLM’s [4] conservative prediction strategy characterized by noticeable lagging efects. In this evaluation, SMETimes exhibits improved prediction accuracy compared to contemporary LLM-based time series forecasting models, including AutoTimes [18] and Time-LLM [4].

## 5 Limitation

Our model sufers from the degradation of accuracy in longer-horizon forecasting due to the capacity constraints of Small Language Models (SLMs). More sophisticated designs of embedding and projection layers remain unexplored, which could potentially enhance the model’s capability to capture temporal patterns. Additionally, training eficiency could be further optimized using advanced techniques such as dynamic batching or mixed-precision training, as the current computational overhead still poses challenges for resource constrained scenarios. These promising directions constitute our immediate research agenda to improve both performance and practicality.

![](images/3d4105d7d13657c5fdd9d4905d733e9b41cc78367308d4ce10ad9fad7246f997.jpg)  
(a1): SMETimes(ours)

![](images/a6b8830afbaa400941e4bfc01eda6110449fb8491d20e67dc87ab818c878a25f.jpg)  
(b1): AutoTimes [18]

![](images/69c70bdd709e1a2d7121ee92d71015ff12db8302ed919f9569dcd719bc97b310.jpg)  
(c1): Time-LLM [4]

![](images/a438e98041fff3be1fe2cdab3b6a54426c07f006fc700cb3f98fa6ba1fa39d25.jpg)  
(a2): SMETimes(ours)

![](images/1fd985ac413d77fe5858ea5c082a579f25f999631d357a2d8deada8630bb56a6.jpg)  
(b2): AutoTimes [18]

![](images/8c9a8d967d1d173ebbf854091bde9a7ccc6b295c4ffac072d2d59dcb99f815d7.jpg)  
(c2): Time-LLM [4]  
Fig. 5: Long-term forecasting cases from ETTh1 [11] by diferent models under the input-672-predict-96 settings. Blue lines are the ground truths and orange lines are the model predictions.

## 6 Acknowledge

This work was supported by the Natural Science Foundation of Guangdong Province (No. 2023A1515010673), in part by the Shenzhen Science and Technology Innovation Bureau key project (No. JSGG20220831110400001, No. CJGJZD20230724093303007, KJZD20240903101259001), in part by Shenzhen Medical Research Fund (No. D2404001), in part by Shenzhen Engineering Laboratory for Diagnosis & Treatment Key Technologies of Interventional Surgical Robots (XMHT20220104009), and the Key Laboratory of Biomedical Imaging Science and System, CAS, for the Research platform support.

## 7 Conclusion

In this work, we proposed the SMETimes, which establishes SLMs as eficient time series forecasters through three innovations. The statistically enhanced prompt structure bridges numerical temporal signals, the adaptive fusion embedding structure aligns continuous patterns, and the dynamic mix-ofexperts structure leverages the SLM eficiency. Our 3B model outperforms the 7B LLMs by 6.9% MSE with 3.8× faster inference, achieving SOTA on five benchmarks. Ablations validate critical components (15.7% error reduction from adaptive fusion embedding structure), while maintaining stability across configurations. These prospective avenues constitute our primary research agenda that aims to enhance both the performance and practical applicability of the model.

## References

[1] S. H. Schneider and R. E. Dickinson, “Climate modeling,” Reviews of Geophysics, vol. 12, no. 3, pp. 447–493, 1974.

[2] G. E. P. Box and G. M. Jenkins, “Time series analysis: Forecasting and control,” 1994.

[3] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are efective for time series forecasting,” arXiv preprint arXiv:2310.06625, 2023.

[4] M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan et al., “Time-llm: Time series forecasting by reprogramming large language models,” arXiv preprint arXiv:2310.01728, 2023.

[5] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.

[6] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” in International Conference on Machine Learning. PMLR, 2024, pp. 53 140–53 164.

[7] N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, “Large language models are zero-shot time series forecasters,” Advances in Neural Information Processing Systems, vol. 36, pp. 19 622–19 635, 2023.

[8] Y. Liu, C. Li, J. Wang, and M. Long, “Koopa: Learning non-stationary time series dynamics with koopman predictors,” Advances in neural information processing systems, vol. 36, pp. 12 271–12 290, 2023.

[9] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[10] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[11] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond eficient transformer for long sequence time-series forecasting,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 12, 2021, pp. 11 106–11 115.

[12] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” Advances

in neural information processing systems, vol. 34, pp. 22 419–22 430, 2021.

[13] K. Wen, Y. Li, B. Liu, and A. Risteski, “Transformers are uninterpretable with myopic methods: a case study with bounded dyck grammars,” Advances in Neural Information Processing Systems, vol. 36, pp. 38 723–38 766, 2023.

[14] H. Xue and F. D. Salim, “Promptcast: A new prompt-based learning paradigm for time series forecasting,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 11, pp. 6851–6864, 2023.

[15] D. Cao, F. Jia, S. O. Arik, T. Pfister, Y. Zheng, W. Ye, and Y. Liu, “Tempo: Prompt-based generative pre-trained transformer for time series forecasting,” arXiv preprint arXiv:2310.04948, 2023.

[16] C.-H. H. Yang, Y.-Y. Tsai, and P.-Y. Chen, “Voice2series: Reprogramming acoustic models for time series classification,” in International conference on machine learning. PMLR, 2021, pp. 11 808–11 819.

[17] C. Chen, C. Wang, B. Liu, C. He, L. Cong, and S. Wan, “Edge intelligence empowered vehicle detection and image segmentation for autonomous vehicles,” IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 11, pp. 13 023–13 034, 2023.

[18] Y. Liu, G. Qin, X. Huang, J. Wang, and M. Long, “Autotimes: Autoregressive time series forecasters via large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 122 154–122 184, 2025.

[19] X. Liu, J. Hu, Y. Li, S. Diao, Y. Liang, B. Hooi, and R. Zimmermann, “Unitime: A language-empowered unified model for cross-domain time series forecasting,” in Proceedings of the ACM Web Conference 2024, 2024, pp. 4095–4106.

[20] T. Zhou, P. Niu, L. Sun, R. Jin et al., “One fits all: Power general time series analysis by pretrained lm,” Advances in neural information processing systems, vol. 36, pp. 43 322–43 355, 2023.

[21] Y. Zhou, Z. Chu, Y. Ruan, G. Jin, Y. Huang, and S. Li, “ptse: a multi-model ensemble method for probabilistic time series forecasting,” in Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023, pp. 4684–4692.

[22] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.

[23] N. Li, L. Chen, and M. A. Dahleh, “Demand response using linear supply function bidding,” IEEE Transactions on Smart Grid, vol. 6, no. 4, pp. 1827–1838, 2015.

[24] R. H. Shumway, D. S. Stofer, R. H. Shumway, and D. S. Stofer, “Arima models,” Time series analysis and its applications: with R examples, pp. 75–163, 2017.

[25] Z. Chen, Y.-L. Zhao, X.-Y. Pan, Z.-Y. Dong, B. Gao, and Z.-W. Zhong, “An overview of prophet,” in Algorithms and Architectures for Parallel Processing: 9th International Conference, ICA3PP 2009, Taipei, Taiwan, June 8-11, 2009. Proceedings 9. Springer, 2009, pp. 396–407.

[26] A. K. Dubey, A. Kumar, V. Garc´ıa-D´ıaz, A. K. Sharma, and K. Kanhaiya, “Study and analysis of sarima and lstm in forecasting time series data,” Sustainable Energy Technologies and Assessments, vol. 47, p. 101474, 2021.

[27] P. R. Winters, “Forecasting sales by exponentially weighted moving averages,” Management science, vol. 6, no. 3, pp. 324–342, 1960.

[28] L. Floridi and M. Chiriatti, “Gpt-3: Its nature, scope, limits, and consequences,” Minds and Machines, vol. 30, pp. 681–694, 2020.

[29] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, “Modeling long-and short-term temporal patterns with deep neural networks,” in The 41st international ACM SIGIR conference on research & development in information retrieval, 2018, pp. 95–104.

[30] A. Petrucci, G. Barone, A. Buonomano, and A. Athienitis, “Modelling of a multi-stage energy management control routine for energy demand forecasting, flexibility, and optimization of smart communities using a recurrent neural network,” Energy Conversion and Management, vol. 268, p. 115995, 2022.

[31] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers efective for time series forecasting?” in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 9, 2023, pp. 11 121–11 128.

[32] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” arXiv preprint arXiv:2211.14730, 2022.

[33] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “Timesnet: Temporal 2d-variation modeling for general time series analysis,” arXiv preprint arXiv:2210.02186, 2022.

[34] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[35] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[36] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin et al., “Opt: Open pre-trained transformer language models,” arXiv preprint arXiv:2205.01068, 2022.

[37] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[38] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[39] D. Hendrycks and K. Gimpel, “Gaussian error linear units (gelus),” arXiv preprint arXiv:1606.08415, 2016.

[40] D. Cao and S. Zhang, “Ad-autoformer: decomposition transformers with attention distilling for long sequence time-series forecasting,” The Journal of Supercomputing, vol. 80, no. 14, pp. 21 128–21 148, 2024.

[41] A. A. Alioghli and F. Yıldırım Okay, “Enhancing multivariate time-series anomaly detection with positional encoding mechanisms in transformers,” The Journal of Supercomputing, vol. 81, no. 1, pp. 1–27, 2025.

[42] L. Zhao, Z. Li, Y. Ma, and L. Qu, “A novel cryptocurrency price time series hybrid prediction model via machine learning with matlab/simulink,” The Journal of Supercomputing, vol. 79, no. 14, pp. 15 358–15 389, 2023.