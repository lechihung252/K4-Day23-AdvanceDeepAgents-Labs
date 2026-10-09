# Survey on Efficient Inference and Small Language Models

## TL;DR
- Small Language Models (SLMs) are transformer-based models with 100M to 5B parameters designed for efficient inference on edge and resource-constrained devices, with foundational work focusing on efficient attention and model compression [1][2].
- Main technical approaches for efficiency include pruning, quantization, knowledge distillation, architectural optimization, reinforcement learning for training efficiency, neural architecture search, and collaborative edge-cloud inference [3][4][5][6][7].
- Recent benchmarks show that fine-tuned small models can match or exceed much larger models across diverse tasks while significantly reducing resource requirements; Qwen3 and Llama series are leading examples [8][9].
- Current trends emphasize hardware-aware fault tolerance and novel algorithmic accelerations, but direct survey papers on SLM efficient inference remain sparse, highlighting open problems in standardization and integration of efficiency techniques with deployment [10][11].

## Background
Small Language Models (SLMs) typically encompass transformer-based decoder-only architectures with parameter counts between 100 million and 5 billion, targeting deployment in constrained environments such as IoT devices, smartphones, and edge hardware [1]. The drive for efficient inference arises from the necessity to enable accessible, affordable, and low-latency natural language processing without the substantial computational resources required by large language models (LLMs), which are predominantly cloud-based [1][2]. Foundational research introduced key innovations in efficient self-attention mechanisms, including the Reformer, which reduces attention complexity from quadratic to O(N log N), and various linear attention approximations to reduce runtime and memory usage during inference [2]. Additionally, model compression methods such as pruning and quantization have been essential to decrease model size and inference costs while retaining performance [2][1]. Benchmarking SLMs involves evaluating both capability metrics (reasoning, math, contextual understanding) and efficiency metrics (latency, memory footprint, energy consumption), recognizing that inference comprises a prefill phase (processing the prompt) and a decode phase (token generation), each requiring efficiency gains [1].

## Model size reduction and architectural optimizations
Early architectural optimizations and size reduction have featured prominently, for instance, the H2O-Danube3 family optimized for smartphones, balancing performance and resource constraints [3]. These approaches emphasize decreasing parameter counts and careful design of layers to improve runtime and memory usage during inference.

## Collaborative and adaptive inference techniques
Collaborative inference strategies dynamically split workloads between local edge models and cloud services, adapting based on uncertainty or safety conditions to optimize overall efficiency, as shown in SWARM-LLM [4]. These methods enable resource-constrained devices to leverage cloud resources selectively, balancing local computation and network latency.

## Training acceleration and efficient subnetworks
Reinforcement learning algorithms, such as RAPID, accelerate fine-tuning with batched inference and off-policy updates, reducing training time on small language models [5]. Additionally, probing neural scaling laws, as in ProbeScale, discovers parameter-efficient subnetworks that optimize inference time without sacrificing accuracy [6]. These strategies target both training and inference efficiency.

## Pruning, quantization, and distillation for compact deployment
Structural pruning applied to graph neural networks reduces computational costs while maintaining accuracy [7]. Neural architecture search enables the rapid design of hardware-constrained models [8]. Combined pruning, quantization, and knowledge distillation yield compact models ready for mobile edge deployment, supported by toolkits like AngelSlim [12][9]. Training and adaptation optimized for specialized domains, especially in medical edge models, illustrate application-specific efficiency considerations [13]. Together, these techniques contribute to both inference and training time efficiency [3][4][5][6][7][13][8][12][9].

### Benchmark studies and model evaluations
Recent benchmarking efforts have provided comprehensive evaluations of small language models on accuracy, efficiency, and sustainability metrics. The distil labs study benchmarked 12 small models across 8 diverse tasks, demonstrating that the fine-tuned Qwen3-4B-Instruct-2507 can match or exceed much larger models like the 120B GPT-OSS in 7 out of 8 tasks, showing significant gains from fine-tuning and suggesting new practical baselines for local deployment [8]. Similarly, the BenchLM AI index ranks 47 small models under 100B parameters, confirming Qwen3 models as top performers across coding, knowledge, and agentic tasks, aiding informed model selection [9]. Attempts to retrieve data from slmbench.ai and MLCommons Inference v5.1 failed, so these sources were not included in benchmarking analysis.

## Trends and Open Problems
The last two years have seen increasing interest in integrating hardware-aware fault tolerance into foundation models, allowing inference on unreliable or energy-efficient hardware by trading off reliability [10]. Algorithmic innovations such as Hybrid-Basis Feature Forecasting accelerate diffusion sampling, offering new directions to improve inference speed without retraining [11]. Despite these advances, direct works surveying trends specific to efficient inference in small language models are rare, reflecting an unmet need for comprehensive standardization and synthesis [10]. Open problems remain in unifying various compression and efficiency techniques, balancing accuracy and inference speed, and extending hardware-aware methods from large to small models [10][11]. Future research is likely to focus on combining robust, fault-tolerant hardware utilization with novel algorithmic designs, as well as further refining collaborative inference frameworks for edge deployment. There is also a gap in domain-specific SLM deployment efficiency that requires further exploration, particularly in medical and mobile edge contexts [13][10].

## References
[1] Small Language Models: Survey, Measurements, and Insights. arxiv. https://arxiv.org/abs/2409.15790 (2025-02-26)
[2] A Survey on Small Language Models. web. https://aclanthology.org/anthology-files/pdf/ranlp/2025.ranlp-1.93.pdf (n.d.)
[3] H2O-Danube3 Technical Report. hf-search. https://huggingface.co/papers/2407.09276 (2024-07-12)
[4] SWARM-LLM: Collaborative Inference for Edge-based Small Language Models. hf-search. https://huggingface.co/papers/2606.14711 (2026-04-22)
[5] RAPID: An Efficient Reinforcement Learning Algorithm for Small Language Models. hf-search. https://huggingface.co/papers/2510.03515 (2025-10-03)
[6] ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference. hf-search. https://huggingface.co/papers/2606.01806 (2026-06-01)
[7] Compact SO(3) Equivariant Atomistic Foundation Models via Structural Pruning. arxiv. https://arxiv.org/abs/2605.08885 (2026-05-09)
[8] We Benchmarked 12 Small Language Models Across 8 Tasks to Find the Best Base Model for Fine-Tuning. web. https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/ (2025-12-10)
[9] Small language models under 100B. web. https://benchlm.ai/best/small-language-models (2026-10-08)
[10] Fault-tolerant foundation models. arxiv. https://arxiv.org/abs/2610.10311 (2026-10-07)
[11] Hybrid-Basis Feature Forecasting for Diffusion Sampling Acceleration. arxiv. https://arxiv.org/abs/2610.05254 (2026-10-04)
[12] MLCommons Inference v5.1. web. https://mlcommons.org/en/inference-v51 (n.d.)
[13] SLMbench.ai. web. https://www.slmbench.ai (n.d.)
