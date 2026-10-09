# Survey on World Models

## TL;DR
- World models are internal representations used to understand and predict the dynamics of the external world, enabling intelligent planning, robotics, and decision-making tasks [1][2][3].
- The main families of world model approaches include latent variable models (e.g., RSSM-based PlaNet and Dreamer), neural simulators (transformer and diffusion-based), and specialized application models such as for embodied AI and autonomous driving, each with trade-offs in interpretability, fidelity, and utility for planning [4][5][6].
- Recent advances have emphasized closed-loop approaches that adapt dynamically to observed states during long-horizon task execution, exemplified by models like WorldGuide, which close the gap between decision-making and procedural execution [7].
- Open challenges include the unclear definitions of world models, difficulties with long-horizon planning and causality, action conditioning, sim-to-real transfer, and achieving physical consistency beyond visual realism [8][2].
- Promising future directions involve integrating multiple model families, fostering embodied interactive environments, and enhancing models to support practical decision-making in complex, dynamic real-world settings [1][8].

## Background
World models are a class of internal representations that allow AI systems to represent, understand, and predict the dynamics of the external world relevant for planning and control. This concept has roots dating back to Marvin Minsky's frame representation in the 1960s and the psychological theory of mental models, which suggest that intelligent agents create internal simulations to navigate their environments [2].

Early foundational work in the modern era was popularized by Ha and Schmidhuber's 2018 neural-network latent world models that learned implicit representations conducive to model-based reinforcement learning (MBRL) [2]. These approaches have evolved to define capability tiers such as one-step predictors, multi-step simulators, and evolving models that adapt autonomously [1].

Recent architectures emphasize physical and digital world laws and accommodate applications across robotics, scientific discovery, and social simulation, progressively targeting interactive real-time control on consumer hardware [1][3][9]. 

## Main Families of World Model Approaches
Latent variable models such as PlaNet and Dreamer employ recurrent state-space models (RSSM) with stochastic latent representations that capture uncertainty and temporal dynamics concisely. These are particularly effective for reinforcement learning and planning tasks, balancing compactness and uncertainty management but often sacrifice interpretability [4][5].

Neural simulators consist of autoregressive transformers and diffusion-based generators that produce rich video or token-based predictions. These models enhance fidelity and realism but can struggle with maintaining physical consistency and long-term stability compared to classical simulators [6].

Specialized domain-focused models target embodied AI, autonomous driving, and other contexts where environmental complexity and partial observation necessitate customized architectures. The diversity of world model representations includes latent vectors, video frames, 3D scenes, and symbolic structures, reflecting different trade-offs between visual realism, planning utility, and generalization [10][5].

Further improvements have been noted in planning through latent subgoal conditioning enabling more effective long-horizon action generation within latent spaces [11]. This line of work aims to bridge the gap between compact latent dynamics and explicit planning strategies.

## Recent Results and Benchmarks
Recent empirical work has focused on enhancing world models for complex procedural task execution, particularly the move towards closed-loop formulations where the model dynamically adapts to state changes rather than relying on open-loop generation or pretrained executors [7].

WorldGuide (2026) exemplifies this shift, showcasing a video-based world model capable of goal-directed procedural task execution that recognizes task completion based on closed-loop feedback [7]. 

## Trends and Open Problems
There is a dual perspective in world modeling focused either on understanding current world states or predicting future ones, with applications spanning autonomous driving, robotics, and social simulation [2].

Key open challenges include the lack of clear definitions for world models, limitations in long-horizon reasoning and causal insight, difficulties in action conditioning, and barriers in transferring models from simulation to real world environments [8].

Models often conflate the goals of visual fidelity (e.g., video models) with the deeper need for predictive and causal accuracy, leading to ongoing research to reconcile these aspects [2][8].

Future directions emphasize developing dynamic, interactive embodied environments where agents can learn and act effectively, and integrating diverse modeling approaches to produce world models that support practical decision-making beyond purely visual simulation [1][8].

## References
[1] Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond. web. https://arxiv.org/html/2604.22748v1 (2026-04-24)
[2] Understanding World or Predicting Future? A Comprehensive Survey of World Models | ACM Computing Surveys. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[3] DreamForge-World 0.1 Preview: A Low-Compute Real-Time Controllable World Model. hf-search. https://huggingface.co/papers/2606.30292 (2026-06-29)
[4] World Models: A Comprehensive Survey of .... web. https://arxiv.org/abs/2606.00133 (2026-06-01)
[5] World Models Survey | Comprehensive Overview. web. https://world-models.io/en/research/world-models-survey/ (n.d.)
[6] From Generation to Simulation: How Far Are World Models from Being True Simulators?. hf-search. https://huggingface.co/papers/2608.23070 (2026-08-24)
[7] WorldGuide: Goal-Directed Video World Model for Procedural Task Execution. hf-daily. https://huggingface.co/papers/2610.12459 (2026-10-08)
[8] State of World Models 2026 | Landscape Report. web. https://world-models.io/reports/state-of-world-models-2026/ (n.d.)
[9] InternW0: A Foundational Physical World Model for Efficient Real-World Interactions. hf-search. https://huggingface.co/papers/2609.27656 (2026-09-23)
[10] State of World Models 2026: Taxonomy, Benchmarks and Open Challenges. web. https://world-models.io/reports/state-of-world-models-2026/state-of-world-models-2026-v1.0.pdf (n.d.)
[11] SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning. hf-search. https://huggingface.co/papers/2607.17973 (2026-07-20)
