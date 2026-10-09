# Survey on Video and Multimodal Generation

## TL;DR
- Video and multimodal generation have evolved from GANs and VAEs toward diffusion models and multimodal foundation models offering improved stability, fidelity, and scalability [1][2][3].
- In video generation, GANs produce sharp images but suffer temporal consistency issues, VAEs favor efficiency but blur, and diffusion models and transformers achieve higher quality with computational trade-offs; hybrids combine these strengths [2][4][3][5].
- Multimodal generation integrates heterogeneous pipelines managing text, audio, video, and images, with architectures leveraging hierarchical skills, style adaptation, and modality fusion strategies for coherent cross-modal outputs [6][7][8].
- Recent benchmarks like EvalCrafter and T2VSafetyBench assess video generation models on visual quality, motion, alignment, and safety, revealing strengths of models like Gen2 and challenges in camera motion and malicious content handling [9][10].
- Open problems include long video generation constraints, data efficiency, evaluation metrics aligned with human preferences, and adaptive multimodal alignment for interactive systems [11][12][13][14].

## Background
Video and multimodal generation involve producing coherent, high-quality video frames and integrating multiple modalities such as text, images, audio, and video within generative models. Early foundational approaches include variational autoencoders (VAEs) and generative adversarial networks (GANs), the latter excelling in sharp image synthesis but suffering from instability and mode collapse [1][2]. Progression toward diffusion models has brought improved stability and fidelity in video and image generation tasks [1][3]. Multimodal foundation models, combining large language models with vision and other sensory inputs, enable versatile text-to-image, text-to-video, and broader text-to-media capabilities, thus expanding the generation paradigm [1][15]. This trajectory reflects a hierarchical evolution from foundational fusion strategies and embeddings, to generative mechanisms, and finally to applications in text-to-video and creative media generation [1][16][15].

## Main Families of Video Generation Approaches
Video generation methods primarily fall into GAN-based, VAE-based, diffusion models, transformer-based, and hybrid approaches [2][4][3]. GANs provide sharp frame quality but struggle with temporal consistency, training stability, and mode collapse. VAEs offer efficient latent representations and easier training but generally produce blurrier or less detailed videos [2][4]. Diffusion models are increasingly dominant for their high visual fidelity and stable generation of temporally coherent videos, though costly in computational resources [3]. Transformers model long-range dependencies effectively and are often incorporated with latent diffusion or VAE frameworks to enhance temporal coherence and scalability [17][5][18]. Hybrid models combine advantages across families, such as latent VAE compression with diffusion denoising and transformer context modeling, to offset individual weaknesses [5][18]. This integration trend is prominent in state-of-the-art video synthesis models aiming for real-time, high-resolution, and semantically coherent video generation [5].

## Main Families of Multimodal Generation Approaches
Multimodal generative methods handle complex heterogeneous inputs like text, images, video, and audio with multi-stage pipelines coordinating cross-modal dependencies [6][7]. Architectures frequently adopt modular plug-and-play hierarchical skills or style-adaptive frameworks to accommodate modality-specific generative tasks while enabling interaction [7][19]. Training schemes largely employ general-purpose objectives derived from large-scale multimodal datasets but still grapple with capturing fine-grained physical-world semantics and style information [6][20][19]. Fusion methodologies operate either at the model level, combining multiple modalities in unified networks, or at the workflow level, orchestrating separate specialized models for each modality to produce coherent outputs [6][8]. Applications include text-to-video, native audio-visual dialogue systems, and expressive speech synthesis with style adaptation, illustrating the ongoing expansion and challenges in multimodal AI generation [19][8].

## Recent Results and Benchmarks
Benchmarking efforts from 2024 onward emphasize comprehensive and multidimensional evaluations of video generation models [9][10]. EvalCrafter introduces a large prompt set with 17 objective metrics addressing visual quality, motion, content fidelity, and text alignment, validated against human preferences. Models such as Gen2 and PikaLab score highly on visual appeal and temporal consistency but struggle with nuanced camera motions and consistent text generation across frames [9]. T2VSafetyBench targets safety considerations, benchmarking models on malicious and jailbreak prompts, highlighting trade-offs between usability and security but no model satisfies all safety metrics perfectly [10]. These benchmarks provide critical guidance for assessing advances and limitations in text-to-video generation, supporting model development focused on human-aligned quality and safety.

## Trends and Open Problems
Current research trends highlight the challenges of long video generation beyond minute scales due to complexity in planning, semantic understanding, and memory [11][12]. There is an emphasis on data efficiency and scalable training frameworks like MUG-V 10B to facilitate large video model training [14]. Evaluation remains a key open problem, with recent benchmarks advancing hierarchical, human-aligned metrics but gaps persist in fully replicating human judgment [13][21]. Multimodal alignment and adaptive interfaces are emerging directions to enhance interactive generative applications spanning video, audio, and text [22]. Future research aims to improve cross-modal semantic coherence, temporal dynamics in extended sequences, and safety, with human-centered evaluation frameworks guiding responsible deployment [11][22].

## References
[1] Generative AI for multimodal content: a survey with empirical and experimental evaluations. web. https://link.springer.com/article/10.1007/s10462-026-11525-6 (2026-03-19)
[2] Image Generation Models: A Technical History. arxiv. https://arxiv.org/abs/2603.07455 (2026-03-08)
[3] A Survey on Video Diffusion Models. hf-search. https://huggingface.co/papers/2310.10647 (2023-10-16)
[4] Bridging Text and Video Generation: A Survey. arxiv. https://arxiv.org/abs/2510.04999 (2025-10-06)
[5] LTX-Video: Realtime Video Latent Diffusion. hf-search. https://huggingface.co/papers/2501.00103 (2024-12-30)
[6] vLLM-Omni Technical Report: A Unified Serving Runtime for Omni-Modality Generation. arxiv. https://arxiv.org/abs/2610.09307 (2026-10-07)
[7] Omni-IO Skills: Harnessing Your Agent Omni-Native. arxiv. https://arxiv.org/abs/2609.31847 (2026-09-25)
[8] OmniVChat: Synthesizing, Benchmarking, and Training for Native Audio-Visual Dialogue. arxiv. https://arxiv.org/abs/2609.21465 (2026-09-18)
[9] EvalCrafter: Benchmarking and Evaluating Large Video Generation Models. web. https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_EvalCrafter_Benchmarking_and_Evaluating_Large_Video_Generation_Models_CVPR_2024_paper.pdf (2024)
[10] T2VSafetyBench: Evaluating the safety of text-to-video generative models. web. http://dl.acm.org/doi/10.5555/3737916.3739955 (2024-12-10)
[11] A Survey of Generative Categories and Techniques in Multimodal Generative Models. arxiv. https://arxiv.org/abs/2506.10016 (2025-05-29)
[12] Video Is Worth a Thousand Images: Exploring the Latest Trends in Long Video Generation. arxiv. https://arxiv.org/abs/2412.18688 (2024-12-24)
[13] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2411.13503 (2024-11-20)
[14] MUG-V 10B: High-efficiency Training Pipeline for Large Video Generation Models. hf-search. https://huggingface.co/papers/2510.17519 (2025-10-20)
[15] Next-Gen AIGC: a review of multimodal foundation models for text-to-media innovations. web. https://link.springer.com/article/10.1007/s11704-025-51171-9 (2026-02-20)
[16] Text-to-video generators: a comprehensive survey. web. https://link.springer.com/article/10.1186/s40537-025-01314-3 (2025-11-14)
[17] VACE: All-in-One Video Creation and Editing. hf-search. https://huggingface.co/papers/2503.07598 (2025-03-10)
[18] CV-VAE: A Compatible Video VAE for Latent Generative Video Models. hf-search. https://huggingface.co/papers/2405.20279 (2024-05-30)
[19] Interactive TTS: Dynamic Speaking Style Adaptation for Expressive Speech Synthesis. arxiv. https://arxiv.org/abs/2609.25707 (2026-09-22)
[20] OmniFysics-Nano-V2 Technical Report: Understanding the Physical World Across Modalities. arxiv. https://arxiv.org/abs/2609.25738 (2026-09-22)
[21] EvalCrafter: Benchmarking and Evaluating Large Video Generation Models. hf-search. https://huggingface.co/papers/2310.11440 (2023-10-17)
[22] Generative AI in Multimodal User Interfaces: Trends, Challenges, and Cross-Platform Adaptability. arxiv. https://arxiv.org/abs/2411.10234 (2024-11-15)
