# CT-to-MRI Cross-Modality Synthesis with Multi-Condition Stable Diffusion
This repository implements a multi-condition diffusion-based generative model for cross-modality medical imaging.
The system integrates CT image latents and text prompts to guide MRI synthesis, enabling reconstruction and translation between imaging modalities.
The approach leverages Stable Diffusion to augment datasets and improve MRI analysis via CT-informed generation.

🧠 Problem Statement

MRI scans provide detailed soft tissue contrast, but acquiring them is costly, time-consuming, and sometimes not feasible for certain patients. CT scans, on the other hand, are more accessible but lack the same tissue detail.
This project aims to bridge the gap by using CT scans as input to synthesize corresponding MRI images, improving dataset diversity and enabling cross-modality analysis without requiring additional patient scans.

🛠 Features

✅ Multi-condition Stable Diffusion with CT latents + text prompts

✅ Latent-space VAE encoding for efficient medical image processing

✅ Dataset augmentation for downstream MRI analysis

✅ Cross-modality reconstruction & translation

✅ Training & inference pipeline for paired CT–MRI datasets

✅ Evaluation metrics: SSIM, PSNR, visual inspection

📂 Dataset
The dataset used in this project is from the SynthRAD2023 Challenge
"A Benchmark for Cross-Modality Medical Image Translation" ((https://arxiv.org/abs/2303.16320?utm_source=chatgpt.com)).
It includes paired CT and MRI brain images for supervised training and evaluation of generative models.

📄 License & Usage
Please follow the dataset license and usage guidelines provided by the SynthRAD2023 Challenge organizers.
Model code and pipeline are for research purposes only.



