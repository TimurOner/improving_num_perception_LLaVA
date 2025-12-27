# 🔢 Improving Visual Numerosity in Multimodal Models (LLaVA)

## 🔍 Key Engineering Problem

The Multimodal Large Language Models (MMLLMs) 
![Results](Readme%20Figures/llava_architecture.png)
## 📊 Datasets

![Results](Readme%20Figures/ccnl1_dataset_illustration.png)
![Results](Readme%20Figures/ccnl2_obj_num_pairs.png)
![Results](Readme%20Figures/clevr_dataset_illustration.png)


## 📉 Baseline Results (Untuned LLaVA)
LLaVA has some baseline counting (ennumeration) capabilities but the counting accuracy remains comparatively low especially for images containing more than 5 objects.


## 🔧 Fine-Tuning Approaches

The finteuning scenarious were considered.

- **Multimodal Projector:** First, the multimodal projector was fine-tuned with LoRA updates.
- **Vision Transformer (ViT):** After observing limited improvements and limited robustness to distribution shift, the vision transformer was fine-tuned on CLEVR.
- **Projector + ViT:** The entire pipeline with the tuned transformer was trained again, resulting in improved enumeration accuracy and robustness to distribution shifts.
- **Language Model Ablations:** The effect of language model tuning was tested via ablations.
- **Learning Rate Ablations:** The effect of different learning rates when tuning the LLaVA pipeline was studied.


## 🔄 Generalization & Transfer Experiments

In this work, I have conducted experiments to see how the performance improvement in enumeration transfers. The following transfer scenarious were considered:

- Cross-category transfer
- Even → Odd numerosity transfer
- Cross Dataset Transfer (CCNL2 to CCNL1)
  
## 🧩 Representation Analysis
- Few-shot linear probing
- PCA embedding analysis

## 📐 Hyperparameter Analysis

## 🔁 Transfer to Downstream Numerical Tasks

## 📌 Key Improvements and Discussion

## ⚠️ Limitations and Future Work


## 📄 Thesis Reference


## 🙏 Acknowledgements
