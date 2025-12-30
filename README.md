# 🔢 Improving Visual Numerosity in Multimodal Models (LLaVA)

## 🔍 Key Engineering Problem

The Multimodal Large Language Models (MMLLMs) struggle with counting number of items in images (visual enumeration). In this repo, a systematic tuning approach is presented to tackle this problem. One of key aims is to achieve an improvement that is robust to different object catgories and different backgrounds. I show that more numerosity aware vision transformer representations result in not only better and more robust counting performance but also result in better performance in binary comparison task. To see the full results and the methods, please check the the excrepts from my thesis in this repo. 

<p align="center">
  <img src="Readme Figures/llava_architecture.png" width="600">
</p>


## 📊 Datasets

3 datasets are used in this work. Some of the samples from these datasets are shown below.
<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="Readme Figures/ccnl1_dataset_illustration.png" width="100%"><br>
      <sub><b>(a)</b> CCNL-1 dataset illustration</sub>
    </td>
    <td align="center" width="33%">
      <img src="Readme Figures/ccnl2_obj_num_pairs.png" width="100%"><br>
      <sub><b>(b)</b> Object–number pair distribution (CCNL-2)</sub>
    </td>
    <td align="center" width="33%">
      <img src="Readme Figures/clevr_dataset_illustration.png" width="100%"><br>
      <sub><b>(c)</b> CLEVR dataset illustration</sub>
    </td>
  </tr>
</table>

## 📌 Organization of this Repo

This repo is organized as follows:

📂 ViT Zoo: The tuned adapters for tuned vision transformers.

📂 trained_adapters_for_llava: The trained PEFT adapters for the entire LLaVA pipeline.

📂 Modules: The helper functions and the class definitions for the training and inference code.

In addition to these folders, there is training code for the LLaVA pipeline, the training code for the ViT, and Jupyter notebooks where the performance of LLaVA is demonstrated for different datasets and different training settings.





## 🔧 Fine-Tuning Approaches

The finetuning scenarios were considered.

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
  

## 📉 Improvements Demo

LLaVA has some baseline counting (ennumeration) capabilities but the counting accuracy remains comparatively low especially for images containing more than 5 objects.
<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="Readme Figures/cm_all_CCNL1.png" width="100%"><br>
      <sub><b>(a)</b> Counting accuracy on CCNL-2 (Baseline LLaVA)</sub>
    </td>
    <td align="center" width="50%">
      <img src="Readme Figures/cm_all.png" width="100%"><br>
      <sub><b>(b)</b>Counting accuracy on CCNL-2 (Tuned LLaVA)</sub>
    </td>
  </tr>
</table>



## 🙏 Acknowledgements

Many thanks to the members of CCNL at University of Padova for their supervision and guidance during this work.
