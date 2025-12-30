

# Built-in libraries
import os
import json
import copy
import random,time
import re
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from functools import partial
from pathlib import Path

# Scientific computing
import math
import numpy as np
import pandas as pd
from scipy.stats import zscore, linregress
from scipy.spatial import ConvexHull

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torch.cuda import amp

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.pyplot import Normalize

# Machine Learning / Dimensionality Reduction
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

#LR Scheduling
from torch.optim.lr_scheduler import CosineAnnealingLR


# Transformers / Hugging Face
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPImageProcessor,
    BitsAndBytesConfig,
    AutoModelForVision2Seq,
    pipeline,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    LlavaForConditionalGeneration,
    LlavaProcessor
)

# PEFT Imports
from peft import (
    PeftModel, 
    PeftConfig, 
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
)
# HuggingFace Datasets
from datasets import Dataset as HFDataset, load_dataset

# UMAP
import umap

# Image processing
from PIL import Image

# SafeTensors
from safetensors.torch import load_file, save_file

# Accelerate for multi-GPU training
from accelerate import Accelerator

# Weights & Biases for experiment tracking
import wandb

# Additional utilities
import contextlib
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load and unpack configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Unpack configuration
    learning_rate = config['learning_rate']
    total_epochs = config['total_epochs']
    
    # Dataset names
    train_dataset_name = config['train_dataset_name']
    val_dataset_name = config['val_dataset_name']
    test_dataset_name = config['test_dataset_name']
    
    # Save paths
    results_save_path = config['preds_save_path']
    adapter_save_path = config['adapter_save_path']
    prompt_inference = config['prompt_inference']
    
    # Data splits
    train_nums = config['train_nums']
    val_nums = config['val_nums']
    test_nums = config['test_nums']
    
    # Object categories
    train_objs = config['train_objs']
    val_objs = config['val_objs']
    test_objs = config['test_objs']

    # Data Augmentation
    num_augment_train =config['num_augment_train']
    num_augment_factor = config['num_augment_factor']
    extra_basic_aug = config['extra_basic_aug']
    

    # LoRA parameters
    r = config['r']
    lora_alpha = config['lora_alpha']
    target_modules = config['target_modules']
    lora_dropout = config['lora_dropout']
    bias = config['bias']
    task_type = config['task_type']
    
    # Training parameters
    batch_size = config['batch_size']
    early_stop_tolerance = config['early_stop_tolerance']
    early_stop_trig = config['early_stop_trig']

    custom_vit_path = config['custom_vit_path']
    
    # Print summary
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 DATASET & TRAINING:")
    print(f"  Train Dataset: {train_dataset_name}")
    print(f"  Val Dataset: {val_dataset_name}")
    print(f"  Test Dataset: {test_dataset_name}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Total Epochs: {total_epochs}")
    print(f"  Batch Size: {batch_size}")
    
    print(f"\n💾 SAVE PATHS:")
    print(f"  Predictions Save Path: {results_save_path}")
    print(f"  Adapter Save Path: {adapter_save_path}")
    print(f"  Prompt Inference: {prompt_inference}")
    
    print(f"\n🔢 DATA SPLITS:")
    print(f"  Train Numbers: {train_nums} (Total: {len(train_nums)})")
    print(f"  Val Numbers: {val_nums} (Total: {len(val_nums)})")
    print(f"  Test Numbers: {test_nums} (Total: {len(test_nums)})")
    
    print(f"\n🧩 DATA AUGMENTATION:")
    print(f"  Augmentation Enabled: {num_augment_train}")
    print(f"  Augmentation Factor: {num_augment_factor}")
    print(f"  Extra Basic Aug: {extra_basic_aug}")


    print(f"\n🎯 OBJECT CATEGORIES:")
    print(f"  Train Objects: {train_objs} (Total: {len(train_objs)})")
    print(f"  Val Objects: {val_objs} (Total: {len(val_objs)})")
    print(f"  Test Objects: {test_objs} (Total: {len(test_objs)})")
    
    print(f"\n🔧 LORA CONFIGURATION:")
    print(f"  Rank (r): {r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Dropout: {lora_dropout}")
    print(f"  Bias: {bias}")
    print(f"  Task Type: {task_type}")
    print(f"  Target Modules: {', '.join(target_modules)}")
    
    print(f"\n⏹️ EARLY STOPPING:")
    print(f"  Tolerance: {early_stop_tolerance} epochs")
    print(f"  Threshold: {early_stop_trig}")
    
    print("=" * 60)
    
    return config


def get_inference_ready_model(model_name, adapter_path, vit_adapt_path, quantization_config, 
                               load_from_local=False, local_model_path=None, no_quantize=False):
  
    model_name = model_name or "llava-hf/llava-1.5-7b-hf"

    if load_from_local and local_model_path:
        load_source = local_model_path
    else:
        load_source = model_name

    processor = AutoProcessor.from_pretrained(model_name)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"
    tmp_dir = Path("/home/timur_oner/tmp_llava")
    tmp_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist

    if vit_adapt_path is not None:
        baseline_model = LlavaForConditionalGeneration.from_pretrained(
            load_source,
            device_map="auto",
            torch_dtype=torch.float16
        )
        baseline_vit = baseline_model.vision_tower.vision_model
        _, pefted_vit = load_peft_vit_classifier(
            baseline_vit, vit_adapt_path, 
            freeze_vit=True, device='cuda', load_class_head=False
        )

        baseline_vit = pefted_vit.merge_and_unload()
        baseline_model.vision_tower.vision_model = baseline_vit

        if no_quantize == False:
                print("Quantizing merged model...")
                baseline_model.save_pretrained(tmp_dir)
                del baseline_model
                torch.cuda.empty_cache()
                baseline_model = LlavaForConditionalGeneration.from_pretrained(
                    tmp_dir,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
    else:
        if no_quantize == False:
            baseline_model = LlavaForConditionalGeneration.from_pretrained(
                load_source,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            baseline_model = LlavaForConditionalGeneration.from_pretrained(
                load_source,
                device_map="auto",
                torch_dtype=torch.float16
            )
    
    if adapter_path is not None:
        print(f"Loading language model adapter from {adapter_path}")
        baseline_model = PeftModel.from_pretrained(baseline_model, adapter_path)
        if no_quantize == False:
            baseline_model = baseline_model.merge_and_unload()
    
    return processor, baseline_model


def load_peft_vit_classifier(base_vit, save_path, freeze_vit=False, device='cuda', load_class_head=True):
    peft_model_path = save_path
    base_vit_fresh = copy.deepcopy(base_vit)
    peft_model_loaded = PeftModel.from_pretrained(base_vit_fresh, peft_model_path)
    return base_vit_fresh,peft_model_loaded 

def sanity_check_llava_model(model, train_dataset, processor, device):
    ex_prompt = (
        "USER: <image>\n"
        "Count the total number of distinct objects in the photo. "
        "Answer only with the final count as a numeral.\n"
        "ASSISTANT:"
    )
    
    n = len(train_dataset)
    rand_id = random.randint(0, n - 1)
    rand_sample = train_dataset[rand_id]
   
    image_path = rand_sample[5]
    image_PIL = Image.open(image_path).convert("RGB")
    true_num = rand_sample[4]
    
    inputs = processor(text=ex_prompt, images=image_PIL, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )
    
    decoded = processor.decode(output[0], skip_special_tokens=True)
    
    print(f"Image: {image_path}")
    print(f"True count: {true_num}")
    print(f"Model output: {decoded}")
  

# This is the standard ImageTextDataset


import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageTextDataset(Dataset):
    def __init__(
        self,
        image_base_path,
        dataset,
        processor,
        nums,
        categories,
        test_mode=False,
        undersamp_ratio=1.0,
        sample_idx_range=None  # <-- NEW ARGUMENT
    ):
        self.base_path = image_base_path
        self.processor = processor
        self.dataset = dataset
        self.test_mode = test_mode
        self.sample_idx_range = sample_idx_range  # store the range

        self.base_prompt = (
            "USER: <image>\nHow many objects are there in the image? "
            "Respond only with the number.\nASSISTANT:"
        )

        self.image_paths = []
        self.labels_int = []

        all_img_files = os.listdir(image_base_path)
        img_path_list = []
        labels_list = []

        for img_file in all_img_files:
            true_num = self.__get_true_num(img_file)
            cat_name = self.__get_category(img_file)

            # Usual dataset filtering
            keep = False
            if self.dataset == 'CLEVR':
                keep = (true_num is not None and true_num in nums)

            elif self.dataset in ['CCNL1', 'CCNL2']:
                keep = (true_num is not None and true_num in nums and cat_name in categories)

            if not keep:
                continue

            # ----- NEW SAMPLE IDX FILTERING -----
            if self.dataset == "CCNL2" and self.sample_idx_range is not None:
                sample_idx = self.__get_sample_idx(img_file)
                if sample_idx not in self.sample_idx_range:
                    continue
            # -----------------------------------

            img_path_list.append(os.path.join(image_base_path, img_file))
            labels_list.append(true_num)

        # Optional undersampling
        if undersamp_ratio < 1.0 and img_path_list:
            combined = list(zip(img_path_list, labels_list))
            random.shuffle(combined)
            keep_len = max(1, int(len(combined) * undersamp_ratio))
            combined = combined[:keep_len]
            img_path_list, labels_list = zip(*combined)
            img_path_list, labels_list = list(img_path_list), list(labels_list)

        self.image_paths = img_path_list
        self.labels_int = labels_list

    def __len__(self):
        return len(self.image_paths)

    def __get_true_num(self, img_name):
        parts = img_name.split('_')
        if self.dataset == 'CHLDBOOK':
            return int(parts[1])
        elif self.dataset in ['CCNL1', 'CCNL2', 'CLEVR']:
            for part in parts:
                if part.isdigit() and int(part) < 11:
                    return int(part)
        return None

    def __get_category(self, img_name):
        parts = img_name.split('_')
        if parts and self.dataset in ['CCNL1', 'CCNL2']:
            return parts[1] if len(parts) > 1 else ""
        return ""

    # FIXED: actually use img_name
    def __get_sample_idx(self, img_name):
        if self.dataset == 'CCNL2':
            parts = img_name.split('_')
            return int(parts[-1].split('.')[0]) % 50
        return None

    def get_item_metadata(self, idx):
        return self.labels_int[idx], self.image_paths[idx]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        original_numerical_label = self.labels_int[idx]
        full_text_training = f"{self.base_prompt} {original_numerical_label}"
        full_text_inference = f"{self.base_prompt} "

        if not self.test_mode:
            processed = self.processor(
                images=image, text=full_text_training, return_tensors="pt"
            )
            input_ids = processed.input_ids.squeeze(0)
            attention_mask = processed.attention_mask.squeeze(0)
            pixel_values = processed.pixel_values.squeeze(0)

            # mask prompt part
            prompt_only = self.processor(
                images=image, text=full_text_inference, return_tensors="pt"
            )
            mask_len = prompt_only.input_ids.squeeze(0).shape[0]

            labels = input_ids.clone()
            labels[:mask_len] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        else:
            processed = self.processor(
                images=image, text=full_text_inference, return_tensors="pt"
            )
            input_ids = processed.input_ids.squeeze(0)
            attention_mask = processed.attention_mask.squeeze(0)
            pixel_values = processed.pixel_values.squeeze(0)
            labels = -100 * torch.ones_like(input_ids)

        return pixel_values, input_ids, attention_mask, labels, original_numerical_label, image_path


class ExpandedDifferenceDataset(Dataset):
    def __init__(self, dataset_instance, diff_nums_per_image=2, use_basic_augmentation=False):
        self.dataset = dataset_instance
        self.diff_nums_per_image = diff_nums_per_image
        self.use_basic_augmentation = use_basic_augmentation

        # Define basic augmentation if enabled
        if self.use_basic_augmentation:
            self.augmentation = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augmentation = None

        self.prompt_template_more = (
            "USER: <image>\n"
            "How many more objects are there in the image than %d?\n"
            "ASSISTANT:"
        )
        self.prompt_template_fewer = (
            "USER: <image>\n"
            "How many fewer objects are there in the image than %d?\n"
            "ASSISTANT:"
        )

        self.samples_per_image = 1 + self.diff_nums_per_image
        self.effective_len = len(self.dataset) * self.samples_per_image

        print(f"[ExpandedDifferenceDataset] Each image generates {self.samples_per_image} samples")
        print(f"Effective dataset length: {self.effective_len}")

    def __len__(self):
        return self.effective_len

    def __getitem__(self, idx):
        image_idx = idx // self.samples_per_image
        variation_idx = idx % self.samples_per_image

        original_label, image_path = self.dataset.get_item_metadata(image_idx)
        image = Image.open(image_path).convert("RGB")

        # Apply basic augmentation if enabled
        if self.use_basic_augmentation and not self.dataset.test_mode:
            image = self.augmentation(image)

        if variation_idx == 0:
            # Original count prompt
            prompt_text = f"{self.dataset.base_prompt} {original_label}"
            answer = original_label
        else:
            # Bidirectional difference prompt
            if random.random() < 0.5:
                compare_number = random.randint(1, original_label)
                prompt_text = self.prompt_template_more % compare_number
                answer = max(original_label - compare_number, 0)
            else:
                compare_number = random.randint(original_label, 11)
                prompt_text = self.prompt_template_fewer % compare_number
                answer = max(compare_number - original_label, 0)

            prompt_text = f"{prompt_text} {answer}"

        # Process image + text
        processed = self.dataset.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        )
        input_ids = processed.input_ids.squeeze(0)
        attention_mask = processed.attention_mask.squeeze(0)
        pixel_values = processed.pixel_values.squeeze(0)

        # Mask prompt tokens
        prompt_only_text = prompt_text.rsplit(" ", 1)[0]
        prompt_only_processed = self.dataset.processor(
            images=image,
            text=prompt_only_text,
            return_tensors="pt"
        )
        mask_len = prompt_only_processed.input_ids.squeeze(0).shape[0]

        labels = input_ids.clone()
        labels[:mask_len] = -100
        labels[labels == self.dataset.processor.tokenizer.pad_token_id] = -100

        # Return consistent tuple
        meta = {
            "image_path": image_path,
            "original_label": original_label,
            "variation_idx": variation_idx
        }
        if variation_idx != 0:
            meta["compare_number"] = compare_number
            meta["difference_answer"] = answer

        return pixel_values, input_ids, attention_mask, labels, answer, image_path

def custom_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])  # shape: (B, C, H, W)
    
    input_ids = [item[1] for item in batch]
    attention_masks = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    
    # Assumes tokenizer.pad_token_id == 0, which is typical
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    numerical_labels = torch.tensor([item[4] for item in batch], dtype=torch.long)
    image_paths = [item[5] for item in batch]

    return (
       pixel_values,
       input_ids_padded,
       attention_mask_padded,
       labels_padded,
       numerical_labels,
       image_paths)
def analyze_peft_model(peft_model):
    """Analyze PEFT model parameters and display adapter information"""
    
    total_params = 0
    trainable_params = 0
    adapter_modules = []
    
    print("=" * 70)
    print("PEFT MODEL ANALYSIS")
    print("=" * 70)
    
    # Count parameters and identify adapter modules
    for name, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # Extract module name for adapters
            if "lora_" in name:
                module_base = name.split(".lora_")[0]
                if module_base not in adapter_modules:
                    adapter_modules.append(module_base)
    
    # Calculate percentage
    trainable_percentage = (trainable_params / total_params) * 100
    
    print(f"\n📊 PARAMETER STATISTICS:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Trainable Percentage: {trainable_percentage:.4f}%")
    print(f"  Memory Reduction: {100 - trainable_percentage:.2f}%")
    
    print(f"\n🔧 PEFT ADAPTER MODULES:")
    print(f"  Total Adapter Modules: {len(adapter_modules)}")
    for i, module in enumerate(sorted(adapter_modules), 1):
        print(f"  {i:2d}. {module}")
    
    # Additional PEFT info
    if hasattr(peft_model, 'peft_config'):
        config = peft_model.peft_config
        if config:
            peft_type = list(config.values())[0].peft_type if config else "Unknown"
            print(f"\n⚙️  PEFT CONFIGURATION:")
            print(f"  PEFT Type: {peft_type}")
            if hasattr(list(config.values())[0], 'r'):
                print(f"  LoRA Rank (r): {list(config.values())[0].r}")
            if hasattr(list(config.values())[0], 'lora_alpha'):
                print(f"  LoRA Alpha: {list(config.values())[0].lora_alpha}")
    
    print("=" * 70)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_percentage': trainable_percentage,
        'adapter_modules': adapter_modules
    }


def print_dataset_info(dataset, name):
    print(f"\n{name} Dataset Info:")
    print(f"  Number of samples: {len(dataset)}")

    # Count numerosities and categories from dataset attributes
    from collections import Counter
    numerosity_counts = Counter(dataset.labels_int)
    print(f"  Numerosity distribution: {dict(sorted(numerosity_counts.items()))}")

    # Collect categories from image filenames
    categories = [dataset._ImageTextDataset__get_category(os.path.basename(p)) for p in dataset.image_paths]
    category_counts = Counter(categories)
    print(f"  Category distribution: {dict(sorted(category_counts.items()))}")

def print_dataloader_info(dataloader, name):
    print(f"\n{name} DataLoader Info:")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Number of workers: {dataloader.num_workers}")
    print(f"  Pin memory: {dataloader.pin_memory}")




def extract_number(response):
    """Extract integer from model response"""
    numbers = re.findall(r'\d+', response.split("ASSISTANT:")[-1] if "ASSISTANT:" in response else response)
    return int(numbers[-1]) if numbers else 0


def sanity_check_dataset(dataset, model, tokenizer, device):
    import random
    
    n = len(dataset)
    rand_id = random.randint(0, n - 1)
    
    pixel_values, input_ids, attention_mask, labels, original_numerical_label, image_path = dataset[rand_id]
    
    print(f"\n=== Dataset Sanity Check ===")
    print(f"Sample index: {rand_id}/{n}")
    print(f"Image path: {image_path}")
    print(f"Original numerical label: {original_numerical_label}")
    
    print(f"\nPixel values shape: {pixel_values.shape}")
    print(f"Pixel values: min={pixel_values.min():.3f}, max={pixel_values.max():.3f}, mean={pixel_values.mean():.3f}")
    
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[:30]}")
    print(f"Decoded input: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
    
    print(f"\nAttention mask shape: {attention_mask.shape}")
    print(f"Attention mask sum: {attention_mask.sum()}")
    
    print(f"\nLabels shape: {labels.shape}")
    print(f"Labels (non -100): {labels[labels != -100]}")
    print(f"Decoded labels: {tokenizer.decode(labels[labels != -100], skip_special_tokens=True)}")
    
    print(f"\n=== Testing with model ===")
    model.eval()
    with torch.no_grad():
        pixel_values_batch = pixel_values.unsqueeze(0).to(device)
        input_ids_batch = input_ids.unsqueeze(0).to(device)
        attention_mask_batch = attention_mask.unsqueeze(0).to(device)
        
        output = model.generate(
            pixel_values=pixel_values_batch,
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            max_new_tokens=5,
            do_sample=False
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Model output: {decoded_output}")

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    device,
    epochs=10,
    early_stop_tolerance=3,
    early_stop_perc_thr=0.01,
    use_mixed_precision=False
):
    train_loss_log = []
    val_loss_log = []
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    hardcoded_numbers_of_interest = [1,2,7,8,9]
    model_initial  =  copy.deepcopy(model)
    scaler = amp.GradScaler() if use_mixed_precision else None

    # Fixed: Single scheduler for all parameter groups
    initial_lr = optimizer.param_groups[0]['lr']  # Use first group's LR as reference
    min_lr = initial_lr / 10
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_lr
    )
     

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for pixel_values, input_ids, attn, labels, *_ in pbar:
                pixel_values = pixel_values.to(device)
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if use_mixed_precision:
                    with amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attn,
                            pixel_values=pixel_values,
                            labels=labels,
                        )
                        loss = outputs.loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        # Step scheduler once per epoch
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_loss_log.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pixel_values, input_ids, attn, labels, *_ in val_loader:
                pixel_values = pixel_values.to(device)
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)

                context = amp.autocast() if use_mixed_precision else contextlib.nullcontext()
                with context:
                    val_loss += model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        pixel_values=pixel_values,
                        labels=labels,
                    ).loss.item()

        val_loss /= len(val_loader)
        val_loss_log.append(val_loss)



        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        # Improved early stopping logic
        if val_loss < best_loss:
            # Calculate relative improvement
            if best_loss != float('inf'):
                improvement = (best_loss - val_loss)
            else:
                improvement = float('inf')  # First epoch always counts as improvement

            if improvement >= early_stop_perc_thr or best_loss == float('inf'):
                best_loss = val_loss
                epochs_without_improvement = 0
                print(f"→ New best validation loss: {val_loss:.4f} (improvement: {improvement:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"→ Early stopping counter: {epochs_without_improvement}/{early_stop_tolerance} (improvement: {improvement:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"→ Early stopping counter: {epochs_without_improvement}/{early_stop_tolerance} (no improvement)")

        if epochs_without_improvement >= early_stop_tolerance:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    return model, train_loss_log, val_loss_log





def get_predictions_batched(
    model_untuned, 
    model_tuned, 
    tokenizer, 
    test_dataloader, 
    device, 
    sample_rate=0.1, 
    numbers_of_interest=[7, 8, 9]
):
    model_untuned.eval()
    model_tuned.eval()
    
    preds_untuned = []
    preds_tuned = []
    labels = []
    start_time_total = time.time()
    
    target_tokens = [29955, 29947, 29929]  # token IDs corresponding to your numbers
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            pixel_values = batch[0].to(device)
            input_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            numerical_labels = batch[4].to(device)
            original_length = input_ids.shape[1]

            # --- Untuned model ---
            outputs_untuned = model_untuned.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                return_dict_in_generate=True,
                output_scores=True
            )
            sequences_untuned = outputs_untuned.sequences
            new_tokens_untuned = sequences_untuned[:, original_length:]
            decoded_untuned = tokenizer.batch_decode(new_tokens_untuned, skip_special_tokens=True)

            # --- Tuned model ---
            outputs_tuned = model_tuned.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                return_dict_in_generate=True,
                output_scores=True
            )
            sequences_tuned = outputs_tuned.sequences
            new_tokens_tuned = sequences_tuned[:, original_length:]
            decoded_tuned = tokenizer.batch_decode(new_tokens_tuned, skip_special_tokens=True)

            # --- Optional: inspect probabilities of target tokens (first generation step) ---
            probs_tuned = F.softmax(outputs_tuned.scores[1], dim=-1)
            probs_untuned = F.softmax(outputs_untuned.scores[1], dim=-1)
            _probability_hook(
                probs_tuned[:, target_tokens],
                probs_untuned[:, target_tokens],
                numerical_labels,
                sample_rate=sample_rate,
                numbers_of_interest=numbers_of_interest
            )

            # --- Extract numbers from decoded text (like manual inference) ---
            preds_untuned.extend([extract_number(seq.strip()) for seq in decoded_untuned])
            preds_tuned.extend([extract_number(seq.strip()) for seq in decoded_tuned])
            labels.extend(numerical_labels.tolist())
    
    total_time = time.time() - start_time_total
    print(f"Total inference time: {total_time:.2f}s")
    
    return preds_untuned, preds_tuned, labels


def _probability_hook(tuned_probs_batch, untuned_probs_batch, ground_truth_batch, numbers_of_interest, sample_rate=0.1):
    batch_size = tuned_probs_batch.shape[0]
    num_samples_to_print = max(1, int(batch_size * sample_rate))
    sample_indices = random.sample(range(batch_size), num_samples_to_print)
    
    for sample_idx in sample_indices:
        tuned_probs = tuned_probs_batch[sample_idx]
        untuned_probs = untuned_probs_batch[sample_idx]
        ground_truth = ground_truth_batch[sample_idx]

        print("\n--- Ground Truth: %d ---" % ground_truth)

        print("Tuned probs:")
        for idx, i in enumerate(numbers_of_interest):
            prob = tuned_probs[idx].item()
            print("  Class %d: %.4f" % (i, prob))

        print("Untuned probs:")
        for idx, i in enumerate(numbers_of_interest):
            prob = untuned_probs[idx].item()
            print("  Class %d: %.4f" % (i, prob))

        print("Difference (tuned - untuned):")
        for idx, i in enumerate(numbers_of_interest):
            diff = (tuned_probs[idx] - untuned_probs[idx]).item()
            print("  Class %d: %+0.4f" % (i, diff))


def check_rmse_filtered(model_untuned, model_tuned, test_dataloader, device, numbers_of_interest):
    model_untuned.eval()
    model_tuned.eval()
    
    num2tok = {
        1: 29896, 2: 29906, 3: 29941, 4: 29946, 5: 29945,
        6: 29953, 7: 29955, 8: 29947, 9: 29929
    }
    target_tokens = [num2tok[n] for n in numbers_of_interest if n in num2tok]
    
    per_sample_squared_errors = []
    start_time_total = time.time()
    
    with torch.no_grad():
        for batch in test_dataloader:
            pixel_values = batch[0].to(device)
            input_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            gt_labels = batch[4].to(device)

            outputs_untuned = model_untuned.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2,
                return_dict_in_generate=True,
                output_scores=True
            )

            outputs_tuned = model_tuned.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2,
                return_dict_in_generate=True,
                output_scores=True
            )

            probs_tuned = F.softmax(outputs_tuned.scores[1], dim=-1)
            probs_untuned = F.softmax(outputs_untuned.scores[1], dim=-1)
            tuned_subset = probs_tuned[:, target_tokens]
            untuned_subset = probs_untuned[:, target_tokens]
            _probability_hook(
                tuned_subset, 
                untuned_subset, 
               gt_labels,   
               numbers_of_interest,
               0.01
            )

            mse_per_sample = (tuned_subset - untuned_subset).pow(2).mean(dim=1)
            per_sample_squared_errors.append(mse_per_sample)
    
    per_sample_squared_errors = torch.cat(per_sample_squared_errors, dim=0)
    overall_rmse = torch.sqrt(per_sample_squared_errors.mean()).item()
    
    total_time = time.time() - start_time_total
    print(f"Total inference time: {total_time:.2f}s | RMSE for numbers {numbers_of_interest}: {overall_rmse:.4f}")
    model_untuned.train()
    model_tuned.train()

    return overall_rmse



def load_quantized_models(model_untuned_path, model_tuned_path):
    from transformers import BitsAndBytesConfig, AutoModelForVision2Seq
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print("Loading untuned model with 4-bit quantization...")
    model_untuned = AutoModelForVision2Seq.from_pretrained(
        model_untuned_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("Loading tuned model with 4-bit quantization...")
    model_tuned = AutoModelForVision2Seq.from_pretrained(
        model_tuned_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return model_untuned, model_tuned


def brief_analyze_preds(preds_untuned, preds_tuned, labels):
   
    assert len(preds_untuned) == len(preds_tuned) == len(labels), "Mismatched lengths"

    total = len(labels)
    correct_untuned = 0
    correct_tuned = 0

    per_label_stats = defaultdict(lambda: {
        "total": 0,
        "tuned_correct": 0,
        "untuned_correct": 0,
        "tuned_errors": [],
        "untuned_errors": []
    })

    for pu, pt, gt in zip(preds_untuned, preds_tuned, labels):
        per_label_stats[gt]["total"] += 1
        
        error_untuned = abs(pu - gt)
        error_tuned = abs(pt - gt)
        
        per_label_stats[gt]["untuned_errors"].append(error_untuned)
        per_label_stats[gt]["tuned_errors"].append(error_tuned)
        
        if pu == gt:
            correct_untuned += 1
            per_label_stats[gt]["untuned_correct"] += 1
        if pt == gt:
            correct_tuned += 1
            per_label_stats[gt]["tuned_correct"] += 1

    print(f"Total Accuracy (Untuned): {correct_untuned / total:.2%}")
    print(f"Total Accuracy (Tuned):   {correct_tuned / total:.2%}")
    print("\nPer-label Accuracy and MAE:")
    
    for label in sorted(per_label_stats):
        stats = per_label_stats[label]
        total_label = stats["total"]
        tuned_acc = stats["tuned_correct"] / total_label
        untuned_acc = stats["untuned_correct"] / total_label
        tuned_mae = np.mean(stats["tuned_errors"])
        untuned_mae = np.mean(stats["untuned_errors"])
        print(f"Label {label}: Tuned Acc={tuned_acc:.2%} | Untuned Acc={untuned_acc:.2%} | "
              f"Tuned MAE={tuned_mae:.3f} | Untuned MAE={untuned_mae:.3f} | Count={total_label}")


def save_preds_and_labels(path, preds_untuned, preds_tuned, scores_untuned, scores_tuned, labels):
    os.makedirs(path, exist_ok=True)  # create dir if not exists

    # Open all files at once
    with open(os.path.join(path, "untuned_preds.txt"), "w") as f_untuned_preds, \
         open(os.path.join(path, "tuned_preds.txt"), "w") as f_tuned_preds, \
         open(os.path.join(path, "untuned_scores.txt"), "w") as f_untuned_scores, \
         open(os.path.join(path, "tuned_scores.txt"), "w") as f_tuned_scores, \
         open(os.path.join(path, "labels.txt"), "w") as f_labels:

        for pred_u, pred_t, score_u, score_t, label in zip(preds_untuned, preds_tuned, scores_untuned, scores_tuned, labels):
            f_untuned_preds.write(f"{pred_u}\n")
            f_tuned_preds.write(f"{pred_t}\n")
            f_untuned_scores.write(f"{score_u}\n")
            f_tuned_scores.write(f"{score_t}\n")
            f_labels.write(f"{label}\n")

def save_losses(path, train_loss, val_loss):
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, "train_loss.txt"), "w") as f_train, \
         open(os.path.join(path, "val_loss.txt"), "w") as f_val:
        
        for t, v in zip(train_loss, val_loss):
            f_train.write(f"{t}\n")
            f_val.write(f"{v}\n")
