


from llava_finetune_funs_new import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"



config_path = '/home/timur_oner/train_config_new.json'
config = load_config(config_path)


learning_rate = config['learning_rate']

total_epochs = config['total_epochs']
train_dataset_name = config['train_dataset_name']
val_dataset_name = config['val_dataset_name']
test_dataset_name = config['test_dataset_name']

results_save_path = config['preds_save_path']
train_nums = config['train_nums']
val_nums = config['val_nums']
test_nums = config['test_nums']
train_objs = config['train_objs']
val_objs = config['val_objs']
test_objs = config['test_objs']
num_augment_train = config['num_augment_train']
num_augment_factor = config['num_augment_factor']
extra_basic_aug = config['extra_basic_aug']


r = config['r']
lora_alpha = config['lora_alpha']
target_modules = config['target_modules']
lora_dropout = config['lora_dropout']
bias = config['bias']
task_type = config['task_type']
batch_size = config['batch_size']
early_stop_tolerance = config['early_stop_tolerance']
early_stop_trig = config['early_stop_trig']


adapter_save_path = config['adapter_save_path'] 
prompt_inference = config['prompt_inference']
custom_vit_path = config['custom_vit_path']
print("Prompt Inference:", config['prompt_inference'])

if train_dataset_name == 'CCNL1':
 train_dataset_path = "/home/timur_oner/datasets/CCNL"
elif train_dataset_name == 'CCNL2':
 train_dataset_path = "/home/timur_oner/datasets/CCNL2" 
elif train_dataset_name == 'CLEVR':
 train_dataset_path = "/home/timur_oner/datasets/CLEVR" 
elif train_dataset_name  == 'CHLDBOOK':
 raise ValueError("CHLDBOOK dataset is not implemented yet.")

if val_dataset_name == 'CCNL1':
 val_dataset_path =  "/home/timur_oner/datasets/CCNL"
elif val_dataset_name == 'CCNL2':
 val_dataset_path =  "/home/timur_oner/datasets/CCNL2"
elif val_dataset_name == 'CLEVR':
 val_dataset_path = "/home/timur_oner/datasets/CLEVR"
elif val_dataset_name == 'CHLDBOOK':
 raise ValueError("CHLDBOOK dataset is not implemented yet.")

if test_dataset_name == 'CCNL1':
 test_dataset_path =  "/home/timur_oner/datasets/CCNL"
elif test_dataset_name == 'CCNL2':
 test_dataset_path = "/home/timur_oner/datasets/CCNL2"
elif test_dataset_name == 'CLEVR':
 test_dataset_path = "/home/timur_oner/datasets/CLEVR"
elif test_dataset_name == 'CHLDBOOK':
 raise ValueError("CHLDBOOK dataset is not implemented yet.")
 
#  Model name, tokenizers and image processor
model_name = "llava-hf/llava-1.5-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
image_processor = AutoProcessor.from_pretrained(model_name).image_processor


# Load base model with the quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


llava_processor, base_model =  get_inference_ready_model(model_name,None,custom_vit_path,quantization_config)
device = base_model.device

# Loading PEFT paramters and integrating them with the base model

lora_config = LoraConfig(
    r=r,
    lora_alpha=lora_alpha,
    target_modules=target_modules,
    lora_dropout=lora_dropout,
    bias=bias,
    task_type=task_type,
)

base_model_copy = copy.deepcopy(base_model)
base_model_copy = prepare_model_for_kbit_training(base_model_copy)
peft_model = get_peft_model(base_model_copy, lora_config) 

# Loading the data

import random
from torch.utils.data import Subset

train_dataset = ImageTextDataset(
    image_base_path=train_dataset_path,
    dataset=train_dataset_name,            # or 'CLEVR'
    processor=llava_processor,
    nums=train_nums,
    categories=train_objs,
    undersamp_ratio=1,
   sample_idx_range=range(0,40)
)

if num_augment_train == 'diff':
  train_dataset = ExpandedDifferenceDataset(train_dataset,num_augment_factor-1,extra_basic_aug)

val_dataset = ImageTextDataset(
    image_base_path=val_dataset_path,
    dataset=val_dataset_name,            # or 'CLEVR'
    processor=llava_processor,
    nums=val_nums,
    categories=val_objs,
    undersamp_ratio=1,
     sample_idx_range=range(40,50)
)



test_dataset = ImageTextDataset(
    image_base_path=test_dataset_path,
    dataset=test_dataset_name,            
    processor=llava_processor,
    nums=test_nums,
    categories=test_objs,
   test_mode=True,
   undersamp_ratio=0.1
)



train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,       # adjust batch size as needed
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate_fn
)

def print_gpu_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # in GB
    free = reserved - allocated
    print(f"{prefix} GPU memory — Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB, Free: {free:.3f} GB")

# Before clearing cache
print_gpu_memory("Before clearing cache:")

# Clear cache
torch.cuda.empty_cache()

# After clearing cache
print_gpu_memory("After clearing cache:")

# Sanity check before training
# sanity_check_llava_model(peft_model,train_dataset,llava_processor,device)
print('Sanity Checking test dataset...')
sanity_check_dataset(test_dataset,peft_model,tokenizer,device)

# Print dataloader info
print_dataloader_info(train_loader, "Train")
print_dataloader_info(val_loader, "Validation")
print_dataloader_info(test_loader, "Test")



# The optimizer and the training

optimizer = torch.optim.Adam([param for param in peft_model.parameters() if param.requires_grad],lr=learning_rate,
                             betas=(0.9, 0.98), eps=1e-5, weight_decay=1e-4)
trained_model, train_losses, val_losses = train_model(
    model=peft_model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    device=device,
    epochs=total_epochs,                      # or any number you prefer
    early_stop_tolerance=early_stop_tolerance,
    early_stop_perc_thr=early_stop_trig,
    use_mixed_precision=True,


)


# Saving the model dicts
hardcoded_numbers_of_interest = [7,8,9]
# preds_untuned, preds_tuned,  labels  = get_predictions_batched(base_model, trained_model, tokenizer, test_loader, device, 0.1 ,hardcoded_numbers_of_interest)
# brief_analyze_preds(preds_untuned, preds_tuned, labels)
# save_preds_and_labels(results_save_path, preds_untuned, preds_tuned,preds_untuned,preds_tuned, labels)
save_losses(results_save_path, train_losses, val_losses)
trained_model.save_pretrained(adapter_save_path)
