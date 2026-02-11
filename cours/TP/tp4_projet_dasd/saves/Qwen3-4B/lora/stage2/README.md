---
library_name: peft
license: other
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: stage2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stage2

This model is a fine-tuned version of [unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit) on the train_stage2 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6073

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 4e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 2

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.10.0+cu130
- Datasets 3.6.0
- Tokenizers 0.21.1