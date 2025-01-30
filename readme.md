# LSN: Language-Specific Neurons in Multilingual LLMs

## Overview
This repository contains the implementation of **Language-Specific Neuron (LSN) Analysis** in **Multilingual Large Language Models (LLMs)**. The codebase includes methods for identifying and analyzing **language neurons**, performing **test-time interventions**, and conducting **fine-tuning experiments** using techniques like **LoRA**.

## Features
- **Language-Specific Neuron Identification**  
  - Implements **Language Activation Probability Entropy (LAPE)**  
  - Implements **Activation Probability 90p (Act Prob 90p)**  
- **Test-Time Interventions**
  - Modifies neuron activations using **mean, percentile-based, and zero** interventions  
- **Fine-Tuning of Language Neurons**
  - Uses **LoRA-based fine-tuning**  
  - Compares language-specific neuron fine-tuning with random neuron fine-tuning  
- **Perplexity Change Analysis**
  - Computes the effect of interventions on language **perplexity (PPXC)**  
- **Experiments on XNLI and XQuAD**
  - Evaluates **zero-shot performance**  
  - Benchmarks on **Llama 3.1 (8B) and Mistral Nemo (12B)**  

## Installation
To set up the environment, install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Identify Language-Specific Neurons
To extract neurons specific to a language:
```bash
python code/activation.py
python code/lang_neurons.py
```

### 2. Fine-Tune Language-Specific Neurons
Perform LoRA-based fine-tuning on identified neurons:
```bash
python code/lora_cls_finetune.py
python code/lora_instruct_finetune.py
```

### 3. Test-Time Intervention
Modify activations at inference time using:
```bash
python code/xnli_eval.py
python code/xquad_eval.py
```

### 4. Compute Perplexity Change
To analyze the impact of interventions on perplexity:
```bash
python code/perplexity.py
```
