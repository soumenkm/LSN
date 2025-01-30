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

## Repository Structure
\\```
LSN/
│── code/                    # Main codebase for LSN experiments
│   ├── models/              # Model definitions and loading scripts
│   ├── data/                # Scripts for dataset processing (XNLI, XQuAD, Wikipedia)
│   ├── experiments/         # Scripts for running experiments (test-time intervention, fine-tuning)
│   ├── utils/               # Utility functions for neuron analysis, visualization, and evaluation
│   ├── config/              # Configuration files for model training and evaluation
│   ├── scripts/             # Bash scripts for launching experiments
│   └── notebooks/           # Jupyter notebooks for analysis and visualization
│── results/                 # Stores output results, logs, and figures
│── figures/                 # Contains plots and visualization of neuron activations
│── README.md                # Project documentation (this file)
│── requirements.txt         # Dependencies for setting up the environment
│── setup.py                 # Installation script for the package
\\```

## Installation
To set up the environment, install dependencies using:
\\```bash
pip install -r requirements.txt
\\```
or use a virtual environment:
\\```bash
python -m venv lsn_env
source lsn_env/bin/activate  # On Linux/Mac
lsn_env\\Scripts\\activate     # On Windows
pip install -r requirements.txt
\\```

## Usage

### 1. Identify Language-Specific Neurons
To extract neurons specific to a language:
\\```bash
python code/experiments/extract_language_neurons.py --method lape --output_dir results/lape_neurons
\\```
For **Activation Probability 90p**:
\\```bash
python code/experiments/extract_language_neurons.py --method act_prob_90p --output_dir results/act_prob_neurons
\\```

### 2. Test-Time Intervention
Modify activations at inference time using:
\\```bash
python code/experiments/test_time_intervention.py --method lape --intervention mean
\\```
Supported intervention methods: `mean`, `P90`, `P10`, `zero`

### 3. Fine-Tune Language-Specific Neurons
Perform LoRA-based fine-tuning on identified neurons:
\\```bash
python code/experiments/fine_tune_language_neurons.py --model llama3.1 --task xnli --method lape
\\```

### 4. Compute Perplexity Change
To analyze the impact of interventions on perplexity:
\\```bash
python code/experiments/perplexity_analysis.py --model mistral --task xquad
\\```

## Results
- The results of our experiments show that **test-time interventions do not significantly improve task performance**.
- **Fine-tuning language neurons** using LoRA also **fails to enhance zero-shot transfer performance**.
- **Perplexity changes reveal that interventions affect language modeling** but do not always correlate with task performance.

For detailed numerical results, refer to the **tables in the paper and appendix**.

## Citation
If you find this work useful, please cite:
\\```bibtex
@article{yourcitation2025,
  author    = {Your Name and Co-Authors},
  title     = {Investigating Language-Specific Neurons in Multilingual LLMs},
  journal   = {Conference/Journal Name},
  year      = {2025}
}
\\```

## License
This project is licensed under the MIT License.

## Acknowledgments
We thank the open-source community for their contributions to multilingual language modeling research.
