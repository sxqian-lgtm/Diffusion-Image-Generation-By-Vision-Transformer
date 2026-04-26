# CuddleDiT

Class-balanced cute animal image generation with a diffusion transformer.

## Project Name

**CuddleDiT** is the polished project name for this repository.

Full title:
**CuddleDiT: Class-Balanced Cute Animal Generation with Diffusion Transformers**

If you later want to rename the GitHub repository itself, this is the name I would recommend using.

## Overview

This project builds a 64x64 image generator for cute animal images using a diffusion model with a Vision Transformer style backbone. The implementation started from a course generative modeling assignment and extends it with an auxiliary classifier that encourages better category coverage across the generated samples.

The repository focuses on three goals:

- train a compact classifier from prompt-derived labels
- train a DiT-based diffusion model on cute animal images
- explore a class-balance regularization term to improve diversity

## Core Idea

Each training image is paired with a JSON file containing the prompt used to synthesize it. The code extracts one of 11 animal labels from that prompt and uses those labels to train a small CNN classifier. During diffusion training, the classifier provides an additional signal that nudges the generator toward a more uniform class distribution across a batch.

This creates two training variants:

- a baseline diffusion objective
- a class-balance-regularized diffusion objective

## Repository Structure

```text
.
|-- classifier_utils.py
|-- project4.ipynb
|-- PROJECT_REPORT.md
|-- requirements.txt
|-- smoke_test_pipeline.py
`-- .gitignore
```

## Files

- `project4.ipynb`: main notebook for training and sample generation
- `classifier_utils.py`: dataset loading, label extraction, classifier definition, and classifier training helpers
- `smoke_test_pipeline.py`: quick end-to-end pipeline check on a reduced local subset
- `PROJECT_REPORT.md`: polished project summary and technical report
- `requirements.txt`: Python dependencies
- local dataset clone is expected at `synth-cute/` but is not committed to this repository

## Dataset

The original assignment dataset contains synthetic cute animal images across 11 classes:

- cat
- chicken
- dog
- dragon
- fish
- frog
- gecko
- hamster
- horse
- monkey
- rabbit

This repository does not version the dataset itself. For local development, place the dataset in a folder named `synth-cute/`.

## Method

### 1. Prompt-to-label supervision

The project reads each image's JSON metadata and infers the class label directly from the prompt text.

### 2. Auxiliary classifier

A small CNN classifier is trained on the labeled images to predict the animal category.

### 3. Diffusion transformer

The main generator uses Hugging Face Diffusers with:

- `DiTTransformer2DModel`
- `DDPMScheduler`
- RGB images at `64x64`

### 4. Diversity-aware regularization

In the second training variant, reconstructed samples are passed through the classifier. The average predicted class distribution over the batch is then compared against a uniform target distribution. This penalty is added to the diffusion loss to encourage broader category coverage.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the smoke test

If you do not already have the dataset locally, clone it first:

```bash
git clone https://github.com/dl4ds/synth-cute
```

Then run:

```bash
python smoke_test_pipeline.py --data-root synth-cute --buckets 00 01 --limit 175 --classifier-epochs 1 --batch-size 16
```

The smoke test:

- loads the local subset
- reports class distribution
- trains the classifier briefly
- runs one baseline diffusion update
- runs one class-balance diffusion update
- generates a sample and prints summary statistics

### 3. Open the main notebook

```bash
jupyter notebook project4.ipynb
```

Use the notebook for the full training workflow, preview generation, and fixed-seed image export.

## Development Notes

- The notebook is currently the primary training entry point.
- The repository does not yet include automated FID or Inception Score evaluation.
- Generated samples and checkpoints are intentionally ignored in version control.

## Recommended GitHub Presentation

For GitHub, this repository now presents best as:

- a compact research-style generative modeling project
- a notebook-backed diffusion experiment with reusable Python utilities
- an exploration of classifier-guided diversity control

## Report

The full project write-up is available here:

- [PROJECT_REPORT.md](./PROJECT_REPORT.md)

## Tech Stack

- Python
- PyTorch
- Hugging Face Diffusers
- Transformers
- Jupyter
- Matplotlib
- scikit-image

## Future Improvements

- move notebook training into a dedicated script
- save checkpoints and preview grids automatically
- add quantitative evaluation metrics
- compare baseline and regularized models with reproducible experiments
- publish a gallery of final generated samples
