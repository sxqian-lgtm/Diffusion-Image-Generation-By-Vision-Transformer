# CuddleDiT

## Project Overview

**CuddleDiT** is a diffusion-based image generation project for synthesizing 64x64 cute animal images with a Vision Transformer backbone. The project began as a course assignment on generative modeling and evolved into a compact research prototype that explores whether class-balance feedback can improve diversity across generated animal categories.

The repository combines three main components:

1. A notebook-based training workflow in `project4.ipynb`
2. A lightweight image classifier trained from prompt-derived labels in `classifier_utils.py`
3. A smoke-test pipeline in `smoke_test_pipeline.py` for quickly validating the end-to-end training setup

## Problem Setting

The underlying dataset contains synthetic cute animal images paired with JSON metadata that includes the original text prompt. The assignment objective is to train a generative model that produces visually plausible and diverse samples at 64x64 resolution.

The main challenge is not only sample quality, but also coverage across the 11 semantic classes:

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

## Method

### 1. Label Extraction from Prompt Metadata

Each image has a sibling JSON file containing the prompt used to create it. The project infers class labels by scanning the prompt text for one of the known animal names. This provides supervision without requiring a separate annotation file.

### 2. Auxiliary Classifier

`classifier_utils.py` defines:

- `JsonLabelDataset` to load image and JSON pairs
- `split_dataset` to build a label-aware train/validation split
- `SmallCNNClassifier` as a compact classifier for the 11 animal categories
- `train_classifier` to fit the classifier and retain the best validation checkpoint

This classifier is used for two purposes:

- checking whether prompt-derived categories are learnable from the images
- providing a class-distribution signal during diffusion training

### 3. Diffusion Backbone

The generative model in `project4.ipynb` uses:

- `DiTTransformer2DModel` from Hugging Face Diffusers
- `DDPMScheduler` for forward and reverse diffusion
- image resolution of 64x64 with RGB inputs

The notebook configures a compact DiT-style architecture with:

- patch size 4
- 8 transformer layers
- 8 attention heads
- attention head dimension 64
- 1000 diffusion timesteps

### 4. Two Training Objectives

The notebook implements and compares two diffusion training strategies:

#### Baseline Diffusion Loss

The first version trains the model with the standard denoising objective:

- sample Gaussian noise
- perturb clean images at random timesteps
- predict the injected noise
- minimize MSE between predicted and true noise

#### Class-Balance-Regularized Diffusion Loss

The second version adds an auxiliary balance term:

- reconstruct an `x0` estimate from the noisy image and predicted noise
- run the classifier on the reconstructed estimate
- average class probabilities over the batch
- compare the mean distribution to a uniform target
- add the penalty to the diffusion loss with a small coefficient

This encourages the generator to avoid collapsing toward only a few categories and is the key experimental contribution of the project.

## Repository Contents

- `project4.ipynb`: main notebook with dataset setup, classifier training, diffusion training, and sample generation
- `classifier_utils.py`: reusable dataset and classifier helpers
- `smoke_test_pipeline.py`: small-scale end-to-end pipeline for fast verification
- `requirements.txt`: Python dependencies
- `synth-cute/`: expected local dataset directory for development and smoke testing, not versioned in this repository

## Smoke Test Workflow

The smoke test script exists to answer a practical question early: does the pipeline work before committing to a long GPU run?

It performs the following steps:

1. Load a subset of local images and prompt metadata
2. Report class counts
3. Train the small CNN classifier for a short run
4. Run one baseline diffusion optimization step
5. Run one balance-regularized diffusion optimization step
6. Generate a short sample and print summary statistics

This makes the repository easier to reproduce and debug on limited hardware.

## Implementation Notes

### Strengths

- Uses prompt metadata to create labels without extra annotation effort
- Separates reusable classifier utilities from notebook experimentation
- Includes a small pipeline test instead of relying only on a large training notebook
- Experiments with a meaningful diversity-oriented regularizer rather than only vanilla diffusion training

### Current Limitations

- The main training flow is notebook-centric rather than packaged as a CLI training script
- Evaluation metrics such as FID or Inception Score are not automated in the repository
- Generated image outputs are not stored in version control
- The dataset is expected to be cloned or mounted locally rather than stored directly in this repository

## Suggested Next Steps

- move notebook training code into a standalone `train.py`
- add checkpoint saving and loading for the DiT model
- save preview grids during training
- log quantitative metrics over epochs
- compare baseline and class-balance training with a consistent evaluation protocol
- add a final results gallery once the model is trained on the full dataset

## Conclusion

CuddleDiT is a compact but thoughtful diffusion project that goes beyond a basic course implementation. Its main idea is to combine a DiT-style diffusion model with classifier-based class-balance regularization so that the generated samples are not only sharp, but also better distributed across animal categories. Even in its current form, the repository demonstrates clear modeling intent, modular helper code, and a reproducible validation path through the smoke test pipeline.
