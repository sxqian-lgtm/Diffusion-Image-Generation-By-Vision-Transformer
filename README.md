# CuddleDiT

Diffusion-based cute animal image generation with a compact DiT backbone, prompt-derived labels, checkpointed notebook training, and lightweight evaluation utilities.

## Overview

This repository contains the final working state of a course project on image generation. The goal is to train a 64x64 diffusion model on the `synth-cute` dataset and generate diverse, visually plausible cartoon animal images.

The project evolved from a basic notebook experiment into a more complete workflow with:

- prompt-to-label supervision for 11 animal classes
- a small auxiliary classifier trained from the dataset metadata
- DiT-style diffusion training in Jupyter
- periodic checkpoint saving during training
- fixed-seed sample export for qualitative review
- deterministic fixed-seed sample export for qualitative review

## Dataset

The project uses the `synth-cute` dataset from the course materials. Each image has a sibling JSON file containing the prompt used to generate it. Those prompts are used to infer one of 11 labels:

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

The dataset is not versioned in this repository. On SCC, the real path used during evaluation is:

```text
/projectnb/dl4ds/materials/datasets/synth-cute
```

## Main Components

### 1. `project4.ipynb`

This is the primary training notebook. It contains:

- dataset loading and visualization
- classifier training
- DiT model setup
- diffusion training with preview generation
- checkpoint saving every few epochs
- fixed-seed export for final image sets

Current training behavior includes:

- RGB images at 64x64
- `DiTTransformer2DModel` backbone
- `DDPMScheduler`
- 800 diffusion timesteps
- checkpoint names such as `dit_model1_epoch168.pt`

### 2. `classifier_utils.py`

Reusable classifier-side utilities:

- `JsonLabelDataset`
- prompt-to-label parsing
- train/validation splitting
- `SmallCNNClassifier`
- classifier training helpers

The classifier serves two purposes:

- providing a compact category recognizer for analysis
- checking whether prompt-derived categories are learnable from the image set

### 3. `view_generated_fixed.ipynb`

A small utility notebook for quickly viewing all `fixed-*.png` images in `generated_fixed/`.

## Training Workflow

The project was trained in stages rather than a single uninterrupted run. In practice, the workflow is:

1. Train the classifier and save `classifier_state_dict.pt`
2. Train the diffusion model in `project4.ipynb`
3. Save epoch checkpoints periodically
4. Resume from later checkpoints when needed
5. Export fixed-seed samples for manual review
6. Inspect generated fixed-seed outputs and select representative examples

Because of this staged process, the notebook execution history may not match the exact chronological order of every experiment, but the repository preserves the full working pipeline.

## Fixed-Seed Outputs

The repository includes a `generated_fixed/` folder for deterministic qualitative inspection. These images are useful for:

- comparing model quality across checkpoints
- visually tracking progression over training
- selecting favorite outputs for reporting

The project also includes a small set of `favorite-*.png` images at the repository root for quick showcase examples.

## How To Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Open the training notebook

```bash
jupyter notebook project4.ipynb
```

## Repository Contents

```text
.
|-- classifier_utils.py
|-- generated_fixed/
|-- project4.ipynb
|-- view_generated_fixed.ipynb
|-- .gitignore
`-- classifier_state_dict.pt
```

Checkpoint `.pt` files are intentionally not described as part of the permanent source tree, even though local working copies may exist during experiments.

## Limitations

- The main training flow is notebook-centric rather than packaged as a CLI training script.
- The final project relied on staged checkpoint continuation rather than one completely clean end-to-end run.
- Large training artifacts are kept local and are not intended to be committed in full.

## Final Status

At this stage, the repository represents a working end-to-end generative modeling project:

- classifier training works
- diffusion training works
- checkpoint saving works
- fixed-seed sample generation works

The codebase is now in a good state for submission, review, and presentation.
