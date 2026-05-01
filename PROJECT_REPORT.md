# CuddleDiT Final Project Report

## 1. Project Summary

This project studies diffusion-based image generation on a synthetic cute-animal dataset. The goal is to generate 64x64 cartoon-style animal images that are both visually plausible and reasonably diverse across semantic classes.

The final system combines:

- a prompt-derived label extraction pipeline
- a compact CNN classifier trained on those labels
- a DiT-style denoising model trained in a notebook workflow
- periodic checkpointing and resumption
- fixed-seed sample export for qualitative comparison
- a deterministic fixed-seed export workflow for qualitative comparison

The project began as a course assignment but gradually became a more complete experimental pipeline for training, continuing, comparing, and exporting models.

## 2. Problem Setting

The dataset consists of synthetic images generated from prompts. Each image is paired with JSON metadata containing the original prompt text. Rather than relying on separate label annotations, the project derives labels directly from those prompts.

The 11 target classes are:

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

This setup introduces two goals at once:

1. image realism and visual coherence
2. broad category coverage rather than collapsing to a small subset of animals

## 3. Method

### 3.1 Prompt-to-label supervision

The first step is to read the sibling JSON metadata for each image and detect which animal class is present in the prompt text. This makes it possible to train a simple classifier without additional manual annotation.

### 3.2 Auxiliary classifier

The classifier is implemented in `classifier_utils.py` as `SmallCNNClassifier`. It is trained on 64x64 RGB images and used as a compact semantic recognizer.

Its role in the final project is mainly analytical:

- validating that the prompt-derived labels are learnable
- providing class probabilities for evaluating generated samples
- supporting lightweight semantic inspection of generated outputs

The trained classifier checkpoint is saved as `classifier_state_dict.pt`.

### 3.3 Diffusion model

The generative model is built in `project4.ipynb` with Hugging Face Diffusers:

- `DiTTransformer2DModel`
- `DDPMScheduler`

The final notebook configuration uses:

- image size: 64x64
- RGB inputs
- patch size: 4
- transformer layers: 8
- attention heads: 8
- attention head dimension: 64
- diffusion timesteps: 800

The training target is standard denoising MSE between predicted noise and sampled Gaussian noise.

## 4. Training Workflow

The final workflow was not a single perfectly linear from-scratch run. Instead, it evolved into a staged notebook-based process with checkpoint continuation.

The typical pattern was:

1. configure the model and scheduler
2. train for a block of epochs
3. preview fixed-seed samples every few epochs
4. save the current model checkpoint
5. later resume from a selected checkpoint and continue training

This approach made the project more practical to manage, especially when comparing mid-training quality. The notebook was updated so that preview epochs also trigger checkpoint saving, making later comparison easier.

Examples of saved checkpoints include:

- `dit_model1_epoch152.pt`
- `dit_model1_epoch168.pt`
- `dit_model1_epoch192.pt`

The notebook also exports deterministic sample sets from chosen checkpoints into `generated_fixed/`.

## 5. Evaluation Strategy

### 5.1 Qualitative evaluation

The most immediate feedback came from fixed-seed previews. Looking at the same seeds across multiple checkpoints made it easier to observe:

- when animal structure became recognizable
- when failure cases became less frequent
- whether the model improved in detail and stability over time

This was especially useful because diffusion loss alone did not fully capture sample quality.

### 5.2 Real dataset path on SCC

For SCC evaluation, the real dataset path used in practice is:

```text
/projectnb/dl4ds/materials/datasets/synth-cute
```

This is important because the evaluation notebook needs access both to the trained classifier and to the real images when computing feature-based comparisons.

## 6. Project Outcomes

By the end of the project, the repository supports the full loop:

- train classifier
- train diffusion model
- resume training from checkpoints
- generate fixed-seed image sets
- review outputs visually
- inspect generated outputs through fixed seeds and exported samples

This is a meaningful improvement over the initial prototype, which was focused mainly on getting diffusion training to run at all.

## 7. Strengths

- The codebase now preserves an end-to-end working workflow rather than isolated experiments.
- Prompt metadata is used effectively to add semantic structure to the project.
- Checkpoint saving and continuation make long notebook training more manageable.
- Fixed-seed exports create a reproducible qualitative comparison protocol.

## 8. Limitations

- The main training code remains notebook-centric rather than being refactored into a dedicated CLI training script.
- The project workflow reflects real experimentation, including resumed runs, rather than a single polished reproducible script.
- Large model files are intentionally kept outside version control for practical reasons.

## 9. Conclusion

CuddleDiT ended as a solid course-scale generative modeling project. It demonstrates not only a working DiT-based diffusion model, but also a practical research workflow: semantic preprocessing, auxiliary analysis tools, staged training, and deterministic sample export.

The final repository is suitable for submission because it now documents both the modeling idea and the actual process used to train, compare, and present the results.
