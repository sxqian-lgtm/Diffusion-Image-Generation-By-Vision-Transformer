import argparse
import json
import pathlib
from collections import Counter
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, DiTTransformer2DModel
from torch.utils.data import DataLoader, Subset

from classifier_utils import CLASS_NAMES, JsonLabelDataset, SmallCNNClassifier, split_dataset


def collect_image_paths(data_root, buckets):
    image_paths = []
    for bucket in buckets:
        bucket_path = pathlib.Path(data_root) / bucket
        if not bucket_path.exists():
            continue
        for image_path in sorted(bucket_path.glob("*.png")):
            if image_path.with_suffix(".json").exists():
                image_paths.append(image_path)
    return image_paths


def label_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    prompt = meta["inputs"].lower()
    for index, class_name in enumerate(CLASS_NAMES):
        if class_name in prompt:
            return index

    raise ValueError(f"Could not infer class label from {json_path}")


def build_subset_dataset(data_root, buckets, limit, image_size):
    dataset = JsonLabelDataset(data_root, image_size=image_size, grayscale=False)
    allowed_paths = collect_image_paths(data_root, buckets)
    path_to_index = {path: idx for idx, path in enumerate(dataset.image_paths)}
    selected_paths = allowed_paths[: min(limit, len(allowed_paths))]
    selected_indices = [path_to_index[path] for path in selected_paths]
    return dataset, Subset(dataset, selected_indices), selected_paths


def class_distribution(image_paths):
    counts = Counter()
    for image_path in image_paths:
        label = label_from_json(image_path.with_suffix(".json"))
        counts[CLASS_NAMES[label]] += 1
    return counts


def train_classifier_smoke(train_set, val_set, device, epochs, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    classifier = SmallCNNClassifier(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        train_correct = 0
        train_count = 0
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = classifier(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += images.size(0)

        classifier.eval()
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = classifier(images)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_count += images.size(0)

        print(
            f"classifier epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss / train_count:.4f} "
            f"train_acc={train_correct / train_count:.4f} "
            f"val_acc={val_correct / val_count:.4f}"
        )

    return classifier


def dit_forward(model, noisy_images, timesteps, class_labels: Optional[torch.LongTensor] = None):
    if class_labels is None:
        raise ValueError(
            "This diffusers DiT implementation requires class_labels. "
            "For unconditional training, pass a constant dummy label tensor instead."
        )
    return model(noisy_images, timesteps, class_labels=class_labels).sample


def run_diffusion_smoke(
    classifier,
    train_set,
    device,
    batch_size,
    image_size,
    unconditional=True,
    run_balance_step=True,
):
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    images, _ = next(iter(dataloader))
    images = images.to(device)
    # In diffusers 0.37.1, DiTTransformer2DModel still requires class_labels internally.
    # To model the unconditional case, we feed a constant dummy label for every sample.
    class_labels = torch.zeros(images.size(0), device=device, dtype=torch.long)

    model = DiTTransformer2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        patch_size=4,
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=32,
        num_embeds_ada_norm=100,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=100)

    model.train()
    noise = torch.randn_like(images)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (images.size(0),),
        device=device,
        dtype=torch.long,
    )
    noisy_images = scheduler.add_noise(images, noise, timesteps)
    predicted_noise = dit_forward(model, noisy_images, timesteps, class_labels=class_labels)

    loss_diff = F.mse_loss(predicted_noise, noise)
    optimizer.zero_grad()
    loss_diff.backward()
    optimizer.step()

    mode_name = "unconditional (dummy-label DiT)" if unconditional else "class-conditioned"
    print(f"{mode_name} diffusion smoke loss={loss_diff.item():.4f}")

    if run_balance_step:
        model.train()
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (images.size(0),),
            device=device,
            dtype=torch.long,
        )
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        predicted_noise = dit_forward(model, noisy_images, timesteps, class_labels=class_labels)
        loss_diff = F.mse_loss(predicted_noise, noise)

        alpha_bar = scheduler.alphas_cumprod[timesteps].to(device).view(-1, 1, 1, 1)
        x0_hat = (noisy_images - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        x0_hat = x0_hat.clamp(0, 1)

        if classifier is None:
            raise RuntimeError("Balance regularization requires a trained classifier.")

        classifier.eval()
        logits = classifier(x0_hat)
        probs = torch.softmax(logits, dim=1)
        avg_probs = probs.mean(dim=0)
        target = torch.full_like(avg_probs, 1.0 / avg_probs.numel())
        loss_balance = F.mse_loss(avg_probs, target)
        total_loss = loss_diff + 0.01 * loss_balance

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(
            f"balance regularized smoke loss_diff={loss_diff.item():.4f} "
            f"loss_balance={loss_balance.item():.4f} total={total_loss.item():.4f}"
        )

    model.eval()
    scheduler.set_timesteps(10)
    generator = torch.Generator(device=device).manual_seed(1)
    sample = torch.randn((1, 3, image_size, image_size), generator=generator, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            sample_labels = torch.zeros(1, device=device, dtype=torch.long)
            model_output = dit_forward(model, sample, t.unsqueeze(0), class_labels=sample_labels)
            sample = scheduler.step(model_output, t, sample).prev_sample

    generated = sample[0].detach().cpu()
    print(
        "generated sample stats "
        f"shape={tuple(generated.shape)} "
        f"min={generated.min().item():.4f} "
        f"max={generated.max().item():.4f} "
        f"mean={generated.mean().item():.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="synth-cute")
    parser.add_argument("--buckets", nargs="+", default=["00", "01"])
    parser.add_argument("--limit", type=int, default=220)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--classifier-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=["unconditional", "conditional"],
        default="unconditional",
        help="Whether DiT should receive class labels during the smoke test.",
    )
    parser.add_argument(
        "--skip-balance-step",
        action="store_true",
        help="Skip the classifier-based balance regularization step.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset, subset, selected_paths = build_subset_dataset(
        data_root=args.data_root,
        buckets=args.buckets,
        limit=args.limit,
        image_size=args.image_size,
    )
    print(
        f"requested_limit={args.limit} "
        f"available_in_buckets={len(collect_image_paths(args.data_root, args.buckets))} "
        f"selected={len(subset)}"
    )

    if len(subset) < 32:
        raise RuntimeError("Need at least 32 samples for the smoke test.")

    counts = class_distribution(selected_paths)
    print("class distribution:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {counts.get(class_name, 0)}")

    dataset.image_paths = selected_paths
    train_set, val_set = split_dataset(dataset, val_fraction=0.1, seed=42)
    print(f"train_samples={len(train_set)} val_samples={len(val_set)}")

    classifier = None
    if args.skip_balance_step:
        print("Skipping classifier training because --skip-balance-step was set.")
    else:
        classifier = train_classifier_smoke(
            train_set=train_set,
            val_set=val_set,
            device=device,
            epochs=args.classifier_epochs,
            batch_size=args.batch_size,
        )

    run_diffusion_smoke(
        classifier=classifier,
        train_set=train_set,
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        unconditional=args.mode == "unconditional",
        run_balance_step=not args.skip_balance_step,
    )


if __name__ == "__main__":
    main()
