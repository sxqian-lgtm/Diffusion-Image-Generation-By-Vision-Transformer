import json
import pathlib
import random

import imageio.v2 as imageio
import skimage.transform
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


CLASS_NAMES = [
    "cat",
    "chicken",
    "dog",
    "dragon",
    "fish",
    "frog",
    "gecko",
    "hamster",
    "horse",
    "monkey",
    "rabbit",
]


def get_label_from_prompt(prompt):
    prompt = prompt.lower()
    for i, name in enumerate(CLASS_NAMES):
        if name in prompt:
            return i
    raise ValueError(f"Could not find class name in prompt: {prompt}")


class JsonLabelDataset(Dataset):
    def __init__(self, data_path, image_size=64, grayscale=False):
        self.image_paths = sorted(pathlib.Path(data_path).glob("*/*.png"))
        self.image_size = image_size
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        json_path = image_path.with_suffix(".json")

        image = imageio.imread(image_path).astype("float32") / 255.0
        image = skimage.transform.resize(image, (self.image_size, self.image_size), anti_aliasing=True)

        if self.grayscale:
            image = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
            image = image[None, ...]
        else:
            image = image.transpose((2, 0, 1))

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        label = get_label_from_prompt(meta["inputs"])

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def split_dataset(dataset, val_fraction=0.1, seed=42):
    label_to_indices = {}
    for i in range(len(CLASS_NAMES)):
        label_to_indices[i] = []

    for idx, image_path in enumerate(dataset.image_paths):
        json_path = image_path.with_suffix(".json")
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        label = get_label_from_prompt(meta["inputs"])
        label_to_indices[label].append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for label in label_to_indices:
        indices = label_to_indices[label]
        rng.shuffle(indices)
        split_point = int(len(indices) * val_fraction)
        if split_point < 1:
            split_point = 1
        val_indices.extend(indices[:split_point])
        train_indices.extend(indices[split_point:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


class SmallCNNClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_classifier(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


def train_classifier(data_path, device, image_size=64, batch_size=64, num_epochs=10, grayscale=False):
    dataset = JsonLabelDataset(data_path, image_size=image_size, grayscale=grayscale)
    train_set, val_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    in_channels = 1 if grayscale else 3
    model = SmallCNNClassifier(in_channels=in_channels, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_state_dict = None
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        val_loss, val_acc = evaluate_classifier(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {}
            for key, value in model.state_dict().items():
                best_state_dict[key] = value.detach().cpu().clone()

    return model, best_state_dict
