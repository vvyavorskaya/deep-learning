import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import random
from tqdm import tqdm
import os
from scripts.dataset import create_dataloaders

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultimodalFoodCaloriesModel(nn.Module):
    def __init__(self, num_ingredients, pretrained=True):
        super(MultimodalFoodCaloriesModel, self).__init__()
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        self.image_feature_dim = 1280

        tabular_input_dim = num_ingredients + 1

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.tabular_feature_dim = 128
        fusion_input_dim = self.image_feature_dim + self.tabular_feature_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, image, ingredients, mass):
        image_features = self.image_encoder(image)
        image_features = image_features.flatten(1)

        mass = mass.unsqueeze(1) if mass.dim() == 1 else mass
        tabular_input = torch.cat([ingredients, mass], dim=1)
        tabular_features = self.tabular_encoder(tabular_input)

        combined = torch.cat([image_features, tabular_features], dim=1)
        fused_features = self.fusion(combined)

        predictions = self.regressor(fused_features)
        return predictions.squeeze(1)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')

    for batch in pbar:
        images = batch['image'].to(device)
        ingredients = batch['ingredients'].to(device)
        mass = batch['mass'].to(device)
        targets = batch['calories'].to(device)

        optimizer.zero_grad()
        predictions = model(images, ingredients, mass)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.2f}'})

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')

        for batch in pbar:
            images = batch['image'].to(device)
            ingredients = batch['ingredients'].to(device)
            mass = batch['mass'].to(device)
            targets = batch['calories'].to(device)

            predictions = model(images, ingredients, mass)
            loss = criterion(predictions, targets)
            mae = torch.abs(predictions - targets).mean()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            num_samples += batch_size

            pbar.set_postfix({
                'loss': f'{loss.item():.2f}',
                'mae': f'{mae.item():.2f}'
            })

    return total_loss / num_samples, total_mae / num_samples


def train():
    data_dish_csv = "data/dish.csv"
    data_ingredients_csv = "data/ingredients.csv"
    data_img_dir = "data/images"

    pretrained = True
    save_path = "model_best.pth"

    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    weight_decay = 0.0001
    num_workers = 4

    seed = 42

    print("Начало обучения")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    print("Загрузка данных")
    train_loader, test_loader, ingredients_df = create_dataloaders(
        dish_csv_path=data_dish_csv,
        ingredients_csv_path=data_ingredients_csv,
        img_dir=data_img_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    num_ingredients = len(ingredients_df)
    print(f"Количество ингредиентов: {num_ingredients}")

    print("Создание модели")
    model = MultimodalFoodCaloriesModel(
        num_ingredients=num_ingredients,
        pretrained=pretrained
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": []
    }

    best_mae = float("inf")

    print(f"Начало обучения на {epochs} эпох...")
    print("=" * 50)

    for epoch in range(epochs):
        print(f"\nЭпоха {epoch + 1}/{epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_mae = validate(
            model,
            test_loader,
            criterion,
            device
        )

        scheduler.step(val_mae)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae
                },
                save_path
            )
            print(f"Сохранена лучшая модель (MAE: {val_mae:.4f})")

    print("\n" + "=" * 50)
    print("Обучение завершено")
    print(f"Лучший Val MAE: {best_mae:.4f}")

    return history, best_mae



def load_model(checkpoint_path, num_ingredients, device='cpu'):
    model = MultimodalFoodCaloriesModel(
        num_ingredients=num_ingredients,
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Модель загружена из {checkpoint_path}")
    print(f"Эпоха: {checkpoint['epoch']}")
    print(f"Val MAE: {checkpoint['val_mae']:.4f}")

    return model

