import torch
from tqdm import tqdm
from config.paths import model_save_path


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.amp.GradScaler()

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training")

        for images, labels in train_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / len(train_loader))

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc="Validation")

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_bar.set_postfix(
                    loss=val_loss / len(val_loader), accuracy=100 * correct / total
                )

        accuracy = 100 * correct / total
        return val_loss / len(val_loader), accuracy

    def save_model(self, path=model_save_path):
        torch.save(self.model, path)
        print("Model saved!")
