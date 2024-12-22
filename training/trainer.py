from data.data_loader import DataLoaderHandler
import torch
from torch import nn, optim
from training.model_trainer import ModelTrainer


class Trainer:
    def __init__(
        self, model, data_dir, epochs=2, batch_size=8, lr=0.001, weight_decay=1e-5
    ):
        self.model = model
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader_handler = DataLoaderHandler(data_dir, batch_size=batch_size)
        self.model_trainer = None

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.1
        )

    def setup_trainer(self):
        self.model.to(self.device)
        self.model_trainer = ModelTrainer(
            self.model, self.criterion, self.optimizer, self.scheduler, self.device
        )

    def start_training(self):
        self.setup_trainer()

        train_loader = self.data_loader_handler.get_train_loader()
        val_loader = self.data_loader_handler.get_val_loader()

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            train_loss = self.model_trainer.train_one_epoch(train_loader)
            print(f"Training Loss: {train_loss:.4f}")

            val_loss, accuracy = self.model_trainer.validate(val_loader)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

            self.scheduler.step()

            torch.cuda.empty_cache()

        self.model_trainer.save_model()
