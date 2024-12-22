from model.model import model
from training.trainer import Trainer
from config.paths import data_path

if __name__ == "__main__":

    data_dir = data_path
    trainer = Trainer(model, data_dir)
    trainer.start_training()
