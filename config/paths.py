from pathlib import Path

root_path = Path(__file__).resolve().parent.parent


model_save_path = root_path / "pretrained/model.pth"
data_path = root_path / "raw/car_data"
pretrained_model_path = root_path / "pretrained/pretrained_model.pth"
class_labels_path = root_path / "config/class_labels.json"
inference_image_path = (
    root_path / "raw/car_data/test/Mercedes-Benz SL-Class Coupe 2009/02615.jpg"
)
