import torch
from torchvision import transforms
from PIL import Image
import json


class Inference:
    def __init__(self, model_path, json_path, device="cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        self.class_labels = self.load_class_labels(json_path)
        self.preprocess = self.define_preprocessing()

    def load_class_labels(self, json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    def load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model

    def define_preprocessing(self):
        return transforms.Compose(
            [
                transforms.Resize((550, 550)),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        if isinstance(output, tuple):
            output = output[0]

        _, predicted_class = torch.max(output, dim=1)
        predicted_class_index = predicted_class.item()
        predicted_class_name = self.class_labels.get(
            str(predicted_class_index), "Unknown Class"
        )

        return predicted_class_name, predicted_class_index
