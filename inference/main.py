from inference import Inference
import torch
from model.features import Features
from model.network_wrapper import Network_Wrapper
import types
import sys
from model.basic_config import BasicConv
from config.paths import pretrained_model_path, class_labels_path, inference_image_path

basic_conv = types.ModuleType("basic_conv")
basic_conv.BasicConv = BasicConv
sys.modules["basic_conv"] = basic_conv

if __name__ == "__main__":
    model_path = pretrained_model_path
    json_path = class_labels_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inference = Inference(model_path=model_path, json_path=json_path, device=device)
    image_path = inference_image_path

    predicted_class_name, predicted_class_index = inference.predict(image_path)

    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted class: {predicted_class_name}")
