from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DataLoaderHandler:
    def __init__(
        self, data_dir, batch_size=8, image_size=(550, 550), crop_size=(448, 448)
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.crop_size = crop_size

    def get_train_loader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                ),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.ImageFolder(f"{self.data_dir}/train", transform=transform)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def get_val_loader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.ImageFolder(f"{self.data_dir}/test", transform=transform)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
