import torch
from torch.utils.data import Dataset
from PIL import Image

from constants import PIC_SIZE
"""
transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        """

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ImageClassifierDataset(Dataset):
    def __init__(self, image_list, image_classes, transformation):
        self.images = []
        self.labels = []
        self.classes = list(set(image_classes))
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        self.image_size = PIC_SIZE
        self.transforms = transformation
        self.np_array_to_pil(image_list)

    def np_array_to_pil(self, image_list):
        for image, image_class in image_list:
            image = Image.fromarray(image)
            transformed_image = self.transforms(image)
            self.images.append(transformed_image)
            label = self.class_to_label[image_class]
            self.labels.append(label)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
