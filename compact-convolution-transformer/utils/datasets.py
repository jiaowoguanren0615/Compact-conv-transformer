import torch
from PIL import Image
from torchvision import transforms
from .split_data import read_split_data
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, image_paths, image_labels, transforms=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transforms = transforms

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert('RGB')
        label = self.image_labels[item]
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}


def build_dataset(args):
    train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(args.data_path)

    train_transform = data_transform['train']
    valid_transform = data_transform['valid']

    train_set = MyDataset(train_image_path, train_image_label, train_transform)
    valid_set = MyDataset(val_image_path, val_image_label, valid_transform)

    return train_set, valid_set