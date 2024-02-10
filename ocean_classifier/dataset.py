import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class OceanDataset(Dataset):
    """
    Dataset for classification of ocean objects in image crops.

    Attributes:
        file_list (list of strings): List of all the file paths.
        labels (list of strings): ?
        transform (callable, optional): Optional transform to be applied on an image.
    """

    def __init__(self, file_list, labels, transform=None):
        """
        Args:
            file_list (list of strings): List of all the file paths.
            labels (list of strings): ?
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Returns length of data as an int."""
        return len(self.file_list)

    def __getitem__(self, item):
        """
        Returns a batch of items from the dataset.

        Args:
            item (torch tensor): Tensor of items to get from the dataset.

        Returns:
            sample (dict): Dictionary of the form {'image': image, "target": torch.tensor}
        """

        if torch.is_tensor(item):
            item = item.tolist()

        # Load and transform the image.
        label, img_name = self.file_list[item]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        # One-hot encoding of the label, using an unknown class if none of the existing labels.
        target_vector = torch.zeros(len(self.labels))
        if label in self.labels:
            target_vector[self.labels.index(label)] = 1
        else:
            RuntimeError("Unknown label {}".format(label))

        sample = {'image': image, 'target': target_vector}

        return sample
