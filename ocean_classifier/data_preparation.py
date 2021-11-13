import os
import random
import numpy as np


class OceanDataPreparation():
    """
    Prepares ocean data for training and evaluation by splitting datasets.

    Analysis of dataset statistics to set normalization or class balancing parameters
    and other operations could also happen in this class.

    Attributes:
        root_dir (string): Directory with all the images.
        classes (list): List of class names as strings.
        splits (dictionary): Dict with the percentage of the dataset to include in the train, val and test splits.
                             Must sum to 1.

    """

    def __init__(self, root_dir, classes, splits={"train": 0.8, "val": 0.1, "test": 0.1}):
        """
        Initializes the DataPreparationClass.

        Args:
            root_dir (string): Directory with all the images.
            classes (list): List of class names as strings.
            splits (dictionary): Dict with the percentage of the dataset to include in the train, val and test splits.
                                 Must sum to 1.
        """
        if sum(splits.values()) != 1:
            RuntimeError("Split percentages must add up to 1 not {} = {}".format( sum(splits.values()), splits))

        random.seed(42) # Can be any integer.

        self.root_dir = root_dir
        self.classes = classes
        self.splits = splits

        self.all_data = self.get_data_file_names()

    def get_data_file_names(self):
        """
        Returns all data (not split) as a dictionary of filenames organized by class.

        Returns:
            datalist (dictionary): Dictionary with filenames for each class, specified by key.
        """

        datalist = {}
        for class_name in self.classes:
            datalist[class_name] = [
                file_name for file_name in os.listdir(os.path.join(self.root_dir, class_name)) if ".jpeg" in file_name
            ]

        return datalist

    def get_all_data(self):
        """
        Returns all data (not split) as a dictionary of tuples organized by class.

        Returns:
            datalist (list): List of tuples (class_name, file_name)
        """
        dataset = []
        for class_name in self.classes:
            class_files = self.all_data[class_name]
            dataset += [(class_name, os.path.join(self.root_dir, class_name, file_name)) for file_name in class_files]

        return dataset

    def split_data(self):
        """
        Randomly splits the data into training, validation and test splits.

        Returns:
            trainset (list): List of tuples containing (label, filename) for training set.
            valset (list): List of tuples containing (label, filename) for validation set.
            evalset (list): List of tuples containing (label, filename) for evaluation set.
        """

        trainset = []
        valset = []
        testset = []
        for class_name in self.all_data.keys():
            class_files = self.all_data[class_name]
            random.shuffle(class_files)

            test_size = int(np.ceil(self.splits["test"] * len(class_files)))
            val_size = int(np.ceil(self.splits["val"] * len(class_files)))

            testset += [(class_name, os.path.join(self.root_dir, class_name, file_name)) for file_name in class_files[0:test_size]]
            valset += [(class_name, os.path.join(self.root_dir, class_name, file_name)) for file_name in class_files[test_size:test_size + val_size]]
            trainset += [(class_name, os.path.join(self.root_dir, class_name, file_name)) for file_name in class_files[test_size + val_size:]]

        return trainset, valset, testset


