import argparse

import torch
import torchvision.transforms as transforms

from data_preparation import OceanDataPreparation
from dataset import OceanDataset
from evaluator import Evaluator


class OceanClassifier():
    """
    Classifier for ocean object data.

    Loads a model file and runs evaluation on a directory of images.

    Attributes:
        classes (list of strings): Class names.
        confidence_threshold (float): The value confidence needs to exceed for a class to be classified as not unknown.

    """

    def __init__(self, confidence_threshold):
        """
        OceanClassifier for classifying images.

        Params:
            confidence_threshold (float): The value confidence needs to exceed for a class to be classified as not unknown.
        """

        self.classes = ["bird", "boat", "saildrone"]
        self.confidence_threshold = confidence_threshold

    def prepare_data(self, data_directory, batch_size):
        """
        Prepares the datasest and dataloader from a data directory.

        Data directory is expected to be organized as:

        Args:
            data_directory (string): Directory containing data to prepare.
            batch_size (int): Size of the batches.

        Returns:
            dataloader (Pytorch DataLoader): Dataloader ready to provide data for evaluation.
        """

        # Data preparation
        data_prepper = OceanDataPreparation(root_dir=data_directory, classes=self.classes)

        transform = transforms.Compose(  # composing several transforms together
            [transforms.ToTensor(),  # to tensor object
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # mean = 0.5, std = 0.5

        # Set up data loaders
        num_workers = 1
        dataset = OceanDataset(data_prepper.get_all_data(), data_prepper.classes, transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return dataloader

    def classify(self, dataloader, model_path, output_directory):
        """
        Classifies images in dataloader using the model at model_path and stored the results in the output_directory.

        Args:
            dataloader (Pytorch DataLoader): Dataloader ready to provide data for evaluation.
            model_path (string): Path to pytorch model (*.pt)
            output_directory (string): Path to location to store result files.

        Returns:
            None
        """

        # Set up the evaluator
        evaluator = Evaluator(output_directory, self.classes, self.confidence_threshold)

        # Load the model.
        model = torch.load(model_path)

        # Evaluation
        _, _, target_classes, predicted_classes = evaluator.evaluate(model, dataloader)

        # Get the metrics
        accuracy, confusion_matrix, classification_report = evaluator.get_metrics(target_classes, predicted_classes)
        print(classification_report)

        # Save metric visualizations.
        evaluator.visualize(dataloader.dataset, target_classes, predicted_classes)

        print("Images for review saved in {}.".format(output_directory))


if __name__ == "__main__":

    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_path", default="./model.pt",
                        help="Path to pytorch model file (*.pt). Default: ./model.pt")
    parser.add_argument("-d", "--data", dest="data_directory", default="./data",
                        help="Path to directory of test images organized in folders by class. Default: ./data")
    parser.add_argument("-o", "--output", dest="output_directory", default="./results",
                        help="Path to directory for model output. If not provided, no images are saved. Default: ./results")
    parser.add_argument("-c", "--confidence_threshold", dest="confidence_threshold", default="0.5",
                        type=float,
                        help="Confidence threshold for predictions to be classified. Default: 0.5")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default="128",
                        type=int,
                        help="Batch size for evaluation batches. Set based on system constraints. Default: 128")
    args = parser.parse_args()

    # Setup the classifier and dataloader, then classify.
    ocean_classifier = OceanClassifier(args.confidence_threshold)
    dataloader = ocean_classifier.prepare_data(args.data_directory, args.batch_size)
    ocean_classifier.classify(dataloader, args.model_path, args.output_directory)