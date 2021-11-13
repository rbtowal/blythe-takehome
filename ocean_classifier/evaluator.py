import copy
import os
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torchvision

class Evaluator():
    """
    Class for evaluating a model.

    Attributes:
        output_directory (string): Path to location to store result files.
        classes (list of strings): Class names.
        confidence_threshold (float): The value confidence needs to exceed for a class to be classified as not unknown.
        neptune_run (neptune run): Neptune.ai run object for meta data logging. Optional.
    """

    def __init__(self, output_directory, classes, confidence_threshold, neptune_run=None):
        """ Initialize Evaluator. """

        self.output_directory = output_directory
        self.neptune_run = neptune_run
        self.class_names = classes
        self.confidence_threshold = confidence_threshold

    def get_images_per_class_by_index(self, dataset, indices):
        """
        Returns a dictionary of images organized by class name.

        Args:
            dataset (Pytorch Dataset): Dataset with images corresponding to indices.
            indices (list): List of image indices to pull into the image.

        Returns:
            images (dict of images): Dictionary containing keys for each class and a list of images for each key.
        """

        images = {}
        for idx in indices:
            sample = dataset[idx]
            class_name = self.get_class_names(torch.unsqueeze(sample['target'], 0))[0]
            if class_name in images.keys():
                images[class_name].append(sample['image'])
            else:
                images[class_name] = [sample['image']]

        return images

    def get_class_names(self, class_tensor):
        """
        Get class name from prediction tensor.

        Args:
            class_tensor (torch tensor): [batch_size, num_classes] torch tensor.

        Returns:
            class_names (string): Name of the class. Unknown if prediction is ambiguous.
        """

        max_indices = torch.max(class_tensor, 1).indices.detach().tolist()
        max_values = torch.max(class_tensor, 1).values.detach().tolist()

        # If there is a prediction value that meets the confidence criteria,
        # update the class name before returning.
        class_names = [self.class_names[idx] if val >= self.confidence_threshold else "unknown"
                       for idx, val in zip(max_indices, max_values)]

        return class_names

    def visualize(self, dataset, target_classes, predicted_classes):
        """
        Creates visualizations of the model output as a confusion matrix and image grids of correct and incorrect images.

        Args:
            dataset (Pytorch Dataset): Dataset with images corresponding to indices.
            target_classes (list): List of strings of the target classes.
            predicted_classes (list): List of strings of the predicted classes.

        Returns:
            None
        """

        fig, ax = plt.subplots()
        # Add the unknown class if there is an unknown prediction.
        display_labels = copy.deepcopy(self.class_names)
        if "unknown" in set(predicted_classes):
            display_labels.append("unknown")
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(target_classes,
                                                                predicted_classes,
                                                                display_labels=display_labels,
                                                                ax=ax)
        plt.savefig(os.path.join(self.output_directory, "confusion_matrix.png"))
        if self.neptune_run: self.neptune_run["results/confusion_matrix"].upload(fig)

        for result in ["correct", "wrong"]:
            if result == "wrong":
                predictions = [idx for idx, target, predicted in
                                     zip(range(len(target_classes)), target_classes, predicted_classes) if
                                     target != predicted]
            else:
                predictions = [idx for idx, target, predicted in
                                     zip(range(len(target_classes)), target_classes, predicted_classes) if
                                     target == predicted]

            images = self.get_images_per_class_by_index(dataset, predictions)

            for class_name in images.keys():

                # Plot images, save and upload if using Neptune.
                image_grid = torchvision.utils.make_grid(images[class_name])
                torchvision.utils.save_image(image_grid, os.path.join(self.output_directory, "{}_{}.png".format(class_name, result)))
                if self.neptune_run:
                    fig = plt.figure()
                    plt.imshow(image_grid.permute((1, 2, 0)))
                    if self.neptune_run: self.neptune_run["results/{}_{}".format(result, class_name)].upload(fig)

    def get_metrics(self, target_classes, predicted_classes):
        """
        Gets the defined metrics and prints a classification report.

        Args:
            target_classes (list): List of strings of the target classes.
            predicted_classes (list): List of strings of the predicted classes.

        Returns:
            accuracy (float): Accuracy
            confusion_matrix (numpy array): Confusion matrix
            classification_report (scikit-learn classification_report): Report for printing.
        """

        accuracy = sklearn.metrics.balanced_accuracy_score(target_classes, predicted_classes)
        confusion_matrix = sklearn.metrics.confusion_matrix(target_classes, predicted_classes)

        # Add the unknown class if there is an unknown prediction.
        target_names = copy.deepcopy(self.class_names)
        if "unknown" in set(predicted_classes):
            target_names.append("unknown")
        classification_report = sklearn.metrics.classification_report(target_classes,
                                                                      predicted_classes,
                                                                      target_names=target_names,
                                                                      zero_division=0)

        return accuracy, confusion_matrix, classification_report

    def evaluate(self, model, dataloader):
        """
        Evaluates the model using the given dataloader.

        Args:
            model (pytorch model): PyTorch model
            dataloader (pytorch dataloader): PyTorch dataloader

        Returns:
            targets(tensor): Targets for the input images.
            predictions (tensor): Predictions from the network.
            target_classes (list): List of integers of the target classes.
            predicted_classes (list): List of integers of the predicted classes.
        """

        model.eval()
        targets = []
        predictions = []
        target_classes = []
        predicted_classes = []
        for idx, batch in enumerate(iter(dataloader)):

            # Infer
            data = batch["image"]
            target = batch["target"]
            prediction = model(data)

            # Collect and assign class names.
            targets += [target]
            predictions += [prediction]
            target_classes += self.get_class_names(target)
            predicted_classes += self.get_class_names(prediction)

        target_tensors = torch.cat(targets)
        prediction_tensors = torch.cat(predictions)

        return target_tensors, prediction_tensors, target_classes, predicted_classes
