import argparse
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim

from data_preparation import OceanDataPreparation
from dataset import OceanDataset
from evaluator import Evaluator
from model import OceanClassifierModel

# For experiment tracking and monitoring.
USE_NEPTUNE=False
try:
    import neptune.new as neptune
    USE_NEPTUNE=True
except ImportError:
    print("Could not find neptune installed, proceeding without it.")


def prepare_data(data_directory, batch_size):
    """
    Prepares the datasest and dataloader from a data directory.

    Data directory is expected to be organized as:

    Args:
        data_directory (string): Directory containing data to prepare.
        batch_size (int): Size of the batches.

    Returns:
        dataloaders (dict of Pytorch DataLoader): Dataloader for training, validation and test data.
    """

    dataloaders = {}

    # Data preparation
    data_prepper = OceanDataPreparation(root_dir=data_directory, classes=classes)
    traindata, valdata, testdata = data_prepper.split_data()
    data_sampler = None #torch.utils.data.WeightedRandomSampler(data_prepper.train_set_weights, batch_size, replacement=True)

    transform = transforms.Compose(  # composing several transforms together
        [transforms.ToTensor(),  # to tensor object
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # mean = 0.5, std = 0.5
    # TODO: Add augmentations to increase regularization and robustness.

    # Set up data loaders
    trainset = OceanDataset(traindata, data_prepper.classes, transform)
    valset = OceanDataset(valdata, data_prepper.classes, transform)
    testset = OceanDataset(testdata, data_prepper.classes, transform)

    num_workers = 8 # Set based on limitations of my machine. Could be configurable.
    dataloaders["train"] = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers,
                                              sampler=data_sampler)
    dataloaders["val"] = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    dataloaders["test"] = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return dataloaders

def train(model, dataloaders, params, output_directory, use_neptune=False):
    """
    Trains a model.

    Args:
        model (Pytorch model): Model file.
        dataloaders (dict of Pytorch DataLoader): Dataloader for training, validation and test data.
        params (dict): Dictionary of parameters used for training.
        output_directory (string): Path to location to store result files.
        use_neptune (bool): True if using Neptune for experiment tracking, False if not.

    Returns:
        None
    """

    if use_neptune:
        neptune_run = neptune.init(
            project='saildrone',
        )
    else:
        neptune_run = None

    # Set up evaluator
    evaluator = Evaluator(output_directory, classes, params["confidence_threshold"])


    # Training Loop
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params["learning_rate"], total_steps=params["epochs"])

    for e in range(params["epochs"]):
        epoch_time = time.time()
        if use_neptune: neptune_run["learning_rate"].log(scheduler.get_last_lr())

        # Train
        train_loss = 0.0
        model.train()
        for idx, batch in enumerate(iter(dataloaders["train"])):

            data = batch["image"]
            targets = batch["target"]
            optimizer.zero_grad()

            # forward_time = time.time()
            predictions = model(data)
            # print("Forward pass took {} seconds".format(time.time() - forward_time))

            loss = loss_function(predictions, targets)

            # backward_time = time.time()
            loss.backward()
            # print("Backward pass took {} seconds".format(time.time() - backward_time))

            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        if use_neptune: neptune_run["train/loss"].log(train_loss)
        print("Epoch took {} seconds".format(time.time() - epoch_time))
        print("Epoch {} Train Loss: {}".format(e, train_loss))

        # Validate
        targets, predictions, target_classes, predicted_classes = evaluator.evaluate(model, dataloaders["val"])
        validation_loss = loss_function(predictions, targets)
        accuracy, _, _ = evaluator.get_metrics(target_classes, predicted_classes)
        torch.save(model, 'model.pt')

        if use_neptune:
            neptune_run["val/loss"].log(validation_loss)
            neptune_run["val/accuracy"].log(accuracy)
        print("Epoch {} Validation Loss: {}".format(e, validation_loss))

    # Evaluation
    _, _, target_classes, predicted_classes = evaluator.evaluate(model, dataloaders["test"])
    accuracy, confusion_matrix, classification_report = evaluator.get_metrics(target_classes, predicted_classes)
    evaluator.visualize(dataloaders["test"].dataset, target_classes, predicted_classes)

    if use_neptune: neptune_run["test/accuracy"].log(accuracy)
    print(classification_report)

    # Save the model.
    torch.save(model, 'model.pt')

    if use_neptune: neptune_run.stop()

if __name__ == "__main__":

    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_directory", default="./data",
                        help="Path to directory of test images organized in folders by class. Default: ./data")
    parser.add_argument("-o", "--output", dest="output_directory", default="./results",
                        help="Path to directory for model output. If not provided, no images are saved. Default: ./results")

    # Note: For frequent and repeatable experimentation, these parameters would be specified in a config file
    # that could be referenced with the model output.
    parser.add_argument("-c", "--confidence_threshold", dest="confidence_threshold", default="0.5",
                        type=float,
                        help="Confidence threshold for predictions to be classified. Default: 0.5")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default="256",
                        type=int,
                        help="Batch size for evaluation batches. Set based on system constraints. Default: 256")
    parser.add_argument("-e", "--epochs", dest="epochs", default="50",
                        type=int,
                        help="Epochs to run for training. Default: 50")
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate", default="0.01",
                        type=float,
                        help="Learning rate for training. Default: 0.01")
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", default="0.001",
                        type=float,
                        help="Weight decay for training. Default: 0.001")
    parser.add_argument("-un", "--use_neptune", dest="use_neptune", default="False",
                        help="Whether to use neptune.ai for experiment monitoring. Default: False",
                        action='store_false')
    args = parser.parse_args()

    params = {}
    params["batch_size"] = args.batch_size
    params["epochs"] = args.epochs
    params["learning_rate"] = args.learning_rate
    params["weight_decay"] = args.weight_decay
    params["confidence_threshold"] = args.confidence_threshold

    # If user asked to use neptune, and it is installed, use it.
    use_neptune = False
    if USE_NEPTUNE and (args.use_neptune != 'False'):
        use_neptune = True

    classes = ["bird", "boat", "saildrone"]
    model = OceanClassifierModel(classes)
    dataloaders = prepare_data(args.data_directory, params["batch_size"])
    train(model, dataloaders, params, args.output_directory, use_neptune)