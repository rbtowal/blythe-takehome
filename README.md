## Ocean Image Classifier

This repo contains an ocean image classifier that will classify images as one of 'bird', 'boat', 'saildrone' or 'unknown'.

### Model Selection and Training

The model is based on the EfficientNet B7 implementation because it is fast and can be trained on a laptop.
With more training resources, one could train a bigger model, but for edge deployments smaller
models are generally preferred. There are many small models out there, chose this particular
one as it currently is the highest performing small model on ImageNet. To train on this specific
task, the last layer of the model was converted from 1000 outputs to 3 outputs that go through a softmax.
Only this last layer was trained and the rest of the network was kept intact. This is possible due to the
similarity of the ImageNet task and this task.

The model was trained on a MacBook Air using only CPU resources for 50 epochs in 5:30 hours. The training time
needed to reach this level of performance could be drastically shortened with more powerful machines. Training
metrics and efficiency can be viewed [here](https://app.neptune.ai/rbtowal/saildrone/e/SAIL-109).

Performance was measured by precision, recall and f1-score. This model achieves the following:

```
              precision    recall  f1-score   support

        bird       0.89      0.89      0.89        35
        boat       0.90      0.93      0.92        60
   saildrone       1.00      0.91      0.96        35
     unknown       0.00      0.00      0.00         0

    accuracy                           0.92       130
   macro avg       0.70      0.68      0.69       130
weighted avg       0.92      0.92      0.92       130
```

Reasonable performance was reached with a relatively simple training regimen that does not involve
augmentation and uses a standard normalization to a 0.5 mean and 0.5 std. Since this is a specialized
domain where most of the images may contain similar non-relevant information (like blue textures)
it is possible a more tailored pre-processing strategy could improve performance along with augmentation.

### Prerequisites

The requirements.txt file contains the required libraries to run this code. Install with

```buildoutcfg
pip3 install -r requirements.txt
```

Note: Neptune is a third party tool for monitoring experiments. If you do not have an account (it's free for individuals), please run the training script with `--use_neptune=False`.

Download the model from [dropbox](https://www.dropbox.com/s/w36i68z1j6aogr0/model.pt?dl=0) and place in the same directory as this readme. Email rbtowal@gmail.com if you need access to this location.

### Running the classifier

Running the classifier will produce a classification report printed to the screen, a confusion matrix and images
containing a grid of correctly and wrongly classified images for each class. The output will be structured like this:

```
├── results
│   ├── bird_correct.png
│   ├── bird_wrong.png
│   ├── boat_correct.png
│   ├── boat_wrong.png
│   ├── confusion_matrix.png
│   ├── saildrone_correct.png
│   └── saildrone_wrong.png
```
Note that for larger datasets, packing all the correct and wrongly classified images into a single image will not
be practical, but it was handy for a dataset of this size. For larger datasets
using single file outputs organized by directories, or a database with image references
would be more scalable.

To run the classifier, run the following command:

```buildoutcfg
python3 ocean_classifier/ocean_classifier.py
```

Running without arguments will use the data in this repo under `./data` and the model `./model.pt` by default.
To change the test data location, output directory and other parameters, please see the help by running:

```buildoutcfg
bash-3.2$ python3 ocean_classifier/ocean_classifier.py --help
usage: ocean_classifier.py [-h] [-m MODEL_PATH] [-d DATA_DIRECTORY] [-o OUTPUT_DIRECTORY] [-c CONFIDENCE_THRESHOLD] [-b BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model MODEL_PATH
                        Path to pytorch model file (*.pt). Default: ./model.pt
  -d DATA_DIRECTORY, --data DATA_DIRECTORY
                        Path to directory of test images organized in folders by class. Default: ./data
  -o OUTPUT_DIRECTORY, --output OUTPUT_DIRECTORY
                        Path to directory for model output. If not provided, no images are saved. Default: ./results
  -c CONFIDENCE_THRESHOLD, --confidence_threshold CONFIDENCE_THRESHOLD
                        Confidence threshold for predictions to be classified. Default: 0.5
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for evaluation batches. Set based on system constraints. Default: 128
```

### Training the classifier

The model checked in here was trained using the code under `train.py` and  `model.py`. To train a new model, 
run the following command:

```buildoutcfg
python3 ocean_classifier/train.py
```

Specify new data, training regimens, output locations or confidence thresholds on the command line:

```buildoutcfg
bash-3.2$ python3 ocean_classifier/train.py --help
usage: train.py [-h] [-d DATA_DIRECTORY] [-o OUTPUT_DIRECTORY] [-c CONFIDENCE_THRESHOLD] [-b BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE] [-wd WEIGHT_DECAY] [-un USE_NEPTUNE]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIRECTORY, --data DATA_DIRECTORY
                        Path to directory of test images organized in folders by class. Default: ./data
  -o OUTPUT_DIRECTORY, --output OUTPUT_DIRECTORY
                        Path to directory for model output. If not provided, no images are saved. Default: ./results
  -c CONFIDENCE_THRESHOLD, --confidence_threshold CONFIDENCE_THRESHOLD
                        Confidence threshold for predictions to be classified. Default: 0.5
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for evaluation batches. Set based on system constraints. Default: 256
  -e EPOCHS, --epochs EPOCHS
                        Epochs to run for training. Default: 50
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for training. Default: 0.01
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        Weight decay for training. Default: 0.001
  -un USE_NEPTUNE, --use_neptune USE_NEPTUNE
                        Whether to use neptune.ai for experiment monitoring. Default: False
```

#### Monitoring model training

This model was trained using [neptune.ai](https://neptune.ai/) to monitor the training process. If you have 
this installed, you can run with neptune by specifying `--use_neptune` on the command line. It is 
defaulted to off.

Note that this code currently has zero test coverage as that appeared out of the scope of this challenge.

If you have any questions please feel free to email me at rbtowal@gmail.com
