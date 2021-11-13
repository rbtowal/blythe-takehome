## Saildrone Machine Learning Engineer Challenge

This coding challenge is intended to gauge the level of your experience as a software developer and give us an example
of how you approach machine learning problems.

We expect that this challenge will take about 4-6 hours to complete. We typically expect to get your completed submission
within one week of receiving the problem, please communicate with us immediately if you need more time to complete the
challenge.


### Setup

In the `/data/` folder you will find  200px by 150px crops from real images taken by Saildrones. Each crop contains one
of three classes: `bird`, `boat`, or `saildrone`.

Your task is to build a classifier that labels new image crops as one of these three classes. This is an opportunity to
show us how you think, so please describe for us your process of data exploration, choice of preprocessing steps,
approach to model selection, evaluation methods, etc.

A complete solution should provide an easy way for the reviewer to run evaluation of your model on a test set of new image crops
of the same size and organized in directories the same way as the training data.


### Guidelines

Please submit your solution as a pull request to this Github repo.

You should write your solution to the challenge in Python. You may use any ML framework you prefer, but let us know why
you chose what you chose.

Your final solution must include instructions for evaluating your model on a new test set. If it's difficult to set up
and run this evaluation, our assessment of your solution will reflect as much.

If you do not have access to a GPU, please let us know so we can evaluate your challenge with this in mind. Training a
deep learning model for this task is feasible on CPU but we'd like to understand how you deal with the additional
challenge of longer training cycles.

Feel free to use any online resources such as Stack Overflow, Google, etc. If you borrow code make sure to provide
attribution.

If you have any questions please feel free to email us at cory@saildrone.com
