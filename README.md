[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Aj7Sf-j_)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10866179&assignment_repo_type=AssignmentRepo)
# Using pretrained CNNs for image classification

This repo contains code which trains a pretrained CNN on a dataset on images of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). The dataset has a accompanying paper on *arXiv.org* which can be read [here](https://arxiv.org/abs/2104.02830).

The code in this repository does the following:

- Trains a classifier on the *Indo fashion* dataset using the *pretrained CNN* called *VGG16*
- Saves training and validation history plots
- Saves a classification report

## User instructions

1. The user needs to download the dataset and have the folder open on the same level as this repository.
The dataset can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset).

**NB** The code includes absolute paths which means that the code may not be able to run if the the paths to the images do not match up with the ones defined in the code (example: /work/images/test/0.jpeg). The user may therefore change this section of the code if necessary. 

2. Run the ```setup.sh``` file to install necessary packages to run the code in this repository:

```bash setup.sh```

3. Run the ```run.sh``` file from command line to run the code:

```bash run.sh```

This installs the necessary packages into a virtual enviroment and runs the code which trains the classifier, saves the training and validation history plots as well as the classification report.

## Repository structure

| Item | Description |
| --- | --- |
| data | folder in which the user may save the dataset |
| out | folder where the history plots and classification report will be saved to |
| src | folder which contains the .py script |
| requirements.txt | text file listing the necessary packages to run the code |
| run.sh | shell file which runs the code in src |
| setup.sh | shell file which installs the necessary packages |

## Discussion

The goal of this repository is to train a pretrained classifier on a unseen dataset which is a concept known as *transfer learning*. Pretrained classifiers like VGG16 are effective because they have been trained on very large amounts of data which requires a vast amount of ressources. A pretrained classifier therefore makes classification tasks more efficient and available to the average user.

First I attempted to train the classifier with 10 epochs. But after four hours of training on a 32 CPU machine, the code had only run through approximately 1 1/3 epoch. I had expected that the process would go significantly faster after the first epoch but that did not seem to be the case. I assessed that training 10 epochs would not be feasible and therefore decided to cut down the amount of data and epochs. 
I used pandas to get an equally distributed sampling from each class, thereby reducing the size of the dataset by 50%. Additionally I reduced the amount of epochs from 10 to 3.
