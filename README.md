[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Aj7Sf-j_)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10866179&assignment_repo_type=AssignmentRepo)
# Using pretrained CNNs for image classification

This repo contains code which trains a classifier on a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). The dataset has a accompanying paper on *arXiv.org* which can be read [here](https://arxiv.org/abs/2104.02830).

The code in this repository does the following:

- Trains a classifier on the *Indo fashion* dataset using the *pretrained CNN* called *VGG16*
- Saves training and validation history plots
- Saves a classification report

## User instructions

1. The user needs to download the dataset and save it into the folder called "**data**" in this repository. The dataset can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset).

2. Run the ```setup.sh```file to install necessary packages to run the code in this repository:

```bash setup.sh```

3. Run the ```run.sh``` file from command line to run the code:

```bash run.sh```

This installs the necessary packages into a virtual enviroment and runs the code which trains the classifier, saves the training and validation history plots as well as the classification report.

## Repository structure

| item | description | 
| --- | --- |
| data | folder in which the user may save the dataset |
| out | folder where the history plots and classification report will be saved to |
| src | folder which contains the .py script | 
| requirements.txt | text file listing the necessary packages to run the code | 
| run.sh | shell file which runs the code in src |
| setup.sh | shell file which installs the necessary packages | 

.................................................

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Aj7Sf-j_)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10866179&assignment_repo_type=AssignmentRepo)
# Using pretrained CNNs for image classification

In the previous assignments involving classification, we were performing a kind of simple feature extraction on images by making them greyscale and flattening them to a single vector. This vector of pixel values was then used as the input for some kind of classification model.

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which trains a classifier on this dataset using a *pretrained CNN like VGG16*
- Save the training and validation history plots
- Save the classification report

## Tips

- You should not upload the data to your repo - it's around 3GB in size.
  - Instead, you should document in the README file where your data comes from, how a user should find it, and where it should be saved in order for your code to work correctly.
- The data comes already split into training, test, and validation datasets. You can use these in a ```TensorFlow``` data generator pipeline like we saw in class this week - you can see an example of that [here](https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator).
- There are a lot of images, around 106k in total. Make sure to reserve enough time for running your code!
- The image labels are in the metadata folder, stored as JSON files. These can be read into ```pandas``` using ```read_json()```. You can find the documentation for that online.