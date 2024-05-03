# Image Classification of Sclerotic and non-sclerotic glomeruli

The repo contains the research, findings, codebase, traiinng and evaluation of certain models for binary classification of sclerotic and non-sclerotic glomeruli using microscopic images. A short summary of the models trained, these will discussed below:

|          Model        | Test Set Accuracy |
|-----------------------|-------------------|
|   Logistic Regression | 90.5%             |
|   Simple CNN          | 98.3%             |
|   ResNet-50           | 99%               |

Lets walk though the steps as they were implemented:

### Step 1: (LOGISTIC REGRESSION)

I started with a linear classification of the images by flattening them and using scikit.learn to fit a logistic regression (LR) model on it. Even though CNNs are the holy grail of image classification problems but I still feel trying to fit a LR model gives a good insight in the data and helps understand it more. The following issues were discovered in doing LR;

1. The image dataset is not homogenous in size.
2. The dataset is highly unbalanced. Class0 (non-sclerotic) is very much outnumbered (4704 vs 1054).
3. The dataset contains .png images which contains additional layer of intensity information which can be a boon or a bain. Boon as it contains extra information to be processed on, bain is my PC hardware has limited capability hence adding extra dimentionality. 

### Step 2: (Overcome dataset challenges)

I. Various methods were tried to homogenise the image size, I tried cropping to a specific size, resizing, crop to the minimum size of image, add padding to the image depending on various parameters. After applying these operation LR accuracy was calculated to determine which method suits the best. 

II. Images were augmented by adding little variations (i.e. rotaating, fliping, shear..) to augment the cardinality of each class.

III. The shape of images used for training of LR was 256x256x4

With this I was able to reach an accuracy of 90% with LR (pretty good I'd say, but we still have CNNs). This LR model can be downloaded from [here](). 20% of the data set was split using test_train_split for validation.

### Step 3: (CNN Model)

Due to limited computing power, creating and running a CNN made me find the optimal image size without hammpering much of the accuray, a dataset of dimention (~10000 x 256 x 256 x 4) would simply crach my kernel just on creating and loading this data (m1 mac- 8gb ram). A beefier machine coould have handled it. I hence settled with image dimentions of 128x128x3 and the following was the summary, accuracy and confusion matrix. A batch_normalisation layer felt neccessary here to have a normalization effect on the images accross.

Model can be downloaded from [here]()


#### Visualisation of the conv2d layer, and its effect on images

1st Conv layer

### Step 4: (Using pretrained model)
Knowing the capablitites of ResNet50 on image datatset which is part of the residual network family. I tried tweaking it to maximize accuracy. Trained it for 20 epochs, and its metrics are shown below.



