# Kaggle-Tabular_Playground_Series_Feb_2021
The goal is to predict a continuous target based on a number of feature columns given in the data. All of the feature columns, cat0 - cat9 are categorical, and the feature columns cont0 - cont13 are continuous. The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting the amount of an insurance claim. Although the features are anonymized, they have properties relating to real-world features.

A lot of the techniques used are drawn from what I have used and learnt from the [Tabular Playground Series - January Competition](https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Jan_2021).

## EDA
Note: since github cannot handle interactive plots, please go to [Tabular Playground Feb EDA.ipynb](https://nbviewer.jupyter.org/github/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/Tabular%20Playground%20Feb%20EDA.ipynb).

#### 1. Adversarial validation 
<img src="https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/Adversarial%20validation.png" width="600" />

The trained model did not extract any information to distinguish train vs test data for both the training and holdout dataset - i.e. It is confirmed that the training data has identical distribution as and very representative of testing data.

#### 2. Exploratory data analysis 
Similar to Tabular Playground 1, the data is very clean but anonymised. This competition's data also contains categorical variables. Based on January's top performers, data engineering was mostly futile (except the use of DAE as feature engineering/selection implemented below) and that is probably the case for this competition as well.

[See Pandas Profiling EDA](https://htmlpreview.github.io/?https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/Pandas%20Profiling%20EDA.html#interactions)

## Submission 1: Kaggle Tuned LGBM + XGB
Borrowed from kaggle public notebooks [Ensemble Starter](https://www.kaggle.com/tunguz/ensembling-starter-tps-feb-2021), a high-performance baseline ensemble of two tuned LGBM models + XGB model. PB score @ 8.4218

## Submission (fail): Optuna Tuned CatBoost
150 Optuna trials with learning rate of 1. After picking the top performing parameters, the models are fitted with learning rate of 0.0075 for fine-tuning. 5 folds is used throughout the hyper parameter search process (since others have pointed out that higher folds will have better OOF scores but become less and less representative of test data). 

Due to time contrainst and lack-luster performance (OOF of 0.844), this submission was abandoned. 

## Submission 2: Optuna Tuned LGBM 
After numerous attempts to tune using 10CV-averaged LGBM, the parameters seem to have trouble converging. It maybe due to hyperband pruner being too aggressive and discarded promising trials. Regardless, 10CV takes too long to hyperparameter search and will use hyperparameter shared by Kagglers Shogosuzuki and Hamza. PB score @ 8.4220

## Submission 3: Deepstack/Hybrid Denoising Autoencoder (DAE) + Multilayer Perceptron (ANN):
I wanted to implement   Tabular Playground Series January Competition's [1st place - turn your data into DAEta](https://www.kaggle.com/springmanndaniel/1st-place-turn-your-data-into-daeta/comments). The data distribution is very similar to January's data, except there are now 10 categorical varaibles. This should be noted since transforming categorical data may make it too sparse for DAE.   Since the original solution is in R and also did not provide code, I have the chance to work out the code from scratch using Python!

**Current model architecture:**<br>
Deepstack Autoencoder:
Input --> Encode Layer 1 (1500 Dense Relu) --> Encode Layer 2 (1500 Dense Relu) --> Deepstack 1 (500 Dense Relu) --> Deepstack 2 (500 Dense Relu) --> Deepstack 3 (500 Dense Relu) --> Decode Layer 1 (1500 Dense Relu) --> Decode Layer 2 (1500 Dense Relu)--> output


Encoder                    |  Decoder                  | Autoencoder               |
:-------------------------:|:-------------------------:|:-------------------------:|
![](https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/encoder.png)  |  ![](https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/decoder.png) |  ![](https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/DAE.png)


Due to OOM issues that I currently do not know how to resolve, I shrank the original DAE model by Danzel from 1500 * 3 deepstack layers into a hybrid bottleneck into 500 * 3 deepstack layers. 



The three deepstack layers are fed as input into the MLP model. I am using optuna + keras for MLP hyperparameter tuning, which I found to have a more intuitive pruning operation than kerastuner's hyperband. 



