# Kaggle-Tabular_Playground_Series_Feb_2021
The goal is to predict a continuous target based on a number of feature columns given in the data. All of the feature columns, cat0 - cat9 are categorical, and the feature columns cont0 - cont13 are continuous. The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting the amount of an insurance claim. Although the features are anonymized, they have properties relating to real-world features.

## EDA
#### 1. Adversarial validation 
It is confirmed that the training data has identical distribution as and very representative of testing data.

#### 2. Exploratory data analysis 
Similar to Tabular Playground 1, the data is very clean but anonymised. This competition's data also contains categorical variables. Based on January's top performers, data engineering was mostly futile and that is probably the case for this competition as well.

## Submission 1: Kaggle Tuned LGBM + XGB
Borrowed from kaggle public notebooks (https://www.kaggle.com/tunguz/ensembling-starter-tps-feb-2021), a high-performance baseline ensemble of two tuned LGBM models + XGB model. 

## Submission (fail): Optuna Tuned CatBoost
150 Optuna trials with learning rate of 1. After picking the top performing parameters, the models are fitted with learning rate of 0.0075 for fine-tuning. 5 folds is used throughout the hyper parameter search process (since others have pointed out that higher folds will have better OOF scores but become less and less representative of test data). 

Due to time contrainst and lack-luster performance (OOF of 0.844), this submission was abandoned. 

## Submission 2: Optuna Tuned LGBM 

