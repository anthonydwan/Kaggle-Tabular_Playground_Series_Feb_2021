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

## Attempt 3: Deepstack/Hybrid Denoising Autoencoder (DAE) + Multilayer Perceptron (ANN):
I wanted to implement   Tabular Playground Series January Competition's [1st place - turn your data into DAEta](https://www.kaggle.com/springmanndaniel/1st-place-turn-your-data-into-daeta/comments). The data distribution is very similar to January's data, except there are now 10 categorical varaibles. This should be noted since transforming categorical data may make it too sparse for DAE.   Since the original solution is in R and also did not provide code, I have the chance to work out the code from scratch using Python!

**model architecture:**<br>
Original architecture - Deepstack Autoencoder:
Input --> Encode Layer 1 (1500 Dense Relu) --> Encode Layer 2 (1500 Dense Relu) --> Deepstack 1 (500 Dense Relu) --> Deepstack 2 (500 Dense Relu) --> Deepstack 3 (500 Dense Relu) --> Decode Layer 1 (1500 Dense Relu) --> Decode Layer 2 (1500 Dense Relu)--> output


Visual representation of final DAE architecture
Encoder                    |  Decoder                  | Autoencoder               |
:-------------------------:|:-------------------------:|:-------------------------:|
![](https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/encoder.png)  |  ![](https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/decoder.png) |  ![](https://github.com/anthonydwan/Kaggle-Tabular_Playground_Series_Feb_2021/blob/main/DAE.png)


Due to OOM issues that I currently do not know how to resolve, I shrank the original DAE model by Danzel from 1500 * 3 deepstack layers into a hybrid bottleneck into 500 * 3 deepstack layers. It seemed that the Denzel's computer spec was simply better than mine (64GB ram).  

The three deepstack layers are fed as input into the MLP model. I am using optuna + keras for MLP hyperparameter tuning, which I initially thought to have a more intuitive pruning operation than kerastuner's hyperband. However, after writing i noted that the optuna integration callback function for keras has already been deprecated. 

I was unable to crack 0.88 with my DAE --> NN architecture. I have tried various size of DAE (smallest at around 64/64/32) and it was difficult to tell if it was working or not since the MSE for the DAE remained at around 0.076 regardless on the amount of noise or the complexity of the DAE architecture. When the DAE model was too complex, it seemed that the decoder was able to predict even when the encoder creates meaningless output (all 0 or same value). 

## Attempt 4 Bottleneck Denosing Autoencoder + LGBM:
To see whether it was the problem of ANN, I tried making a bottleneck DAE where the encode output layer is fed to a tuned LGBM model. There is only slight performance improvement to 0.87 RMSE. With such poor performance, I suspect that the inclusion of categorical features in the Feb dataset may have made it incredibly difficult to use DAE since the transformed data are generally more sparse. This did not drop below the 0.87.


## Attempt 5: GBDT Ensemble with incremental overfit improvement 
Credits to Siavash from Kaggle (@siavrez) for testing out this idea. Taking the tuned parameters from public discussion for the popular GBDT models - XGB, LGBM, Catboost and using average ensemble. The key idea is that for each of the models,  after one training cycle, the model is retrained with the same weights but with lower regularizations to try to further fit to the training data. This is done 10 times within 1 CV fold. While this theoretically should overfit to the training data, the results showed sufficient improvement over the regularized versions. 

# Post-competition: Refining DAE architecture 
Once again, the [winning solution](https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222745) involves a DAE architecture which is not surprising since the data is created synthetically with noise using CTGAN. 

## Attempt 6: Implementing DAE - from the basics 
Before attempting to construct the DAE created by Ren (@ryanzhang on Kaggle) which involved using transformers, I am going to perform DAE which can at least perform conparably to the GBDT models. 


