# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree, In which we build and optimizing an Azure Machine Learning pipeline using the Python SDK and Scikit-learn model and then this model will be compared to an Azure AutoML run and do some analysis on it.

## Summary
In This project we have used Azure ML Studio and a "Bank Marketing" Dataset which is related to direct marketing campaigns of a Portuguese banking sector & campaigns were based on phone calls.

The best performing model was a **Voting Ensemble**. This was found using **AutoML** feature of Azure.

### Files Used to perform the Analysis are 
- train.py
- udacity-project.ipynb

### Steps followed for completion of Project
1. Firstly we have to Load the dataset into Registered Dataset using given Dataset URL.
2. Created a compute instance for the Notebook execution.
3. We build a Machine learning model using skikit learn (Logistic Regression) & tune the hyper parameters to find the best model using azure ML python SDK and Hyper Drive.
4. We have used the Azure AutoML Feature to find the best model and best Hyperparameters by creating a series of experiments.
5. We have build a report by comparing both the above model & Make a solid understanding on AzureML Studio with the best practices of MLOps.

**Pipeline Architect**
![Pipeline Architect](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/Pipeline_Architect.JPG?raw=true "Pipeline Architect")

## Scikit-learn Pipeline
**The pipeline architecture**
- Initially we retrieve the dataset in registered dataset from the url provided using Azure TabularDatasetfactory class.
- To clean the data, We have processed our dataset using the provided 'clean_data' function in which it has pass through 'converting categorical variable to binary encoding', 'one hot encoding' etc. 
- we have split our dataset into training & testing part with the ratio of 70:30 respectively and used the LogisticRegression class from skikit learn to define Logistic Regression model.
- With the help of Azure ML Hyper Drive we have tuned the inverse regularization(C) and maximum iterations(max_iter) hyperparamters to find the best combination for maximizing the accuracy.
- The classification algorithm used here is Logistic Regression with accuracy as the primary metric for classification which is completely defined in the train.py file
- Finally, we have noted the best run of the hyperdrive and saved the best model.

**The benefits of the parameter sampler**
- We have used Random Parameter Sampler to provide random sampling over a hyperparameter search space.
- It has also performing equally as Grid Search with lesser compute power requirements.

**Hyperparameters**
- Inverse regularization parameter(C)- A control variable that retains strength modification of Regularization by being inversely positioned to the Lambda regulator. The relationship, **[ C = 1/λ ]** would be that lowering C - would strengthen the Lambda regulator.
- No of iterations(max_iter):The number of times we want the learning to happen. This helps is solving high complex problems with large training hours.

**The benefits of the early stopping policy**
- Useful in stopping the HyperDrive run before it gets Overfitted or if the accuracy of the model is not improving from the best accuracy by a certain defined amount after every given number of iterations
- Here we have used the **BanditPolicy** for early stopping policy. you can check [here](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py)  
  
  
  ![Child Runs](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/Child_runs.png?raw=true "Child Runs")
  
  ![HyperDrive Metrics](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/HyperDrive_Metrics.png?raw=true "HyperDrive Metrics")

## AutoML
  1.  AutmoML means that we can Automating the process, it reduces the time consumed by the training (Traditional) process. It also helps in performing iterative tasks of ML models. It is known for its incredible flexibility
  2.  With the help of AutoML we can accelerate the time taken for deployment of models into production with great efficency.
  3.  The models used here were RandomForests,Boosted Trees,XGBoost,LightGBM,SGD Classifier,Voting Ensemble.
  4.  It also used different pre- processing techniques like Standard Scaling, Min Max Scaling, Sparse Normalizer, MaxAbsScaler. It also has efficiently balanced the class Imabalance in the data.
  5.  It also has the feature of crossvalidation where number of cross_validation split is specified using which it performs validation on the dataset.

In this project, we set the AutoML configuration with accuracy as primary metric and cross validation. The cross validation is important to avoid overfitting and helps generalize the model better. For computational reasons, in this experiment, the experiment timeout was set to 30 Minutes which have limited the number of Models that could be built.

 ![AutoML_Models](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/AutoML_Models.png?raw=true "AutoML_Models")
 ![Best_AutoML_Model](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/Best_AutoML_Model.png?raw=true "Best_AutoML_Model")
 ![Best_AutoML_Metrics](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/Best_AutoML_Metrics.png?raw=true "Best_AutoML_Metrics")
 ![AutoML_Features](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/AutoML_Features.png?raw=true "AutoML_Features")
 ![AutoML_BoxPlot](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/AutoML_BoxPlot.png?raw=true "AutoML_BoxPlot")
 

## Pipeline comparison
- Overall,the difference in accuracy between the AutoML model and the Hyperdrive tuned custom model is very small. AutoML accuracy was 0.917238 while the Hyperdrive accuracy was 0.910268
- With Respect to architecture AutoML was better than hyperdrive because it tried a lot of different models, which is quite impossible to do with Hyperdrive, as that would require us to create a new pipeline for each model.
- The architecture of both pipelines are different, but the ideas are close: Load the data, instanciate the infrastructure to compute, set the parameters and call the compute method. The main difference is the that using AutoML we have infitine possibilities to increase the search for a better algorithm or a hyperparameter combination.

## Future work
We can notice that the dataset has Class Imbalance, we need to treat this by applying few balancing techniques like SMOTE,ADASYN and try to find further on this. For improvements and future experiments it is important to explore other algorithms and validate more metrics. Accuracy in this case is very important, but using a confusion matrix can be important.We can also used balanced_accuracy which is a primary metric that calculates for the arithmetic mean of recall for each class. Another example is AUC_weighted, which gets the arithmetic mean of the score for each class , weighted by the true number of true instances in each class. We can modify the experiments with other hyperparameters and increase the cross validation to enhance models performance and generalization.

## Proof of cluster clean up
![Delete_Cluster](https://github.com/yash872/ML_with_Azure_Nanodegree/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Images/Delete_Cluster.png?raw=true "Delete_Cluster")
