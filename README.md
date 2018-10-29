# Machine Learning Project 1 - The Higgs Boson Challenge 

This project contains the code used for an academic Kaggle competition on The Higgs Boson Machine Learning Challenge organized during the Machine Learning course at EPFL.
It contains several implementations of Machine Learning methods and optimization functions for achieving the best result we could get on the competition. 

## Project overview

The project is organized in several files in order to guarantee modularity and have a clear structure: 

 - `implementations.py` contains the functions of six Machine Learning methods (Gradient Descent,
    Stochastic Gradient Descent, Least Squares, Ridge Regression, Logistic Regression,
    Regularized Logistic Regression).
 - `costs.py` contains the error functions used for computing the loss (Mean Square Error,
 	 Root Mean Square Error, Mean Absolute Error) and the loss methods. 
 - `gradients.py` contains the code for computing the gradients.
 - `data_processing.py` contains the code for preprocessing data before training the model,
  	i.e. cleaning, standardizing, feature engineering and augmentation through polynomials.
 - `helpers.py` contains different tools for different purposes, such as "batch_iter" for Stochastic
 	Gradient Descent, "sigmoid" for computing the loss and the gradient for Logistic Regression and 
 	"compute_accuracy" for assessing model performance.
 - `proj1_helpers.py` is a script with some useful functions for loading data and generating the predictions. 
 - `cross_validation.py` contains some utlities for the local testing of the models and some
 	hyper-parameters tuning methods for getting the best parameters. 
 - `plots.py` contains tools for plotting and testing hyper-parameters.
 - `run.py` is the script for generating the best submission achieved (accuracy = 0.817).


## Getting Started

These instructions will provide you all the information to get the best result achieved on your local machine, as described above.

### Training dataset preparation 

It is necessary to include a `data` folder in the project with 2 csv files:

 - `train.csv`: Training set of 250000 events. The file starts with the ID and label column, then the 30 feature columns.
 - `test.csv`: The test set of around 568238 events without labels.

  
### Creating the prediction

Once the dataset have been set-up, just execute `run.py`:

```
python run.py
```

You will see some output on the screen. Once "Done" appears, you will be able to see that a `prediction.csv` file has been generated. This file contains the predictions with the best model and parameters we could find and replicates exactly our best submission on the Kaggle competition.

## Notes

The hyper-parameters and the model used for the best prediction are hard coded in `run.py` and they can be changed in order to try other configurations and obtain different results.

## Authors 

 - Marshall Cooper
 - Andrea Piccione
 - Divij Pherwani

## Acknowledgments

The entire project used some utlities functions provided during the course lab sessions that can be found on https://github.com/epfml/ML_course