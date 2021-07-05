# long_to_short_assessment_anxiety

Code for experiments conducted for the study of Developing Efficient, Convenient, and Non-repetitive Short Assessments to Replace a Long Assessment with Similar Accuracies, in the context of predicting the levels of anxiety in adults.

A demo dataset (`data_filtered.csv`) is included under the `/data` folder. The full dataset used for this study cannot be released due to data privacy laws from certain countries.

The experiment code files are numbered in the order of steps.

## Instructions to run code

First git clone this repository onto your computer following the instructions on the GitHub page.

Make sure you have Python (version 3.6 or later) installed on your computer. Install all the libraries listed in `requirements.txt` (If you have pip, use the command `pip install -r requirements.txt`). Note that in Linux, you may need to install `python3-dev` and `python3-tk` for some libraries to work.

Start running the code files from 2 through 5. If you don't run the files in order, it may result in errors. In `5_train_sklearn.py`, you need to provide an argument for model type (`--type`), which can be `lr` (Logistic Regression), `xgb` (XGBoost), `rf` (Random Forest), `svm` (Support Vector Machine), and `mlp` (Multilayer Perceptron Neural Network). If the argument is not provided, it uses `lr` by default. In Linux, the full command for running the code is `python 5_train_sklearn.py --type lr`.