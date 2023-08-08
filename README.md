# long_to_short_dass

Code for experiments conducted for the study of Developing Efficient, Convenient, and Non-repetitive Short Assessments to Replace a Long Assessment with Similar Accuracies, in the context of predicting anxiety levels in adults.

To get the dataset used for this study, please let me (Bill) know via email.

Demo: https://kangleelab-surveys.herokuapp.com/anxiety_moderate

# Abstract

Background: Assessments and surveys are used to measure various aspects of individuals such as abilities, attitudes, skills, and wellbeing. These tools contain multiple items and can take varying amounts of time to complete. Valid assessments provide reliable measurements, but incomplete responses due to time constraints or impatience can lower participation and measurement quality.

This study aims to propose the use of a Long to Short approach that uses machine learning to develop efficient and convenient short assessments or surveys to approximate an equivalent long one with acceptable accuracies.

Methods: The dataset used for this study contained a total of 31,715 participants worldwide containing demographic information, their responses to the DASS-42 questionnaire, and their corresponding anxiety scores and status. The most important questions on the DASS-42 were extracted using feature selection from this dataset, and seven different machine learning algorithms were then trained and tested on 90% of the dataset. The models are then evaluated against the pristine dataset which is not used in training or testing. 

Results: After combining the results of feature selection using MRMR and Extra Tree, the 10 items numbered {11, 13, 34, 23, 17, 15, 12, 4, 2, 33} were the most important questions in DASS-42. The best-performing models were Ensemble and Random Forest based on the metrics of AUC score and F1 score. Therefore, it is proposed that a minimum of 5 items from DASS-42 plus 3 demographics items could be used for rapid assessment of the levels of anxiety if the use case allows for collecting demographic data, and 7 items from DASS-42 if the use case does not call for the input of demographics.

Interpretation: The study suggests that it is certainly possible to reduce the number of questions in a questionnaire-based assessment while retaining a relatively high accuracy using machine learning models.  

# Python environment and Jupyter Notebook setup

## 1. Set up Anaconda 

To run the Jupyter Notebooks in this repository, you will need to install Python and Anaconda. One way to do this is to install Miniconda from https://docs.conda.io/en/latest/miniconda.html. The study was conducted on Python version 3.11, so it is recommended to install a matching environment. 

After installing miniconda, open the Anaconda prompt and install Jupyter Notebook using the following command: 
```
conda install -c conda-forge notebook
```

Open Jupyter Notebook by opening the Anaconda prompt and use the following command: 
```
jupyter notebook
```

## 2. Install the required packages

Afterwards, you can install the required libraries using pip: 
```
pip install 'package name'
```

This will install the latest version of the package. 

You can install a specific version of a required library using the following command: 
```
pip install 'package name' == x.x.x
```

The following is the full list of libraries you need to install/reinstall and their version used during the study. You can copy each pip command and run it in the Anaconda Prompt.

```
pip install numpy==1.25.0
pip install pandas==1.5.3
pip install pycountry_convert==0.7.2
pip install sklearn==1.2.2
pip install matplotlib==3.7.2
pip install pymrmr==0.1.1
pip install scipy==1.10.1
pip install xgboost==1.7.4
```
If the code crashes and raises the error ModuleNotFoundError: No module named 'module name', copy the module name and use pip to install the package.

# Link to publication
doi.org/10.3758/s13428-021-01771-7 