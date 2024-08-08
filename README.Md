![header](header.png)
# Addressing Customer Churn in an E-Commerce Company

## About
In this project, our goal is to reduce customer churn rate from 16.8% to 10% in an e-commerce company. While the current churn rate at 16.8% is below the global churn rate in an e-commerce industry, if not addressed soon and properly, this churn problem will pose a financial risk to the company. Here, we used a diagnostic analysis focusing on understanding possible factors driving churn by comparing the demographics, satisfaction, and behaviors of churned and non-churned customers. We also developed a classification model to predict churn, prioritizing the minimization of false negatives due to their financial impacts (false negative = 5 x false positive). 

After series of experiments exploiting resampling techniques (i.e., SMOTE, ADASYN, NearMiss v3) and 10 machine learning algorithms, we selected XGBoost as the final model based on both business (customer acquisition cost and retention cost) and machine learning metrics (F2 score), and identified two key factors which can be intervened for reducing the churn rate, namely tenure and cashback amount. The interventions can be done to customers predicted as churn by the model by increasing the cashback amount and lengthening the tenure. A simulation using 50% of intervention success rate, demonstrated the churn rate reduction to 10% is possible by utilizing the machine learning model and an intervention from the business stakeholder side.

>![important] Notebook
> The notebook for storing every process of this project can be seen in folder [notebook](https://github.com/LingAdeu/customer-churn-prediction/blob/main/notebook/notebook.ipynb). Alternatively, kindly see [NBViewer version](https://nbviewer.org/github/LingAdeu/customer-churn-prediction/blob/main/notebook/notebook.ipynb) for a better display.

## Data
The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) on an online retail company. This dataset consists of 15 numerical columns, including the target variable `Churn`, and 5 categorical columns. 

## Folder Organization
    .
    ├── README.Md                                   <- The top-level README for using this project.
    ├── data
    │   ├── E Commerce Dataset.xlsx                 <- The raw dataset for the analysis on the notebook.
    │   └── clean_data_with_updated_values.csv      <- The clean dataset for a Tableau analysis.
    ├── img                                         <- Folder containing images for the notebook
    ├── model
    │   └── clf_final.pkl                           <- The final model
    ├── notebook
    │   └── notebook.ipynb                          <- Jupyter notebook file for data analysis and modeling
    ├── requirements.txt                            <- The requirements file for reproducing the environment.
    └── tableau
        └── workbook.twb                            <- The Tableau workbook

## Contributor
- [Fariz Rachman Hadi](https://github.com/FarizHadi)
- [Adelia Januarto](https://github.com/LingAdeu)