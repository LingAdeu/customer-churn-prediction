![header](header.png)
# Addressing Customer Churn in an E-Commerce Company

## About
In this project, we address a customer churn problem of 16.8% in an e-commerce company. Although the current churn rate is considered good by e-commerce standards (common churn rate: 22%), it results in substantial financial losses if not treated properly. We conducted a diagnostic analysis on historical data to understand the factors driving customer churn, focusing on comparing churned and non-churned customers in terms of their demographics and behavior.

In addition to the data analysis, we built a classification model to predict customer churn focusing more on minimizing false negatives due to their higher financial cost. After conducting a series experiments and evaluating model performance and financial impacts, we selected XGBoost as the final model. Key features influencing churn predictions include tenure, cashback amount, recency of last order, number of addresses, and marital status. By using these insights and our model's predictions, companies can implement more targeted retention strategies and effectively reduce customer churn.

## Data
The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

## Folder Organization
    .
    ├── README.Md                                   <- The top-level README for using this project.
    ├── data
    │   ├── E Commerce Dataset.xlsx                 <- The raw dataset for the analysis on the notebook.
    │   └── clean_data_with_updated_values.csv      <- The clean dataset for a Tableau analysis.
    ├── img                                         <- Folder containing images for the notebook
    ├── model
    │   ├── calibrated_best_model.pkl
    │   └── clf_final.pkl                           <- The final model
    ├── notebook
    │   └── notebook.ipynb                          <- Jupyter notebook file for data analysis and modeling
    ├── requirements.txt                            <- The requirements file for reproducing the environment.
    └── tableau
        └── workbook.twb                            <- The Tableau workbook

## Contributor
- [Fariz Rachman Hadi](https://github.com/FarizHadi)
- [Adelia Januarto](https://github.com/LingAdeu)