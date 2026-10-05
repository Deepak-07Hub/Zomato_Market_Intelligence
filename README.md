# Zomato Restaurant Rating Prediction & Market Intelligence

## Overview
This project analyzes a real restaurant dataset to understand how location, pricing, cuisine, digital adoption, and customer engagement relate to restaurant ratings. The notebook combines data cleaning, exploratory analysis, feature engineering, and machine learning into one portfolio-ready workflow.

## Business Problem
Restaurant operators and analysts often want to know which characteristics are associated with higher customer satisfaction and stronger restaurant performance. This project uses the actual Zomato dataset to answer that question with a practical, explainable workflow.

## Objectives
- Clean and validate a real restaurant dataset
- Explore market patterns by location, cuisine, and price
- Engineer features for classification
- Train and compare machine learning models
- Predict whether a restaurant is likely to be highly rated
- Produce a business-facing market intelligence summary

## Dataset
The project uses a real Zomato-style restaurant dataset stored as `zomato.csv` when present, with fallback support for the original local dataset filename if needed. The notebook checks the available CSV file automatically.

## Key Questions
- Which locations appear strongest?
- Which cuisines are popular and/or highly rated?
- What pricing patterns are visible?
- Does online ordering or table booking show a relationship with ratings?
- Which features are most useful for predicting a high-rated restaurant?

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- scikit-learn
- Jupyter Notebook
- ipywidgets (when available)

## Project Workflow
1. Load the dataset and inspect the real schema.
2. Clean invalid values and standardize categories.
3. Explore the market with EDA and Plotly charts.
4. Engineer predictive features.
5. Train several classification models.
6. Compare performance with F1 and ROC-AUC as primary decisions.
7. Create a prediction demo and business-facing findings.

## Exploratory Analysis
The EDA section examines market trends in restaurant count, ratings, pricing, geographic concentration, cuisine popularity, and digital adoption patterns.

## Feature Engineering
The working dataset includes engineered features such as price category, cuisine count, votes log, online-order flag, table-booking flag, location frequency, and cuisine frequency. The original rating is not used as a feature to avoid leakage.

## Machine Learning
The notebook uses a classification setup with a stratified train/test split and a preprocessing pipeline. The target is defined as a restaurant being highly rated when its aggregate rating is at least 4.0.

## Model Evaluation
The project compares logistic regression, decision tree, random forest, and gradient boosting models using accuracy, precision, recall, F1 score, and ROC-AUC.

## Interactive Visualization
The notebook contains interactive Plotly charts for rating distribution, city analysis, cuisine analysis, pricing relationships, votes, and digital adoption patterns. A polished notebook-style HTML hero section is also included.

## Market Intelligence
The notebook summarizes the strongest observed business patterns and interprets them as relationships that are associated with performance rather than causal effects.

## Business Recommendations
The recommendations are derived from actual findings in the dataset rather than generic startup advice.

## Repository Structure
```text
zomato-market-intelligence/
??? Zomato_Market_Intelligence.ipynb
??? zomato.csv
??? README.md
??? requirements.txt
??? docs.md
??? .gitignore
??? LICENSE
```

## How to Run
1. Open the notebook in Jupyter or VS Code.
2. Ensure the dataset is available as `zomato.csv` or update the dataset path variable near the top of the notebook.
3. Run all cells from top to bottom.
4. Review the analysis, model output, and generated repository files.

## Requirements
See `requirements.txt` for the project dependency list.

## Limitations
This is a classification-focused business intelligence project and not a causal analysis of restaurant performance.

## Future Improvements
Future versions could add richer geospatial analysis, review text sentiment, and more advanced model tuning.

## Skills Demonstrated
- Data cleaning and validation
- Exploratory data analysis
- Feature engineering
- Machine learning with scikit-learn
- Plotly and notebook storytelling
- Business analysis and recommendations

## Author
Project created in a Jupyter Notebook format for GitHub-ready portfolio presentation.
