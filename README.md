# Using Healthcare Risk Factors to Predict Length of Hospital Stay
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Jupyter Notebook](https://img.shields.io/badge/notebook-Jupyter-orange.svg)
![Last Commit](https://img.shields.io/github/last-commit/skadamcik/hospital-stay-prediction.svg)


### Data manipulation and analysis on a synthetic hospital record data set

This project explores how medical risk factors can be analyzed to predict the length of hospital stay. It demonstrates end-to-end data science practices, including cleaning noisy data, visualizing trends, and comparing linear vs. nonlinear regression nethods for predictive accuracy.

While patient vitals are often highly variable, the ability to predict how long they will stay after initial admittance can aid in hospital resource planning and allocation, as well as provide solid indicators that may signal a longer duration of hospitalization.

## Project Goals

- Clean and validate synthetic patient records with noisy columns, null values, and implausible entries
- Visualize medical vitals/conditions against hospital stay length
- Build and evaluate linear and nonlinear regression models
- Compare model performance and determine feature performance

## Executive Summary
Linear regression acheived the strongest performance (R<sup>2</sup> = 0.66, $\pm$ 1.25 days error), drawing predictive power primarily from medical conditions. Random Forest performed slightly worse (R<sup>2</sup> = 0.65, $\pm$ 1.27 days error), but emphasized patient vitals. Together, these results suggest that in this hospital, medical conditions provided clearer predictive signals for hospital stay, while vitals contributed complementary but less consistent information.

## Repository Contents

* `hospital_stay_prediction.ipynb` &rarr; Jupyter notebook with full Python workflow
* `images/` &rarr; folder with `.png` files of all visual outputs
* `patient_records.csv` &rarr; original Kaggle dataset, renamed for simplicity purposes
* `requirements.txt` &rarr; Python dependencies used for the project

## Dataset Description

The original synthetic dataset can be found [here](https://www.kaggle.com/datasets/abdallaahmed77/healthcare-risk-factors-dataset/data), and includes 30,000 unique records with 20 features.

| Feature Name      | Short Description                                            | Data Type |
| :---------------- | :----------------------------------------------------------- | :-------- |
| `Age`               | Patient's age in years                                       | `float`     |
| `Gender`            | Male or Female                                               | `object`    |
| `Medical Condition` | Reported health condition (Healthy, Arthritis, Cancer, etc.) | `object`    |
| `Glucose`           | Blood glucose level (mg/dL)                                  | `float`     |
| `Blood Pressure`    | Systolic blood pressure                                      | `float`     |
| `BMI`               | Body Mass Index                                              | `float`     |
| `Oxygen Saturation` | Blood oxygen saturation level (%)                            | `float`     |
| `LengthOfStay`      | Number of days spent in hospital                             | `int`       |
| `Cholesterol`       | Cholesterol level (mg/dL)                                    | `float`     |
| `Triglycerides`     | Triglyceride level (mg/dL)                                   | `float`     |
| `HbA1c`             | Hemoglobin A1c (%)                                           | `float`     |
| `Smoking`           | Whether the patient smokes (0 = non-smoker, 1 = smoker)      | `int`       |
| `Alcohol`           | Whether the patient consumes alcohol (0 = no, 1 = yes)       | `int`       |
| `Physical Activity` | Hours/week of physical activity                              | `float`     |
| `Diet score`        | Quality of diet (numeric)                                    | `float`     |
| `Family History`    | Family medical history (0 = no, 1 = yes)                     | `int`       |
| `Sleep Hours`       | Average hours of sleep per day                               | `float`     |
| `random_notes`      | Various random strings                                       | `object`    |
| `noise_col`         | Unrelated, random values                                     | `float`     |

## Workflow Overview

### Part 1: Libraries Used

- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib, seaborn (data visualizations)
- scikit-learn (model development):
  - `LinearRegression`, `Ridge`, `Lasso`
  - `RandomForestRegressor`
  - `train_test_split`,  accuracy metrics (e.g. `r2_score`, `mean_absolute_error`, `root_mean_squared_error`)
- statsmodels (diagnostics):
  - `variable_inflation_factor`
  - `add_constant`

### Part 2: Data Cleaning
Several measures were taken to prime the dataset for regression analysis.
1. Dropped unrelated columns (e.g. `random_notes`, `noise_col`)

2. Removed records with any number of missing values. Without medical expertise, attempting to impute values risked introducing bias. Given the size of the dataset, regression models could still generalize well.

3. Standardized feature names (e.g. `Medical Condition` &rarr; `medical_condition`)

4. Established a validation dictionary with plausible ranges for vitals (healthy and extreme). Overlayed validation boundaries on histograms and dropped records outside these ranges.
    * Ex. Oxygen Saturation

![Distribution of oxygen saturation with boundary lines](images/o2sat_dist.png)![Cleaned distribution of oxygen saturation after validation](images/o2sat_cleandist.png)

5. Encoded categorical features:
    * `gender`: binary encoding &rarr; `gender_encoded`
    * `medical_condition`: one-hot encoding (baseline = "Healthy"), then converted from `boolean` to `int`

**Final dataset:** 12,341 entries, 23 columns

### Part 3: Exploratory Data Analysis
To assess relationships between features, an initial correlation heatmap was generated in the notebook. This highlighted potential sources of multicollinearity. For example, `diabetes`, `glucose`, and `hb_a1c` showed strong, positive correlations with each other, while `age` and `oxygen_saturation` had moderate negative correlations with `asthma`.

A second heatmap was generated with `length_of_stay` as the target focus.

![Heatmap visualizing all features against length of stay](images/LoS_heatmap.png)

**Key findings:**
* **`cancer` was most strongly correlated with `length_of_stay`**
* Moderate correlations shown for `hb_a1c`, `diet_score`, `stress_level`, `sleep_score`, and `diabetes`

Further exploring these relationships, scatterplots were generated for each feature against `length_of_stay`, with trendlines added for clarity.
* Ex. `cancer` (`r = 0.69`) and `stress_level` (`r = -0.26`) against `length_of_stay`

![Scatterplot of cancer against length of stay, r = 0.69](images/cancer_scatter.png)
![Scatterplot of diet score against length of stay, r = -0.26](images/diet_scatter.png)

**Feature selection:**
* Dropped features with correlation magnitude < 0.1
* Retained medical condition features to preserve one-hot encoding integrity
* Although best practice suggests dropping features with |r| < 0.25, borderline features (`glucose`) and others with potential real-world significance (`age`, `oxygen_saturation`, `physical_activity`) were kept

**Regression dataset:** 12,341 records, 15 features

### Part 4: Linear Regression model development and evaluation
Using the refined dataset, a baseline linear regression model acheived an **R<sup>2</sup> of 0.66**, with an average prediction error of **$\pm$ 1.25 days**. As expected, the model performed better with shorter lengths of stay, but variability increased for longer stays. Residual analysis (not shown here) confirmed this, highlighting greater error at higher lengths of stay.

![Scatterplot of predicted vs. actual length of stay, linear model](images/lr_pred_act.png)

**Key findings:**
* Medical conditions were stronger predictors of hospital stay than vital metrics, (due to binary encoding). 
* Compared to a healthy patient:
    * Cancer patients stay almost **12 days longer**
    * Diabetes patients stay about **4 days longer**
    * Asthma and hypertension patients stay almost **3 days longer**
    * Obesity and arthritis patients stay about **2 days longer**

To assess multicollinearity, the variable inflation factor (VIF) was calculated. Only `diabetes` exceeded 10 (10.81), while `hypertension` and `obesity` had borderline-high values (6.22 and 4.90). These features were maintained to preserve one-hot integrity.

Ridge and Lasso regression models were tuned and tested. 
* Ridge regression ($\alpha$ = 0.01) performed identically to the baseline linear model
* Lasso regression ($\alpha$ = 0.001) performed marginally better, but not enough to justify replacing the initial linear model. 

### Part 5: Random Forest regression model development and evaluation
To address the limited variation in predictions from the linear model, a Random Forest regression was tested on the initial, cleaned dataset. The model produced more variable predictions but slightly lower accuracy, with **R<sup>2</sup> = 0.65** and average error of **$\pm$ 1.27 days**.

![Scatterplot of predicted vs. actual length of stay, random forest model](images/rf_pred_act.png)

### Part 6: Compare models and summarize findings
Comparing linear coefficients and Random Forest feature importances revealed consistent top predictors: `cancer` and `diabetes`.Beyond these, **the linear model relied more on medical conditions**, while **the Random Forest model emphasized patient vitals**.

![Bar chart comparing top 10 linear coefficients vs. RF importances](images/lr_top10_features.png)

A broader union of top predictors of both models was also analyzed, confirming these trends.

**Summary of Model Performances**
|Model|R<sup>2</sup>|Mean Absolute Error|Root Mean Squared Error|
|:---|:---|:---|:---|
|Linear|0.6625|1.2549|1.5709|
|Ridge|0.6625|1.2549|1.5709|
|Lasso|0.6628|1.2543|1.5702|
|Random Forest|0.6506|1.2680|1.5904|

Based on these values, it is reasonable to conclude that both the linear and Lasso models were comparably effective in predicting the length of hospital stay. Since their metrics are marginally different, implementing one over the other would be up to personal discretion. In practice, linear regression is often chosen for simplicity and interpretability, while Lasso regression is ideal when featurization and regularization are important for reducing noise or overfitting.

## Limitations and Further Investigations
**Limitations**
* Results are based on a synthetic dataset and may not fully reflect real hospital data.
* Dropping missing values reduced dataset size by over half, which may limit generalizablity.
* Linear regression relied heavily on medical conditions due to one-hot/binary encoding, while Random Forest emphasized vitals - this may reflect encoding choices rather than true predictive power.
* Important predictors of hospital stay (i.e. socioeconomic status, insurance policy, comorbidities beyond what is listed) were not included, suggesting that the variability observed could be explained by these external factors.

**Further Investigations**
* Implement mean, KNN, or regression-based imputation strategies to retain more records.
* Engineer new features and interaction terms to improve predictive accuracy.
* Validate findings against real-world hospital datasets.
* Explore other nonlinear models (e.g. Gradient Boosting, XGBoost, GAMs)
* Apply k-fold cross-validation for more robust evaluation.

While results are promising, further work on real-world data and advanced models is needed.

## Viewing and running the project
1. **Clone the repository**
```bash
git clone https://github.com/skadamcik/hospital-stay-prediction.git
cd hospital-stay-prediction
```
2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate # On Mac/Linux
venv\Scripts\activate # On Windows
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the Jupyter notebook**
```bash
jupyter notebook hospital_stay_prediction.ipynb
```
5. **Explore results**
* The notebook walks through data cleaning, exploratory analysis, and model development.
* Visualizations and outputs are saved as .png files in the project folder for quick reference.

## Acknowledgements
This project was completed as part of my final portfolio for my degree in Applied Mathematics, Programming Concentration.
I would like to thank  Dr. Diane Bansbach, Mr. Darrell Dow, and Dr. Jodee Vallone for their guidance and support during this process.
Special thanks to the Wilmington University Applied Mathematics program for providing me with the opportunity to explore data science and data analytics in depth.