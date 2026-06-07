# Case Study: Diabetes Mellitus Preprocessing

## Project overview

This project cleans and prepares a pregnancy cohort dataset for downstream machine learning analysis. The dataset contains maternal visceral adipose tissue (VAT) measurements collected during routine obstetric ultrasound in the first 20 weeks of pregnancy, together with maternal clinical variables and delivery outcomes.

The main outcome of interest is **gestational diabetes mellitus (GDM)**, stored in the dataset as:

```text
gestational dm
```

The case study asks for preprocessing rather than model building. The notebook therefore focuses on data quality assessment, missing-data treatment, feature engineering, class-imbalance preparation, and scaling.

## Files

| File | Description |
|---|---|
| `CaseStudy_Diabetes.ipynb` | Main Jupyter notebook containing the preprocessing workflow. |
| `visceral_fat.csv` | Input dataset expected by the notebook. |
| `README.md` | Project documentation and run instructions. |

The notebook currently expects the CSV file at:

```text
visceral-adipose-tissue-measurements-during-pregnancy-1.0.0/visceral_fat.csv
```

If your CSV is stored somewhere else, update the `data_path` variable in the notebook.

## Dataset description

The dataset contains intake variables measured during the first 20 weeks of pregnancy and outcome variables recorded at delivery.

Key variables include:

| Variable | Meaning |
|---|---|
| `number` | Unique case ID. |
| `age (years)` | Maternal age in years. |
| `ethnicity` | Ethnicity, coded as `0 = white`, `1 = not white`. |
| `diabetes mellitus` | Previous diabetes mellitus, coded as `0 = no`, `1 = yes`. |
| `mean diastolic bp (mmhg)` | Mean diastolic blood pressure. |
| `mean systolic bp (mmhg)` | Mean systolic blood pressure. |
| `central armellini fat (mm)` | Maternal visceral adipose tissue measurement. |
| `current gestational age` | Gestational age at inclusion, recorded as weeks and days. |
| `pregnancies (number)` | Number of pregnancies. |
| `first fasting glucose (mg/dl)` | First measured fasting glucose. |
| `bmi pregestational (kg/m)` | Pregestational body mass index. |
| `gestational age at birth` | Gestational age at delivery, recorded as weeks and days. |
| `type of delivery` | `0 = vaginal birth`, `1 = caesarean section`. |
| `child birth weight (g)` | Child birth weight in grams. |
| `gestational dm` | Current gestational diabetes, coded as `0 = no`, `1 = yes`. |

Missing values are represented by empty cells.

## Notebook workflow

### 1. Import libraries

The notebook uses the following main Python libraries:

```python
numpy
pandas
seaborn
matplotlib
pathlib
scikit-learn
imbalanced-learn
```

### 2. Read the data

The dataset is loaded using `pandas.read_csv()`.

```python
data_path = Path("visceral-adipose-tissue-measurements-during-pregnancy-1.0.0/visceral_fat.csv")
diabetes = pd.read_csv(data_path)
```

### 3. Initial data assessment

The notebook performs an initial exploratory assessment using:

- `diabetes.info()`
- `diabetes.head()`
- `diabetes.describe()`
- value counts for categorical or discrete variables
- missing-value counts by column

Initial findings from the notebook:

| Issue | Finding |
|---|---|
| Dataset size | 133 rows and 15 columns before exclusion cleaning. |
| Previous diabetes case | 1 participant has previous diabetes mellitus and is removed based on the exclusion criteria. |
| Missing ethnicity | 1 missing value. |
| Missing pregnancies | 5 missing values before exclusion cleaning. |
| Missing first fasting glucose | 30 missing values before exclusion cleaning. |
| Missing pregestational BMI | 1 missing value. |
| GDM imbalance | After excluding the previous diabetes case, the target has 115 non-GDM cases and 17 GDM cases. |

### 4. Exclusion cleaning

The case-study brief excludes participants with pre-existing type 1 or type 2 diabetes. The notebook removes the one participant with previous diabetes mellitus:

```python
diabetes_clean = diabetes[diabetes["diabetes mellitus"] == 0].copy()
```

After this, the `diabetes mellitus` column is dropped because it no longer contains useful variation.

### 5. Missing-data handling

The notebook evaluates several missing-value strategies, including:

- mode imputation
- median imputation
- mean imputation
- deterministic ratio imputation
- KNN imputation

Final selected approach:

| Feature | Final handling |
|---|---|
| `ethnicity` | Filled with modal class, `0`. |
| `bmi pregestational (kg/m)` | Filled with the mean. |
| `pregnancies (number)` | KNN imputation, then rounded because it is a count variable. |
| `first fasting glucose (mg/dl)` | KNN imputation, kept continuous because glucose is physiologically continuous. |

The notebook chooses KNN imputation for `pregnancies (number)` and `first fasting glucose (mg/dl)` because it better preserves relationships between variables compared with simpler approaches.

### 6. Gestational-age conversion

The notebook converts gestational-age string values from weeks and days into numeric day counts:

```python
current gestational age (days)
gestational age at birth (days)
```

This makes the variables easier to use in downstream machine learning models.

### 7. Feature enrichment

The notebook creates additional calculated features:

| New feature | Formula / logic | Purpose |
|---|---|---|
| `mean pulse pressure (mmhg)` | systolic BP minus diastolic BP | Captures blood pressure spread. |
| `hypertensive flag` | systolic BP > 140 or diastolic BP > 90 | Binary indicator for high blood pressure. |
| `visceral_fat_to_bmi_ratio` | central Armellini fat divided by BMI | Captures VAT relative to general body size. |

### 8. Class imbalance

The cleaned dataset is highly imbalanced:

```text
gestational dm = 0: 115 cases
gestational dm = 1: 17 cases
```

The notebook discusses several possible approaches:

- random oversampling
- undersampling
- SMOTE
- class weights in downstream models
- optimizing for recall or sensitivity

The notebook implements a stratified train-test split first, then applies SMOTE only to the training data to avoid data leakage.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=10, stratify=y
)

smote = SMOTE(random_state=10)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
```

### 9. Normalisation and standardisation

The notebook creates both normalised and standardised versions of the training and test data.

Normalisation uses:

```python
MinMaxScaler()
```

Standardisation uses:

```python
StandardScaler()
```

The notebook notes that standardisation may be more suitable because it handles outliers better and does not force features into a strict 0 to 1 range.

## How to run the notebook

### 1. Create an environment

Example using `conda`:

```bash
conda create -n diabetes-preprocessing python=3.11
conda activate diabetes-preprocessing
```

Or using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 3. Check the data path

Make sure the CSV file is available at:

```text
visceral-adipose-tissue-measurements-during-pregnancy-1.0.0/visceral_fat.csv
```

Alternatively, update the notebook line:

```python
data_path = Path("your/path/to/visceral_fat.csv")
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Then open:

```text
CaseStudy_Diabetes.ipynb
```

Run the notebook from top to bottom.


## Outputs produced

The notebook prepares the following useful objects for future modelling:

| Object | Description |
|---|---|
| `diabetes_clean` | Cleaned full dataset after exclusion, missing-data handling, gestational-age conversion, and feature enrichment. |
| `x_train`, `x_test` | Stratified train-test feature split. |
| `y_train`, `y_test` | Stratified train-test target split. |
| `x_train_smote`, `y_train_smote` | SMOTE-balanced training features and target. |
| `x_train_normalised`, `x_test_normalised` | Min-max normalised feature sets. |
| `x_train_standardised`, `x_test_standardised` | Standardised feature sets. |

## Project status

This notebook completes the preprocessing stage requested by the case-study brief. It does not train or evaluate predictive models. The prepared datasets can be used next for classification models that predict gestational diabetes mellitus.

Recommended next modelling steps:

1. Select a baseline classifier, such as logistic regression.
2. Compare models trained on the original, class-weighted, and SMOTE-balanced training data.
3. Prioritise recall or sensitivity because missing GDM-positive cases may be more clinically important than false positives.
4. Evaluate using confusion matrix, recall, precision, F1-score, ROC-AUC, and PR-AUC.
