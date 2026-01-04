import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns

df_num = train_df[numerical_cols]
df_cat = train_df[categorical_cols]

# df_num is your numerical dataframe
missing_percent = df_num.isnull().mean() * 100
print(missing_percent)
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_num_imputed = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

# Encode categorical columns
encoder = OrdinalEncoder()
df_cat_encoded = encoder.fit_transform(df_cat)

# Impute
imputer = IterativeImputer()
df_cat_imputed = imputer.fit_transform(df_cat_encoded)
df_cat_encoded = pd.get_dummies(df_cat, drop_first=True)
df_num = df_num.reset_index(drop=True)
df_cat_encoded = df_cat_encoded.reset_index(drop=True)

df_clean = pd.concat([df_num_imputed, df_cat_encoded], axis=1)


df_clean_fe = df_clean.copy()
def create_profit_margin(df):
    """
    Calculate profit margin ratio for each business.
    This indicates financial efficiency and sustainability.
    """
    # Handle division by zero and missing values
    profit_margin = []

    for idx, row in df.iterrows():
        income = row.get('personal_income', np.nan)
        expenses = row.get('business_expenses', np.nan)

        # Check if both values exist and income is not zero
        if pd.notna(income) and pd.notna(expenses) and income != 0:
            margin = (income - expenses) / income
            # Cap extreme values
            margin = max(-1, min(margin, 1))  # Keep between -1 and 1
        else:
            margin = np.nan

        profit_margin.append(margin)

    return profit_margin
df_clean_fe['profit_margin'] = create_profit_margin(df_clean_fe)

def create_financial_access_score(df):
    """
    Calculate a composite score of financial service access.
    Higher score = better access to formal financial services.
    """
    financial_features = [
        'has_bank_account', 'has_loan_account', 'has_internet_banking',
        'has_debit_card', 'medical_insurance', 'funeral_insurance'
    ]

    # Available features in the dataset
    available_features = [col for col in financial_features if col in df.columns]

    scores = []
    for idx, row in df.iterrows():
        score = 0
        valid_features = 0

        for feature in available_features:
            value = row.get(feature, np.nan)

            if pd.notna(value):
                valid_features += 1
                # Check for positive responses
                if value in ['Yes', 'Have now', 'have now']:
                    score += 1
                elif value in ['Used to have but don\'t have now', 'used to have']:
                    score += 0.5  # Partial credit for past access

        # Normalize by number of valid features checked
        if valid_features > 0:
            normalized_score = score / valid_features
        else:
            normalized_score = np.nan

        scores.append(normalized_score)

    return scores # Changed to return the scores list
df_clean_fe['financial_access_score'] = create_financial_access_score(df_clean_fe)

import pandas as pd
import numpy as np

# Convert boolean True/False to integer 1/0
df_clean_fe['Target_Low_int'] = df_clean_fe['Target_Low'].astype(int)
df_clean_fe['Target_Medium_int'] = df_clean_fe['Target_Medium'].astype(int)

# Create single target column
# Logic:
# 0 -> Low
# 1 -> Medium
# 2 -> High (if neither Low nor Medium)
df_clean_fe['Target'] = np.where(
    df_clean_fe['Target_Low_int'] == 1, 0,  # Low
    np.where(df_clean_fe['Target_Medium_int'] == 1, 1, 2)  # Medium or High
)

# Drop original columns
df_clean_fe = df_clean_fe.drop(['Target_Low','Target_Medium','Target_Low_int','Target_Medium_int'], axis=1)

# Check the new target distribution
print(df_clean_fe['Target'].value_counts())
