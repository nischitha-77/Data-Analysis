import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath):
    """Load dataset from a CSV file"""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Fill missing values: mean for numerical, mode for categorical"""
    for col in df.columns:
        if df[col].dtype == 'object':  # Categorical
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])
        else:  # Numerical
            df.loc[:, col] = df[col].fillna(df[col].mean().round())  # Ensure integer remains integer
    return df

def remove_duplicates(df):
    """Remove duplicate rows"""
    df.drop_duplicates(inplace=True)
    return df

def handle_outliers(df):
    """Replace outliers using IQR method"""
    for col in df.select_dtypes(include=np.number).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df.loc[:, col] = np.where((df[col] < lower) | (df[col] > upper), round(df[col].median()), df[col])
    return df

def scale_numerical_features(df):
    """Normalize numerical features using StandardScaler"""
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df.loc[:, num_cols] = scaler.fit_transform(df[num_cols])
    return df

def encode_categorical_variables(df):
    """One-hot encode categorical variables"""
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        df.drop(columns=cat_cols, inplace=True)
        df = pd.concat([df, encoded_df], axis=1)
    return df

def preprocess_data(df):
    """Run all preprocessing steps"""
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = scale_numerical_features(df)
    df = encode_categorical_variables(df)

    # 🔥 Ensure integer columns remain Int64
    for col in df.select_dtypes(include=['float']).columns:
        if df[col].dropna().apply(lambda x: x.is_integer()).all():
            df[col] = df[col].astype('Int64')  # Convert safely

    return df
