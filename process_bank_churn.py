import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

def split_data(raw_df: pd.DataFrame, target_column: str = "Exited") -> tuple:
    """
    Splits the raw dataframe into training and validation sets.

    Parameters:
    raw_df (DataFrame): The raw input dataframe to be split.
    target_column (str): The column name of the target variable.

    Returns:
    tuple: Training and validation dataframes.
    """
    train_df, val_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df[target_column]
    )
    return train_df, val_df

def separate_columns(train_df: pd.DataFrame) -> tuple:
    """
    Separates columns into input columns and target column.

    Parameters:
    train_df (DataFrame): The dataframe from which columns will be separated.

    Returns:
    tuple: Input columns (list) and target column (str).
    """
    input_cols = list(train_df.columns[1:-1])
    target_col = train_df.columns[-1]
    return input_cols, target_col

def encode_categorical_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, categorical_cols: list) -> tuple:
    """
    Encodes categorical features using OneHotEncoder.

    Parameters:
    train_inputs (DataFrame): The training data with features to be encoded.
    val_inputs (DataFrame): The validation data with features to be encoded.
    categorical_cols (list): List of categorical columns to encode.

    Returns:
    tuple: Transformed training and validation inputs.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoder_cols = encoder.get_feature_names_out(categorical_cols)
    train_inputs[encoder_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoder_cols] = encoder.transform(val_inputs[categorical_cols])
    return train_inputs, val_inputs, encoder

def scale_numeric_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, numeric_cols: list) -> tuple:
    """
    Scales numeric features using MinMaxScaler.

    Parameters:
    train_inputs (DataFrame): The training data with numeric features.
    val_inputs (DataFrame): The validation data with numeric features.
    numeric_cols (list): List of numeric columns to scale.

    Returns:
    tuple: Scaled training and validation inputs.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])

    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Preprocesses raw data by splitting, encoding, scaling (optionally), and cleaning.

    Parameters:
    raw_df (DataFrame): The raw input dataframe to preprocess.
    scaler_numeric (bool): Whether to scale numeric features (default is True).

    Returns:
    dict: A dictionary containing preprocessed data for training and validation.
    """
    # Step 1: Split data
    train_df, val_df = split_data(raw_df)
    input_cols, target_col = separate_columns(train_df)

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    # Step 2: Identify numeric and categorical columns
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include="object").columns.tolist()

    # Step 3: Encode categorical features
    train_inputs, val_inputs, encoder = encode_categorical_features(train_inputs, val_inputs, categorical_cols)

    # Step 4: Scale numeric features if requested
    if scaler_numeric:
        train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
    else:
        scaler = None

    # Step 5: Remove unwanted columns (Geography and Gender)

    train_inputs.drop(columns=categorical_cols, inplace=True)
    val_inputs.drop(columns=categorical_cols, inplace=True)

    # Step 6: Return preprocessed data
    return {
        "X_train": train_inputs,
        "train_targets": train_targets,
        "X_val": val_inputs,
        "val_targets": val_targets,
        "input_cols": input_cols,
        "encoder": encoder,
        "scaler": scaler
    }


def preprocess_new_data(new_df: pd.DataFrame, encoder, scaler, input_cols) -> pd.DataFrame:
    """
    Preprocesses new data using the already trained encoder and scaler.

    Parameters:
    new_df (DataFrame): The new input data to preprocess.
    encoder (OneHotEncoder): The trained OneHotEncoder to transform categorical features.
    scaler (MinMaxScaler): The trained MinMaxScaler to scale numeric features.
    input_cols (list): List of input column names.

    Returns:
    DataFrame: The preprocessed new data.
    """

    new_inputs = new_df[input_cols].copy()

    # Encode categorical features using the trained encoder
    categorical_cols = new_inputs.select_dtypes(include="object").columns.tolist()
    encoded_categorical_data = encoder.transform(new_inputs[categorical_cols])
    # Add encoded categorical features to new_inputs
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_cols))
    new_inputs = pd.concat([new_inputs, encoded_categorical_df], axis=1)
    
    # Drop the original categorical columns
    new_inputs.drop(columns=categorical_cols, inplace=True)

    # Scale numeric features using the trained scaler
    numeric_cols = new_inputs.select_dtypes(include=np.number).columns.tolist()
    new_inputs[numeric_cols] = scaler.transform(new_inputs[numeric_cols])

    return new_inputs


