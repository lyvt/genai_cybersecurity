import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def prepare_data(dir_path, data_type='train'):
    """
    Prepare data for training and testing.

    :param dir_path: Path to the data folder containing 'train.csv' and 'test.csv'.
    :param data_type: 'train' or 'test' to specify the type of data.
    :return: Tuple (X, y) where features are normalized and labels are binary.
    """
    # Load the data
    data = pd.read_csv(f'{dir_path}/{data_type}.csv', sep=",", header=None)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Split into features (X) and labels (y)
    X, y = data.iloc[:, :-1].to_numpy(), (data.iloc[:, -1] > 0).astype(int)

    # For training, only use normal data (label 0)
    if data_type == 'train':
        X, y = X[y == 0], y[y == 0]

    # Define scaler file path
    scaler_path = os.path.join("checkpoints", "scaler.pkl")

    # Normalize the features
    scaler = MinMaxScaler()
    if data_type == 'train':
        X = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)

    return X, y


if __name__ == '__main__':
    dir_path = 'data/UNSW'
    X, y = prepare_data(dir_path=dir_path)
