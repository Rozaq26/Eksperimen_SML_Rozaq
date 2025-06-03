import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_dir):
    """
    Loads data from input_path, performs outlier removal, splitting, and scaling,
    and saves the preprocessed data to output_dir.

    Args:
        input_path (str): Path to the raw input CSV file.
        output_dir (str): Directory to save the preprocessed data and scalers.
    """
    # Load dataset
    try:
        df = pd.read_csv(input_path)
        print(f"Dataset loaded successfully from: {input_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Assuming 'Outcome' is the target variable
    if 'Outcome' not in df.columns:
        print("Error: 'Outcome' column not found in the dataset.")
        return

    df_processed = df.copy()

    # Handle outliers using IQR method before splitting
    print("Handling outliers...")
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    # Exclude the target column from outlier detection
    if 'Outcome' in numerical_cols:
        numerical_cols.remove('Outcome')

    initial_rows = len(df_processed)
    for feature in numerical_cols:
        Q1 = df_processed[feature].quantile(0.25)
        Q3 = df_processed[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter data that are within the normal bounds for the current feature
        df_processed = df_processed[(df_processed[feature] >= lower_bound) & (df_processed[feature] <= upper_bound)]

    rows_after_outlier_removal = len(df_processed)
    print(f"Jumlah data sebelum hapus outlier: {initial_rows}")
    print(f"Jumlah data setelah hapus outlier: {rows_after_outlier_removal}")
    if initial_rows > rows_after_outlier_removal:
        print(f"{initial_rows - rows_after_outlier_removal} rows removed due to outliers.")


    # Pisahkan fitur dan target
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    print("Features (X) and target (y) separated.")

    # Split data (80% train, 20% test)
    # Ensure stratify is used for classification tasks to maintain target distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")


    # Standardisasi fitur (hanya pada kolom numerik)
    # Re-identify numerical columns in case any were removed during outlier handling (though unlikely with this method)
    numerical_columns_after_split = X_train.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()

    # Fit scaler on the training data and transform both train and test sets
    X_train_scaled = scaler.fit_transform(X_train[numerical_columns_after_split])
    X_test_scaled = scaler.transform(X_test[numerical_columns_after_split])
    print("Numerical features scaled.")

    # Update the dataframes with scaled values
    X_train[numerical_columns_after_split] = X_train_scaled
    X_test[numerical_columns_after_split] = X_test_scaled


    # Pastikan folder output ada
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created or already exists: {output_dir}")

    # Simpan hasil preprocessing ke CSV
    X_train.to_csv(f"{output_dir}/x_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/x_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    # Simpan scaler untuk penggunaan di masa depan
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    print(f"Scaler saved to: {os.path.join(output_dir, 'scaler.joblib')}")


    print("✅ Preprocessing selesai. Dataset dan scaler disimpan di folder:", output_dir)

if __name__ == '__main__':
    # Contoh penggunaan fungsi
    # Pastikan file 'diabetes.csv' ada di lokasi ini atau sesuaikan path
    input_file = '/dataset_raw/diabetes.csv' # sesuaikan path
    output_folder = 'dataset_preprocessing/diabetes_preprocessing' # Sesuaikan nama folder output jika perlu
    preprocess_data(input_file, output_folder)