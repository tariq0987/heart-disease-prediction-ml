"""
Script to download the UCI Heart Disease dataset.
"""
import os
import requests
import pandas as pd
from pathlib import Path

def download_heart_disease_data():
    """Download the Cleveland Heart Disease dataset from UCI."""
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)
    
    # UCI dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    print(f"Downloading dataset from {url}...")
    
    try:
        # Download the data
        response = requests.get(url)
        response.raise_for_status()
        
        # Save raw data
        raw_file = data_dir / "heart_disease_raw.data"
        with open(raw_file, 'wb') as f:
            f.write(response.content)
        print(f"Raw data saved to {raw_file}")
        
        # Load as DataFrame
        df = pd.read_csv(raw_file, names=column_names, na_values='?')
        
        # Save as CSV
        csv_file = data_dir / "heart_disease.csv"
        df.to_csv(csv_file, index=False)
        print(f"CSV file saved to {csv_file}")
        print(f"Dataset shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    df = download_heart_disease_data()
    if df is not None:
        print("\nFirst 5 rows:")
        print(df.head())