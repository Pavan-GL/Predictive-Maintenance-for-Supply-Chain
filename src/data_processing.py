import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, filepath):
        """Initialize with the path to the dataset."""
        self.filepath = filepath
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load the dataset."""
        try:
            self.data = pd.read_csv(self.filepath)
            logging.info("Dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def preprocess_data(self):
        """Preprocess the dataset: handle missing values, split into features and target."""
        try:
            if self.data is None:
                raise ValueError("Data not loaded. Please call load_data() first.")
            self.data.fillna(method='ffill', inplace=True)  # Fill missing values
            
            # Print the column names for debugging
            logging.info(f"Columns in the dataset: {self.data.columns.tolist()}")
            
            if 'Failure' not in self.data.columns:
                logging.error("'failure' column not found in the dataset.")
                raise KeyError("'failure' column not found in the dataset.")
            

            self.data = self.data.select_dtypes(include=[np.number]) 
            X = self.data.drop('Failure', axis=1)  # Features
            y = self.data['Failure']  # Target variable
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data preprocessed successfully.")
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise

    def scale_data(self):
        """Scale the data using StandardScaler."""
        try:
            if self.X_train is None or self.X_test is None:
                raise ValueError("Data not split. Please call preprocess_data() first.")
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            logging.info("Data scaled successfully.")
            return self.X_train_scaled, self.X_test_scaled
        except Exception as e:
            logging.error(f"Error in scaling data: {e}")
            raise

    def save_processed_data(self, filename):
        """Save the processed data as a pickle file."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'X_train': self.X_train,
                    'X_test': self.X_test,
                    'y_train': self.y_train,
                    'y_test': self.y_test,
                    'scaler': self.scaler
                }, f)
            logging.info(f"Processed data saved to {filename}.")
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor('D:\Predictive Maintenance for Supply Chain\data\cleaned_predictive_maintenance_data.csv')
    processor.load_data()
    processor.preprocess_data()
    X_train_scaled, X_test_scaled = processor.scale_data()
    processor.save_processed_data('D:\Predictive Maintenance for Supply Chain\models\processed_data.pkl')
