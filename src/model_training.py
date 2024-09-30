import pandas as pd
import logging
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, filepath=None):
        """Initialize with the path to the dataset."""
        self.filepath = filepath
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load the dataset from a CSV file."""
        try:
            self.data = pd.read_csv(self.filepath)
            logging.info("Dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def load_processed_data(self, pickle_file):
        """Load the preprocessed data from a pickle file."""
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
                self.X_train = data['X_train']
                self.X_test = data['X_test']
                self.y_train = data['y_train']
                self.y_test = data['y_test']
                self.scaler = data['scaler']
            logging.info("Processed data loaded successfully from pickle file.")
        except Exception as e:
            logging.error(f"Error loading processed data: {e}")
            raise

    def preprocess_data(self):
        """Preprocess the dataset: handle missing values, split into features and target."""
        try:
            if self.data is None:
                raise ValueError("Data not loaded. Please call load_data() first.")
            self.data.fillna(method='ffill', inplace=True)  # Fill missing values
            
            if 'failure' not in self.data.columns:
                logging.error("'failure' column not found in the dataset.")
                raise KeyError("'failure' column not found in the dataset.")

            self.data = self.data.select_dtypes(include=[np.number])  # Keep only numeric columns
            X = self.data.drop('failure', axis=1)  # Features
            y = self.data['failure']  # Target variable
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
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            logging.info("Data scaled successfully.")
            return X_train_scaled, X_test_scaled
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

class ModelTrainer:
    def __init__(self, input_shape):
        """Initialize the model trainer."""
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Assuming binary classification
        ])

    def compile_model(self):
        """Compile the Keras model."""
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("Model compiled successfully.")

    def train_model(self, X_train_scaled, y_train):
        """Train the model."""
        history = self.model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)
        logging.info("Model trained successfully.")
        return history

    def save_model(self, filename):
        """Save the trained model."""
        self.model.save(filename)
        logging.info(f"Model saved to {filename}.")

if __name__ == "__main__":
    # Example usage
    data_processor = DataProcessor()
    
    # Load preprocessed data from Pickle
    data_processor.load_processed_data('D:\Predictive Maintenance for Supply Chain\models\processed_data.pkl')
    
    # Scale the data (if needed)
    X_train_scaled, X_test_scaled = data_processor.scale_data()
    
    # Initialize and train the model
    model_trainer = ModelTrainer(input_shape=X_train_scaled.shape[1])
    model_trainer.compile_model()
    model_trainer.train_model(X_train_scaled, data_processor.y_train)
    model_trainer.save_model('D:\Predictive Maintenance for Supply Chain\models\predictive_maintenance_model.h5')
