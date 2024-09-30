import logging
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    def __init__(self, model_path, pickle_path):
        """Initialize with model and data paths."""
        self.model_path = model_path
        self.pickle_path = pickle_path
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_model(self):
        """Load the trained model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def load_processed_data(self):
        """Load preprocessed data from a pickle file."""
        try:
            with open(self.pickle_path, 'rb') as f:
                data = pickle.load(f)
                self.X_test = data['X_test']
                self.y_test = data['y_test']
            logging.info("Processed data loaded successfully from pickle file.")
        except Exception as e:
            logging.error(f"Error loading processed data: {e}")
            raise

    def make_predictions(self):
        """Make predictions using the loaded model."""
        try:
            y_pred = self.model.predict(self.X_test)
            y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to binary classes
            logging.info("Predictions made successfully.")
            return y_pred_classes
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise

    def evaluate_model(self, y_pred_classes):
        """Evaluate the model and print the classification report and confusion matrix."""
        try:
            print(confusion_matrix(self.y_test, y_pred_classes))
            print(classification_report(self.y_test, y_pred_classes))
            logging.info("Model evaluation completed successfully.")
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator(
        model_path='D:\Predictive Maintenance for Supply Chain\models\predictive_maintenance_model.h5',
        pickle_path='D:\Predictive Maintenance for Supply Chain\models\processed_data.pkl'
    )
    
    evaluator.load_model()
    evaluator.load_processed_data()
    
    # Make predictions and evaluate
    y_pred_classes = evaluator.make_predictions()
    evaluator.evaluate_model(y_pred_classes)
