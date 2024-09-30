import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_predictive_maintenance_data(num_rows=5000):
    """Generate a synthetic predictive maintenance dataset."""
    try:
        np.random.seed(42)  # For reproducibility

        # Create a unique identifier for each machine
        machine_ids = [f'Machine_{i}' for i in range(1, num_rows + 1)]

        # Randomly generate operating hours
        operating_hours = np.random.randint(100, 5000, size=num_rows)

        # Randomly generate temperature (20 to 100 degrees Celsius)
        temperature = np.random.uniform(20, 100, size=num_rows)

        # Randomly generate vibration (0.5 to 10 mm/s)
        vibration = np.random.uniform(0.5, 10, size=num_rows)

        # Randomly generate pressure (30 to 150 psi)
        pressure = np.random.uniform(30, 150, size=num_rows)

        # Generate Maintenance Required based on certain conditions
        maintenance_required = (temperature > 80).astype(int)  # Maintenance needed if temperature > 80

        # Generate failure based on conditions: higher likelihood with increased operating hours and vibration
        failure = ((operating_hours > 3000) & (vibration > 7)).astype(int)

        # Create a DataFrame
        data = pd.DataFrame({
            'Machine_ID': machine_ids,
            'Operating_Hours': operating_hours,
            'Temperature': temperature,
            'Vibration': vibration,
            'Pressure': pressure,
            'Maintenance_Required': maintenance_required,
            'Failure': failure
        })

        logging.info("Synthetic predictive maintenance data generated successfully.")
        return data

    except Exception as e:
        logging.error(f"An error occurred while generating data: {e}")
        return None

def save_to_csv(data, filename):
    """Save DataFrame to a CSV file."""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(filename)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        # Save the DataFrame to a CSV file
        data.to_csv(filename,header= True, index=False)
        logging.info(f"Data saved to {filename} successfully.")

    except Exception as e:
        logging.error(f"An error occurred while saving to CSV: {e}")

if __name__ == "__main__":
    # Generate the dataset
    predictive_maintenance_data = generate_predictive_maintenance_data()

    if predictive_maintenance_data is not None:
        # Save the dataset to a CSV file
        save_to_csv(predictive_maintenance_data, 'data/predictive_maintenance_data.csv')

    logging.info("Process completed.")
