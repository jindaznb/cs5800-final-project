# Jinda Zhang CS5800 SU24

import pandas as pd
import numpy as np


def generate_dataset(filename='data.csv', num_points=100, num_variables=3, value_range=(0, 10)):
    """
    Generates a dataset with continuous variables and saves it to a CSV file.

    Parameters:
    - filename: Name of the CSV file to save the dataset.
    - num_points: Number of data points to generate.
    - num_variables: Number of continuous variables.
    - value_range: Tuple specifying the range (min, max) for the continuous values.
    """
    variable_names = [f'x{i+1}' for i in range(num_variables)]
    data = np.random.uniform(
        low=value_range[0], high=value_range[1], size=(num_points, num_variables))

    df = pd.DataFrame(data, columns=variable_names)
    df.to_csv(filename, index=False)
    print(f'Dataset saved to {filename}')

def main():
	generate_dataset(filename='../data/data.csv', num_points=100, num_variables=3, value_range=(0, 10))
      

if __name__ == "__main__":
	main()