import pandas as pd
from datetime import datetime


# Function to calculate the sum of power consumption between a start and end time
def calculate_power_consumption(csv_file, start_time, end_time):
    # Read the CSV file without a header, assign column names manually
    data = pd.read_csv(csv_file, header=None, names=['timestamp', 'power_consumption'])

    # Convert the 'timestamp' column to datetime objects
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Convert the input times to datetime objects
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Filter the rows where the timestamp is between the start and end time
    filtered_data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]

    # Calculate the sum of power consumption in the filtered data
    total_power_consumption = filtered_data['power_consumption'].sum()

    return total_power_consumption/60


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Job energy consumption.')
    parser.add_argument('file_path', type=str, help='Full path of .csv file containing energy consumption data')
    parser.add_argument('start_time', type=str, help='Job start timestamp')
    parser.add_argument('end_time', type=str, help='Job end timestamp')
    args = parser.parse_args()



    csv_file = args.file_path
    start_time = args.start_time
    end_time = args.end_time

    # Calculate the sum of power consumption
    total_power = calculate_power_consumption(csv_file, start_time, end_time)

    print(f"Total power consumption between {start_time} and {end_time}: {total_power}Wh")
