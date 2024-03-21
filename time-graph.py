import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import time


def plot_graph(csv_filename):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_filename)

    # Convert 'Time' column to datetime
    df["Time"] = pd.to_datetime(df["Time"])

    # Plot the graph
    plt.plot(
        df["Time"],
        df["People Inside"],
    )
    plt.title("Number of People Inside Over Time")
    plt.xlabel("Time")
    plt.ylabel("People Inside")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_data(csv_filename):
    while True:
        # Plot the graph
        plot_graph(csv_filename)

        # Wait for 30 seconds before plotting again
        time.sleep(2)


# Example usage:
csv_filename = "entry_exit_log_3.csv"
visualize_data(csv_filename)
