import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Plotting
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    # Load CSV data into a DataFrame
    df = pd.read_csv("entry_exit_log.csv")

    # Convert 'time' column to datetime format
    df["time"] = pd.to_datetime(df["time"])

    # Extract hour from the timestamp
    df["hour"] = df["time"].dt.hour

    # Calculate average people inside per hour
    avg_people_inside = df.groupby("hour")["people_inside"].mean()
    avg_people_inside.plot(kind="bar", color="skyblue")
    plt.title("Average People Inside per Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Average People Inside")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()


ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)

plt.show()
