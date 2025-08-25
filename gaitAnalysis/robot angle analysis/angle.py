import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to your DLC output file
dlc_file = "C:/Users/martv/Documents/robot video/processed/all/1 (18)DLC_Resnet50_robot1Aug21shuffle1_snapshot_020.csv"

# Load DLC output (CSV format)
df = pd.read_csv(dlc_file, header=[0, 1, 2])

# Extract x, y, and likelihood columns for left and right
left_x = df[('DLC_Resnet50_robot1Aug21shuffle1_snapshot_020', 'left', 'x')].copy()
left_y = df[('DLC_Resnet50_robot1Aug21shuffle1_snapshot_020', 'left', 'y')].copy()
left_likelihood = df[('DLC_Resnet50_robot1Aug21shuffle1_snapshot_020', 'left', 'likelihood')]
right_x = df[('DLC_Resnet50_robot1Aug21shuffle1_snapshot_020', 'right', 'x')].copy()
right_y = df[('DLC_Resnet50_robot1Aug21shuffle1_snapshot_020', 'right', 'y')].copy()
right_likelihood = df[('DLC_Resnet50_robot1Aug21shuffle1_snapshot_020', 'right', 'likelihood')]

# Set values with likelihood < 0.6 to NaN
left_x[left_likelihood < 0.6] = np.nan
left_y[left_likelihood < 0.6] = np.nan
right_x[right_likelihood < 0.6] = np.nan
right_y[right_likelihood < 0.6] = np.nan

# Interpolate missing values (linear interpolation)
left_x = left_x.interpolate(limit_direction='both')
left_y = left_y.interpolate(limit_direction='both')
right_x = right_x.interpolate(limit_direction='both')
right_y = right_y.interpolate(limit_direction='both')

# --- Outlier detection and interpolation ---
def interpolate_outliers(series, threshold=4):
    # Calculate rolling median and MAD (Median Absolute Deviation)
    rolling_median = series.rolling(window=5, center=True, min_periods=1).median()
    mad = (series - rolling_median).abs().rolling(window=5, center=True, min_periods=1).median()
    # Identify outliers
    outliers = ((series - rolling_median).abs() > threshold * mad)
    # Set outliers to NaN
    series[outliers] = np.nan
    # Interpolate again
    return series.interpolate(limit_direction='both')

left_x = interpolate_outliers(left_x)
left_y = interpolate_outliers(left_y)
right_x = interpolate_outliers(right_x)
right_y = interpolate_outliers(right_y)

# Get frame numbers and time
frames = df.index
time = frames / 60  # 60 frames per second

# Compute vector from left to right
dx = right_x - left_x
dy = right_y - left_y

# Compute angle (in radians) relative to the horizontal axis (x-axis)
angles = np.arctan2(dy, dx)  # angle in radians

# --- Apply smoothing filter to the angle ---
window_size = 7  # You can adjust this (must be odd)
angles_smooth = pd.Series(angles).rolling(window=window_size, center=True, min_periods=1).mean().values

# Convert to degrees for easier interpretation
angles_deg = np.degrees(angles_smooth)

# Unwrap angles to avoid discontinuities
angles_unwrapped = np.unwrap(angles_smooth)

# --- Compute angular velocity over a window of half a second (30 frames) ---
window = 9  # half a second at 60 fps

# Use central difference over the window
angular_velocity = np.full_like(angles_unwrapped, np.nan)
for i in range(window, len(angles_unwrapped) - window):
    delta_angle = angles_unwrapped[i + window] - angles_unwrapped[i - window]
    delta_time = time[i + window] - time[i - window]
    angular_velocity[i] = delta_angle / delta_time  # radians per second

angular_velocity_deg = np.degrees(angular_velocity)  # convert to degrees per second

# Plot angle over time
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, angles_deg)
plt.ylabel('Angle (degrees)')
plt.title('Smoothed Angle between Left and Right Points Over Time')
plt.xlim((time[0], time[-1]))  # Ensure full x-axis

# Plot angular velocity over time
plt.subplot(2, 1, 2)
plt.plot(time, angular_velocity_deg)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (degrees/s)')
plt.title(f'Angular Velocity Over Time ({window/60}s window)')
plt.xlim((time[0], time[-1]))  # Ensure full x-axis

plt.tight_layout()
plt.savefig("angle_and_angular_velocity.png")  # Save the figure to a file in the current folder
# plt.show()  # Optionally comment this out if you don't want to display