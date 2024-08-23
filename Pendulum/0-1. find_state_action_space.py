import numpy as np

angle_num_bins = 17
velocity_num_bins = 17
action_num_bin = 11
angle_bins = np.linspace(-np.pi, np.pi, angle_num_bins)
velocity_bins = np.linspace(-8.0, 8.0, velocity_num_bins)
action_bins = np.linspace(-2.0, 2.0, action_num_bin)