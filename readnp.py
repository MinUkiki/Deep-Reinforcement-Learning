# import numpy as np
# import matplotlib.pyplot as plt

# # File path from which to load Q-table
# file_path = 'q_table.npy'

# # Load Q-table
# q_table_loaded = np.load(file_path)

# print(f"Q-table loaded from {file_path}")

# def plot_q_values(q_table, angle_idx, velocity_idx):
#     actions = np.linspace(-2.0, 2.0, len(q_table[angle_idx, velocity_idx]))
    
#     # Adjusting bar width to add space between bars
#     bar_width = 0.1  # Arbitrary value for bar width
#     plt.bar(actions, q_table[angle_idx, velocity_idx], width=bar_width)
    
#     plt.xlabel('Actions')
#     plt.ylabel('Q-values')
#     plt.title(f'Q-values for State (Angle Index: {angle_idx}, Velocity Index: {velocity_idx})')
#     plt.show()

# # Example usage: plot Q-values for a specific state
# num = np.random.randint(0, 17)
# plot_q_values(q_table_loaded, num, num)

# file_path = 'q_table.npy'
# np.save(file_path, q_table)
# print(f"Q-table saved to {file_path}")

# import numpy as np
# import matplotlib.pyplot as plt

# # File path from which to load Q-table
# file_path = 'q_table.npy'

# # Load Q-table
# q_table_loaded = np.load(file_path)

# print(f"Q-table loaded from {file_path}")

# def plot_q_values(q_table):
#     angle_num_bins, velocity_num_bins, action_num_bins = q_table.shape
#     actions = np.linspace(-2.0, 2.0, action_num_bins)
#     bar_width = 0.2  # Arbitrary value for bar width

#     fig, axs = plt.subplots(angle_num_bins, velocity_num_bins, figsize=(20, 20))

#     for i in range(angle_num_bins):
#         for j in range(velocity_num_bins):
#             axs[i, j].bar(actions, q_table[i, j], width=bar_width)
#             axs[i, j].set_xlabel('Actions')
#             axs[i, j].set_ylabel('Q-values')
#             axs[i, j].set_title(f'Q-values for State ({i}, {j})')
    
#     plt.tight_layout()
#     plt.show()

# # Plot Q-values for all states and actions
# plot_q_values(q_table_loaded)