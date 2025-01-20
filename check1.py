import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    # Load the dataset
    dataset_path = r"C:\Users\SAKTHIVEL\OneDrive\Documents\check\synthetic_telecom_data_with_anomalies.csv"
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset not found. Please check the file path and try again.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: The dataset file is empty. Please provide a valid dataset.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset: {e}")
    exit()

# Verify required columns
required_columns = ['Time_of_Day', 'BS ID', 'Location', 'Mode', 'Power', 'Active Users',
                    'Throughput', 'Traffic', 'Bandwidth', 'Latency', 'Cooling Power',
                    'SNR', 'Ambient Temp', 'Signal Strength']

missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Error: Missing columns in dataset - {missing_columns}. Please ensure all required columns are present.")
    exit()

# Initialize Q-learning parameters
try:
    n_states = len(data)  # Each row is treated as a state
    n_actions = 3  # Possible actions: decrease, maintain, increase power
    q_table = np.zeros((n_states, n_actions))

    learning_rate = 0.1
    discount_factor = 0.9
    exploration_rate = 1.0
    exploration_decay = 0.99
    min_exploration_rate = 0.01
    n_episodes = 1000
    max_steps_per_episode = 100

    # Map actions
    actions = {
        0: -1,  # Decrease power
        1: 0,   # Maintain power
        2: 1    # Increase power
    }

    # Track energy consumption before and after optimization
    total_energy_before = data['Power'].sum()
    cumulative_q_score = 0

    # Define reward function
    def calculate_reward(row, action):
        current_power = row['Power']
        reward = 0
        if action == 0 and current_power > 0:  # Decrease power
            reward = 10
        elif action == 1:  # Maintain power
            reward = 5
        elif action == 2:  # Increase power
            reward = -10  # Penalize increase
        return reward

    # Q-learning algorithm
    for episode in range(n_episodes):
        state = np.random.randint(0, n_states)  # Start from a random state
        total_reward = 0  # Track total reward for the episode

        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            if np.random.uniform(0, 1) < exploration_rate:
                action = np.random.randint(0, n_actions)  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit

            # Take action and observe new state and reward
            next_state = (state + 1) % n_states
            reward = calculate_reward(data.iloc[state], action)
            total_reward += reward

            # Update Q-table
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action]
            )

            # Accumulate Q-value
            cumulative_q_score += np.sum(q_table[state, :])

            state = next_state

        # Decay exploration rate
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

        # Log the total reward at the end of the episode
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Apply learned policy to optimize power
    for index, row in data.iterrows():
        state = index
        best_action = np.argmax(q_table[state, :])
        data.at[index, 'Power'] += actions[best_action]  # Adjust power based on the best action

    total_energy_after = data['Power'].sum()

    # Output results
    print(f"\nTotal Energy Consumption Before Optimization: {total_energy_before:.2f}")
    print(f"Total Energy Consumption After Optimization: {total_energy_after:.2f}")
    print(f"Cumulative Q-Score (Sum of all Q-values): {cumulative_q_score:.2f}")

    # Display Q-table as a DataFrame
    q_table_df = pd.DataFrame(q_table, columns=['Action: Decrease', 'Action: Maintain', 'Action: Increase'])
    print("\nQ-Table (Final State-Action Values):")
    print(q_table_df)

    # Save Q-table as a CSV file for further analysis
    q_table_df.to_csv("q_table.csv", index_label="State")
    print("\nQ-Table has been saved as 'q_table.csv'.")

    # Plot energy consumption
    try:
        plt.bar(['Before Optimization', 'After Optimization'], [total_energy_before, total_energy_after], color=['blue', 'green'])
        plt.title('Energy Consumption Comparison')
        plt.ylabel('Energy (Units)')
        plt.yticks(np.arange(0, max(total_energy_before, total_energy_after) + 10000, 10000))  # Set y-axis scale to increments of 10,000
        plt.show()
    except Exception as e:
        print(f"An error occurred while generating the graph: {e}")

except KeyError as ke:
    print(f"KeyError: Missing required column in dataset: {ke}")
except IndexError as ie:
    print(f"IndexError: An index operation failed: {ie}")
except Exception as e:
    print(f"An unexpected error occurred during Q-learning: {e}")
