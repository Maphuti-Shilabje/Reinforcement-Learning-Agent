# plot_results_S1_only.py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_s1_learning_curves(data_to_plot, plot_title_prefix="", output_dir="results/scenario1", base_filename="s1_comparison"):
    """Plots learning curves specifically for Scenario 1 data."""
    plt.figure(figsize=(16, 6)) # Adjusted figure size slightly

    # Plot Rewards
    plt.subplot(1, 2, 1)
    for strategy_name, data in data_to_plot.items():
        if 'avg_rewards' in data and data['avg_rewards'] is not None:
            # Smooth the curve for better readability
            window_size = 50 # Adjust if needed
            if len(data['avg_rewards']) >= window_size:
                smoothed_rewards = np.convolve(data['avg_rewards'], np.ones(window_size)/window_size, mode='valid')
                epochs_axis = np.arange(len(smoothed_rewards)) + window_size // 2 # Adjust axis for smoothed data
            else:
                smoothed_rewards = data['avg_rewards'] # Not enough data to smooth
                epochs_axis = np.arange(len(smoothed_rewards))
            
            plt.plot(epochs_axis, smoothed_rewards, label=f"{strategy_name} Rewards")

            if 'std_rewards' in data and data['std_rewards'] is not None:
                if len(data['std_rewards']) >= window_size:
                    smoothed_std_rewards = np.convolve(data['std_rewards'], np.ones(window_size)/window_size, mode='valid')
                else:
                    smoothed_std_rewards = data['std_rewards']
                
                # Ensure smoothed_rewards and smoothed_std_rewards have compatible lengths for fill_between
                len_to_use = min(len(smoothed_rewards), len(smoothed_std_rewards))
                plt.fill_between(epochs_axis[:len_to_use],
                                 smoothed_rewards[:len_to_use] - smoothed_std_rewards[:len_to_use],
                                 smoothed_rewards[:len_to_use] + smoothed_std_rewards[:len_to_use],
                                 alpha=0.2)
    plt.xlabel("Epochs")
    plt.ylabel("Average Reward (Smoothed)")
    plt.title(f"{plot_title_prefix}Learning Curve (Avg Rewards)")
    plt.legend()
    plt.grid(True)

    # Plot Steps
    plt.subplot(1, 2, 2)
    for strategy_name, data in data_to_plot.items():
        if 'avg_steps' in data and data['avg_steps'] is not None:
            window_size = 50
            if len(data['avg_steps']) >= window_size:
                smoothed_steps = np.convolve(data['avg_steps'], np.ones(window_size)/window_size, mode='valid')
                epochs_axis = np.arange(len(smoothed_steps)) + window_size // 2
            else:
                smoothed_steps = data['avg_steps']
                epochs_axis = np.arange(len(smoothed_steps))

            plt.plot(epochs_axis, smoothed_steps, label=f"{strategy_name} Steps")

            if 'std_steps' in data and data['std_steps'] is not None:
                if len(data['std_steps']) >= window_size:
                    smoothed_std_steps = np.convolve(data['std_steps'], np.ones(window_size)/window_size, mode='valid')
                else:
                    smoothed_std_steps = data['std_steps']

                len_to_use = min(len(smoothed_steps), len(smoothed_std_steps))
                plt.fill_between(epochs_axis[:len_to_use],
                                 smoothed_steps[:len_to_use] - smoothed_std_steps[:len_to_use],
                                 smoothed_steps[:len_to_use] + smoothed_std_steps[:len_to_use],
                                 alpha=0.2)
    plt.xlabel("Epochs")
    plt.ylabel("Average Steps per Epoch (Smoothed)")
    plt.title(f"{plot_title_prefix}Learning Curve (Avg Steps)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    plot_output_path = os.path.join(output_dir, f"{base_filename}.png")
    os.makedirs(output_dir, exist_ok=True) 
    plt.savefig(plot_output_path)
    print(f"Plot saved to {plot_output_path}")
    plt.show() # Show plot interactively

def load_s1_data(strategies, num_runs, results_data_dir, is_stochastic):
    """Loads data specifically for Scenario 1."""
    data_collection = {}
    stochastic_suffix_file = "_stochastic" if is_stochastic else ""
    scenario_prefix = "s1" # Hardcoded for S1
    print(f"\nAttempting to load S1 data: Stochastic: {is_stochastic}, Runs: {num_runs}, Dir: {results_data_dir}")

    for strategy in strategies:
        base_filename_data = f"{scenario_prefix}_{strategy}{stochastic_suffix_file}_over_{num_runs}_runs"
        # print(f"  Strategy: {strategy}, Base Filename Stem: {base_filename_data}") # Debug

        avg_rewards_file = os.path.join(results_data_dir, f"{base_filename_data}_avg_rewards.npy")
        avg_steps_file = os.path.join(results_data_dir, f"{base_filename_data}_avg_steps.npy")
        std_rewards_file = os.path.join(results_data_dir, f"{base_filename_data}_std_rewards.npy")
        std_steps_file = os.path.join(results_data_dir, f"{base_filename_data}_std_steps.npy")

        if os.path.exists(avg_rewards_file) and os.path.exists(avg_steps_file):
            strategy_label = f"{strategy.capitalize()} Decay"
            data_collection[strategy_label] = {
                "avg_rewards": np.load(avg_rewards_file),
                "avg_steps": np.load(avg_steps_file),
                "std_rewards": np.load(std_rewards_file) if os.path.exists(std_rewards_file) else None,
                "std_steps": np.load(std_steps_file) if os.path.exists(std_steps_file) else None
            }
            print(f"    Successfully loaded S1 data for {strategy_label} (Stochastic: {is_stochastic}).")
        else:
            print(f"    COULD NOT LOAD S1 data for {strategy.capitalize()} Decay (Stochastic: {is_stochastic}). Files not found (e.g., {avg_rewards_file})")
            
    return data_collection

if __name__ == "__main__":
    S1_NUM_RUNS = 3  # Ensure this matches the --runs used for Scenario1.py
    S1_RESULTS_DIR = "results/scenario1"
    S1_STRATEGIES = ["multiplicative", "linear"]

    print("--- Generating Plots for Scenario 1 (Deterministic) ---")
    s1_data_det = load_s1_data(S1_STRATEGIES, S1_NUM_RUNS, S1_RESULTS_DIR, is_stochastic=False)
    if s1_data_det:
        plot_s1_learning_curves(s1_data_det, 
                             plot_title_prefix="Scenario 1 (Deterministic): ", 
                             output_dir=S1_RESULTS_DIR, 
                             base_filename="s1_deterministic_comparison")
    else:
        print("No deterministic data found for Scenario 1 to plot.")

    print("\n--- Generating Plots for Scenario 1 (Stochastic) ---")
    s1_data_sto = load_s1_data(S1_STRATEGIES, S1_NUM_RUNS, S1_RESULTS_DIR, is_stochastic=True)
    if s1_data_sto:
        plot_s1_learning_curves(s1_data_sto,
                             plot_title_prefix="Scenario 1 (Stochastic): ",
                             output_dir=S1_RESULTS_DIR,
                             base_filename="s1_stochastic_comparison")
    else:
        print("No stochastic data found for Scenario 1 to plot.")
    
    print("\nPlotting script finished.")