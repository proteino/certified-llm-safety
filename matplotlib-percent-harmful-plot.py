import matplotlib.pyplot as plt
import json
import sys

def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def process_data(data):
    numeric_keys = [key for key in data.keys() if key.isdigit()]
    iterations = sorted(map(int, numeric_keys))
    sequence_lengths = sorted(map(int, data[str(iterations[0])].keys()))
    
    processed_data = {
        iteration: [data[str(iteration)][str(length)]["percent_harmful"] 
                    for length in sequence_lengths]
        for iteration in iterations
    }
    
    return iterations, sequence_lengths, processed_data

def create_plot(iterations, sequence_lengths, data):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, iteration in enumerate(iterations):
        ax.plot(sequence_lengths, data[iteration], marker='o', linestyle='-', 
                color=colors[i % len(colors)], label=f'{iteration} Iterations')

    ax.set_xlabel('Adversarial Sequence Length (in tokens)')
    ax.set_ylabel('Percent Harmful')
    ax.set_title('Percent Harmful vs Adversarial Sequence Length')
    ax.legend(title='# Iterations')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    raw_data = load_data(filename)
    iterations, sequence_lengths, processed_data = process_data(raw_data)
    plot = create_plot(iterations, sequence_lengths, processed_data)
    
    # You can choose to show the plot
    plt.show()
    
    # Or save it to a file
    # plot.savefig('percent_harmful_plot.png')

if __name__ == "__main__":
    main()