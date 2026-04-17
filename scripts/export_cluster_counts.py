import json
import csv
import os

def export_counts():
    input_file = 'scripts/cluster_to_patterns.json'
    output_file = 'cluster_counts.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Sort by count descending
    sorted_data = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cluster_ID', 'Pattern_Count'])
        for cluster_id, patterns in sorted_data:
            writer.writerow([cluster_id, len(patterns)])

    print(f"Successfully created {output_file} with {len(data)} clusters.")

if __name__ == "__main__":
    export_counts()
