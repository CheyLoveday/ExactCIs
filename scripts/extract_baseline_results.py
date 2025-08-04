#!/usr/bin/env python3
"""
Extract baseline benchmark results from the output file and save them in JSON format.
"""

import re
import json
from pathlib import Path

def extract_results(output_file):
    """Extract benchmark results from the output file."""
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Define patterns to extract information
    table_pattern = r"Benchmarking (\w+) table: \((\d+), (\d+), (\d+), (\d+)\)"
    avg_time_pattern = r"Average time: (\d+\.\d+)ms"
    min_time_pattern = r"Min time: (\d+\.\d+)ms"
    max_time_pattern = r"Max time: (\d+\.\d+)ms"
    std_dev_pattern = r"Std deviation: (\d+\.\d+)ms"
    
    # Find all table sections
    table_sections = re.findall(r"Benchmarking.*?Std deviation: \d+\.\d+ms", content, re.DOTALL)
    
    results = []
    
    for section in table_sections:
        # Extract table info
        table_match = re.search(table_pattern, section)
        if not table_match:
            continue
        
        name = table_match.group(1)
        a, b, c, d = map(int, table_match.groups()[1:])
        
        # Extract timing info
        avg_match = re.search(avg_time_pattern, section)
        min_match = re.search(min_time_pattern, section)
        max_match = re.search(max_time_pattern, section)
        std_match = re.search(std_dev_pattern, section)
        
        if not all([avg_match, min_match, max_match, std_match]):
            continue
        
        avg_time = float(avg_match.group(1))
        min_time = float(min_match.group(1))
        max_time = float(max_match.group(1))
        std_dev = float(std_match.group(1))
        
        # Create result entry
        result = {
            'name': name,
            'table': [a, b, c, d],
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_dev': std_dev,
            # Placeholder for individual times and results
            'times': [avg_time] * 5,  # Approximate with 5 identical times
            'results': [[0.0, 0.0]] * 5  # Placeholder for results
        }
        
        results.append(result)
    
    return results

def main():
    """Extract and save baseline results."""
    output_file = Path("baseline_output.txt")
    if not output_file.exists():
        print(f"Error: {output_file} not found")
        return
    
    results = extract_results(output_file)
    
    if not results:
        print("Error: No results extracted")
        return
    
    # Save results
    with open("baseline_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Extracted {len(results)} benchmark results and saved to baseline_benchmark_results.json")
    
    # Print summary
    print("\nBaseline Results Summary:")
    print(f"{'Table Type':<20} {'Avg Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<20} {result['avg_time']:<15.2f} {result['min_time']:<15.2f} {result['max_time']:<15.2f}")

if __name__ == "__main__":
    main()