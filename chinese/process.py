import json
import csv

# Path to your JSONL file and output CSV file
input_file_path = 'all.jsonl'
output_file_path = 'final.csv'

with open(input_file_path, 'r', encoding='utf-8') as jsonl_file, \
     open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    
    # Prepare CSV writer
    csv_writer = csv.writer(csv_file)
    # Write headers to CSV file
    csv_writer.writerow(['text', 'label'])
    
    # Process each line in the JSONL file
    for line in jsonl_file:
        # Parse the JSON line
        data = json.loads(line)
        
        # Process human answers
        for answer in data.get('human_answers', []):  # Using .get to avoid KeyError if key doesn't exist
            if answer is not None:  # Check if the answer is not None
                clean_answer = answer.replace('\r', '').replace('\n', '')
                csv_writer.writerow([clean_answer, 1])  # Label 1 for human answers
        
        # Process ChatGPT answers
        for answer in data.get('chatgpt_answers', []):  # Using .get to avoid KeyError if key doesn't exist
            if answer is not None:  # Check if the answer is not None
                clean_answer = answer.replace('\r', '').replace('\n', '')
                csv_writer.writerow([clean_answer, 0])
