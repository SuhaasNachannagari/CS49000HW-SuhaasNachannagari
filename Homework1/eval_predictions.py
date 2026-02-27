import json

def evaluate(prediction_path, ground_truth_path):
    predictions = []
    with open(prediction_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
            
    ground_truth = []
    with open(ground_truth_path, 'r') as f:
        for line in f:
            ground_truth.append(json.loads(line))
            
    if len(predictions) != len(ground_truth):
        print(f"Error: Number of predictions ({len(predictions)}) does not match number of ground truth samples ({len(ground_truth)})")
        return

    correct = 0
    total = len(predictions)
    
    for p, g in zip(predictions, ground_truth):
        if p['prediction'] == g['is_sarcastic']:
            correct += 1
            
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    if accuracy >= 0.70:
        print("Success! Your model achieved > 70% accuracy.")
    else:
        print("Keep trying! You need > 70% accuracy.")

if __name__ == "__main__":
    evaluate('prediction.jsonl', 'valid.jsonl')
