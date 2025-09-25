import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import ImprovedForceClassificationEnsemble, AdvancedTextPreprocessor, prepare_data

def evaluate_model():
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = ImprovedForceClassificationEnsemble(num_labels=6)
    model.load_state_dict(torch.load("./uof_model/pytorch_model.bin"))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("./uof_model")
    preprocessor = AdvancedTextPreprocessor()
    
    # Load and preprocess test data
    print("Loading and preprocessing test data...")
    test_df = pd.read_csv('test_set.csv')
    test_df = prepare_data(test_df)  # Apply same preprocessing as training
    
    # Prepare test data with context
    texts = test_df['speech_turn'].tolist()
    contexts = [f"{prev} {curr} {next_}" for prev, curr, next_ in 
               zip(test_df['prev_text'], test_df['speech_turn'], test_df['next_text'])]
    labels = test_df['use_of_force_level'].tolist()
    
    # Tokenize and predict
    print("Running predictions...")
    predictions = []
    confidence_scores = []
    
    with torch.no_grad():
        for text, context in zip(texts, contexts):
            # Tokenize main text and context
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            context_inputs = tokenizer(
                context,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Get model predictions
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                context_ids=context_inputs['input_ids'],
                context_mask=context_inputs['attention_mask']
            )
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            conf = torch.max(probs).item()
            
            predictions.append(pred)
            confidence_scores.append(conf)
    
    # Calculate and display detailed metrics
    print("\nDetailed Classification Report:")
    print(classification_report(labels, predictions, target_names=[
        'Level 0 (Non-force)', 'Level 1 (Verbal)', 'Level 2 (Soft)',
        'Level 3 (Hard)', 'Level 4 (Less Lethal)', 'Level 5 (Lethal)'
    ], digits=4))
    
    # Create enhanced confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
                yticklabels=['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'])
    plt.title('Enhanced Confusion Matrix on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('test_confusion_matrix.png')
    plt.close()
    
    # Calculate and display detailed per-class metrics
    print("\nDetailed Per-class Analysis:")
    for i in range(6):
        class_indices = [j for j, label in enumerate(labels) if label == i]
        if class_indices:
            class_predictions = [predictions[j] for j in class_indices]
            class_confidences = [confidence_scores[j] for j in class_indices]
            
            class_accuracy = sum(1 for p in class_predictions if p == i) / len(class_predictions)
            avg_confidence = sum(class_confidences) / len(class_confidences)
            
            print(f"\nLevel {i}:")
            print(f"  Accuracy: {class_accuracy:.2%}")
            print(f"  Avg Confidence: {avg_confidence:.2%}")
            print(f"  Sample Size: {len(class_indices)}")
            
            # Analyze error patterns
            errors = [(pred, conf) for pred, conf in zip(class_predictions, class_confidences) if pred != i]
            if errors:
                print("  Error Analysis:")
                error_counts = {}
                for pred, conf in errors:
                    error_counts[pred] = error_counts.get(pred, 0) + 1
                for pred, count in sorted(error_counts.items()):
                    print(f"    Misclassified as Level {pred}: {count} times ({count/len(class_indices):.1%})")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Text': texts,
        'True_Label': labels,
        'Predicted_Label': predictions,
        'Confidence': confidence_scores
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nDetailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    evaluate_model() 