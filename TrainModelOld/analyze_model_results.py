"""
Model Results Analysis Script
Loads saved models and generates comprehensive evaluation metrics

Usage: python3.12 analyze_model_results.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    accuracy_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import json
import csv
from datetime import datetime
import tensorflow as tf

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# Configuration
LANDMARK_PATH = "../gte9_landmarks"
MODEL_PATH = "../testing/model_comparison_results"
OUTPUT_PATH = "../testing/detailed_analysis"
os.makedirs(OUTPUT_PATH, exist_ok=True)

SEQUENCE_LENGTH = 70
NUM_LANDMARKS = 63
NUM_FEATURES = 4
FEATURE_DIM = NUM_LANDMARKS * NUM_FEATURES

print("\n" + "="*80)
print("MODEL RESULTS ANALYSIS")
print("="*80)

# ==================== LOAD DATA ====================

def load_test_data():
    """Load test data"""
    print("\n📂 Loading test data...")
    
    # Load preprocessed data
    X = np.load(f"{LANDMARK_PATH}/x.npy")
    y = np.load(f"{LANDMARK_PATH}/y.npy")
    y_encoded = np.load(f"{LANDMARK_PATH}/y_encoded.npy")
    y_onehot = np.load(f"{LANDMARK_PATH}/y_onehot.npy")
    
    # Reshape to proper format
    X = X.reshape(-1, SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # Split data (same split as training)
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=123, stratify=y_encoded
    )
    
    print(f"✅ Test data loaded: {X_test.shape[0]} samples")
    print(f"✅ Number of classes: {len(label_encoder.classes_)}")
    
    return X_test, y_test, label_encoder


def find_saved_models():
    MODEL_PATH="../testing/best_bigru"
    """Find all saved model files"""
    print(f"\n🔍 Searching for saved models in {MODEL_PATH}...")
    
    model_files = []
    for filename in os.listdir(MODEL_PATH):
        if filename.endswith('_best.keras'):
            model_path = os.path.join(MODEL_PATH, filename)
            model_name = filename.replace('_best.keras', '')
            model_files.append({
                'name': model_name,
                'path': model_path,
                'filename': filename
            })
    
    if not model_files:
        print("❌ No saved models found!")
        return []
    
    print(f"✅ Found {len(model_files)} saved models:")
    for m in model_files:
        print(f"   - {m['name']}")
    
    return model_files


# ==================== EVALUATION FUNCTIONS ====================

def evaluate_model_comprehensive(model, X_test, y_test, label_encoder, model_name):
    """Comprehensive evaluation of a single model"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    # Prepare data in correct shape
    if 'CNN' in model_name or 'Hybrid' in model_name:
        X_test_input = X_test
    else:
        X_test_input = X_test.reshape(-1, SEQUENCE_LENGTH, FEATURE_DIM)
    
    # Get predictions
    print("🔮 Generating predictions...")
    y_pred_probs = model.predict(X_test_input, verbose=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    print("📊 Calculating metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    
    # Top-k accuracy
    top3_acc = top_k_accuracy_score(y_true_labels, y_pred_probs, k=3)
    top5_acc = top_k_accuracy_score(y_true_labels, y_pred_probs, k=5)
    
    # Precision, Recall, F1-Score (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average=None, zero_division=0, labels=range(len(label_encoder.classes_))
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Loss
    loss = model.evaluate(X_test_input, y_test, verbose=0)[0]
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'loss': loss,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'per_class_metrics': {
            'classes': label_encoder.classes_.tolist(),
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        },
        'predictions': {
            'y_true': y_true_labels.tolist(),
            'y_pred': y_pred_labels.tolist(),
            'y_pred_probs': y_pred_probs.tolist()
        }
    }
    
    # Print summary
    print(f"\n📊 Results Summary:")
    print(f"   Accuracy:          {accuracy*100:.2f}%")
    print(f"   Top-3 Accuracy:    {top3_acc*100:.2f}%")
    print(f"   Top-5 Accuracy:    {top5_acc*100:.2f}%")
    print(f"   Loss:              {loss:.4f}")
    print(f"   Precision (macro): {precision_macro:.4f}")
    print(f"   Recall (macro):    {recall_macro:.4f}")
    print(f"   F1-Score (macro):  {f1_macro:.4f}")
    print(f"   Precision (weighted): {precision_weighted:.4f}")
    print(f"   Recall (weighted):    {recall_weighted:.4f}")
    print(f"   F1-Score (weighted):  {f1_weighted:.4f}")
    
    return results


# ==================== REPORTING FUNCTIONS ====================

def save_incremental_result(result, output_file="incremental_results.json"):
    """
    Save or append a single model result to JSON file incrementally
    This allows viewing results as models are evaluated
    """
    result_file = os.path.join(OUTPUT_PATH, output_file)
    
    # Prepare serializable result (convert numpy arrays)
    serializable_result = {
        'model_name': result['model_name'],
        'accuracy': float(result['accuracy']),
        'top3_accuracy': float(result['top3_accuracy']),
        'top5_accuracy': float(result['top5_accuracy']),
        'loss': float(result['loss']),
        'precision_macro': float(result['precision_macro']),
        'recall_macro': float(result['recall_macro']),
        'f1_macro': float(result['f1_macro']),
        'precision_weighted': float(result['precision_weighted']),
        'recall_weighted': float(result['recall_weighted']),
        'f1_weighted': float(result['f1_weighted']),
        'per_class_metrics': result['per_class_metrics'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Load existing results if file exists
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                existing_results = json.load(f)
        except:
            existing_results = []
    else:
        existing_results = []
    
    # Append new result
    existing_results.append(serializable_result)
    
    # Save updated results
    with open(result_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"✅ Results saved incrementally to: {result_file}")
    
    # Also save a simple text summary for quick viewing
    summary_file = os.path.join(OUTPUT_PATH, "incremental_summary.txt")
    with open(summary_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Model: {result['model_name']}\n")
        f.write(f"Evaluated: {serializable_result['timestamp']}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Accuracy:          {result['accuracy']*100:>6.2f}%\n")
        f.write(f"Top-3 Accuracy:    {result['top3_accuracy']*100:>6.2f}%\n")
        f.write(f"Top-5 Accuracy:    {result['top5_accuracy']*100:>6.2f}%\n")
        f.write(f"Loss:              {result['loss']:>8.4f}\n")
        f.write(f"Precision (macro): {result['precision_macro']:>8.4f}\n")
        f.write(f"Recall (macro):    {result['recall_macro']:>8.4f}\n")
        f.write(f"F1-Score (macro):  {result['f1_macro']:>8.4f}\n")
        f.write(f"Precision (weighted): {result['precision_weighted']:>8.4f}\n")
        f.write(f"Recall (weighted):    {result['recall_weighted']:>8.4f}\n")
        f.write(f"F1-Score (weighted):  {result['f1_weighted']:>8.4f}\n")
    
    print(f"✅ Summary appended to: {summary_file}")


def save_top5_predictions(result, label_encoder):
    """Save detailed Top-5 predictions per sample to CSV and JSON"""
    model_name = result['model_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    y_true_labels = np.array(result['predictions']['y_true'])
    y_pred_probs = np.array(result['predictions']['y_pred_probs'])
    
    # Get top-5 class indices for each sample
    top5_indices = np.argsort(y_pred_probs, axis=1)[:, -5:][:, ::-1]
    
    # Get top-5 probabilities
    top5_probs = np.take_along_axis(y_pred_probs, top5_indices, axis=1)
    
    # Convert indices to class names
    top5_class_names = label_encoder.inverse_transform(top5_indices.flatten()).reshape(-1, 5)
    
    # True class names
    true_class_names = label_encoder.inverse_transform(y_true_labels)
    
    # Prepare data for export
    results_list = []
    for i in range(len(y_true_labels)):
        result_row = {
            'sample_id': i,
            'true_class': true_class_names[i],
            'correct': true_class_names[i] in top5_class_names[i],
            'top1_class': top5_class_names[i][0],
            'top1_prob': float(top5_probs[i][0]),
            'top2_class': top5_class_names[i][1],
            'top2_prob': float(top5_probs[i][1]),
            'top3_class': top5_class_names[i][2],
            'top3_prob': float(top5_probs[i][2]),
            'top4_class': top5_class_names[i][3],
            'top4_prob': float(top5_probs[i][3]),
            'top5_class': top5_class_names[i][4],
            'top5_prob': float(top5_probs[i][4]),
        }
        results_list.append(result_row)
    
    # Save as CSV
    csv_file = f"{OUTPUT_PATH}/{model_name}_{timestamp}_top5_predictions.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)
    print(f"✅ Top-5 predictions saved to CSV: {csv_file}")
    
    # Save as JSON
    json_file = f"{OUTPUT_PATH}/{model_name}_{timestamp}_top5_predictions.json"
    with open(json_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"✅ Top-5 predictions saved to JSON: {json_file}")
    
    # Print summary statistics
    correct_top1 = sum(1 for r in results_list if r['top1_class'] == r['true_class'])
    correct_top5 = sum(1 for r in results_list if r['correct'])
    print(f"\n📊 Top-5 Analysis:")
    print(f"   Top-1 Correct: {correct_top1}/{len(results_list)} ({correct_top1/len(results_list)*100:.2f}%)")
    print(f"   Top-5 Correct: {correct_top5}/{len(results_list)} ({correct_top5/len(results_list)*100:.2f}%)")
    print(f"   Improvement: +{(correct_top5-correct_top1)/len(results_list)*100:.2f}% from Top-1 to Top-5")
    
    return results_list


def generate_detailed_report(all_results):
    """Generate comprehensive text report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{OUTPUT_PATH}/detailed_analysis_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MODEL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Models Analyzed: {len(all_results)}\n\n")
        
        # Overall comparison
        f.write("="*80 + "\n")
        f.write("OVERALL COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        # Sort by accuracy
        sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        
        f.write(f"{'Rank':<6} {'Model':<40} {'Acc':<8} {'Top-3':<8} {'Top-5':<8} {'F1':<8} {'Loss':<8}\n")
        f.write("-"*80 + "\n")
        
        for rank, result in enumerate(sorted_results, 1):
            f.write(f"{rank:<6} {result['model_name']:<40} "
                   f"{result['accuracy']*100:>6.2f}% "
                   f"{result['top3_accuracy']*100:>6.2f}% "
                   f"{result['top5_accuracy']*100:>6.2f}% "
                   f"{result['f1_weighted']:>6.4f} "
                   f"{result['loss']:>6.4f}\n")
        
        # Detailed per-model analysis
        f.write("\n\n" + "="*80 + "\n")
        f.write("DETAILED PER-MODEL ANALYSIS\n")
        f.write("="*80 + "\n")
        
        for result in sorted_results:
            f.write(f"\n\n{'='*80}\n")
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Overall Metrics:\n")
            f.write(f"  Accuracy:              {result['accuracy']*100:.2f}%\n")
            f.write(f"  Top-3 Accuracy:        {result['top3_accuracy']*100:.2f}%\n")
            f.write(f"  Top-5 Accuracy:        {result['top5_accuracy']*100:.2f}%\n")
            f.write(f"  Loss:                  {result['loss']:.4f}\n\n")
            
            f.write(f"Macro-averaged Metrics:\n")
            f.write(f"  Precision:             {result['precision_macro']:.4f}\n")
            f.write(f"  Recall:                {result['recall_macro']:.4f}\n")
            f.write(f"  F1-Score:              {result['f1_macro']:.4f}\n\n")
            
            f.write(f"Weighted-averaged Metrics:\n")
            f.write(f"  Precision:             {result['precision_weighted']:.4f}\n")
            f.write(f"  Recall:                {result['recall_weighted']:.4f}\n")
            f.write(f"  F1-Score:              {result['f1_weighted']:.4f}\n\n")
            
            # Best and worst performing classes
            per_class = result['per_class_metrics']
            f1_scores = np.array(per_class['f1_score'])
            classes = np.array(per_class['classes'])
            support = np.array(per_class['support'])
            
            # Filter out classes with no samples
            valid_idx = support > 0
            if np.sum(valid_idx) > 0:
                valid_f1 = f1_scores[valid_idx]
                valid_classes = classes[valid_idx]
                valid_support = support[valid_idx]
                
                # Best performing
                best_idx = np.argsort(valid_f1)[-5:][::-1]
                f.write(f"Top 5 Best Performing Classes:\n")
                for idx in best_idx:
                    f.write(f"  {valid_classes[idx]:<20} F1: {valid_f1[idx]:.4f} (n={int(valid_support[idx])})\n")
                
                f.write(f"\n")
                
                # Worst performing
                worst_idx = np.argsort(valid_f1)[:5]
                f.write(f"Top 5 Worst Performing Classes:\n")
                for idx in worst_idx:
                    f.write(f"  {valid_classes[idx]:<20} F1: {valid_f1[idx]:.4f} (n={int(valid_support[idx])})\n")
    
    print(f"\n✅ Detailed report saved to: {report_file}")
    return report_file


def save_json_results(all_results):
    """Save results as JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"{OUTPUT_PATH}/results_{timestamp}.json"
    
    # Remove numpy arrays for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {k: v for k, v in result.items() if k not in ['confusion_matrix', 'predictions']}
        json_results.append(json_result)
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✅ JSON results saved to: {json_file}")


def create_comparison_plots(all_results):
    """Create comparison visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sort by accuracy
    sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    model_names = [r['model_name'] for r in sorted_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    accuracies = [r['accuracy']*100 for r in sorted_results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    ax.barh(model_names, accuracies, color=colors)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(accuracies):
        ax.text(v + 0.5, i, f'{v:.2f}%', va='center')
    
    # 2. Top-k accuracy comparison
    ax = axes[0, 1]
    x = np.arange(len(model_names))
    width = 0.25
    acc = [r['accuracy']*100 for r in sorted_results]
    top3 = [r['top3_accuracy']*100 for r in sorted_results]
    top5 = [r['top5_accuracy']*100 for r in sorted_results]
    
    ax.bar(x - width, acc, width, label='Top-1', alpha=0.8)
    ax.bar(x, top3, width, label='Top-3', alpha=0.8)
    ax.bar(x + width, top5, width, label='Top-5', alpha=0.8)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Top-K Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name[:20] for name in model_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Precision, Recall, F1 comparison
    ax = axes[1, 0]
    x = np.arange(len(model_names))
    precision = [r['precision_weighted'] for r in sorted_results]
    recall = [r['recall_weighted'] for r in sorted_results]
    f1 = [r['f1_weighted'] for r in sorted_results]
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name[:20] for name in model_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Loss comparison
    ax = axes[1, 1]
    losses = [r['loss'] for r in sorted_results]
    ax.barh(model_names, losses, color=colors)
    ax.set_xlabel('Loss', fontsize=12)
    ax.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(losses):
        ax.text(v + 0.05, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plot_file = f"{OUTPUT_PATH}/comparison_plots_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plots saved to: {plot_file}")


# ==================== MAIN EXECUTION ====================

def main():
    """Main analysis function"""
    
    # Load test data
    X_test, y_test, label_encoder = load_test_data()
    
    # Find saved models
    model_files = find_saved_models()
    
    if not model_files:
        print("\n❌ No models to analyze. Train models first!")
        return
    
    # Clear previous incremental files
    incremental_json = os.path.join(OUTPUT_PATH, "incremental_results.json")
    incremental_txt = os.path.join(OUTPUT_PATH, "incremental_summary.txt")
    
    if os.path.exists(incremental_json):
        os.remove(incremental_json)
        print(f"🗑️  Cleared previous incremental JSON file")
    
    if os.path.exists(incremental_txt):
        os.remove(incremental_txt)
        print(f"🗑️  Cleared previous incremental summary file")
    
    # Write header to summary file
    with open(incremental_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL ANALYSIS - INCREMENTAL RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models to analyze: {len(model_files)}\n")
    
    # Analyze each model
    all_results = []
    
    for model_info in model_files:
        try:
            print(f"\n{'='*80}")
            print(f"Loading model: {model_info['name']}")
            print(f"{'='*80}")
            
            # Load model with custom objects for Lambda layers
            if 'LSTM_Attention' in model_info['name']:
                # Custom Lambda function for LSTM_Attention
                custom_objects = {
                    'reduce_sum': lambda x: tf.reduce_sum(x, axis=1)
                }
                model = tf.keras.models.load_model(model_info['path'], 
                                                   custom_objects=custom_objects,
                                                   compile=False)
                # Recompile with proper metrics
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model = tf.keras.models.load_model(model_info['path'])
            
            # Evaluate
            results = evaluate_model_comprehensive(
                model, X_test, y_test, label_encoder, model_info['name']
            )
            
            all_results.append(results)
            
            # Save results incrementally
            save_incremental_result(results)
            
            # Save Top-5 predictions
            save_top5_predictions(results, label_encoder)
            
            # Clean up
            del model
            
        except Exception as e:
            print(f"❌ Error analyzing {model_info['name']}: {e}")
            print(f"   Skipping this model...")
            continue
    
    if not all_results:
        print("\n❌ No models were successfully analyzed!")
        return
    
    # Generate reports
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    generate_detailed_report(all_results)
    save_json_results(all_results)
    create_comparison_plots(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   Models analyzed: {len(all_results)}")
    
    best_model = max(all_results, key=lambda x: x['accuracy'])
    print(f"\n🏆 Best Model: {best_model['model_name']}")
    print(f"   Accuracy: {best_model['accuracy']*100:.2f}%")
    print(f"   F1-Score: {best_model['f1_weighted']:.4f}")
    
    print(f"\n📁 All results saved to: {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()
