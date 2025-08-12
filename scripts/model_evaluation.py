#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detector import EnsembleAnomalyDetector, IsolationForestDetector, AutoencoderDetector, LSTMAnomalyDetector, generate_synthetic_transactions

class ModelEvaluator:
    def __init__(self):
        self.ensemble = EnsembleAnomalyDetector()
        self.results = {}
    
    def prepare_test_data(self, n_samples: int = 2000):
        """Prepare test dataset with known anomalies"""
        print("Generating test data...")
        
        # Generate normal transactions
        normal_transactions = generate_synthetic_transactions(int(n_samples * 0.8))
        
        # Generate anomalous transactions
        anomalous_transactions = []
        for i in range(int(n_samples * 0.2)):
            transaction = generate_synthetic_transactions(1)[0]
            
            # Make it anomalous
            if random.random() < 0.5:
                transaction.amount *= random.uniform(5, 20)  # Unusually high amount
            else:
                transaction.hour_of_day = random.choice([2, 3, 4])  # Night transactions
                transaction.merchant = "Suspicious_Merchant"
            
            anomalous_transactions.append(transaction)
        
        # Combine and create labels
        all_transactions = normal_transactions + anomalous_transactions
        labels = [0] * len(normal_transactions) + [1] * len(anomalous_transactions)
        
        return all_transactions, labels
    
    def evaluate_ensemble(self, transactions, true_labels):
        """Evaluate ensemble model performance"""
        print("Evaluating ensemble model...")
        
        predictions = []
        scores = []
        
        for transaction in transactions:
            result = self.ensemble.predict(transaction)
            predictions.append(1 if result.anomaly_type.value != "normal" else 0)
            scores.append(result.anomaly_score)
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        auc_score = roc_auc_score(true_labels, scores)
        
        self.results['ensemble'] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'predictions': predictions,
            'scores': scores,
            'classification_report': classification_report(true_labels, predictions)
        }
        
        print(f"Ensemble Accuracy: {accuracy:.3f}")
        print(f"Ensemble AUC: {auc_score:.3f}")
    
    def plot_results(self, true_labels):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ROC Curve
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(true_labels, results['scores'])
            axes[0, 0].plot(fpr, tpr, label=f"{model_name} (AUC = {results['auc']:.3f})")
        
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend()
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, self.results['ensemble']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # Score Distribution
        scores = self.results['ensemble']['scores']
        normal_scores = [scores[i] for i, label in enumerate(true_labels) if label == 0]
        anomaly_scores = [scores[i] for i, label in enumerate(true_labels) if label == 1]
        
        axes[1, 0].hist(normal_scores, alpha=0.7, label='Normal', bins=30)
        axes[1, 0].hist(anomaly_scores, alpha=0.7, label='Anomaly', bins=30)
        axes[1, 0].set_xlabel('Anomaly Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        
        # Accuracy Comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        axes[1, 1].bar(model_names, accuracies)
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Model Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        print("Results saved to model_evaluation_results.png")
    
    def run_evaluation(self):
        """Run complete model evaluation"""
        # Prepare models
        isolation_forest = IsolationForestDetector()
        autoencoder = AutoencoderDetector(input_dim=15)
        lstm_detector = LSTMAnomalyDetector()
        
        self.ensemble.add_model("isolation_forest", isolation_forest)
        self.ensemble.add_model("autoencoder", autoencoder)
        self.ensemble.add_model("lstm", lstm_detector)
        
        # Generate training and test data
        train_transactions = generate_synthetic_transactions(5000)
        test_transactions, true_labels = self.prepare_test_data(2000)
        
        # Train models
        self.ensemble.train_all_models(train_transactions)
        
        # Evaluate
        self.evaluate_ensemble(test_transactions, true_labels)
        
        # Plot results
        self.plot_results(true_labels)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(self.results['ensemble']['classification_report'])

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
