# -*- coding: utf-8 -*-
"""
Robust Road Accident Severity Prediction Model
Training Script with Error Handling and Model Persistence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, f1_score
)

warnings.filterwarnings('ignore')

class AccidentSeverityModel:
    """
    Production-ready model for accident severity prediction
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.feature_names = None
        self.threshold = None
        self.metrics = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        try:
            print(f"Loading data from {self.data_path}...")
            df = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(df)} records")
            
            # Drop unnecessary columns
            cols_to_drop = ['Severity_Binary', 'ID', 'Start_Lat', 'Start_Lng', 
                           'Unnamed: 0', 'Distance(mi)']
            cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            
            X = df.drop(columns=cols_to_drop)
            y = df['Severity_Binary']
            
            # Handle missing values
            if y.isnull().any():
                print(f"⚠ Found {y.isnull().sum()} null values in target. Dropping...")
                X = X[~y.isnull()]
                y = y.dropna()
            
            # One-hot encoding
            cat_cols = X.select_dtypes(include='object').columns.tolist()
            print(f"✓ Found {len(cat_cols)} categorical columns")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
            # Fill any remaining nulls
            if X.isnull().any().any():
                print("⚠ Filling remaining null values...")
                X = X.fillna(X.median())
            
            print(f"✓ Final feature shape: {X.shape}")
            print(f"✓ Class distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error in data preprocessing: {str(e)}")
            raise
    
    def train_model(self, X, y):
        """Train Random Forest model with optimal parameters"""
        try:
            print("\n" + "="*50)
            print("TRAINING MODEL")
            print("="*50)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            
            print(f"✓ Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Train model
            print("\n Training Random Forest...")
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                min_samples_leaf=50,
                n_jobs=-1,
                class_weight='balanced',
                verbose=0
            )
            
            self.model.fit(X_train, y_train)
            print("✓ Model training completed")
            
            # Evaluate
            self._evaluate_model(X_test, y_test)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f" Error training model: {str(e)}")
            raise
    
    def _evaluate_model(self, X_test, y_test):
        """Evaluate model and find optimal threshold"""
        try:
            print("\n Evaluating model...")
            
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            # Base metrics
            self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
            self.metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            self.metrics['f1_score'] = f1_score(y_test, y_pred)
            
            print(f"✓ Base Accuracy: {self.metrics['accuracy']:.4f}")
            print(f"✓ ROC-AUC: {self.metrics['roc_auc']:.4f}")
            print(f"✓ F1 Score: {self.metrics['f1_score']:.4f}")
            
            # Find optimal threshold
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            
            pr_df = pd.DataFrame({
                'threshold': thresholds,
                'precision': precision[:-1],
                'recall': recall[:-1]
            })
            
            # Find best threshold (recall >= 0.70, precision >= 0.50)
            best_zone = pr_df[
                (pr_df['recall'] >= 0.70) & 
                (pr_df['precision'] >= 0.50)
            ]
            
            if not best_zone.empty:
                # Choose threshold with best F1 score
                best_zone['f1'] = 2 * (best_zone['precision'] * best_zone['recall']) / \
                                 (best_zone['precision'] + best_zone['recall'])
                self.threshold = best_zone.loc[best_zone['f1'].idxmax(), 'threshold']
            else:
                print(" No threshold meets criteria. Using default 0.46")
                self.threshold = 0.46
            
            print(f"✓ Optimal threshold: {self.threshold:.4f}")
            
            # Evaluate with optimal threshold
            y_pred_tuned = (y_prob >= self.threshold).astype(int)
            
            self.metrics['tuned_accuracy'] = accuracy_score(y_test, y_pred_tuned)
            self.metrics['tuned_f1'] = f1_score(y_test, y_pred_tuned)
            self.metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_tuned)
            
            print(f"\n✓ Tuned Accuracy: {self.metrics['tuned_accuracy']:.4f}")
            print(f"✓ Tuned F1 Score: {self.metrics['tuned_f1']:.4f}")
            print(f"\nConfusion Matrix:\n{self.metrics['confusion_matrix']}")
            
        except Exception as e:
            print(f"❌ Error evaluating model: {str(e)}")
            raise
    
    def get_feature_importance(self, top_n=20):
        """Get top N feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        return importances.head(top_n)
    
    def save_model(self, output_dir='Trained_Models'):
        """Save trained model and metadata"""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{output_dir}/accident_model_{timestamp}.pkl"
            
            # Save model and metadata
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'threshold': self.threshold,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"\n Model saved to {model_path}")
            
            # Save feature importance
            importance_path = f"{output_dir}/feature_importance_{timestamp}.csv"
            self.get_feature_importance().to_csv(importance_path)
            print(f" Feature importance saved to {importance_path}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            raise

def main():
    """Main training pipeline"""
    try:
        print("\n" + "="*60)
        print("ROAD ACCIDENT SEVERITY PREDICTION - TRAINING PIPELINE")
        print("="*60)
        
        # Initialize model
        data_path = "Data/accident_clean_data.csv"
        
        model = AccidentSeverityModel(data_path)
        
        # Load and preprocess data
        X, y = model.load_and_preprocess_data()
        
        # Train model
        X_train, X_test, y_train, y_test = model.train_model(X, y)
        
        # Display feature importance
        print("\n" + "="*50)
        print("TOP 20 MOST IMPORTANT FEATURES")
        print("="*50)
        print(model.get_feature_importance())
        
        # Save model
        model_path = model.save_model()
        
        print("\n" + "="*60)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n Final Metrics:")
        print(f"   - Accuracy: {model.metrics['tuned_accuracy']:.4f}")
        print(f"   - F1 Score: {model.metrics['tuned_f1']:.4f}")
        print(f"   - ROC-AUC: {model.metrics['roc_auc']:.4f}")
        print(f"\n Model saved at: {model_path}")
        print("\n You can now use this model with the Streamlit app!")
        
        return model
        
    except Exception as e:
        print(f"\n TRAINING FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    trained_model = main()
