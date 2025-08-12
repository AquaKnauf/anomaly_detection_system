#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import argparse
import os

class DataPipeline:
    def __init__(self, db_path: str = "fraud_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT,
                amount REAL,
                merchant TEXT,
                category TEXT,
                timestamp TEXT,
                location TEXT,
                device_id TEXT,
                is_weekend INTEGER,
                hour_of_day INTEGER,
                is_anomalous INTEGER,
                anomaly_score REAL,
                anomaly_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                avg_amount REAL,
                std_amount REAL,
                transaction_count INTEGER,
                last_transaction_date TEXT,
                risk_score REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                model_name TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                auc_score REAL,
                evaluation_date TEXT,
                sample_size INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized: {self.db_path}")
    
    def load_kaggle_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess Kaggle credit card fraud dataset"""
        print(f"Loading data from {file_path}...")
        
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Rename columns to match our schema
        column_mapping = {
            'Time': 'timestamp_seconds',
            'Amount': 'amount',
            'Class': 'is_fraudulent'
        }
        
        # If it's the standard Kaggle dataset, it has V1-V28 features
        if 'V1' in df.columns:
            df = df.rename(columns=column_mapping)
            
            # Create synthetic transaction details
            df['transaction_id'] = df.index.map(lambda x: f"kaggle_{x:08d}")
            df['user_id'] = df.index.map(lambda x: f"user_{x % 10000}")
            df['merchant'] = np.random.choice(
                ["Amazon", "Walmart", "Starbucks", "ATM", "Unknown"], 
                size=len(df)
            )
            df['category'] = np.random.choice(
                ["shopping", "food", "cash", "gas"], 
                size=len(df)
            )
            df['location'] = np.random.choice(
                ["New York", "Los Angeles", "Chicago", "Unknown"], 
                size=len(df)
            )
            df['device_id'] = df.index.map(lambda x: f"device_{x % 1000}")
            
            # Convert timestamp
            base_time = datetime.now() - timedelta(days=30)
            df['timestamp'] = df['timestamp_seconds'].apply(
                lambda x: (base_time + timedelta(seconds=x)).isoformat()
            )
            
            # Add time-based features
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['hour_of_day'] = df['datetime'].dt.hour
            df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(int)
            
        print(f"Loaded {len(df)} transactions ({df['is_fraudulent'].sum()} fraudulent)")
        return df
    
    def save_to_database(self, df: pd.DataFrame):
        """Save preprocessed data to database"""
        conn = sqlite3.connect(self.db_path)
        
        # Select relevant columns for our schema
        transaction_cols = [
            'transaction_id', 'user_id', 'amount', 'merchant', 'category',
            'timestamp', 'location', 'device_id', 'is_weekend', 'hour_of_day'
        ]
        
        if 'is_fraudulent' in df.columns:
            transaction_cols.append('is_fraudulent')
        
        # Save transactions
        df[transaction_cols].to_sql('transactions', conn, if_exists='append', index=False)
        
        # Create user profiles
        user_profiles = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'count'],
            'timestamp': 'max'
        }).round(2)
        
        user_profiles.columns = ['avg_amount', 'std_amount', 'transaction_count', 'last_transaction_date']
        user_profiles['risk_score'] = 0.0  # Will be updated by ML models
        user_profiles['updated_at'] = datetime.now().isoformat()
        
        user_profiles.to_sql('user_profiles', conn, if_exists='replace')
        
        conn.close()
        print(f"Data saved to database: {len(df)} transactions, {len(user_profiles)} user profiles")
    
    def export_training_data(self, output_path: str, sample_size: int = None):
        """Export training data for model development"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM transactions WHERE is_fraudulent IS NOT NULL"
        if sample_size:
            query += f" ORDER BY RANDOM() LIMIT {sample_size}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Export to various formats
        df.to_csv(f"{output_path}.csv", index=False)
        df.to_parquet(f"{output_path}.parquet", index=False)
        
        print(f"Training data exported: {len(df)} samples")
        print(f"Files created: {output_path}.csv, {output_path}.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data pipeline for fraud detection')
    parser.add_argument('--action', choices=['load', 'export'], required=True)
    parser.add_argument('--input-file', help='Input CSV file path')
    parser.add_argument('--output-path', help='Output file path (without extension)')
    parser.add_argument('--sample-size', type=int, help='Sample size for export')
    parser.add_argument('--db-path', default='fraud_detection.db', help='Database path')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(args.db_path)
    
    if args.action == 'load' and args.input_file:
        df = pipeline.load_kaggle_data(args.input_file)
        pipeline.save_to_database(df)
    elif args.action == 'export' and args.output_path:
        pipeline.export_training_data(args.output_path, args.sample_size)
    else:
        print("Invalid arguments. Use --help for usage information.")