#!/usr/bin/env python3

import json
import time
import random
import argparse
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

class TransactionProducer:
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8'),
            acks='all',
            retries=3,
            batch_size=16384,
            linger_ms=10
        )
    
    def generate_transaction(self, transaction_id: str) -> dict:
        """Generate a realistic transaction"""
        users = [f"user_{i}" for i in range(1000)]
        merchants = [
            "Amazon", "Walmart", "Target", "Starbucks", "Shell", 
            "McDonald's", "Best Buy", "Home Depot", "Costco", "ATM_Withdrawal"
        ]
        categories = ["shopping", "food", "gas", "entertainment", "bills", "cash"]
        locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        
        # 5% chance of creating an anomalous transaction
        is_anomalous = random.random() < 0.05
        
        if is_anomalous:
            amount = random.uniform(5000, 50000)  # Very high amount
            merchant = random.choice(["Unknown_Merchant", "Suspicious_ATM", "Foreign_Card"])
            category = "cash"
            location = "Unknown"
        else:
            amount = random.lognormal(mean=3, sigma=1)  # Normal distribution
            merchant = random.choice(merchants)
            category = random.choice(categories)
            location = random.choice(locations)
        
        return {
            "transaction_id": transaction_id,
            "user_id": random.choice(users),
            "amount": round(max(1.0, amount), 2),
            "merchant": merchant,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "device_id": f"device_{random.randint(1000, 9999)}",
            "is_weekend": datetime.now().weekday() >= 5,
            "hour_of_day": datetime.now().hour,
            "is_anomalous": is_anomalous  # For testing purposes
        }
    
    def send_transactions(self, topic: str, rate_per_second: int, duration_seconds: int):
        """Send transactions at specified rate"""
        print(f"Sending transactions to topic '{topic}' at {rate_per_second} TPS for {duration_seconds}s")
        
        interval = 1.0 / rate_per_second
        start_time = time.time()
        transaction_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                transaction = self.generate_transaction(f"txn_{transaction_count:08d}")
                
                # Send to Kafka
                future = self.producer.send(
                    topic,
                    value=transaction,
                    key=transaction["transaction_id"]
                )
                
                transaction_count += 1
                
                # Print progress
                if transaction_count % 100 == 0:
                    elapsed = time.time() - start_time
                    current_rate = transaction_count / elapsed
                    print(f"Sent {transaction_count} transactions ({current_rate:.1f} TPS)")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nStopping producer...")
        
        finally:
            # Ensure all messages are sent
            self.producer.flush()
            self.producer.close()
            
            total_time = time.time() - start_time
            actual_rate = transaction_count / total_time
            print(f"\nCompleted: {transaction_count} transactions in {total_time:.1f}s ({actual_rate:.1f} TPS)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kafka transaction producer')
    parser.add_argument('--servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='transactions', help='Kafka topic')
    parser.add_argument('--rate', type=int, default=10, help='Transactions per second')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    
    args = parser.parse_args()
    
    producer = TransactionProducer(args.servers)
    producer.send_transactions(args.topic, args.rate, args.duration)