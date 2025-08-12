#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import random
import time
from datetime import datetime
import argparse

class LoadGenerator:
    def __init__(self, base_url: str, requests_per_second: int = 10):
        self.base_url = base_url.rstrip('/')
        self.rps = requests_per_second
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'anomalies_detected': 0
        }
    
    def generate_transaction(self, transaction_id: str) -> dict:
        """Generate a realistic transaction"""
        users = [f"user_{i}" for i in range(1000)]
        merchants = ["Amazon", "Walmart", "Starbucks", "Shell", "ATM_Unknown"]
        categories = ["shopping", "food", "gas", "entertainment", "cash"]
        locations = ["New York", "Los Angeles", "Chicago", "Unknown"]
        
        # 10% chance of suspicious transaction
        if random.random() < 0.1:
            amount = random.uniform(5000, 25000)
            merchant = "Unknown_Merchant"
            category = "cash"
            location = "Unknown"
        else:
            amount = random.lognormal(3, 1)
            merchant = random.choice(merchants)
            category = random.choice(categories)
            location = random.choice(locations)
        
        return {
            "transaction_id": transaction_id,
            "user_id": random.choice(users),
            "amount": round(max(1.0, amount), 2),
            "merchant": merchant,
            "category": category,
            "location": location,
            "device_id": f"device_{random.randint(1000, 9999)}"
        }
    
    async def send_transaction(self, session: aiohttp.ClientSession, transaction: dict):
        """Send a single transaction"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/predict",
                json=transaction,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                self.stats['response_times'].append(response_time)
                self.stats['total_requests'] += 1
                
                if response.status == 200:
                    self.stats['successful_requests'] += 1
                    data = await response.json()
                    
                    if data.get('anomaly_type') in ['suspicious', 'fraudulent']:
                        self.stats['anomalies_detected'] += 1
                        
                else:
                    self.stats['failed_requests'] += 1
                    
        except Exception as e:
            self.stats['failed_requests'] += 1
            self.stats['total_requests'] += 1
    
    async def generate_load(self, duration_seconds: int):
        """Generate load for specified duration"""
        print(f"Generating load: {self.rps} RPS for {duration_seconds} seconds")
        print(f"Target: {self.base_url}")
        print("=" * 60)
        
        start_time = time.time()
        request_interval = 1.0 / self.rps
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            request_count = 0
            
            while time.time() - start_time < duration_seconds:
                # Generate transaction
                transaction = self.generate_transaction(f"load_test_{request_count:06d}")
                
                # Send request
                task = asyncio.create_task(self.send_transaction(session, transaction))
                tasks.append(task)
                
                request_count += 1
                
                # Wait for interval
                await asyncio.sleep(request_interval)
                
                # Print progress every 100 requests
                if request_count % 100 == 0:
                    elapsed = time.time() - start_time
                    current_rps = request_count / elapsed
                    print(f"Sent {request_count} requests ({current_rps:.1f} RPS)")
            
            # Wait for all requests to complete
            print("Waiting for all requests to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def print_statistics(self):
        """Print load test statistics"""
        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        
        print(f"Total Requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_requests']}")
        print(f"Failed: {self.stats['failed_requests']}")
        
        if self.stats['total_requests'] > 0:
            success_rate = self.stats['successful_requests'] / self.stats['total_requests']
            print(f"Success Rate: {success_rate:.1%}")
        
        if self.stats['response_times']:
            response_times = self.stats['response_times']
            print(f"Avg Response Time: {sum(response_times)/len(response_times):.3f}s")
            print(f"Min Response Time: {min(response_times):.3f}s")
            print(f"Max Response Time: {max(response_times):.3f}s")
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p50 = sorted_times[len(sorted_times)//2]
            p95 = sorted_times[int(len(sorted_times)*0.95)]
            p99 = sorted_times[int(len(sorted_times)*0.99)]
            
            print(f"P50 Response Time: {p50:.3f}s")
            print(f"P95 Response Time: {p95:.3f}s")
            print(f"P99 Response Time: {p99:.3f}s")
        
        print(f"Anomalies Detected: {self.stats['anomalies_detected']}")

async def main():
    parser = argparse.ArgumentParser(description='Generate load for fraud detection API')
    parser.add_argument('--url', required=True, help='API base URL')
    parser.add_argument('--rps', type=int, default=10, help='Requests per second')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    
    args = parser.parse_args()
    
    generator = LoadGenerator(args.url, args.rps)
    
    try:
        await generator.generate_load(args.duration)
    finally:
        generator.print_statistics()

if __name__ == "__main__":
    asyncio.run(main())