#!/usr/bin/env python3

import requests
import time
import json
import argparse
import sys
from typing import Dict, Any

class IntegrationTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def test_health_endpoint(self) -> bool:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            data = response.json()
            assert data["status"] == "healthy"
            print("✓ Health check passed")
            return True
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False
    
    def test_normal_transaction(self) -> bool:
        """Test normal transaction prediction"""
        try:
            transaction = {
                "transaction_id": "integration_test_001",
                "user_id": "test_user_123",
                "amount": 25.50,
                "merchant": "Starbucks",
                "category": "food",
                "location": "New York",
                "device_id": "device_1234"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=transaction,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            assert "anomaly_score" in data
            assert "anomaly_type" in data
            assert data["transaction_id"] == transaction["transaction_id"]
            
            print(f"✓ Normal transaction test passed (score: {data['anomaly_score']:.3f})")
            return True
            
        except Exception as e:
            print(f"✗ Normal transaction test failed: {e}")
            return False
    
    def test_suspicious_transaction(self) -> bool:
        """Test suspicious transaction prediction"""
        try:
            transaction = {
                "transaction_id": "integration_test_002",
                "user_id": "test_user_123",
                "amount": 9999.99,
                "merchant": "Unknown_ATM",
                "category": "cash",
                "location": "Unknown",
                "device_id": "device_suspicious"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=transaction,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            assert "anomaly_score" in data
            assert "anomaly_type" in data
            
            # Should have higher anomaly score
            assert data["anomaly_score"] > 0.1
            
            print(f"✓ Suspicious transaction test passed (score: {data['anomaly_score']:.3f})")
            return True
            
        except Exception as e:
            print(f"✗ Suspicious transaction test failed: {e}")
            return False
    
    def test_invalid_transaction(self) -> bool:
        """Test invalid transaction handling"""
        try:
            invalid_transaction = {
                "transaction_id": "integration_test_003",
                "user_id": "test_user_123",
                "amount": -50.0,  # Invalid negative amount
                "merchant": "Test",
                "category": "test",
                "location": "Test",
                "device_id": "test"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_transaction,
                timeout=10
            )
            
            # Should return 422 for validation error
            assert response.status_code == 422
            
            print("✓ Invalid transaction handling test passed")
            return True
            
        except Exception as e:
            print(f"✗ Invalid transaction test failed: {e}")
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test metrics endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            assert "transactions_processed" in data
            assert "total_anomalies" in data
            assert "uptime_seconds" in data
            
            print("✓ Metrics endpoint test passed")
            return True
            
        except Exception as e:
            print(f"✗ Metrics endpoint test failed: {e}")
            return False
    
    def test_load_performance(self, num_requests: int = 50) -> bool:
        """Test load performance"""
        try:
            transaction = {
                "transaction_id": "load_test_001",
                "user_id": "load_test_user",
                "amount": 100.0,
                "merchant": "TestMerchant",
                "category": "test",
                "location": "TestCity",
                "device_id": "load_test_device"
            }
            
            start_time = time.time()
            successful_requests = 0
            
            for i in range(num_requests):
                transaction["transaction_id"] = f"load_test_{i:03d}"
                
                try:
                    response = self.session.post(
                        f"{self.base_url}/predict",
                        json=transaction,
                        timeout=5
                    )
                    if response.status_code == 200:
                        successful_requests += 1
                except:
                    pass
            
            total_time = time.time() - start_time
            success_rate = successful_requests / num_requests
            avg_response_time = total_time / num_requests
            
            assert success_rate >= 0.95  # 95% success rate
            assert avg_response_time < 1.0  # Average response time under 1 second
            
            print(f"✓ Load test passed: {successful_requests}/{num_requests} requests "
                  f"({success_rate:.1%} success rate, {avg_response_time:.3f}s avg)")
            return True
            
        except Exception as e:
            print(f"✗ Load test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        print(f"Running integration tests against {self.base_url}")
        print("=" * 60)
        
        tests = [
            self.test_health_endpoint,
            self.test_normal_transaction,
            self.test_suspicious_transaction,
            self.test_invalid_transaction,
            self.test_metrics_endpoint,
            self.test_load_performance
        ]
        
        results = []
        for test in tests:
            result = test()
            results.append(result)
            time.sleep(1)  # Brief pause between tests
        
        print("=" * 60)
        passed = sum(results)
        total = len(results)
        
        if passed == total:
            print(f"🎉 All {total} tests passed!")
            return True
        else:
            print(f"❌ {passed}/{total} tests passed")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run integration tests')
    parser.add_argument('--endpoint', required=True, help='API endpoint URL')
    args = parser.parse_args()
    
    tester = IntegrationTester(args.endpoint)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)