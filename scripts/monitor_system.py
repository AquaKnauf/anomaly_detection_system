#!/usr/bin/env python3

import requests
import time
import json
from datetime import datetime
import argparse

class SystemMonitor:
    def __init__(self, api_url: str, prometheus_url: str):
        self.api_url = api_url.rstrip('/')
        self.prometheus_url = prometheus_url.rstrip('/')
        
    def get_api_metrics(self) -> dict:
        """Get metrics from API"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting API metrics: {e}")
            return {}
    
    def get_prometheus_metrics(self, query: str) -> dict:
        """Query Prometheus for metrics"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return {}
    
    def monitor_system(self, interval_seconds: int = 30, duration_minutes: int = 60):
        """Monitor system metrics"""
        print(f"Monitoring system for {duration_minutes} minutes...")
        print("Metrics will be collected every {interval_seconds} seconds")
        print("=" * 80)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        metrics_history = []
        
        while time.time() < end_time:
            timestamp = datetime.now()
            
            # Get API metrics
            api_metrics = self.get_api_metrics()
            
            # Get Prometheus metrics
            cpu_usage = self.get_prometheus_metrics(
                'rate(container_cpu_usage_seconds_total[5m]) * 100'
            )
            memory_usage = self.get_prometheus_metrics(
                'container_memory_usage_bytes / container_spec_memory_limit_bytes * 100'
            )
            request_rate = self.get_prometheus_metrics(
                'rate(transactions_processed_total[5m])'
            )
            
            # Combine metrics
            current_metrics = {
                'timestamp': timestamp.isoformat(),
                'api_metrics': api_metrics,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'request_rate': request_rate
            }
            
            metrics_history.append(current_metrics)
            
            # Print current status
            print(f"\n[{timestamp.strftime('%H:%M:%S')}] System Status:")
            if api_metrics:
                print(f"  Transactions Processed: {api_metrics.get('transactions_processed', 'N/A')}")
                print(f"  Total Anomalies: {api_metrics.get('total_anomalies', 'N/A')}")
                print(f"  Uptime: {api_metrics.get('uptime_seconds', 'N/A')}s")
            
            # Check for alerts
            self.check_alerts(current_metrics)
            
            time.sleep(interval_seconds)
        
        # Save metrics history
        with open(f'monitoring_results_{int(start_time)}.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        print(f"\nMonitoring completed. Results saved to monitoring_results_{int(start_time)}.json")
    
    def check_alerts(self, metrics: dict):
        """Check for alert conditions"""
        alerts = []
        
        # Check API health
        if not metrics['api_metrics']:
            alerts.append("🚨 API not responding")
        
        # Add more alert conditions based on your requirements
        uptime = metrics['api_metrics'].get('uptime_seconds', 0)
        if uptime < 60:  # Less than 1 minute uptime might indicate restart
            alerts.append("⚠️  Recent restart detected")
        
        if alerts:
            print("  ALERTS:", "; ".join(alerts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor fraud detection system')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API URL')
    parser.add_argument('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
    parser.add_argument('--interval', type=int, default=30, help='Collection interval (seconds)')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration (minutes)')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.api_url, args.prometheus_url)
    monitor.monitor_system(args.interval, args.duration)