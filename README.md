# Real-time Financial Transaction Anomaly Detection System

A production-ready system for detecting fraudulent and suspicious financial transactions in real-time using machine learning and streaming architecture.

## 🚀 Features

- **Real-time Processing**: Stream processing with Apache Kafka
- **Ensemble ML Models**: Isolation Forest, Autoencoders, and LSTM networks
- **Scalable Architecture**: Kubernetes deployment with auto-scaling
- **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards
- **Production Ready**: CI/CD pipeline, security scanning, and health checks

## 🏗️ Architecture

```
[Transaction Stream] → [Kafka] → [Anomaly Detector] → [Database]
                                        ↓
[Grafana] ← [Prometheus] ← [Metrics Endpoint]
```

## 📊 Machine Learning Models

1. **Isolation Forest**: Unsupervised anomaly detection
2. **Autoencoder**: Neural network for reconstruction-based detection
3. **LSTM**: Sequential pattern analysis for time-series anomalies
4. **Ensemble**: Weighted combination of all models

## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone <your-repo-url>
cd fraud-detection

# Setup development environment
make dev-setup

# Run locally with Docker Compose
make deploy-local

# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"test_001","user_id":"user_123","amount":45.67,"merchant":"Starbucks","category":"food","location":"New York","device_id":"device_1234"}'
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
make deploy-k8s

# Monitor the deployment
kubectl get pods -n fraud-detection

# Check logs
kubectl logs -l app=anomaly-detector -n fraud-detection
```

## 📈 Monitoring

- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090
- **API Health Check**: http://localhost:8000/health

## 🧪 Testing

```bash
# Run unit tests
make test

# Run integration tests
python scripts/integration_tests.py --endpoint http://localhost:8000

# Generate load for testing
make generate-load

# Evaluate model performance
make evaluate-models
```

## 📊 Performance Metrics

The system tracks several key metrics:

- **Throughput**: Transactions processed per second
- **Latency**: Response time percentiles (P50, P95, P99)
- **Accuracy**: Model prediction accuracy
- **Anomaly Detection Rate**: Percentage of transactions flagged

## 🔧 Configuration

### Environment Variables
- `KAFKA_SERVERS`: Kafka bootstrap servers
- `REDIS_HOST`: Redis host for caching
- `POSTGRES_URL`: Database connection string

### Model Parameters
- Contamination rate: 10% (configurable)
- Ensemble weights: Isolation Forest (40%), Autoencoder (40%), LSTM (20%)
- Feature engineering: 15+ derived features

## 🛡️ Security

- Container vulnerability scanning with Trivy
- Non-root container execution
- Network policies and RBAC
- Secrets management with Kubernetes secrets

## 📚 API Documentation

### POST /predict
Predict if a transaction is anomalous.

**Request Body:**
```json
{
  "transaction_id": "txn_001",
  "user_id": "user_123",
  "amount": 45.67,
  "merchant": "Starbucks",
  "category": "food",
  "location": "New York",
  "device_id": "device_1234"
}
```

**Response:**
```json
{
  "transaction_id": "txn_001",
  "anomaly_score": 0.15,
  "anomaly_type": "normal",
  "confidence": 0.85,
  "explanation": "No specific risk factors identified",
  "processing_time_ms": 23.5,
  "timestamp": "2025-08-10T10:30:00"
}
```

## 🔄 CI/CD Pipeline

The system includes a complete CI/CD pipeline with:

1. **Code Quality**: Linting, type checking, security scanning
2. **Testing**: Unit tests, integration tests, load testing
3. **Building**: Multi-architecture Docker images
4. **Deployment**: Automated Kubernetes deployment with Helm
5. **Monitoring**: Automatic rollback on health check failures

## 🎯 Business Value

- **Fraud Prevention**: Detect suspicious transactions in real-time
- **Cost Reduction**: Automated screening reduces manual review
- **Scalability**: Handle millions of transactions per day
- **Compliance**: Audit trails and explainable AI decisions

## 📈 Scaling Considerations

- **Horizontal Scaling**: Auto-scaling based on CPU/memory usage
- **Database Sharding**: Partition by user_id for high volume
- **Model Serving**: A/B testing for model improvements
- **Caching**: Redis for user profiles and frequent queries

## 🔍 Troubleshooting

### Common Issues

1. **Models not loading**: Check model file permissions and paths
2. **Kafka connection issues**: Verify Kafka broker accessibility
3. **High memory usage**: Adjust batch sizes and model parameters
4. **Slow predictions**: Enable model caching and feature precomputation

### Debugging Commands
```bash
# Check pod logs
kubectl logs -l app=anomaly-detector -n fraud-detection

# Check resource usage
kubectl top pods -n fraud-detection

# Test connectivity
kubectl exec -it <pod-name> -n fraud-detection -- curl localhost:8000/health
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run `make test` and `make lint`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details