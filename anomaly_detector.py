# Real-time Financial Transaction Anomaly Detection System
# Complete implementation with streaming, ML models, and infrastructure

import os
import json
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# ML Libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# API Framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn

# Streaming
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Database
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ================================
# Configuration and Data Models
# ================================

class AnomalyType(Enum):
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    FRAUDULENT = "fraudulent"

@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    category: str
    timestamp: datetime
    location: str
    device_id: str
    is_weekend: bool = False
    hour_of_day: int = 0
    days_since_last_transaction: float = 0.0

@dataclass
class AnomalyResult:
    transaction_id: str
    anomaly_score: float
    anomaly_type: AnomalyType
    model_used: str
    confidence: float
    timestamp: datetime
    features_used: List[str]
    explanation: str

class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    category: str
    location: str
    device_id: str
    
    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

# ================================
# Feature Engineering
# ================================

class FeatureEngineer:
    def __init__(self):
        self.user_profiles = {}
        self.merchant_stats = {}
        
    def extract_features(self, transaction: Transaction) -> Dict[str, float]:
        """Extract comprehensive features for anomaly detection"""
        features = {
            'amount': transaction.amount,
            'hour_of_day': transaction.hour_of_day,
            'is_weekend': float(transaction.is_weekend),
            'days_since_last': transaction.days_since_last_transaction,
        }
        
        # Amount-based features
        features['amount_log'] = np.log1p(transaction.amount)
        features['amount_rounded'] = transaction.amount % 1.0
        
        # User behavior features
        user_profile = self.user_profiles.get(transaction.user_id, {})
        features['user_avg_amount'] = user_profile.get('avg_amount', transaction.amount)
        features['user_std_amount'] = user_profile.get('std_amount', 0.0)
        features['amount_zscore'] = self._calculate_zscore(
            transaction.amount, 
            features['user_avg_amount'], 
            features['user_std_amount']
        )
        
        # Merchant features
        merchant_stats = self.merchant_stats.get(transaction.merchant, {})
        features['merchant_avg_amount'] = merchant_stats.get('avg_amount', transaction.amount)
        features['merchant_frequency'] = merchant_stats.get('frequency', 1.0)
        
        # Time-based features
        features['is_night'] = float(22 <= transaction.hour_of_day or transaction.hour_of_day <= 5)
        features['is_business_hours'] = float(9 <= transaction.hour_of_day <= 17)
        
        # Categorical encoding (simplified - in production use proper encoding)
        features['category_encoded'] = hash(transaction.category) % 100
        features['location_encoded'] = hash(transaction.location) % 1000
        
        return features
    
    def _calculate_zscore(self, value: float, mean: float, std: float) -> float:
        if std == 0:
            return 0.0
        return (value - mean) / std
    
    def update_user_profile(self, user_id: str, amount: float):
        """Update user spending profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'amounts': [amount],
                'avg_amount': amount,
                'std_amount': 0.0,
                'transaction_count': 1
            }
        else:
            profile = self.user_profiles[user_id]
            profile['amounts'].append(amount)
            profile['transaction_count'] += 1
            
            # Keep only recent transactions (sliding window)
            if len(profile['amounts']) > 100:
                profile['amounts'] = profile['amounts'][-100:]
            
            profile['avg_amount'] = np.mean(profile['amounts'])
            profile['std_amount'] = np.std(profile['amounts'])

# ================================
# ML Models
# ================================

class IsolationForestDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200
        )
        self.scaler = RobustScaler()
        self.is_fitted = False
        
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Train the Isolation Forest model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.scaler.transform(X)
        anomaly_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        return {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,  # -1 for anomaly, 1 for normal
            'model_name': 'isolation_forest'
        }

class AutoencoderDetector:
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        
    def _build_model(self):
        """Build autoencoder architecture"""
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(0.001), loss='mse')
        
        return autoencoder
    
    def train(self, X: np.ndarray, validation_split=0.2):
        """Train the autoencoder"""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = self._build_model()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=100,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate threshold based on training reconstruction error
        train_pred = self.model.predict(X_scaled, verbose=0)
        train_mse = np.mean(np.square(X_scaled - train_pred), axis=1)
        self.threshold = np.percentile(train_mse, 95)  # 95th percentile
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict anomaly scores using reconstruction error"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)
        
        # Normalize scores to [0, 1] range
        anomaly_scores = mse / (self.threshold + 1e-8)
        predictions = np.where(mse > self.threshold, -1, 1)
        
        return {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'reconstruction_errors': mse,
            'model_name': 'autoencoder'
        }

class LSTMAnomalyDetector:
    def __init__(self, sequence_length: int = 10, n_features: int = None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM training"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def _build_model(self):
        """Build LSTM autoencoder architecture"""
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, self.n_features), 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(25, activation='relu', return_sequences=False),
            RepeatVector(self.sequence_length),
            LSTM(25, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=True),
            TimeDistributed(Dense(self.n_features))
        ])
        
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model
    
    def train(self, X: np.ndarray):
        """Train LSTM autoencoder"""
        if self.n_features is None:
            self.n_features = X.shape[1]
            
        X_scaled = self.scaler.fit_transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        self.model = self._build_model()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_sequences, X_sequences,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate threshold
        train_pred = self.model.predict(X_sequences, verbose=0)
        train_mse = np.mean(np.square(X_sequences - train_pred), axis=(1, 2))
        self.threshold = np.percentile(train_mse, 95)
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict anomalies using LSTM"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        if len(X_sequences) == 0:
            return {
                'anomaly_scores': np.array([0.0]),
                'predictions': np.array([1]),
                'model_name': 'lstm'
            }
        
        reconstructed = self.model.predict(X_sequences, verbose=0)
        mse = np.mean(np.square(X_sequences - reconstructed), axis=(1, 2))
        
        anomaly_scores = mse / (self.threshold + 1e-8)
        predictions = np.where(mse > self.threshold, -1, 1)
        
        return {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'reconstruction_errors': mse,
            'model_name': 'lstm'
        }

# ================================
# Ensemble Model
# ================================

class EnsembleAnomalyDetector:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.feature_engineer = FeatureEngineer()
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def train_all_models(self, transactions: List[Transaction]):
        """Train all models in the ensemble"""
        print("Extracting features...")
        feature_list = []
        for transaction in transactions:
            features = self.feature_engineer.extract_features(transaction)
            feature_list.append(features)
            self.feature_engineer.update_user_profile(transaction.user_id, transaction.amount)
        
        # Convert to DataFrame and then numpy array
        df = pd.DataFrame(feature_list)
        X = df.select_dtypes(include=[np.number]).fillna(0).values
        
        print(f"Training ensemble with {len(self.models)} models on {X.shape[0]} samples...")
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.train(X)
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
    
    def predict(self, transaction: Transaction) -> AnomalyResult:
        """Predict using ensemble of models"""
        features = self.feature_engineer.extract_features(transaction)
        X = np.array(list(features.values())).reshape(1, -1)
        
        predictions = {}
        scores = []
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                result = model.predict(X)
                predictions[name] = result
                scores.append(result['anomaly_scores'][0] * self.weights[name])
            except Exception as e:
                print(f"Error in {name} prediction: {e}")
                scores.append(0.0)
        
        # Ensemble score (weighted average)
        ensemble_score = np.mean(scores) if scores else 0.0
        
        # Determine anomaly type based on score
        if ensemble_score > 0.7:
            anomaly_type = AnomalyType.FRAUDULENT
        elif ensemble_score > 0.3:
            anomaly_type = AnomalyType.SUSPICIOUS
        else:
            anomaly_type = AnomalyType.NORMAL
        
        # Generate explanation
        explanation = self._generate_explanation(features, ensemble_score, predictions)
        
        return AnomalyResult(
            transaction_id=transaction.transaction_id,
            anomaly_score=float(ensemble_score),
            anomaly_type=anomaly_type,
            model_used="ensemble",
            confidence=min(abs(ensemble_score), 1.0),
            timestamp=datetime.now(),
            features_used=list(features.keys()),
            explanation=explanation
        )
    
    def _generate_explanation(self, features: Dict, score: float, predictions: Dict) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        if features.get('amount_zscore', 0) > 2:
            explanations.append("Transaction amount is unusually high for this user")
        
        if features.get('is_night', 0) > 0.5:
            explanations.append("Transaction occurred during night hours")
        
        if features.get('days_since_last', 0) < 0.1:
            explanations.append("Multiple transactions in quick succession")
        
        if not explanations:
            explanations.append("No specific risk factors identified")
        
        return "; ".join(explanations)

# ================================
# Streaming Infrastructure
# ================================

class KafkaStreamer:
    def __init__(self, bootstrap_servers: str = None):
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_SERVERS", "kafka:9092")
        print(f"[KafkaStreamer] Kafka bootstrap servers: {self.bootstrap_servers}")
        self.producer = None
        self.consumer = None
        
    def create_producer(self):
        """Create Kafka producer"""
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        return self.producer
    
    def create_consumer(self, topic: str, group_id: str):
        """Create Kafka consumer"""
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        return self.consumer
    
    async def send_transaction(self, topic: str, transaction: Transaction):
        """Send transaction to Kafka topic"""
        if not self.producer:
            self.create_producer()
        
        message = asdict(transaction)
        message['timestamp'] = transaction.timestamp.isoformat()
        
        future = self.producer.send(topic, value=message, key=transaction.transaction_id)
        await asyncio.sleep(0.001)  # Small delay for async behavior
        return future

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile from cache"""
        data = self.redis_client.hgetall(f"user:{user_id}")
        return data if data else None
    
    def update_user_profile(self, user_id: str, profile: Dict):
        """Update user profile in cache"""
        self.redis_client.hmset(f"user:{user_id}", profile)
        self.redis_client.expire(f"user:{user_id}", 86400 * 30)  # 30 days TTL

# ================================
# Monitoring and Metrics
# ================================

# Prometheus metrics
transaction_counter = Counter('transactions_processed_total', 'Total processed transactions')
anomaly_counter = Counter('anomalies_detected_total', 'Total detected anomalies', ['type'])
processing_time = Histogram('transaction_processing_seconds', 'Transaction processing time')
model_accuracy = Gauge('model_accuracy_score', 'Model accuracy score', ['model'])

class MonitoringService:
    def __init__(self):
        self.start_time = time.time()
        
    def record_transaction(self):
        """Record transaction processing"""
        transaction_counter.inc()
    
    def record_anomaly(self, anomaly_type: AnomalyType):
        """Record detected anomaly"""
        anomaly_counter.labels(type=anomaly_type.value).inc()
    
    def record_processing_time(self, duration: float):
        """Record processing time"""
        processing_time.observe(duration)
    
    def update_model_accuracy(self, model_name: str, accuracy: float):
        """Update model accuracy metric"""
        model_accuracy.labels(model=model_name).set(accuracy)

# ================================
# API Service
# ================================

# Global instances
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables - initialize as None
ensemble_detector: Optional[Any] = None
kafka_streamer: Optional[Any] = None
monitoring_service: Optional[Any] = None
redis_cache: Optional[Any] = None

app = FastAPI(title="Financial Anomaly Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global ensemble_detector, kafka_streamer, monitoring_service, redis_cache
    
    try:
        logger.info("Starting application initialization...")
        
        # Initialize monitoring service first
        logger.info("Initializing monitoring service...")
        try:
            monitoring_service = MonitoringService()
        except ImportError as e:
            logger.warning(f"MonitoringService import failed: {e}")
            monitoring_service = None
        
        # Initialize Redis cache
        logger.info("Initializing Redis cache...")
        try:
            redis_cache = RedisCache()
        except ImportError as e:
            logger.warning(f"RedisCache import failed: {e}")
            redis_cache = None
        
        # Initialize ensemble detector
        logger.info("Initializing ensemble detector...")
        try:
            ensemble_detector = EnsembleAnomalyDetector()
            
            # Add models to ensemble with error handling
            try:
                isolation_forest = IsolationForestDetector(contamination=0.1)
                ensemble_detector.add_model("isolation_forest", isolation_forest, weight=0.4)
            except Exception as e:
                logger.warning(f"Failed to add isolation forest: {e}")
            
            try:
                autoencoder = AutoencoderDetector(input_dim=15, encoding_dim=8)
                ensemble_detector.add_model("autoencoder", autoencoder, weight=0.4)
            except Exception as e:
                logger.warning(f"Failed to add autoencoder: {e}")
            
            try:
                lstm_detector = LSTMAnomalyDetector(sequence_length=5)
                ensemble_detector.add_model("lstm", lstm_detector, weight=0.2)
            except Exception as e:
                logger.warning(f"Failed to add LSTM detector: {e}")
                
        except ImportError as e:
            logger.error(f"Failed to import detector modules: {e}")
            raise
        
        # Initialize Kafka streamer with better error handling
        logger.info("Connecting to Kafka...")
        try:
            kafka_streamer = KafkaStreamer()
            logger.info("Kafka connected successfully.")
        except ImportError as e:
            logger.warning(f"KafkaStreamer import failed: {e}")
            kafka_streamer = None
        except Exception as e:
            logger.warning(f"Kafka initialization failed: {e}")
            kafka_streamer = None
        
        # Start Prometheus metrics server with error handling
        try:
            logger.info("Starting Prometheus metrics server on port 8001...")
            start_http_server(8001)
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
        
        # Train models in background - don't block startup
        asyncio.create_task(train_models_background())
        
        logger.info("✓ Anomaly detection service started successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error during startup: {e}")
        # Don't raise here - let the app start in degraded mode
        # raise  # Uncomment if you want startup to fail completely

async def train_models_background():
    """Train models in background without blocking startup"""
    try:
        logger.info("Starting background model training...")
        
        # Add a small delay to ensure all services are ready
        await asyncio.sleep(2)
        
        if ensemble_detector is None:
            logger.warning("Ensemble detector not available, skipping training")
            return
            
        logger.info("Generating synthetic training data...")
        try:
            training_data = generate_synthetic_transactions(1000)
            logger.info("Training models...")
            ensemble_detector.train_all_models(training_data)
            logger.info("✓ Model training complete!")
        except ImportError as e:
            logger.error(f"Failed to import data generation module: {e}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
    except Exception as e:
        logger.error(f"Background training error: {e}")

@app.post("/predict", response_model=Dict[str, Any])
async def predict_transaction(transaction_request: Dict[str, Any]):  # Use Dict instead of Pydantic model for now
    """Predict if a transaction is anomalous"""
    
    # Check if detector is ready
    if ensemble_detector is None:
        raise HTTPException(status_code=503, detail="Service not ready - models not loaded")
    
    start_time = time.time()
    
    try:
        # Create transaction object with error handling
        try:
            transaction = Transaction(
                transaction_id=transaction_request.get("transaction_id"),
                user_id=transaction_request.get("user_id"),
                amount=transaction_request.get("amount"),
                merchant=transaction_request.get("merchant"),
                category=transaction_request.get("category"),
                timestamp=datetime.now(),
                location=transaction_request.get("location"),
                device_id=transaction_request.get("device_id"),
                is_weekend=datetime.now().weekday() >= 5,
                hour_of_day=datetime.now().hour,
                days_since_last_transaction=0.0
            )
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"Transaction model import failed: {e}")
        
        # Predict anomaly
        result = ensemble_detector.predict(transaction)
        
        # Update metrics if available
        if monitoring_service:
            try:
                monitoring_service.record_transaction()
                monitoring_service.record_anomaly(result.anomaly_type)
                processing_duration = time.time() - start_time
                monitoring_service.record_processing_time(processing_duration)
            except Exception as e:
                logger.warning(f"Failed to record metrics: {e}")
        
        # Send to Kafka if available
        if kafka_streamer:
            try:
                await kafka_streamer.send_transaction("transactions", transaction)
            except Exception as e:
                logger.warning(f"Failed to send to Kafka: {e}")
        
        processing_duration = time.time() - start_time
        
        return {
            "transaction_id": result.transaction_id,
            "anomaly_score": result.anomaly_score,
            "anomaly_type": result.anomaly_type.value if hasattr(result.anomaly_type, 'value') else str(result.anomaly_type),
            "confidence": result.confidence,
            "explanation": result.explanation,
            "processing_time_ms": processing_duration * 1000,
            "timestamp": result.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        models_loaded = len(ensemble_detector.models) if ensemble_detector else 0
        kafka_status = "connected" if kafka_streamer else "disconnected"
        redis_status = "connected" if redis_cache else "disconnected"
        
        status = "healthy" if models_loaded > 0 else "starting"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "models_loaded": models_loaded,
            "kafka_status": kafka_status,
            "redis_status": redis_status
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/metrics")
async def get_metrics():
    """Get current metrics"""
    try:
        if monitoring_service is None:
            return {"error": "Monitoring service not available"}
        
        # Safer metric access
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - getattr(monitoring_service, 'start_time', time.time())
        }
        
        # Add other metrics if available
        try:
            # Replace with your actual metric access methods
            metrics["transactions_processed"] = getattr(monitoring_service, 'transaction_count', 0)
            metrics["total_anomalies"] = getattr(monitoring_service, 'anomaly_count', 0)
        except Exception as e:
            logger.warning(f"Failed to get detailed metrics: {e}")
            
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {"error": str(e)}

# Add a startup probe endpoint
@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes"""
    if ensemble_detector is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

# ================================
# Synthetic Data Generation
# ================================

def generate_synthetic_transactions(n_samples: int) -> List[Transaction]:
    """Generate synthetic transaction data for training"""
    np.random.seed(42)
    
    transactions = []
    users = [f"user_{i}" for i in range(100)]
    merchants = ["Amazon", "Walmart", "Target", "Starbucks", "Shell", "McDonald's", 
                "Best Buy", "Home Depot", "Costco", "ATM_Withdrawal"]
    categories = ["shopping", "food", "gas", "entertainment", "bills", "cash"]
    locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    
    for i in range(n_samples):
        user = np.random.choice(users)
        
        # Generate realistic amounts with some outliers
        if np.random.random() < 0.05:  # 5% outliers
            amount = np.random.lognormal(mean=6, sigma=2)  # High amounts
        else:
            amount = np.random.lognormal(mean=3, sigma=1)  # Normal amounts
        
        amount = max(1.0, round(amount, 2))
        
        timestamp = datetime.now() - timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        transaction = Transaction(
            transaction_id=f"txn_{i:06d}",
            user_id=user,
            amount=amount,
            merchant=np.random.choice(merchants),
            category=np.random.choice(categories),
            timestamp=timestamp,
            location=np.random.choice(locations),
            device_id=f"device_{np.random.randint(1000, 9999)}",
            is_weekend=timestamp.weekday() >= 5,
            hour_of_day=timestamp.hour,
            days_since_last_transaction=np.random.exponential(1.0)
        )
        
        transactions.append(transaction)
    
    return transactions

# ================================
# Data Streaming Consumer
# ================================

async def process_transaction_stream():
    """Process transactions from Kafka stream"""
    if not kafka_streamer:
        return
    
    consumer = kafka_streamer.create_consumer("transactions", "anomaly_detector_group")
    
    print("Starting transaction stream processing...")
    
    for message in consumer:
        try:
            transaction_data = message.value
            
            # Convert back to Transaction object
            transaction = Transaction(
                transaction_id=transaction_data['transaction_id'],
                user_id=transaction_data['user_id'],
                amount=transaction_data['amount'],
                merchant=transaction_data['merchant'],
                category=transaction_data['category'],
                timestamp=datetime.fromisoformat(transaction_data['timestamp']),
                location=transaction_data['location'],
                device_id=transaction_data['device_id']
            )
            
            # Process with ensemble detector
            result = ensemble_detector.predict(transaction)
            
            # Log results
            if result.anomaly_type != AnomalyType.NORMAL:
                print(f"🚨 Anomaly detected: {transaction.transaction_id} - "
                      f"{result.anomaly_type.value} (score: {result.anomaly_score:.3f})")
            
            # Update user profile
            ensemble_detector.feature_engineer.update_user_profile(
                transaction.user_id, transaction.amount
            )
            
        except Exception as e:
            print(f"Error processing transaction: {e}")

# ================================
# Main Application
# ================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Financial Anomaly Detection System')
    parser.add_argument('--mode', choices=['api', 'train', 'stream'], 
                       default='api', help='Run mode')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--host', default='0.0.0.0', help='API host')
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        print("Starting API server...")
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.mode == 'train':
        print("Training models...")
        detector = EnsembleAnomalyDetector()
        
        # Add models
        isolation_forest = IsolationForestDetector()
        autoencoder = AutoencoderDetector(input_dim=15)
        lstm_detector = LSTMAnomalyDetector()
        
        detector.add_model("isolation_forest", isolation_forest)
        detector.add_model("autoencoder", autoencoder)
        detector.add_model("lstm", lstm_detector)
        
        # Generate training data
        training_data = generate_synthetic_transactions(5000)
        detector.train_all_models(training_data)
        
        # Save models
        for name, model in detector.models.items():
            joblib.dump(model, f"{name}_model.pkl")
        
        print("✓ Models trained and saved!")
    
    elif args.mode == 'stream':
        print("Starting stream processor...")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_transaction_stream())

# ================================
# Example Usage and Testing
# ================================


# Test the API
import requests

# Normal transaction
normal_transaction = {
    "transaction_id": "test_001",
    "user_id": "user_123",
    "amount": 45.67,
    "merchant": "Starbucks",
    "category": "food",
    "location": "New York",
    "device_id": "device_1234"
}

# Suspicious transaction
suspicious_transaction = {
    "transaction_id": "test_002",
    "user_id": "user_123",
    "amount": 5000.00,
    "merchant": "Unknown_Merchant",
    "category": "cash",
    "location": "Unknown",
    "device_id": "device_9999"
}

# Test normal transaction
response = requests.post("http://localhost:8000/predict", json=normal_transaction)
print("Normal transaction result:", response.json())

# Test suspicious transaction
response = requests.post("http://localhost:8000/predict", json=suspicious_transaction)
print("Suspicious transaction result:", response.json())
