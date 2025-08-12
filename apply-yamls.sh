#!/bin/bash

# List of YAML files to apply
FILES=(
  "anomaly-detector-deployment.yaml"
  "anomaly-detector-hpa.yaml"
  "anomaly-detector-service.yaml"
  "app-config.yaml"
  "configmap.yaml"
  "grafana/grafana-deployment.yaml"
  "grafana/grafana-pvc.yaml"
  "grafana/grafana-service.yaml"
  "grafana/grafana-deployment.yaml"
  "grafana/grafana-pvc.yaml"
  "grafana/grafana-service.yaml"
  "helm/fraud-detection/templates/kafka-headless-service.yaml"
  "helm/fraud-detection/templates/deployment.yml"
  "kafka-deployment.yaml"
  "kafka-service.yaml"
  "kafka-single-node-fixed.yaml"
  "model-pvc.yaml"
  "postgres-deployment.yaml"
  "postgres-pvc.yaml"
  "postgres-service.yaml"
  "prometheus-configmap.yaml"
  "prometheus-deployment.yaml"
  "prometheus-pvc.yaml"
  "prometheus-service.yaml"
  "redis-deployment.yaml"
  "redis-pvc.yaml"
  "redis-service.yaml"
  "zookeeper-deployment.yaml"
  "zookeeper-service.yaml"
  "namespace.yaml"
  "secrets.yaml"
)

# Apply all YAML files
for FILE in "${FILES[@]}"
do
  echo "Applying $FILE"
  kubectl apply -f "$FILE"
done
