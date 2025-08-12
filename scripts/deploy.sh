#!/bin/bash

set -e

# Configuration
NAMESPACE="fraud-detection"
RELEASE_NAME="fraud-detection"
CHART_PATH="./helm/fraud-detection"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not configured or cluster is not accessible"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace ${NAMESPACE}..."
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
}

# Deploy using Helm
deploy_application() {
    log_info "Deploying application using Helm..."
    
    # Add required Helm repositories
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    # Deploy the application
    helm upgrade --install ${RELEASE_NAME} ${CHART_PATH} \
        --namespace ${NAMESPACE} \
        --set image.tag=${IMAGE_TAG:-latest} \
        --wait \
        --timeout=15m
    
    log_info "Application deployed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check deployment status
    kubectl get deployments -n ${NAMESPACE}
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=fraud-detection \
        -n ${NAMESPACE} \
        --timeout=300s
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get svc ${RELEASE_NAME} \
        -n ${NAMESPACE} \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_IP" ]; then
        SERVICE_IP=$(kubectl get svc ${RELEASE_NAME} \
            -n ${NAMESPACE} \
            -o jsonpath='{.spec.clusterIP}')
        log_warn "Using ClusterIP ${SERVICE_IP} (LoadBalancer IP not available)"
    else
        log_info "Service available at ${SERVICE_IP}"
    fi
    
    # Test health endpoint
    sleep 30  # Give service time to start
    
    if curl -f http://${SERVICE_IP}:8000/health &> /dev/null; then
        log_info "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."
    helm rollback ${RELEASE_NAME} -n ${NAMESPACE}
    log_info "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up resources..."
    helm uninstall ${RELEASE_NAME} -n ${NAMESPACE} || true
    kubectl delete namespace ${NAMESPACE} || true
    log_info "Cleanup completed"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        deploy)
            check_prerequisites
            create_namespace
            deploy_application
            verify_deployment
            log_info "🎉 Deployment completed successfully!"
            ;;
        rollback)
            rollback
            ;;
        cleanup)
            cleanup
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|cleanup}"
            exit 1
            ;;
    esac
}

# Trap errors and rollback
trap 'log_error "Deployment failed! Rolling back..."; rollback; exit 1' ERR

main "$@"