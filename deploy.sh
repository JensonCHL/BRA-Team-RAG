#!/bin/bash
# deploy.sh - Deployment script for RAG application

set -e  # Exit on any error

echo "Starting deployment process..."

# Pull latest changes from repository
echo "Pulling latest changes from git..."
git pull origin main

# Check if there are any changes
if [[ -n $(git status --porcelain) ]]; then
    echo "Changes detected, rebuilding containers..."
    
    # Build and start containers
    echo "Building and starting containers..."
    docker-compose up --build -d
    
    # Show container status
    echo "Container status:"
    docker-compose ps
    
    echo "Deployment completed successfully!"
else
    echo "No changes detected, restarting containers..."
    
    # Just restart containers to pick up any config changes
    docker-compose restart
    
    echo "Containers restarted successfully!"
fi

# Show logs
echo "Showing last 20 lines of application logs:"
docker-compose logs --tail=20