# Deployment Instructions

## Prerequisites
1. Docker and Docker Compose installed on your VM
2. Git installed on your VM
3. This repository cloned to your VM

## Initial Setup

1. Clone the repository to your VM:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. Create the required configuration files:
   ```bash
   # Copy and edit the secrets file
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   nano .streamlit/secrets.toml  # Edit with your Google OAuth credentials
   
   # Copy and edit the environment file
   cp .env.example .env
   nano .env  # Edit with your configuration values
   ```

3. Build and start the application:
   ```bash
   docker-compose up --build -d
   ```

## Automatic Updates with Watchtower

This setup includes Watchtower, which automatically updates your containers when the base image changes. Watchtower checks for updates every 30 seconds.

## Manual Deployment

To manually deploy updates:

1. On your development machine:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```

2. On your VM:
   ```bash
   # Using the deployment script
   ./deploy.sh  # Linux/Mac
   deploy.bat   # Windows
   
   # Or manually:
   git pull origin main
   docker-compose up --build -d
   ```

## Environment-Specific Configuration

Different environments (development, staging, production) can use different configuration files:

- Development: Use `.env` and `.streamlit/secrets.toml` locally
- Production VM: Create `.env` and `.streamlit/secrets.toml` on the VM with production values

## Data Persistence

The following directories are mounted as volumes for data persistence:
- `artifacts/` - Application artifacts
- `uploads/` - Uploaded PDF files

These directories will persist between container rebuilds.

## Monitoring

View application logs:
```bash
docker-compose logs -f
```

Check container status:
```bash
docker-compose ps
```

## Security Considerations

1. Never commit `.env` or `.streamlit/secrets.toml` to version control
2. Use strong passwords and secrets
3. Consider using HTTPS in production with a reverse proxy like Nginx
4. Regularly update base Docker images