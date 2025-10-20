@echo off
REM deploy.bat - Deployment script for RAG application (Windows)

echo Starting deployment process...

REM Pull latest changes from repository
echo Pulling latest changes from git...
git pull origin main

REM Check if there are any changes
git status --porcelain > temp_status.txt
findstr /V "^$" temp_status.txt > temp_status_filtered.txt

if %ERRORLEVEL% EQU 0 (
    if exist temp_status_filtered.txt (
        for /F %%A in ('type temp_status_filtered.txt') do (
            set CHANGES_FOUND=1
            goto :build
        )
    )
)

:build
if defined CHANGES_FOUND (
    echo Changes detected, rebuilding containers...
    
    REM Build and start containers
    echo Building and starting containers...
    docker-compose up --build -d
    
    REM Show container status
    echo Container status:
    docker-compose ps
    
    echo Deployment completed successfully!
) else (
    echo No changes detected, restarting containers...
    
    REM Just restart containers to pick up any config changes
    docker-compose restart
    
    echo Containers restarted successfully!
)

REM Show logs
echo Showing last 20 lines of application logs:
docker-compose logs --tail=20

REM Cleanup temp files
if exist temp_status.txt del temp_status.txt
if exist temp_status_filtered.txt del temp_status_filtered.txt