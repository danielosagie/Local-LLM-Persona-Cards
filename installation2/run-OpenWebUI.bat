@echo off
echo Waiting for Docker to be ready...
:DOCKERCHECK
docker info > nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 2 > nul
    goto DOCKERCHECK
)
echo Docker is ready. Starting OpenWebUI...
docker start open-webui || docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
echo Waiting for OpenWebUI to start...
timeout /t 10 > nul
start http://localhost:3000