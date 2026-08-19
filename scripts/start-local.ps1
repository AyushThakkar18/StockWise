$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is required. Install it, start it, then run this script again."
}

if (-not (Test-Path ".env")) {
    throw "Create .env from .env.example and add the required keys first."
}

docker compose up --build --detach
docker compose ps

