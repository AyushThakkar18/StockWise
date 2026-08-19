$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is required."
}

New-Item -ItemType Directory -Force -Path "backups" | Out-Null
docker compose --profile maintenance run --rm backup

