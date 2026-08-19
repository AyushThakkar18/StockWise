# Zero-cost deployment

## Supported target: private self-hosting

Install Docker Desktop, create `.env`, and run `scripts/start-local.ps1`. The API is available at `http://localhost:8000`; the worker consumes durable snapshot jobs. Both use the named `portfoliopilot_data` volume. Secrets are injected from `.env` and are excluded from image build context.

This has no hosting bill beyond electricity and the OpenAI requests you choose to enable. The PC must remain powered on, connected, and awake during scheduled workflows. Run `scripts/backup-local.ps1` regularly and copy backups to another device.

## Optional hosted topology later

Railway is no longer the default because it does not satisfy the strict zero-cost requirement. If a permanent hosted service is desired later, migrate persistence to PostgreSQL and re-evaluate current free tiers; free-tier availability and limits change and must not be assumed.

Keep the API bound to localhost by default. If remote access is later enabled through a free tunnel, first set a randomly generated `PORTFOLIOPILOT_API_TOKEN` of at least 32 characters and treat the tunnel provider's current limits and terms as external dependencies. Live brokerage execution remains absent.

## Free always-on VM option

If Oracle Cloud's Always Free compute capacity is available in your region, a small Ubuntu VM can run the same Docker Compose services without leaving your laptop on. Oracle may require account and payment-method verification, capacity is not guaranteed, and staying inside the current Always Free quotas is your responsibility. Verify the current terms before provisioning.

Keep port 8000 private. Start the stack with both Compose files:

```bash
docker compose -f compose.yaml -f compose.server.yaml up --build -d
```

Access it from your laptop through an SSH tunnel:

```powershell
ssh -L 8000:127.0.0.1:8000 ubuntu@YOUR_VM_IP
```

Then open `http://127.0.0.1:8000`. This avoids purchasing a domain or exposing the API directly. Set `PORTFOLIOPILOT_API_TOKEN` and run the equivalent of `scripts/server-check.ps1` before starting it.

Google Cloud's free `e2-micro` allowance may be another option in eligible regions, but billing-account requirements, network usage, disk limits, and current free-tier terms must be checked carefully. No external provider can be guaranteed to remain free indefinitely.

## Required environment

- `ALPHA_VANTAGE_API_KEY`: required by the market snapshot worker.
- `OPENAI_API_KEY`: optional until AI research jobs are enabled.
- `OPENAI_MODEL`: defaults to `gpt-4o-mini`.
- `FRED_API_KEY`: reserved for the vintage-aware macro adapter.
- `PORTFOLIOPILOT_DB_PATH`: local/staging SQLite path; replaced by a database URL during PostgreSQL migration.
- `PORTFOLIOPILOT_API_TOKEN`: required before exposing the API beyond localhost.
- `PORTFOLIOPILOT_BACKUP_DIR`: verified backup destination.

## Backups and recovery

`scripts/backup-local.ps1` uses SQLite's online backup API and runs `PRAGMA integrity_check` on the result. Stop the containers before manually replacing the active database during a restore. Backups on the same disk do not protect against disk failure.
