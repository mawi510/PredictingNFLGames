# Infrastructure & Deployment

Two things run on the one EC2 box, both behind nginx + Let's Encrypt:
- **Model API** — Docker container at `https://api.promatchpredict.com`. CI builds
  the image, pushes to GHCR, restarts the container over SSH on merge to `main`.
- **Frontend** — static site at `https://promatchpredict.com`. CI builds the Next
  export and rsyncs it to `/var/www/promatchpredict` on merge to `main`.

```
api:  merge -> build image -> push GHCR -> SSH -> docker compose pull && up -d
web:  merge -> next build (export) -> rsync out/ -> /var/www/promatchpredict
```

## One-time EC2 setup for the frontend

```bash
# Web root the deploy rsyncs into (owned by the SSH user so no sudo in CI).
sudo mkdir -p /var/www/promatchpredict
sudo chown ec2-user:ec2-user /var/www/promatchpredict

# nginx vhost for the apex + www. REMOVE any old Streamlit proxy config for this
# domain first (it's what's currently returning 502).
sudo cp infra/nginx-web.conf /etc/nginx/conf.d/promatchpredict.com.conf
sudo nginx -t && sudo systemctl reload nginx

# DNS: A records for promatchpredict.com AND www.promatchpredict.com -> Elastic IP.
# Then issue the cert (after DNS resolves):
sudo certbot --nginx -d promatchpredict.com -d www.promatchpredict.com
```

## Required GitHub Actions secrets

| Secret | Used by | What it is |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | weekly-data.yml | IAM user `nfl-pipeline` keys (S3 access) |
| `EC2_HOST` | deploy-api.yml | EC2 public IP or DNS name |
| `EC2_USER` | deploy-api.yml | SSH user (e.g. `ec2-user`) |
| `EC2_SSH_KEY` | deploy-api.yml | Contents of the `.pem` private key (full file, incl. BEGIN/END lines) |

`GITHUB_TOKEN` is provided automatically and is used to push to / pull from GHCR.

## One-time EC2 provisioning

Done once on the box (the deploy workflow handles everything after).

```bash
# 1. Docker + compose plugin (Amazon Linux 2023)
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user   # re-login after this
sudo dnf install -y docker-compose-plugin   # or the compose-plugin package for your AMI

# 2. App directory + compose file
mkdir -p ~/nfl-spread-model && cd ~/nfl-spread-model
# copy infra/docker-compose.yml here (scp, or curl the raw file from the repo)

# 3. Secrets for the container (NOT committed) — same IAM keys as CI
cat > ~/nfl-spread-model/.env <<'EOF'
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
EOF
chmod 600 ~/nfl-spread-model/.env

# 4. nginx + TLS
sudo dnf install -y nginx
sudo cp infra/nginx.conf /etc/nginx/conf.d/api.promatchpredict.com.conf
sudo systemctl enable --now nginx
# Point an A record api.promatchpredict.com -> this EC2's public IP at your
# DNS provider FIRST, then issue the cert:
sudo dnf install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.promatchpredict.com
```

Open the security group for inbound 80 + 443. Port 8000 stays closed — it's
bound to 127.0.0.1 and only nginx reaches it.

## Manual redeploy

Trigger the **Deploy Model API** workflow via `workflow_dispatch`, or on the box:

```bash
cd ~/nfl-spread-model && docker compose pull && docker compose up -d
```

## Notes

- The model artifact is **not** baked into the image; the container pulls
  `s3://nfl.data/models/cover_classifier.joblib` at boot via `MODEL_S3_URI`.
- The image targets `linux/amd64`. If you move to a Graviton (arm64) instance,
  change `platforms:` in `deploy-api.yml`.
- Cert auto-renewal: certbot installs a systemd timer. The old
  `run_certbot_update.py` SSH script is no longer needed.
