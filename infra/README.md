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

The analyst job (analyst-picks.yml) needs no Claude API key. It signs in with
GitHub's identity token through workload identity federation. One-time setup:

1. Claude Console (platform.claude.com, not claude.ai) > Settings > Workload
   identity > Connect workload > GitHub Actions. In the Federation rule step:
   - GitHub organization `mawi510`, repository `PredictingNFLGames`, branch
     `refs/heads/main`, organization numeric ID `117122865`.
   - Allowed events: the job is started by workflow dispatch, so "Only push"
     rejects it. Pick workflow_dispatch if the list has it. If not, open
     Advanced match options (it then replaces the fields above) and set subject
     pattern `repo:mawi510/PredictingNFLGames:ref:refs/heads/main`, plus claim
     conditions `event_name` = `workflow_dispatch` and `repository_owner_id` =
     `117122865`. Leave the audience blank.
   - Pick one workspace. Keep scope `workspace:developer` and the 10-minute
     token lifetime.
   - Target step: create a service account for the job.
2. Add four **repository secrets** (Settings > Secrets and variables > Actions >
   Secrets, not Environments). They cannot sign anyone in on their own, but run
   logs are public on this repo and secrets show there as `***`:
   `ANTHROPIC_FEDERATION_RULE_ID`
   (`fdrl_...`), `ANTHROPIC_ORGANIZATION_ID` (Console > Settings >
   Organization), `ANTHROPIC_SERVICE_ACCOUNT_ID` (`svac_...`) and
   `ANTHROPIC_WORKSPACE_ID` (`wrkspc_...`).

A rejected sign-in shows as a 401 in the job log. The Console's authentication
history gives the reason; a wrong subject prefix is the most common one.

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

## Starting the scheduled jobs on time

GitHub's cron starts our jobs 4 to 11 hours late (the 11:00 UTC results job ran
between 14:46 and 22:17 UTC in Nov 2026). A cron entry on the box starts them on
time through GitHub's workflow dispatch API instead. The GitHub schedules stay in
the workflows as a backup; a second run of either job changes nothing.

1. Create a fine-grained GitHub token: Settings > Developer settings > Fine-grained
   tokens. Repository access: only `mawi510/PredictingNFLGames`. Permissions:
   Actions, read and write. Nothing else.
2. On the box, as `ec2-user`:

```bash
mkdir -p ~/bin ~/.config/promatchpredict
curl -fsSL https://raw.githubusercontent.com/mawi510/PredictingNFLGames/main/infra/dispatch-workflow.sh -o ~/bin/dispatch-workflow.sh
chmod 700 ~/bin/dispatch-workflow.sh
# Paste the token, then Enter, then Ctrl-D. It never appears in shell history.
cat > ~/.config/promatchpredict/gh-dispatch-token
chmod 600 ~/.config/promatchpredict/gh-dispatch-token
# Test: this starts a results run now (Actions tab shows it within seconds).
~/bin/dispatch-workflow.sh daily-results.yml
sudo curl -fsSL https://raw.githubusercontent.com/mawi510/PredictingNFLGames/main/infra/cron.d/promatchpredict-dispatch -o /etc/cron.d/promatchpredict-dispatch
sudo chmod 644 /etc/cron.d/promatchpredict-dispatch
```

Each dispatch appends a line to `~/dispatch.log`. Re-run the last `sudo curl`
whenever `infra/cron.d/promatchpredict-dispatch` changes.

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
