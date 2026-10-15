# Deploying SARAL for a free demo

One container (the `Dockerfile`) serves the API, the agent's streamed replies and the
built interface from one address, on **Google Cloud Run**. The database is **Neon**
(serverless PostgreSQL with pgvector). Both stay within their free allowances for a
demo. To run it without a card, use GitHub Codespaces (last section).

What you need: a Google account with a billing account attached (Cloud Run's free
allowance requires one; a demo stays inside it), a Neon account, the `gcloud` CLI, and
your Gemini API key. Nothing secret is ever put in the repository or the image.

## 1. Database: Neon

1. Create a project at <https://console.neon.tech>, or use an existing one. **Put the
   app and the database in the same area**: every storage call is a round trip, and a
   turn makes dozens. Measured from India against a Neon project in `us-east-2`
   (Ohio): opening a connection took about 2 s, and a drafting turn about 60 s, even
   with pooling. For Neon `us-east-2`, deploy Cloud Run to `us-east5` (Columbus, Ohio)
   or `us-central1`; for an app in Mumbai, create the Neon project in Singapore.
   The app reads `SARAL_DATABASE_URL`, or else the variables Neon's CLI writes
   (`DATABASE_URL_UNPOOLED`, then `DATABASE_URL`).
2. Copy the **direct** connection string (host without `-pooler`). The agent's memory
   store needs a normal session connection, which the pooled endpoint does not give.
   It looks like `postgresql://USER:PASSWORD@ep-xxx.REGION.aws.neon.tech/neondb?sslmode=require`.
3. Nothing else: on start-up the service creates the `vector` extension, applies its
   migrations and the agent's checkpoint tables.

Neon suspends an idle database after 5 minutes and wakes in well under a second. The
service's connections are checked before use, so the first turn after a pause works
(tested by killing every server connection between two turns).

## 2. Google Cloud: one-time setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com secretmanager.googleapis.com

# Secrets live in Secret Manager, never in the image or the repository.
printf %s "YOUR_GEMINI_API_KEY"      | gcloud secrets create gemini-api-key --data-file=-
printf %s "YOUR_NEON_DIRECT_URL"     | gcloud secrets create saral-database-url --data-file=-

# Let Cloud Run's runtime account read them.
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')
for secret in gemini-api-key saral-database-url; do
  gcloud secrets add-iam-policy-binding "$secret" \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
done
```

In the Google Cloud console, **Billing → Budgets & alerts**, add a budget of $1 with
alerts, so any usage beyond the free allowance is noticed at once.

## 3. Deploy

From the repository root (Cloud Build builds the `Dockerfile`; no local Docker needed):

```bash
gcloud run deploy saral --source . --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 --concurrency 20 \
  --min-instances 0 --max-instances 2 \
  --timeout 900 \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest,SARAL_DATABASE_URL=saral-database-url:latest \
  --set-env-vars SARAL_TEXT_MODEL=gemini-3.5-flash-lite,SARAL_VERIFIER_MODEL=gemini-3.1-flash-lite
```

The command prints the service URL. Open it; `/health` must report `"agent_ready": true`.

Why these settings:

| Setting | Reason |
|---|---|
| `--memory 2Gi` | Measured in this image: about 0.6 GB after two streamed turns (the local embedding model is most of it). 2 GiB leaves room for parsing large uploads; 1 GiB also works and doubles the free hours. |
| `--concurrency 20` | One process streams many turns on one event loop; most of a turn is waiting for Gemini. |
| `--min-instances 0` | Scales to zero when idle, so idle time costs nothing. The first request after idle takes about 10–20 s to start. |
| `--max-instances 2` | Caps the worst case well inside the free allowance. |
| `--timeout 900` | A turn takes seconds, but the streamed connection must not be cut by the default 5 minutes on a slow writer. |

## 4. Before a demo

Cold starts are the only slow part. A minute before presenting, open the URL (or
`curl https://YOUR-URL/health`) so an instance is warm. Upload the deck once; the same
file in a new chat reuses its stored index.

## Cost

Cloud Run's free allowance each month is 2 million requests, 360,000 GB-seconds of
memory and 180,000 vCPU-seconds (request-based billing: an idle instance is not
billed). At 2 GiB that is about 50 hours of active request time, far beyond a demo.
Cloud Build's free build minutes cover a few deploys a day. The image is about 1 GB
and Artifact Registry's free storage is 0.5 GB, so storing it can cost a few cents a
month; delete old images to keep it at that. Gemini usage is billed by Google AI
separately, as in development. Check the dated prices on the provider pages before
relying on these figures.

## Run the same image locally

```bash
docker build -t saral .
docker run --rm -p 8080:8080 -e GEMINI_API_KEY=... -e SARAL_DATABASE_URL=... saral
```

Locally, a database in another container is reached at `host.docker.internal`.

## No card: GitHub Codespaces

A codespace is a cloud machine (2 CPUs, 8 GB) that GitHub gives personal accounts
for free each month, without a card. It runs the whole app; the database stays on
Neon, so documents and conversations persist between codespace sessions.

### One-time setup (about 5 minutes)

1. **Secrets.** github.com → Settings → Codespaces → *Secrets* → New secret, and give
   each access to this repository:
   - `GEMINI_API_KEY`: your Gemini key.
   - `SARAL_DATABASE_URL`: Neon's **direct** connection string (the host without
     `-pooler`).
2. **Create the codespace.** On the repository page: Code → Codespaces → *Create
   codespace on* the branch to demo. The first build installs everything (about
   5 minutes); SARAL then starts on its own on port 8000.
3. **Make the link shareable.** In the codespace, open the *Ports* tab, right-click
   port 8000 → *Port visibility* → **Public**. Copy its address
   (`https://<name>-8000.app.github.dev`) and share that. A private port asks visitors
   to sign in to GitHub as you.

### Before a demo

- Start the codespace a few minutes early (github.com/codespaces → the codespace →
  Open); opening it also starts SARAL if it is not running. It sleeps after 30 minutes
  without activity; raise the timeout to 4 hours (the maximum) in Settings →
  Codespaces → *Default idle timeout*. Visits to the app link do not count as
  activity, only the editor or terminal does.
- Check `https://<name>-8000.app.github.dev/health` shows `"agent_ready": true`.
- Port visibility resets to private when a codespace is recreated; set it again.

### If something is off

- The log: `tail -f /tmp/saral.log` in the codespace terminal.
- Restart the app: `bash .devcontainer/start.sh`.
- Secrets added after the codespace was created only arrive after a restart
  (Codespaces menu → Stop current codespace, then open it again).

### Limits

Free personal usage covers about 60 hours a month on this 2-core machine, plus 15 GB
of storage; stop the codespace when you are done so the hours last. The address
changes if you create a new codespace. For a permanent address, use Cloud Run (above).
