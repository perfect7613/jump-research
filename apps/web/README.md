# JUMP web

Production Next.js interface for declarative visual thought experiments.

```bash
npm install
cp .env.example .env.local
npm run dev
```

The browser calls only `/api/thought-experiments/spec` and `/api/thought-experiments/confirm`. Those server routes validate the exact v2 ingress, stream progress, authenticate to Modal with `JUMP_MODAL_TOKEN`, and validate the returned Spec and Run before releasing them to the client. The bearer token must remain a server-only deployment secret.
