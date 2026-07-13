# pipelines on EKS

Helm chart migrating the Open WebUI **Pipelines** filter service off Porter onto
the self-managed EKS cluster (hub-and-spoke ArgoCD).

## Shape

FastAPI + **uvicorn on :8080** (`main:app`), called in-cluster at
`/v1/perform_filters`. **STATELESS** — no database, no redis, no migrations.
Deployed **PRIVATE**: ClusterIP service (`pipelines-prod:8080`), **no ingress,
no DNS**. It is an arbitrary-code plugin runtime, so it must never be publicly
reachable; the only callers are `ai-gateway` and `ai-gateway-data`. **No IRSA**
(touches no AWS resource at runtime).

- `chart/` — Deployment + Service + HPA + ExternalSecret (+ gated, disabled Ingress).
- `prod/values.yaml` — prod overlay.

## Key decisions

- **Private** — `ingress.enabled: false`; reached only over the ClusterIP service.
- **`PIPELINES_API_KEY`** comes from AWS Secrets Manager (`prod/pipelines`) via
  ESO — never the app's insecure `0p3n-w3bu!` default (`config.py`).
- **Probes** hit `GET /` (`get_status` → unauthenticated `200 {"status": true}`);
  no authed endpoint is probed. Startup probe is generous (5 min) for cold
  model load.
- **No migrate Job** — the service is stateless.
- **CI actions are SHA-pinned** (supply-chain: immutable, tag-retarget proof).

## Deploy prerequisites (operator)

1. **ECR** `k8s/pipelines` — already in `live/_env/ecr.hcl` (apply `live/dev/ecr`).
2. **OIDC trust** — add `pipelines` to `live/dev/github-actions-oidc` `github_repos`
   (companion unbound-infra PR), then `terragrunt apply live/dev/github-actions-oidc`.
3. **Secrets Manager** `prod/pipelines` = `{"PIPELINES_API_KEY":"…"}`.
4. **ArgoCD** — `repo-pipelines` secret (`live/dev/addons`) +
   `argocd-apps/prod/pipelines.yaml` (companion unbound-infra PR).

## Verify (soak, before gateway cutover)

```bash
kubectl -n prod get pods -l app=pipelines-prod
kubectl -n prod port-forward deploy/pipelines-prod 8080:8080 &
curl -s localhost:8080/            # {"status":true}
```

## Cutover

pipelines deploys privately and idle. At the later gateway cutover, flip
`UNBOUND_PIPELINES_URL` → `http://pipelines-prod:8080/v1/perform_filters` in the
`ai-gateway` and `ai-gateway-data` prod values, then scale Porter pipelines to 0.
