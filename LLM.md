# LLM.md — zooai/gym

**Project**: Zoo Gym — LLM fine-tuning trainer (a Zoo fork of Axolotl).
**Org**: zooai. **Image namespace**: `ghcr.io/zooai/gym`.

## What this repo is
Python ML training framework (`zoo-gym`, src layout `src/gym/`) plus a Quarto
documentation website (`_quarto.yml`, `type: website`). CLI entrypoint `gym`.
**gym.zoo.ngo** serves the rendered Quarto docs.

## docs site — how it builds and ships (live: gym.zoo.ngo)
The docs are the 42 hand-written `.qmd` pages (`index.qmd` + `docs/**`). They are
rendered to `_site/` with `quarto render` and served by the canonical static
runtime `ghcr.io/hanzoai/spa:1.2.0` (`COPY _site /public`, port 3000), deployed
to lux-k8s ns `zoo-mainnet` behind a `hanzo.ai/v1alpha1` IngressRoute — the same
pattern as every other Zoo site in that namespace (zoo-computer is the template).

- **`Dockerfile.site`** — the shipping build. Quarto CLI + `jupyter` only (no
  torch, no editable install). The only Python executed at render time is two
  stdlib cells: `index.qmd` splices in `README.md`; `docs/custom_integrations.qmd`
  lists plugins by reading `src/gym/integrations/*/README.md`. Tiny, CPU-only,
  framework-free build.
- **`deploy/k8s.yaml`** — Deployment + Service + IngressRoute `zoo-gym` in
  `zoo-mainnet` (container `spa`, :3000, `/health` probes,
  `imagePullSecrets: ghcr-luxfi`, IngressRoute websecure `Host(gym.zoo.ngo)`,
  certResolver letsencrypt). `IMAGE_PLACEHOLDER` is patched to the pushed tag.
- **`.github/workflows/docker.yml`** — canonical zooai CI (self-hosted amd64,
  `ghcr.io/zooai/gym:sha-<sha7>`, login `hanzo-dev` via `secrets.GHCR_PAT`).
- **DNS** — Cloudflare zone `zoo.ngo`, record `gym` A → the zoo-mainnet ingress
  origin, proxied (mirrors the `computer` record). CF creds: hanzo-k8s secret
  `hanzo/cloudflare-credentials`.

## What the docs build deliberately omits, and why
The upstream Quarto config also generated two things by **introspecting the
training framework**: a quartodoc **API Reference** (`docs/api`) and a
**Configuration Reference** (`docs/config-reference.qmd`, via the
`pre-render: docs/scripts/generate_config_docs.py` hook). Both require *importing*
the framework, which is not possible in a docs-only, CPU-only build:

1. **Half-rename: the package imports itself as `axolotl`.** The dir is
   `src/gym/` but all 174 internal modules do `from axolotl... import` (nothing
   imports `gym.*`). So `import gym.train` → `ModuleNotFoundError: axolotl`.
   quartodoc (`package: gym`) cannot introspect a single module.
2. **GPU-only `mamba_ssm`, unguarded.** `models/mamba/modeling_mamba.py` does an
   unconditional `from mamba_ssm import ...`; `mamba-ssm` has no CPU wheel and
   cannot even build metadata without `nvcc`. quartodoc aborts on it.
3. **The config schemas pull in torch + transformers.** `generate_config_docs.py`
   imports `axolotl.utils.schemas.config`, whose tree imports `torch`
   (`schemas/enums.py`) and `transformers` (`schemas/training.py`,
   `schemas/validation.py`). Generating it needs the full ML stack.

Decision (forward-correct + cheap): `_quarto.yml` no longer runs the
`pre-render` hook or the `quartodoc:` block, so `quarto render` succeeds with no
framework import. `docs/config-reference.qmd` is now a **hand-written**
reference page (force-tracked; `docs/.gitignore` ignores the generated name) so
the six in-doc links to it resolve. `docs/custom_integrations.qmd` had a
hardcoded `../src/axolotl/integrations` path fixed to `../src/gym/integrations`.

### To restore the auto-generated API + config references later
Make the package importable end-to-end **and** install the CPU ML stack:
finish the `axolotl`→`gym` rename across the 174 modules (or rename `src/gym`
back to `src/axolotl` and set quartodoc `package: axolotl`); guard the
`mamba_ssm` import in `models/mamba/modeling_mamba.py` with `try/except
ImportError`; then `pip install -e .` with CPU torch + quartodoc and re-add the
`quartodoc:` block + `pre-render` hook. That is a framework-packaging change,
not a docs change.

---
This file is the single source of truth for AI assistants on this repo.
