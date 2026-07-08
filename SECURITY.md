# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | ✅ |
| < 1.0   | ❌ (pre-release, not supported) |

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security vulnerabilities.

Instead, open a [GitHub security advisory](../../security/advisories/new) on this
repository, or contact the maintainer directly through the profile linked in
[SUPPORT.md](SUPPORT.md). Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce
- Affected version/commit

You should expect an initial response within 5 business days.

## Scope Notes

IKIOS is a research/portfolio project, not a hardened production system. Known,
by-design tradeoffs that are **not** considered vulnerabilities:

- `LLM_PROVIDER=mock` is the default — no external API keys are required or
  contacted unless explicitly configured.
- SQLite is used for metadata/learning/experiment/ML-registry stores in the
  default local deployment; this is a development convenience, not a
  production data store recommendation (see
  [docs/architecture/deployment.md](docs/architecture/deployment.md)).
- Qdrant and Neo4j fall back to embedded/in-memory modes when unreachable —
  this is intentional graceful degradation, not a security boundary.
- **No authentication.** IKIOS assumes a single-tenant, trusted-network
  deployment (e.g. behind a VPN, for one team's internal use). Anyone who can
  reach the API can upload documents and query them. Do not expose this
  deployment directly to the public internet without adding an auth layer.

## Hardening applied in v1.0.1

Following an independent architecture audit prior to the v1.0.1 release:

- **Upload limits**: `/documents/upload` now enforces `MAX_UPLOAD_SIZE_MB`
  (default 25MB, checked while streaming, before the file is fully buffered)
  and validates the file extension before any processing — see
  [docs/api/README.md](docs/api/README.md#error-handling).
- **Indirect prompt injection mitigation**: document-derived text (facts,
  risks, recommendation justifications) reaching the LLM narrator or
  recommendation-refinement calls is now wrapped in explicit
  `<retrieved_evidence>`/`<justifications>` delimiters, with the system
  prompt instructed to treat that content strictly as data, never as
  instructions — see `app/narration/narrator.py` and
  `app/recommendation/engine.py`. **This is a mitigation, not a guarantee**:
  no delimiter-based defense fully eliminates prompt injection risk against a
  sufficiently adversarial document. It only applies when
  `LLM_PROVIDER=anthropic`; the default `mock` provider never calls a real
  model. If you enable a real LLM provider and accept documents from
  untrusted sources, treat narrated output as advisory, not authoritative,
  and consider it alongside the "no authentication" note above.

If you find an actual injection, auth bypass, secret leakage, or similar issue,
please do report it via the process above.
