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
  [docs/deployment/production.md](docs/deployment/production.md)).
- Qdrant and Neo4j fall back to embedded/in-memory modes when unreachable —
  this is intentional graceful degradation, not a security boundary.

If you find an actual injection, auth bypass, secret leakage, or similar issue,
please do report it via the process above.
