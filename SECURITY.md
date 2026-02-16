# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Yule, please report it responsibly.

**Email:** burnersiscool@gmail.com

**What to include:**
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential impact assessment

**Response time:** I'll acknowledge your report within 48 hours and provide a fix timeline within 7 days.

## Scope

Security issues in these areas are especially relevant:

- **Sandbox escapes** — bypassing Windows Job Object or future Linux seccomp restrictions
- **Cryptographic integrity** — weaknesses in Merkle verification, Ed25519 signatures, or attestation chain
- **Model file parsing** — malicious GGUF files that could cause memory corruption or code execution
- **API authentication** — capability token bypass or escalation
- **Unsafe code** — memory safety issues in SIMD intrinsics or mmap handling

## Out of Scope

- Denial of service via large model files (expected behavior)
- Issues requiring physical access to the machine
- Social engineering

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |
