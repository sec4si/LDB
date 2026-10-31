# Security Policy

## Reporting a vulnerability

Please do not open a public issue. Report vulnerabilities privately through
[GitHub security advisories](https://github.com/sec4si/LDB/security/advisories/new).
Include the affected version or commit, reproduction steps, and the impact you observed.

## Scope

LDB attaches to model-serving processes, so the following are in scope:

- the local agent socket and its descriptor files (expected to be user-owned, `0600`);
- runtime plugins and hot attachment (`ldb hot-attach`, which relies on ptrace);
- Neural Patch and RunBundle parsing.

Running LDB requires the same privileges as the process it instruments; issues that
require an attacker who already has those privileges are generally out of scope.
