# rocm-repo-template

The `rocm-repo-template` repository serves as the standard template for creating new repositories within the ROCm GitHub organization. It provides a consistent repository structure, governance artifacts, and security controls to help teams onboard new projects quickly while adhering to ROCm security, compliance, and development standards.

## Included Repository Files

The template includes the following foundational files:

- `README.md` – Project overview and documentation
- `.gitignore` – Standard Git ignore rules
- `LICENSE` – Repository licensing information
- `CONTRIBUTING.md` – Contribution guidelines
- `SECURITY.md` – Security reporting and disclosure process
- `.github/CODEOWNERS` – Code ownership and review requirements

## Included Security Controls

The template provides preconfigured GitHub Actions workflows to enable security scanning (PR and Weekly) and governance by default:

- **CodeQL** (`.github/workflows/codeql.yml`)
  Static application security testing (SAST) for code vulnerabilities.

- **GitLeaks** (`.github/workflows/gitleaks.yml`)
  Secret detection to prevent credential leakage.

- **Zizmor** (`.github/workflows/zizmor.yml`)
  GitHub Actions workflow security analysis.

- **Bandit** (`.github/workflows/bandit.yml`)
  Python security linting and vulnerability detection.

- **PR Security Scan** (`.github/workflows/pr-security-scan.yml`)
  Security Controls executed on pull requests.

- **Weekly Security Scan** (`.github/workflows/weekly-security-scan.yml`)
  Scheduled security scans to continuously monitor repository health.

## Purpose

By inheriting this template, all new ROCm repositories start with a common baseline for:

- Repository governance
- Secure development practices
- Code ownership and review controls
- Automated security scanning
- Compliance and audit readiness

This ensures consistency, reduces onboarding effort, and helps maintain security standards across the ROCm ecosystem
