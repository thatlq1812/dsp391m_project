# Security Policy

## Supported Versions

Currently supported versions for security updates:

| Version | Supported |
| ------- | ------------------ |
| 4.1.x | :white_check_mark: |
| 4.0.x | :white_check_mark: |
| < 4.0 | :x: |

## Reporting a Vulnerability

If you discover a security vulnerability, please follow these steps:

### 1. **Do NOT** create a public GitHub issue

Security vulnerabilities should not be publicly disclosed until a fix is available.

### 2. Email the security team

Send details to: **fxlqthat@gmail.com**

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Wait for acknowledgment

You will receive an acknowledgment within 48 hours.

### 4. Coordinated disclosure

We will:

- Investigate the issue
- Develop a fix
- Release a security update
- Credit you in the release notes (if desired)

## Security Best Practices

### API Keys and Secrets

- **Never** commit API keys or secrets to the repository
- Use `.env` files for local development (not tracked in git)
- Use environment variables for production
- Rotate API keys regularly

### Dependencies

- Keep dependencies up to date
- Run `pip list --outdated` regularly
- Use `safety check` to scan for known vulnerabilities
- Review dependency changes before updating

### Data Security

- Sanitize all user inputs
- Validate data before processing
- Use parameterized queries for database access
- Encrypt sensitive data at rest and in transit

### Code Security

- Use Bandit for security scanning
- Enable all security-related linters
- Follow OWASP security guidelines
- Implement proper error handling

## Security Tools

This project uses:

- **Bandit** - Security linter for Python
- **Safety** - Dependency vulnerability scanner
- **GitHub Dependabot** - Automated dependency updates
- **CodeQL** - Code scanning for vulnerabilities

## Disclosure Policy

- Security issues will be addressed with high priority
- Security patches will be released ASAP
- Public disclosure occurs after fix is available
- Credit will be given to reporters (unless anonymous)

## Contact

Security Team: fxlqthat@gmail.com

## Security Updates

Security updates will be announced in:

- GitHub Security Advisories
- CHANGELOG.md
- Release notes
- Email to known users (if applicable)

Thank you for helping keep Traffic Forecast secure!
