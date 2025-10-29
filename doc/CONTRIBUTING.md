# Contributing to Traffic Forecast

Thank you for considering contributing to the Traffic Forecast project!

## Development Setup

1. **Clone the repository:**

 ```bash
 git clone https://github.com/thatlq1812/dsp391m_project.git
 cd dsp391m_project
 ```

2. **Create conda environment:**

 ```bash
 conda env create -f environment.yml
 conda activate dsp
 ```

3. **Install pre-commit hooks:**
 ```bash
 pip install pre-commit
 pre-commit install
 ```

## Code Standards

### Style Guide

- Follow PEP 8 style guide
- Use Black for code formatting (120 char line length)
- Use isort for import sorting
- Add type hints to all function signatures
- Write docstrings for all public functions/classes

### Testing

- Write unit tests for all new features
- Ensure test coverage >= 70%
- Run tests before submitting PR:
 ```bash
 pytest tests/ -v
 pytest tests/ --cov=traffic_forecast
 ```

### Code Quality

Before submitting, ensure code passes all quality checks:

```bash
# Format code
black traffic_forecast/ tests/
isort traffic_forecast/ tests/

# Run linters
flake8 traffic_forecast/ tests/
pylint traffic_forecast/
mypy traffic_forecast/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Pull Request Process

1. **Create a feature branch:**

 ```bash
 git checkout -b feature/your-feature-name
 ```

2. **Make your changes:**

 - Write clean, well-documented code
 - Add tests for new functionality
 - Update documentation as needed

3. **Commit your changes:**

 ```bash
 git add .
 git commit -m "feat: add your feature description"
 ```

 Use conventional commits:

 - `feat:` new feature
 - `fix:` bug fix
 - `docs:` documentation changes
 - `test:` test additions/changes
 - `refactor:` code refactoring
 - `chore:` maintenance tasks

4. **Push to your fork:**

 ```bash
 git push origin feature/your-feature-name
 ```

5. **Create Pull Request:**
 - Provide clear description of changes
 - Reference related issues
 - Ensure CI checks pass
 - Request review from maintainers

## Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Relevant logs or error messages

## Feature Requests

Feature requests are welcome! Please:

- Check if feature already exists
- Provide clear use case
- Explain expected behavior
- Consider implementation complexity

## Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md with your changes
- Add docstrings to new functions/classes
- Update relevant guides in `doc/`

## Questions?

- Email: fxlqthat@gmail.com
- GitHub Issues: For bug reports and feature requests
- GitHub Discussions: For questions and general discussion

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help create a welcoming community

Thank you for contributing!
