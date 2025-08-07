. Initial Setup
Fork the repository on GitHub to your own account
Clone your fork locally: git clone https://github.com/yourusername/exactcis.git
Set up development environment using uv:
uv venv
source .venv/bin/activate  # macOS/Linux
uv pip install -e ".[dev]"
2. Create Feature Branch
Create a descriptive branch from main:
git checkout -b feature/your-feature-name
Use clear naming like feature/blaker-optimization or feature/new-ci-method
3. Development Standards
Follow coding style: Use Black formatter and PEP 8 guidelines
Add type hints consistently throughout your code
Write comprehensive docstrings for all functions, classes, and modules
Keep changes focused: Each PR should address a single concern
4. Testing Requirements
The project has a unified testing system. You must:
Add tests for new functionality (unit tests, integration tests, edge cases)
Mark tests appropriately:
@pytest.mark.fast for quick tests (< 1 second)
@pytest.mark.slow for comprehensive tests
Run tests locally using the unified test runner:
python scripts/run_tests_unified.py  # Fast tests only
python scripts/run_tests_unified.py --all  # All tests
python scripts/run_tests_unified.py --coverage  # With coverage
5. Documentation Updates
Update docstrings for any modified functions/classes
Update user documentation in the docs/ directory
Add examples for new functionality
Update API reference if you've changed public interfaces
6. Special Considerations for Statistical Methods
If adding/modifying statistical methods:
Validate against established implementations (R, SciPy, etc.)
Document any differences in validation summaries
Include benchmark results for performance changes
Profile your changes using the provided profiling tools
7. Commit and Push
Make clear, descriptive commits:
git commit -m "Add feature: brief description of what was added"
Push your branch to your fork:
git push origin feature/your-feature-name
8. Submit Pull Request
Create a PR from your fork to the main repository
Fill out the PR template with details about your changes
Include test results and validation if applicable
9. Review Process
Automated tests will run (CI/CD pipeline)
Maintainers will review your code
Address feedback promptly
Iterate until approved and merged
Key Success Tips
Start small - break large features into smaller, logical PRs
Test thoroughly - both fast and comprehensive tests
Document well - clear docstrings and user documentation
Follow existing patterns - study the codebase structure first
Validate statistically - especially important for this mathematical library
The project emphasizes quality, statistical accuracy, and comprehensive testing, so take time to ensure your contributions meet these standar