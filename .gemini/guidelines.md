Instructions for Read-Only Code Repository Audit
Primary Directive: You are to perform a comprehensive, read-only audit of the provided Python code repository. Your sole function is to analyze the repository against the checklist below and generate a detailed report of your findings.

Critical Constraint: DO NOT MODIFY ANY FILES. You must not add, delete, or alter any code, comments, or configuration files. Your output is a report, not a set of changes.

## Audit Checklist & Reporting Criteria
For each item, assess the repository and document your findings in the final report.

### 1. README.md Audit 📝
Presence: Verify the existence of a README.md file in the root directory. Report PRESENT or ABSENT.

Content Analysis: Scan the README.md and report on the presence or absence of the following sections. Note any missing elements.

[ ] Project Title

[ ] Project Description/Summary

[ ] Installation Instructions

[ ] Code Usage Example(s)

[ ] License Information

[ ] Contribution Guidelines (Optional, note if present)

### 2. Licensing Audit ⚖️
LICENSE File: Verify the existence of a LICENSE or LICENSE.md file in the root directory. Report PRESENT or ABSENT.

Content Identification: If the file is present, identify the open-source license it contains (e.g., "MIT License", "Apache License 2.0"). Report the identified license or state UNKNOWN if it's non-standard.

Consistency Check: Compare the license identified in the LICENSE file with the license declared in the packaging metadata (pyproject.toml or setup.py). Report MATCH or MISMATCH.

### 3. Project Structure & Naming Audit 📂
Layout Assessment: Analyze the directory structure. Report whether it follows a standard source layout (src/<package_name>) or a direct layout (<package_name>).

__init__.py Files: Scan all subdirectories containing Python modules. Report a list of package directories that are missing an __init__.py file.

Naming Convention Review: Scan all file and directory names within the source code folder. Report any significant deviations from the standard Python snake_case convention.

### 4. Code Quality & Documentation Audit 🧑‍💻
Docstring Coverage: Sample several public functions/classes/modules. Report on the general state of docstring coverage (HIGH, MEDIUM, LOW, or NONE). Note if key functions lack documentation.

Type Hinting Usage: Scan the codebase for the use of type hints. Report on the extent of their usage (EXTENSIVE, PARTIAL, or NONE).

PEP 8 Conformance: Programmatically assess or sample key files for major PEP 8 style guide violations (e.g., indentation, line length, naming). Report a general conformance level (GOOD, NEEDS IMPROVEMENT) and list examples of common violations found.

### 5. Dependency Management Audit 🔗
Dependency File Location: Identify and report the file used for dependency management (e.g., pyproject.toml, requirements.txt).

Dependency Specification: Analyze the listed dependencies. Report whether versions are strictly pinned (package==1.2.3), specified with ranges (package>=1.2.0), or unconstrained (package).

### 6. Testing Audit 🧪
Test Directory Presence: Verify the existence of a tests/ or test/ directory. Report PRESENT or ABSENT.

Test File Identification: If a test directory exists, report on the presence of test files (e.g., files named test_*.py). Report FOUND or NOT FOUND.

### 7. Packaging & Distribution Audit 📦
Packaging File: Verify the presence of a pyproject.toml file. This is the modern standard. Note if only a legacy setup.py is present.

Metadata Completeness: Read the packaging metadata and report on the status (COMPLETE or INCOMPLETE) of the following essential fields:

name

version (Note the current version number, e.g., 0.1.0)

authors

description

readme

license

classifiers

### 8. Version Control (.gitignore) Audit 🌿
.gitignore Presence: Verify the existence of a .gitignore file. Report PRESENT or ABSENT.

Content Analysis: Scan the .gitignore content. Report if it is missing rules for any of the following common patterns:

__pycache__/

Virtual environment directories (.venv/, venv/)

Build artifacts (dist/, build/, *.egg-info/)

Common IDE folders (.idea/, .vscode/)

### Final Output: The Audit Report
Combine all findings into a single, structured markdown report titled "Repository Pre-Publishing Audit". The report should be clearly organized using the section headers above. Each checklist item should have a clear finding associated with it. No other action should be taken.