# Analysis Directory

This directory contains investigation files, diagnostic reports, and validation scripts related to the recent fixes for the ExactCIs package.

## Contents

### Investigation Reports
- `blaker_findings.md` - Initial analysis of Blaker method issues
- `junie_blaker_method_investigation.md` - Detailed investigation by Junie
- `claude_blaker_diagnostics.md` - Claude's diagnostic analysis

### Validation Scripts
- `quick_sense_check_test.py` - Original three-scenario validation test
- `comprehensive_outlier_test.py` - Comprehensive 10-scenario outlier detection test
- `test_blaker_debug_case.py` - Specific debug case for Blaker method

## Usage

These files document the investigation process that led to identifying and fixing critical issues in the Blaker and Unconditional methods. They serve as:

1. **Historical record** of the debugging process
2. **Validation tools** for future regression testing
3. **Reference material** for understanding the fixes implemented

## Key Results

All major outliers have been resolved through targeted fixes to:
- Blaker method root-finding algorithms
- Unconditional method upper bound inflation

See `../RECENT_FIXES_SUMMARY.md` for detailed technical summary.