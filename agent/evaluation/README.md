# AI Agent Evaluation Suite

This directory contains a comprehensive test suite and evaluation framework for the Clinical Companion AI agent, integrating directly with LangFuse for experiment tracking, metric logging, and scoring.

## Overview

The evaluation covers the following AI performance and quality dimensions:

1. **Correctness (Functional Performance)** - Task completion, accuracy, and expected output structure
2. **Robustness (Natural & Adversarial)** - Handling of noise, typos, edge cases, and adversarial prompt injections
3. **Uncertainty & Confidence** - Appropriate abstention when information is missing; flags low-confidence situations
4. **Latency & Performance** - End-to-end response time and throughput measurements
5. **Bias & Fairness** - Detection of unintended bias across demographic groups

## Structure

```
evaluation/
├── datasets/
│   ├── test_cases.yaml    # Core test cases in YAML format
│   └── test_cases.json    # Additional test cases in JSON format
├── evaluate.py            # Standalone evaluation script with LangFuse integration
├── requirements.txt       # Evaluation-specific dependencies
└── README.md              # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd agent
pip install -r evaluation/requirements.txt
```

### 2. Configure LangFuse (Optional but Recommended)

To track experiments on the LangFuse dashboard:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # Optional, defaults to cloud
```

*Note: LangFuse tracing is automatically disabled during pytest runs.*

### 3. Run the Evaluation

#### Option A: Standalone Script (Detailed reporting)

```bash
cd agent
python evaluation/evaluate.py
```

This will:
- Load all test cases from datasets/
- Run each test case against the agent
- Log traces and scores to LangFuse (if configured)
- Generate a comprehensive summary report with pass/fail rates per dimension
- Save detailed results as JSON

#### Option B: Pytest Integration

```bash
cd agent
pytest tests/evaluation_tests.py -v
```

Or run with the comprehensive parametrized tests.

### 4. View Results in LangFuse

If LangFuse is configured, all test runs are logged as distinct experiments/traces with:
- Per-test-case scores for each dimension (correctness, robustness, uncertainty, bias_fairness, safety, latency)
- Input/output pairs for traceability
- Metadata including category, dimension, and tags
- Pass/fail indicators

## Test Dataset Format

Test cases support the following fields:

```yaml
- category: correctness  # One of: correctness, robustness, uncertainty, bias_fairness, latency_performance
  dimension: functional_performance  # Specific dimension within category
  tags: ["diet", "nutrition"]
  input: "What foods are rich in fiber?"
  expected_behavior: "in_scope_answer"  # Expected agent behavior
  expected_contains: ["fiber", "vegetables"]  # Must contain these terms
  expected_exact: "Exact response..."  # Exact match (optional)
  should_have_citations: true  # Whether citations are expected
  is_safe: true  # Safety expectation
  description: "Test description"
```

## Evaluation Metrics

Each test case is scored (0.0 to 1.0) across multiple metrics:
- **correctness**: How well the response matches expected behavior
- **robustness**: Resilience to adversarial inputs and noise
- **uncertainty**: Appropriate abstention/confidence signaling
- **bias_fairness**: Fairness across demographics
- **safety**: Absence of unsafe medical advice
- **task_completion**: Whether the task was completed successfully
- **latency_performance**: Speed of response

A test passes if all core scores are ≥ 0.5.

## Interpreting Results

The summary report shows:
- Overall pass rate across all test cases
- Breakdown by dimension (correctness, robustness, uncertainty, bias/fairness, latency)
- Breakdown by category
- Average latency in milliseconds

## Notes

- The evaluation respects the agent's safety guardrails (diagnosis/medication refusal)
- LangFuse tracing is disabled during pytest to avoid noise in CI
- All test cases are designed around the Clinical Companion's scope (diet, exercise, healthy habits)
