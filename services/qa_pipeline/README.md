# QA Pipeline Service

Statistical test framework for validating and scoring entropy samples from GCP nodes.

## Features

- Modular test framework with plugin architecture for easy test addition
- NIST statistical test implementations
- Real-time quality scoring (0-100 scale)
- Node monitoring and flagging for quality issues
- Integration with NATS messaging system

## Usage

### As a Service

```bash
python main.py
```

The service connects to NATS, subscribes to the input topic, processes samples, and publishes results.

### One-off Testing

Process a single file:

```bash
python main.py --test-file /path/to/entropy.bin --format binary
```

## Architecture

- `framework/`: Core components (test runner, base classes, data types)
- `tests/`: Statistical test implementations
- `config/`: Configuration files
- `main.py`: Service entry point and CLI

## Adding New Tests

1. Create a new module in the `tests/` directory
2. Implement a class that inherits from `StatisticalTest`
3. Add the test to `qa_config.toml`

## Current Tests

- NIST Frequency (Monobit) Test: Basic randomness test checking the proportion of 0s and 1s
