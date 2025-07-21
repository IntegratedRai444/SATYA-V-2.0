#!/bin/bash
# Run all backend tests with coverage
cd $(dirname $0)/..
pytest --cov=backend --cov-report=term-missing 