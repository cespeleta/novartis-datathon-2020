#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "isort"
isort --profile black src

echo "black"
black src || FAILURE=true

echo "pylint"
pylint src || FAILURE=true

echo "pycodestyle"
pycodestyle src || FAILURE=true

echo "mypy"
mypy src || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0