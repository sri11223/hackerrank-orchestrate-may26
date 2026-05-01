"""Compatibility wrapper for evaluators that call ``python code/cli.py``."""

from triage.cli import app


if __name__ == "__main__":
    app()
