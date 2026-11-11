# Contributing

Thank you for helping improve FuseVirome.

1. Create a branch from the current default branch.
2. Install development dependencies with
   `python -m pip install -r requirements-dev.txt`.
3. Keep preprocessing reproducible: use explicit random seeds and derive
   preprocessing statistics from training data only.
4. Run `python -m unittest discover -s tests -v` before submitting a pull request.
5. Document changes that alter feature definitions, sample inclusion, or
   reported metrics.

Do not commit credentials, private sample metadata, virtual environments,
Python caches, or generated files from `outputs/`.
