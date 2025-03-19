# Guidelines for Contributing

Contributions are welcomed in all forms. These may be bugs, feature requests, documentation, or examples. Please feel free to:

- Submit an issue
- Open a pull request
- Help with outstanding issues and pull requests

## Open an Issue

If you find a bug or have a feature request, please [open an issue](https://github.com/williambdean/conjugate/issues/new) on GitHub.
Please check that it is not one of the [open issues](https://github.com/williambdean/conjugate/issues).

## Local Development Steps

### Create a forked branch of the repo

Do this once but keep it up to date

1. [Fork williambdean/conjugate GitHub repo](https://github.com/williambdean/conjugate/fork)
1. Clone forked repo and set upstream

    ```bash
    git clone git@github.com:<your-username>/conjugate.git
    cd conjugate
    git remote add upstream git@github.com:williambdean/conjugate.git
    ```

### Setup Local Development Environment

The project is developed with [Poetry](https://python-poetry.org/).

In the root of the repo, run:

```bash
poetry install
```

And also install the [pre-commit](https://pre-commit.com/) hooks with:

```bash
pre-commit install
```

## Pull Request Checklist

Please check that your pull request meets the following criteria:

- Unit tests pass
- pre-commit hooks pass
- Docstrings and examples render correctly in the documentation

## Documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/).

The docstrings should be of [Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Automations

Formatting will be down with ruff via the pre-commit hooks.

Tests will run on each pull request.

Documentation will be updated with each merge to `main` branch.

Package release to PyPI on every GitHub Release.
