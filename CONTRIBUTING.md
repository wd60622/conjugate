# Guidelines for Contributing

Contributions are welcomed in all forms. These may be bugs, feature requests, documentation, or examples. Please feel free to: 

1. Submit an issue
1. Open a pull request
1. Help with outstanding issues and pull requests

## Open an Issue

If you find a bug or have a feature request, please [open an issue](https://github.com/wd60622/conjugate/issues/new) on GitHub.

## Local Development Steps

### Create a forked branch of the repo

Do this once but keep it up to date 

1. [Fork wd60622/conjugate GitHub repo](https://github.com/wd60622/conjugate/fork)
1. Clone forked repo and set upstream

    ```bash 
    git clone git@github.com:<your-username>/conjugate.git
    cd conjugate
    git remote add upstream git@github.com:wd60622/conjugate.git
    ```

### Setup Local Development Environment

The project is developed with [Poetry](https://python-poetry.org/).

## Pull Request Checklist

Please check that your pull request meets the following criteria: 

- Unit tests pass
- pre-commit hooks pass
- Docstrings and examples render correctly in the documentation

## Documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/)

The docstrings should be of [Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

## Automations

Formatting will be down with [pre-commit](https://pre-commit.com/). 

Tests will run on each pull request

Documentation will be updated with each merge to `main` branch.

Package release to PyPI on every GitHub Release. 

