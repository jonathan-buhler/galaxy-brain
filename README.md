# galaxy-brain
A project with the aim of generating artificial galaxies and worlds


## Installation
We are using Python 3.8 and [Poetry](https://python-poetry.org) for dependency management.

```bash
# Check that your Python version is ^3.8
python --version

# Runs Poetry install script
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# Tells Poetry to use old installer that actually works!
poetry config experimental.new-installer false

# (Optional) Tells Poetry to create the virtualenv in the project's root folder
poetry config virtualenvs.in-project true

# Installs project dependencies
poetry install
```

Other installation options are available in Poetry's [documentation](https://python-poetry.org/docs/)
