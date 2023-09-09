PACKAGE_NAME  := smirnoffee
CONDA_ENV_RUN := conda run --no-capture-output --name $(PACKAGE_NAME)

.PHONY: env lint format test

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file devtools/envs/base.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install

lint:
	$(CONDA_ENV_RUN) isort --check-only $(PACKAGE_NAME) examples
	$(CONDA_ENV_RUN) black --check      $(PACKAGE_NAME) examples
	$(CONDA_ENV_RUN) flake8             $(PACKAGE_NAME) examples

format:
	$(CONDA_ENV_RUN) isort  $(PACKAGE_NAME) examples
	$(CONDA_ENV_RUN) black  $(PACKAGE_NAME) examples
	$(CONDA_ENV_RUN) flake8 $(PACKAGE_NAME) examples

test:
	$(CONDA_ENV_RUN) pytest -v --cov=$(PACKAGE_NAME) --cov-report=xml --color=yes $(PACKAGE_NAME)/tests/
