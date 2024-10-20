PACKAGE_NAME  := smee
CONDA_ENV_RUN := conda run --no-capture-output --name $(PACKAGE_NAME)

EXAMPLES_SKIP := examples/md-simulations.ipynb
EXAMPLES := $(filter-out $(EXAMPLES_SKIP), $(wildcard examples/*.ipynb))

.PHONY: env lint format test test-examples docs docs-deploy

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file devtools/envs/base.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install || true

lint:
	$(CONDA_ENV_RUN) ruff check $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) ruff check examples

format:
	$(CONDA_ENV_RUN) ruff format $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) ruff check --fix --select I $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) ruff format examples
	$(CONDA_ENV_RUN) ruff check --fix --select I examples

test:
	$(CONDA_ENV_RUN) pytest -v --cov=$(PACKAGE_NAME) --cov-append --cov-report=xml --color=yes $(PACKAGE_NAME)/tests/

test-examples:
	$(CONDA_ENV_RUN) jupyter nbconvert --to notebook --execute $(EXAMPLES)

docs:
	$(CONDA_ENV_RUN) mkdocs build

docs-deploy:
ifndef VERSION
	$(error VERSION is not set)
endif
	$(CONDA_ENV_RUN) mike deploy --push --update-aliases $(VERSION)
