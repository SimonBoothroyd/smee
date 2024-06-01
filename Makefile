PACKAGE_NAME  := smee
CONDA_ENV_RUN := conda run --no-capture-output --name $(PACKAGE_NAME)

EXAMPLES_SKIP := examples/md-simulations.ipynb
EXAMPLES := $(filter-out $(EXAMPLES_SKIP), $(wildcard examples/*.ipynb))

.PHONY: pip-install env lint format test test-examples docs-build docs-deploy docs-insiders

pip-install:
	$(CONDA_ENV_RUN) pip install --no-build-isolation --no-deps -e .

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file devtools/envs/base.yaml
	$(CONDA_ENV_RUN) pip install --no-build-isolation --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install || true

lint:
	$(CONDA_ENV_RUN) isort --check-only $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) black --check      $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) flake8             $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) nbqa isort   --check-only  examples
	$(CONDA_ENV_RUN) nbqa black   --check       examples
	$(CONDA_ENV_RUN) nbqa flake8  --ignore=E402 examples

format:
	$(CONDA_ENV_RUN) isort  $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) black  $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) flake8 $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) nbqa isort                examples
	$(CONDA_ENV_RUN) nbqa black                examples
	$(CONDA_ENV_RUN) nbqa flake8 --ignore=E402 examples

test:
	$(CONDA_ENV_RUN) pytest -v --cov=$(PACKAGE_NAME) --cov-report=xml --color=yes $(PACKAGE_NAME)/tests/

test-examples:
	$(CONDA_ENV_RUN) jupyter nbconvert --to notebook --execute $(EXAMPLES)

docs-build:
	$(CONDA_ENV_RUN) mkdocs build

docs-deploy:
ifndef VERSION
	$(error VERSION is not set)
endif
	$(CONDA_ENV_RUN) mike deploy --push --update-aliases $(VERSION)

docs-insiders:
	$(CONDA_ENV_RUN) pip install git+https://$(INSIDER_DOCS_TOKEN)@github.com/SimonBoothroyd/mkdocstrings-python.git \
                    			 git+https://$(INSIDER_DOCS_TOKEN)@github.com/SimonBoothroyd/griffe-pydantic.git
