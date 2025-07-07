-include .env
PROJECT_NAME ?= flow-diffusion
PYTHON_VERSION ?= 3.11
version=$(shell python setup.py --version)

.PHONY: config
config: pyproject.toml setup.py README.md

pyproject.toml: pyproject.toml.template .env
	sed -e "s/{{PYTHON_VERSION}}/$(PYTHON_VERSION)/g" \
		-e "s/{{PY_VERSION_SHORT}}/$(subst .,,$(PYTHON_VERSION))/g" \
		-e "s/{{PROJECT_NAME}}/$(PROJECT_NAME)/g" $< > $@
	@echo "Generated pyproject.toml with Python $(PYTHON_VERSION)"

setup.py: setup.py.template .env
	sed -e "s/{{PROJECT_NAME}}/$(PROJECT_NAME)/g" $< > $@
	@echo "Generated setup.py with project name $(PROJECT_NAME)"

README.md: README.md.template .env
	sed -e "s/{{PROJECT_NAME}}/$(PROJECT_NAME)/g" $< > $@
	@echo "Generated README.md for $(PROJECT_NAME)"

.PHONY: init-project-structure
init-project-structure:
	@if [ ! -d "$(PROJECT_NAME)" ]; then \
		mkdir -p "$(PROJECT_NAME)"; \
		echo '__version__ = "0.1.0"' > "$(PROJECT_NAME)/__init__.py"; \
		echo "Created package directory: $(PROJECT_NAME) with __init__.py"; \
	fi

.PHONY: rename-project
rename-project:
	@if [ -d "take_home" ] && [ "$(PROJECT_NAME)" != "take_home" ]; then \
		mv take_home $(PROJECT_NAME); \
		find . -type f -name "*.py" -exec sed -i "s/from take_home/from $(PROJECT_NAME)/g" {} \; ; \
		find . -type f -name "*.py" -exec sed -i "s/import take_home/import $(PROJECT_NAME)/g" {} \; ; \
		echo "Renamed project from take_home to $(PROJECT_NAME)"; \
	fi

.PHONY: lint
lint:
	pre-commit run --all-files
	nbqa mypy docs/users-guide/ --config setup.cfg

.PHONY: test
test:
	pytest -v --cov-report xml:coverage.xml --cov-report term --cov-report html:cov_html --cov ./tests/

.PHONY: wheel
wheel:
	python setup.py bdist_wheel
	cp \
		`find dist/*.whl -type f -printf '%t@ %p\n' | sort -n | tail -1 | cut -f2- -d" "` \
		app/

.PHONY: requirements
requirements:
	awk '!/^-c /{print$0}' requirements.in > requirements-constraint.in
	awk '!/^-c /{print$0}' requirements-dev.in >> requirements-constraint.in
	pip-compile requirements-constraint.in --resolver=backtracking --strip-extras --upgrade -q
	custom_compile_command="make requirements" pip-compile requirements.in --resolver=backtracking --no-strip-extras --upgrade -q
	custom_compile_command="make requirements" pip-compile requirements-dev.in --resolver=backtracking --no-strip-extras --upgrade -q
	rm requirements-constraint.in requirements-constraint.txt

.PHONY: setup-conda
setup-conda:
	conda create -y --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	@echo "Conda environment '$(PROJECT_NAME)' created with Python $(PYTHON_VERSION)"
	@echo "To activate: conda activate $(PROJECT_NAME)"
	@echo "Then install requirements: pip install -e ."
