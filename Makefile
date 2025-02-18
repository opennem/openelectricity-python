.DEFAULT_GOAL := all
.SHELLFLAGS = -e
UPGRADE_ARGS ?= --upgrade
projectname = openelectricity

# tools
ruff-check = uv run ruff check $(projectname)
mypy = uv run mypy $(projectname)
pytest = uv run pytest tests -v
pyright = uv run pyright -v .venv $(projectname)
hatch = uvx hatch
BUMP ?= dev

.PHONY: test
test:
	$(pytest)


.PHONY: codecov
codecov:
	pytest --cov=./$(projectname)


.PHONY: format
format:
	uv run ruff format $(projectname)
	$(ruff-check) --fix

.PHONY: lint
lint:
	$(ruff-check) --exit-zero

.PHONY: check
check:
	$(pyright)


.PHONY: build
build:
	pip install wheel
	python setup.py sdist bdist_wheel


.PHONY: version
version:
	@if ! echo "release major minor patch fix alpha beta rc rev post dev" | grep -w "$(BUMP)" > /dev/null; then \
		echo "Error: BUMP must be one of: release, major, minor, patch, fix, alpha, beta, rc, rev, post, dev"; \
		exit 1; \
	fi
	# if the branch is main then bump needs to be either major minor patch or release
	if [ "$$current_branch" = "main" ]; then \
		if [ "$$BUMP" != "major" ] && [ "$$BUMP" != "minor" ] && [ "$$BUMP" != "patch" ] && [ "$$BUMP" != "release" ] && [ "$$BUMP" != "rc" ]; then \
			echo "Error: Cannot bump on master branch unless it is major, minor, patch or release"; \
			exit 1; \
		fi \
	fi; \

	# if the current branch is develop then the bump type must be dev
	if [ "$$current_branch" = "develop" ]; then \
		if [ "$$BUMP" != "dev" ]; then \
			echo "Error: Cannot bump on develop branch unless it is dev"; \
			exit 1; \
		fi \
	fi; \

	$(hatch) version $(BUMP)
	@NEW_VERSION=$$(sed -n 's/__version__ = "\([^"]*\)".*/\1/p' opennem/__init__.py); \
	echo "New version: $$NEW_VERSION"; \
	git add opennem/__init__.py; \
	git commit -m "Bump version to $$NEW_VERSION"

.PHONY: tag
tag:
	$(eval CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD))
	$(eval NEW_VERSION := $(shell uvx hatch version))
	@if [ "$(CURRENT_BRANCH)" = "master" ]; then \
		git tag "$(NEW_VERSION)"; \
		echo "Pushing $(NEW_VERSION)"; \
		git push origin "$(NEW_VERSION)" "$(CURRENT_BRANCH)"; \
	else \
		git push -u origin "$(CURRENT_BRANCH)"; \
	fi

.PHONY: publish
publish:
	uvx hatch publish

.PHONY: release-pre
release-pre: format lint test

.PHONY: release
release: release-pre version tag

.PHONY: clean
clean:
	ruff clean
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete -o -type d -name .mypy_cache -delete
	rm -rf build
