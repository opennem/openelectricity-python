.DEFAULT_GOAL := all
.SHELLFLAGS = -e
UPGRADE_ARGS ?= --upgrade
projectname = openelectricity

# tools
ruff-check = uv run ruff check $(projectname)
mypy = uv run mypy $(projectname)
pytest = uv run pytest tests -v
pyright = uv run pyright -v .venv $(projectname)
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


.PHONY: version
version:
	@if ! echo "major minor patch stable alpha beta rc post dev" | grep -w "$(BUMP)" > /dev/null; then \
		echo "Error: BUMP must be one of: major, minor, patch, stable, alpha, beta, rc, post, dev"; \
		exit 1; \
	fi
	# if the branch is main then bump needs to be either major minor patch or stable
	@current_branch=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$current_branch" = "main" ]; then \
		if [ "$(BUMP)" != "major" ] && [ "$(BUMP)" != "minor" ] && [ "$(BUMP)" != "patch" ] && [ "$(BUMP)" != "stable" ] && [ "$(BUMP)" != "rc" ]; then \
			echo "Error: Cannot bump on main branch unless it is major, minor, patch, stable or rc"; \
			exit 1; \
		fi \
	fi; \
	if [ "$$current_branch" = "develop" ]; then \
		if [ "$(BUMP)" != "dev" ]; then \
			echo "Error: Cannot bump on develop branch unless it is dev"; \
			exit 1; \
		fi \
	fi; \
	# Use hatchling's version management
	uvx hatch version $(BUMP); \
	NEW_VERSION=$$(uvx hatch version); \
	echo "New version: $$NEW_VERSION"; \
	git add $(projectname)/__init__.py; \
	git commit -m "Bump version to $$NEW_VERSION"

.PHONY: build
build:
	rm -rf dist/
	uv build

.PHONY: tag
tag:
	$(eval CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD))
	$(eval NEW_VERSION := $(shell uvx hatch version))
	@if [ "$(CURRENT_BRANCH)" = "main" ]; then \
		git tag "v$(NEW_VERSION)"; \
		echo "Pushing v$(NEW_VERSION)"; \
		git push origin "v$(NEW_VERSION)" "$(CURRENT_BRANCH)"; \
	else \
		git push -u origin "$(CURRENT_BRANCH)"; \
	fi

.PHONY: publish
publish: release build
	uv publish

.PHONY: release-pre
release-pre: format lint test

.PHONY: release
release: release-pre version tag

.PHONY: clean
clean:
	ruff clean
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete -o -type d -name .mypy_cache -delete
	rm -rf build dist *.egg-info
