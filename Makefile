VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

SANDBOX_VENV := .opensandbox-venv
SANDBOX_CONFIG := $(HOME)/.sandbox.toml

.PHONY: install install-browser run dev shell build-sandbox install-sandbox-server sandbox-server migrate migration

install:
	python3 -m venv $(VENV)
	$(PIP) install -q -r requirements.txt

install-browser:
	$(PYTHON) -m playwright install chromium

run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000

dev:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

shell:
	$(PYTHON)

build-sandbox:
	docker build -f Dockerfile.sandbox -t jarvis-sandbox .

# opensandbox-server lives in its own venv, isolated from the app's own
# LangChain/LangGraph dependency graph — it's an unrelated daemon process.
install-sandbox-server:
	python3 -m venv $(SANDBOX_VENV)
	$(SANDBOX_VENV)/bin/pip install -q --upgrade pip
	$(SANDBOX_VENV)/bin/pip install -q opensandbox-server opensandbox

sandbox-server:
	@[ -f $(SANDBOX_CONFIG) ] || $(SANDBOX_VENV)/bin/opensandbox-server init-config $(SANDBOX_CONFIG) --example docker
	OPENSANDBOX_INSECURE_SERVER=YES $(SANDBOX_VENV)/bin/opensandbox-server --config $(SANDBOX_CONFIG)

migrate:
	$(VENV)/bin/alembic upgrade head

migration:
	$(VENV)/bin/alembic revision --autogenerate -m "$(name)"
