## Virtual Environment for ConvAI

This folder includes a project-local Python virtual environment at `.venv`.

Quick setup (if not already created):

```bash
cd ConvAI
./setup_venv.sh
```

Activate the environment:

```bash
source .venv/bin/activate
```

Save installed packages:

```bash
pip freeze > requirements.txt
```

To deactivate:

```bash
deactivate
```
