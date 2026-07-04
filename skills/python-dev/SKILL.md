---
name: python-dev
description: Python development in the sandbox. Use when the user asks to write, run, or debug Python scripts; install packages; or work with files using Python.
---

# python-dev

## Environment

The sandbox provides a pre-configured Python venv. Environment variables always injected:
- `WORKSPACE` → `/workspace` (working directory, cwd for all bash commands)
- `OUTPUT` → `/output` (save files here for download)
- `UPLOAD` → `/upload` (user-uploaded files)
- `VIRTUAL_ENV` → venv path (activated automatically)

## Installing packages

```bash
pip install pandas matplotlib requests
```

Multiple packages in one call. Always install before importing in a script.

## Running scripts

Write to a file, then run:
```bash
cat > script.py << 'EOF'
import os
print("Hello from", os.environ["WORKSPACE"])
EOF
python script.py
```

Or pipe directly:
```bash
python - << 'EOF'
print("inline script")
EOF
```

## Saving output files

Never hardcode `/workspace` or `/output` in scripts — use the injected env vars:

```python
import os

output = os.environ["OUTPUT"]      # /output real path
workspace = os.environ["WORKSPACE"] # /workspace real path

# Save a file for download
with open(f"{output}/result.txt", "w") as f:
    f.write("done")

# Or use relative paths (cwd = workspace)
with open("result.txt", "w") as f:
    f.write("saved in workspace")
```

## Common patterns

**Read a CSV from upload:**
```python
import pandas as pd, os
df = pd.read_csv(f"{os.environ['UPLOAD']}/data.csv")
```

**Save a chart:**
```python
import matplotlib.pyplot as plt, os
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
fig.savefig(f"{os.environ['OUTPUT']}/chart.png", dpi=150, bbox_inches="tight")
plt.close()
```

**Check Python and package versions:**
```bash
python --version && pip list | grep -E "pandas|numpy|matplotlib"
```
