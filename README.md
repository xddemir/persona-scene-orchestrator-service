# scene-orchestrator

Coordinator of the pipeline for the M.Sc. thesis *Personality-Based Generation of VR Environments
for Relaxation* (DFKI). It takes a persona, builds a scene specification, has image-gen-service
render a 360° skybox for it, and writes the files Unity reads.

```text
persona → SceneSpec → prompt text → image-gen-service → out/<scene_id>/ → Unity
```

**Status:** HTTP server only. The pipeline is built step by step.

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate     # Linux/macOS
pip install -e ".[dev]"
```

**WSL:** WSL can't use a venv created by Windows Python, and Windows can't use one created in WSL.
Make a separate Linux venv outside the repo:

```bash
python3 -m venv ~/.venvs/scene-orchestrator
source ~/.venvs/scene-orchestrator/bin/activate
pip install -e ".[dev]"
```

## Run

From the project root, with the venv active:

```bash
python -m uvicorn scene_orchestrator.api.app:app --reload --reload-dir src --host 127.0.0.1 --port 8001
```

Then open <http://127.0.0.1:8001/docs>.

- The port is `8001` because image-gen-service's dev server uses `8000`, and the two run side by side.
- `--reload-dir src` limits the reloader to the source tree. Without it, the reloader also scans the
  `.venv` folder (thousands of files). From WSL on `/mnt/c`, that scan is too slow to notice edits.

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /health` | `{"status": "ok", "version": "0.1.0"}` |
| `GET /docs` | Swagger UI; `/` redirects here |

## Tests

```bash
python -m pytest
```
