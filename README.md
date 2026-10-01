# scene-orchestrator

Coordinator of the pipeline for the M.Sc. thesis *Personality-Based Generation of VR Environments
for Relaxation* (DFKI). It takes a persona, builds a scene specification, has image-gen-service
render a 360° skybox for it, and writes the files Unity reads.

```text
persona → SceneSpec → prompt text → image-gen-service → out/<scene_id>/ → Unity
```

**Status:** HTTP server, plus skybox generation as Slurm jobs on Pegasus. A prompt goes in, and the
skybox comes back into `out/`. Persona → SceneSpec → prompt is not built yet.

## Install

Run the orchestrator in **WSL**: it reaches Pegasus through a shared SSH connection, which Windows'
ssh can't provide. WSL can't use a venv created by Windows Python, so make a Linux one outside the
repo:

```bash
python3 -m venv ~/.venvs/scene-orchestrator
source ~/.venvs/scene-orchestrator/bin/activate
pip install -e ".[dev]"
```

The Windows `.venv` still works for running the tests and for VS Code.

## Run

**WSL terminal 1: the shared connection to Pegasus.** Enter your password, then leave it open. It
prints nothing while it's working.

```bash
ssh -NM -o ControlPath=~/.ssh/pegasus.sock demir@login1.pegasus.kl.dfki.de
```

**WSL terminal 2: the orchestrator.**

```bash
cd /mnt/c/Users/Dogukan/persona-scene-orchestrator-service
source ~/.venvs/scene-orchestrator/bin/activate
python -m uvicorn scene_orchestrator.api.app:app --reload --reload-dir src --host 127.0.0.1 --port 8001
```

At startup it reports whether it found the connection:

```text
INFO:     Pegasus: shared SSH connection to demir@login1.pegasus.kl.dfki.de is open
WARNING:  Pegasus: no shared SSH connection to Pegasus. Open it in a WSL terminal and leave it open: ...
```

Then open <http://127.0.0.1:8001/docs> in the Windows browser.

- Off-site you need the DFKI VPN, with NordVPN disconnected (it takes over DNS).
- The port is `8001` because image-gen's own dev server uses `8000`.
- `--reload-dir src` limits the reloader to the source tree. Without it, the reloader also scans the
  `.venv` folder (thousands of files). From WSL on `/mnt/c`, that scan is too slow to notice edits.

## How a scene is made

Each participant gets exactly one scene, built from their own persona. `POST /scenes` runs the
whole pipeline for one participant:

```text
persona (file or inline) → ChatbotV2Adapter → RuleSpecMapper → TemplatePromptBuilder
  → Slurm skybox job (below) → out/<participant_id>/scene_spec.json + out/manifest.json
```

- `scene_id` is the participant id, e.g. `P01`.
- Without a `seed`, it's derived from SHA-256 of the participant id, so it's the same on every run.
- `GET /scenes/P01` returns everything for that participant in one call: status, Slurm progress,
  the converted persona, the spec and the files.
- `scene_spec.json` carries the prompt (`skybox.prompt`) and the image's file name (`skybox.uri`).
- Every outcome ends in `out/manifest.json`, which is what Unity reads. A scene is entered as
  `failed` ("did not finish") when it starts and overwritten when it ends, so even a crash leaves an
  entry. A persona that won't convert (e.g. a missing dimension) is recorded as `failed` with the
  reason.
- Only requests that can't name a scene at all get `422`: no `user.alias`, an alias that isn't a
  safe folder name, or an unreadable file.

```json
{"P01": {"status": "ok", "files": ["scene_spec.json", "skybox_660120003.png", "skybox_660120003.json"],
         "updated_at": "2026-10-01T09:03:58Z"}}
```

`/skyboxes` stays as the low-level way to try a hand-written prompt.

## How a skybox is made

Each skybox is one Slurm job, submitted through the shared connection. Nothing runs on Pegasus
between requests, and the GPU is held only while the picture is made.

```text
POST /skyboxes
  └─ ssh login1: sbatch   GPU job: image-gen gen --backend local_diffusers --kind skybox ...
  └─ ssh login1: sacct    every poll_interval_s, until COMPLETED or failed
  └─ ssh login1: cat      the PNG and its sidecar → out/<scene_id>/
```

Expect the Slurm queue wait plus about 30–60 s of model loading per job. `GET /skyboxes/{id}` shows
where the job is, e.g. `"detail": "Slurm job 123456: PENDING"`. The job's own log is on Pegasus at
`/netscratch/demir/image-gen/out/.slurm/<job id>.out`.

A failed job is retried once, and after that the skybox is marked `failed` with the last lines of
the job's log. These are not retried, because repeating can't help: a missing connection, and a job
Slurm refuses (partition, account). A job still unfinished after `job_timeout_s` is cancelled.

Losing the connection mid-job fails the skybox, but the job keeps running on Pegasus, and its files
still land in `/netscratch/demir/image-gen/out/`.

## Configuration

[`configs/default.yaml`](configs/default.yaml) holds the output folder, the Pegasus login, where
image-gen lives there (repo, venv, weights, output), and the GPU job's partition, account and time
limit. Point `SCENE_ORCHESTRATOR_CONFIG` at another file to use different settings.

## Output

```text
out/<scene_id>/skybox_<seed>.png
out/<scene_id>/skybox_<seed>.json   # image-gen's sidecar: prompt, seed, model + revision, steps, ...
```

The sidecar is image-gen's own, with `file` changed to the new name, plus `upstream_file` (where it
was made on Pegasus) and `slurm_job_id`. A failed skybox writes no files.

## Data contracts

[`models/`](src/scene_orchestrator/models/) holds the Pydantic models: `PersonaProfile` (Big Five
traits, 0..1) as input and `SceneSpec` as output. Every numeric field in `SceneSpec` is
range-constrained, and unknown fields and biomes are rejected.

Unity validates scene files against [`schema/scene_spec.schema.json`](schema/scene_spec.schema.json).
Regenerate it after changing `scene_spec.py`:

```bash
python -m scene_orchestrator.models.schema
```

A test fails while the committed schema is out of date.

## Endpoints

| Route | Purpose |
| --- | --- |
| `POST /scenes` | `{persona_file \| persona, seed?}` → `202` + `{scene_id, status}`; the whole pipeline in the background |
| `GET /scenes/{participant_id}` | Everything for one participant: `queued` → `building_spec` → `generating_image` → `ready` \| `failed`, Slurm progress, persona, spec, files, error |
| `GET /scenes/{participant_id}/files/{name}` | Download one file listed in `files`, e.g. the skybox PNG. Unlisted names get 404 |
| `POST /skyboxes` | `{prompt, seed, scene_id}` → `202` + job; generation runs in the background |
| `GET /skyboxes/{id}` | `queued` → `generating_image` → `ready` \| `failed`, with progress, files or error |
| `GET /health` | Version, and whether the shared connection to Pegasus is open |
| `GET /docs` | Swagger UI; `/` redirects here |

Job state is in memory and lost on restart. The files in `out/` are the durable record.

## Tests

```bash
python -m pytest
```

No cluster needed: the tests simulate the Pegasus login node.
