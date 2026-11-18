# scene-orchestrator

Coordinator of the pipeline for the M.Sc. thesis *Personality-Based Generation of VR Environments
for Relaxation* (DFKI). It takes a persona, builds a scene specification, has image-gen-service
render a 360° skybox for it, and writes the files Unity reads.

```text
persona → SceneSpec → prompt text → image-gen-service → out/<scene_id>/ → Unity
```

**Status:** the whole path runs. A persona goes in; a scene spec, its skybox (a Slurm job on
Pegasus) and a manifest entry come out in `out/`, through the HTTP server or the batch runner.

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
python -m uvicorn scene_orchestrator.api.app:app --reload --reload-dir src --reload-include "*.yaml" --host 127.0.0.1 --port 8001
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
- `--reload-include "*.yaml"` restarts it when a place is marked in
  [`places_v1.yaml`](src/scene_orchestrator/mapping/places_v1.yaml): the server reads the places once,
  at start.

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
- A scene nobody created yet is built by that same `GET`, when the config has an `auto_create`
  block and its `personas_dir` holds `P01.json`. So Unity can ask for a participant straight away.
  The sky is `auto_create.sky_mode`. With `panorama` (the default) the skybox is generated on
  Pegasus if the connection is open, and Unity waits for it; if it isn't, the scene falls back to
  the procedural sky and is `ready` at once.
- That `GET` also builds a scene again when what is on disk can't answer it, or when a scene
  without its image can get one after all: see
  [When a scene is built again](#when-a-scene-is-built-again).
- `scene_spec.json` carries the prompt (`skybox.prompt`) and the image's file name (`skybox.uri`).
- Every outcome ends in `out/manifest.json`, which is what Unity reads. A scene is entered as
  `failed` ("did not finish") when it starts and overwritten when it ends, so even a crash leaves an
  entry. A persona that won't convert (e.g. a missing dimension) is recorded as `failed` with the
  reason.
- Only requests that can't name a scene at all get `422`: no `user.alias`, an alias that isn't a
  safe folder name, or an unreadable file.

```json
{"P01": {"status": "ok", "sky_mode": "panorama",
         "files": ["scene_spec.json", "skybox_660120003.png", "skybox_660120003.json"],
         "updated_at": "2026-10-01T09:03:58Z",
         "place": {"id": "terraces_foot", "status": "candidate", "allow_candidates": true,
                   "places_sha256": "024559f6a23d..."}}}
```

`place` records where the scene was placed: the place, its status in `places_v1.yaml` as the server
had loaded it, the `places.allow_candidates` flag, and the SHA-256 of that file. So each
participant's scene can be traced to the exact list of places and flag it was made with.

`/skyboxes` stays as the low-level way to try a hand-written prompt.

### Two skies

`sky_mode` says which sky is made for the scene:

- `"panorama"` (the default): the generated 360° image, `skybox` in the spec.
- `"procedural"`: a sky drawn by a shader in the Unity project (colour gradient, sun, moon, stars,
  clouds, rain or snow), set up from `procedural_sky`, `lighting` and `weather` in the spec. No
  image is made and Pegasus isn't needed; `POST /scenes` goes from `building_spec` straight to
  `ready`.

`procedural_sky` is filled in on **every** spec, from the same persona, whichever mode was asked
for, so a panorama scene carries both skies. Unity shows the image whenever the spec names one
(`skybox.uri`), and draws the procedural sky when it doesn't. So the two modes can be compared for
one participant, and a panorama that can't be made doesn't cost the scene: after the usual retry,
the spec is written with `sky_mode` flipped to `"procedural"` and the manifest says `fallback`.
Its image is tried again later, when Unity asks for the scene and Pegasus can be reached.

| Manifest `status` | Meaning | `GET /scenes` |
| --- | --- | --- |
| `ok` | The scene as requested | `ready` |
| `fallback` | No panorama; the spec on disk uses the procedural sky. `error` says why | `ready` |
| `failed` | No usable scene (e.g. the persona won't convert). `error` says why | `failed` |

The manifest's `sky_mode` is the mode of the spec on disk, so `"procedural"` for a fallback. Where
the sun and the moon stand, the time of day, the light's colour and strength, fog and rain are in
`lighting` and `weather`, which both modes read; `procedural_sky` holds only what the sky shader
needs on top.

### When a scene is built again

With `auto_create`, `GET /scenes/P01` builds the scene again, from `personas_dir/P01.json`, in
these cases. Unity only has to ask: it waits while the scene is generating.

| What the manifest and `out/P01/` have | What the `GET` does |
| --- | --- |
| A listed file is gone, e.g. `out/P01/` was deleted | Builds it again, with the sky it was asked for |
| `fallback`, and Pegasus can be reached | Builds it again as a panorama, so it gets its image |
| `failed` with "generation started but did not finish": the server stopped mid-way | Picks it up, its Slurm job included |
| `failed` for any other reason | Nothing: the same persona would fail the same way. `POST /scenes` replaces it |
| `ok`, all files there | Nothing |

- A fallback is not tried again within a minute of its last attempt
  (`FALLBACK_RETRY_AFTER_S` in [`app.py`](src/scene_orchestrator/api/app.py)). Unity asks every
  few seconds while it waits, and the question after a failed attempt has to get the procedural
  scene, not start another attempt.
- A scene whose spec is still on disk keeps its seed. One whose files are gone gets the seed
  derived from the participant id, as a new scene does.
- A scene that was asked for as `procedural` is `ok`, not a fallback, so it stays procedural.
- Without the `auto_create` block, or without a persona file for the participant, nothing is
  built again and the `GET` answers from the manifest as it is.

### Time of day and weather

The spec says where the sun and the moon stand and what is in the sky. Both skies follow from
that: the prompt names the hour, the cloud, the moon, the stars and the rain, and the sky shader
draws them.

| What | Driven by | From → to |
| --- | --- | --- |
| `lighting.sun_elevation_deg` | extraversion | −30° (deep night) → 60° (high noon) |
| `lighting.time_of_day` | the sun's elevation, and conscientiousness for the half of the day | see below |
| `procedural_sky.clouds.coverage` | neuroticism | 0.05 (clear) → 0.95 (overcast) |
| `weather.precipitation` | neuroticism | none up to 0.7, then rising to steady at 1.0 |
| `weather.kind` | the place's biome | `snow` in `snowy_valley`, `rain` elsewhere |
| `lighting.moon_phase` | neuroticism | new moon → full moon |
| `procedural_sky.stars.brightness` | neuroticism | 1.0 → 0.35 |
| `procedural_sky.stars.density`, `.milky_way` | openness | a few stars → a crowded sky with the Milky Way |
| `procedural_sky.twilight_color` | conscientiousness, muted by neuroticism | evening amber or morning rose |

| Sun's elevation | `time_of_day` |
| --- | --- |
| below −6° | `night` |
| −6° to 10° | `sunrise` if conscientiousness > 0.5, else `sunset` |
| 10° to 40° | `morning` if conscientiousness > 0.5, else `afternoon` |
| 40° and up | `midday` |

`lighting.color_temperature_k` and `lighting.intensity` describe the key light and follow from the
sun's elevation: warm and weak while it is low, neutral and full once it is high. At night the
key light is the moon: cool (8000 K), and 0.03 to 0.18 of daylight by its phase. The moon stands
as far round from the sun as its phase says (a crescent beside it, a full moon opposite); how
high it stands comes from the seed.

Audio does not follow yet: a night scene still lists its birdsong, and rain has no sound layer.

### What shapes the procedural sky

Every part of it follows from the persona, by the same four themes as the rest of the scene:

| `procedural_sky` field | Driven by | Low → high |
| --- | --- | --- |
| `atmosphere_thickness` | neuroticism | Crisp horizon → haze reaching far up |
| `sky_tint` (colour overhead) | biome (hue), neuroticism (saturation) | Vivid → muted |
| `twilight_color` (the low sun's glow) | conscientiousness (hue), neuroticism (saturation) | Evening amber or morning rose; vivid → muted |
| `clouds.coverage` | neuroticism | Clear → overcast |
| `clouds.softness` | neuroticism | Crisp edges → diffuse |
| `exposure` | extraversion | Dim → bright |
| `sun_size`, `sun_halo` | extraversion | Small sun, faint glow → large sun, strong glow |
| `clouds.brightness` | extraversion | Grey → white |
| `clouds.detail` | openness | Smooth shapes → intricate |
| `stars.density`, `stars.milky_way` | openness | A few stars → a crowded field with the Milky Way |
| `stars.brightness` | neuroticism | A dark clear night → washed out by haze and moon |
| `horizon_color` | openness (plus the sky's own colour) | Paler sky → a second, warmer colour |
| `clouds.banding` | conscientiousness | Scattered puffs → regular rows along the wind |
| `ground_color` | biome | |
| `clouds.offset` | seed | Which clouds, not what kind |

The colours are the daytime ones. The shader takes them down to dusk and night as the sun goes,
and draws the moon, the stars and what falls from `lighting` and `weather`. The clouds drift with
`motion`'s wind (direction and strength), so they have no speed of their own, and the stars
twinkle with it. Agreeableness drives nothing, as elsewhere. The exact numbers are the `procedural_sky.*`
entries of `LINEAR_RULES` in [`rule_mapper.py`](src/scene_orchestrator/mapping/rule_mapper.py).

### The place and the view (spec 0.4)

Unity builds the place itself: the terrain, water, horizon mountains, ground, props, and where the
participant sits. It has six tuned settings, the **archetypes**. A spec picks one, and sets the
place's **axes** on it: the landform, where the participant sits on it, the water and where it lies
from them, how much grows, the season. Whatever the spec leaves out is the archetype's.

The persona chooses **which place**, but only from a fixed list of places, each tuned by hand and
checked in the headset: [`places_v1.yaml`](src/scene_orchestrator/mapping/places_v1.yaml). It never
sets a terrain number. Why: shelter mostly comes from the terrain, so with one fixed landscape a
participant's scene and the yoked one would differ only in light, fog and props (RQ3); and terrain
numbers driven by the answers give odd in-between landscapes, a confound against the hand-made
baseline. The full reasoning is the docstring of
[`rule_mapper.py`](src/scene_orchestrator/mapping/rule_mapper.py).

| Field | What the mapper sends |
| --- | --- |
| `archetype`, `biome` | The place's |
| `terrain.landform`, `position`, `water_relation`, `vegetation`, `season` | The place's. `season` only where the place has one (`snow_hollow`: winter), else `null` |
| `terrain.water.kind` | The place's. `present` agrees with it |
| `terrain.spawn.facing_deg` | A low sun is faced; at night, the moon; by day the sun stands 60–120° to one side |
| `terrain.spawn.placement` | `null`: `terrain.position` says where the participant sits, and Unity reads the facing with it |
| `spatial.prospect_refuge_bias` | From the place's `shelter_level`: 4–5 refuge, 1–2 prospect, 3 balanced |
| `props` | The first 2–5 of the place's own list (openness); `order` = (C − 0.65) / 0.35, sent only with `ringed` (C > 0.65) |
| `focal` | The place's focal prop, straight ahead: 40 m away when calm, 25 m when anxious |
| `framing` | The place's framing prop, 4 m away, 55–80° to one side, at the edge of view |
| `terrain.ground_material`, `macro_shape`, `amplitude_m`, `feature_scale_m`, `roughness`, `water.distance_m`, `water.azimuth_deg`, `spatial.silhouette_distance_m` | Never sent (`null`): the archetype's. For experiments by hand only |

**The pick rule** (`pick_place`, no seed): shelter (today neuroticism) gives a target band,
1 + 4 × shelter rounded half up. The band used is the nearest `shelter_level` with a usable place,
the more sheltered on a tie. In it, the place whose `novelty` is nearest the persona's (today
openness); on a tie, the first in the list. Usable means `verified`, plus `candidate` while
`places.allow_candidates` is true; `rejected` never.

| Shelter | Places (novelty) |
| --- | --- |
| 5 | forest_lake (0.2), hollow_pond (0.4), snow_hollow (0.9) |
| 4 | forest_river (0.3), valley_stream (0.5), terraces_foot (0.8) |
| 3 | meadow_lake (0.1), beach_cove (0.4), valley_lake_view (0.5) |
| 2 | open_meadow (0.0), ridge_lake_below (0.5), rocky_shore_top (0.8) |
| 1 | overlook_crest (0.3), cliff_over_sea (0.7), summit_top (0.9) |

- A spec whose axes Unity would change (a plain with water below, a lake at the summit, the top of a
  valley) is rejected by `SceneSpec`, after each axis is resolved against the archetype's setting.
  The places are checked the same way when they load. The tables are copies of Unity's
  `SceneAxes` and `EnvironmentRules` ([`axes.py`](src/scene_orchestrator/axes.py)).
- Unity reads a missing or `null` number as 0 and a missing string as `""`, and both mean "the
  archetype's". So 0 can't be asked for, and the override fields reject it.
- The fog is capped where horizon mountains stand, so about a third of them still shows
  (`FOG_CAP` = 1.1 / horizon distance). Not in a hollow, where the fog is the shelter, nor in the
  forest, which has no mountains. It never goes below an archetype's fog floor (the forest's 0.02),
  which Unity would raise it to anyway.
- The water's sound follows its kind: a stream with a river, lapping with a lake, waves at the sea.
  The biome now gives only the sounds, the sky's colour and whether rain or snow falls.
- The skybox prompt names the view from the place's landform and position (e.g. "seen from the foot
  of a terraced mountain"), its water from the kind and relation ("a lake far below"), and its
  plants from the props the spec places ("among tall pines and tufts of grass"), so the painted
  panorama agrees with the land.
- Which props a scene has is the place's, picked by hand in `places_v1.yaml`: a focal, a framing
  prop and at least five kinds, most typical first. The persona sets only amounts: how many kinds,
  how many of each, how varied in size, ringed or not. Every name must place something in Unity
  ([`catalogue.py`](src/scene_orchestrator/catalogue.py)). `mountain`, `cliff` and `road_rocks`
  are never sent: Unity places them itself. Nor is `water_lily`: Unity lays a spec's props out on
  dry land, and the ground floats its own lilies on a lake.
- Unity builds `poisson`, `clustered` and `ringed`. Until it builds `scattered`, that is sent as
  `poisson`.
- The seed only decides where and which way: the sun's, moon's and wind's azimuths, the moon's
  elevation, the clouds' offset, and which side (and how far) the day's sun and the framing prop
  stand. The list is in `map_persona`'s docstring.

`rules_markdown()` prints all of these rules as tables for the thesis: the linear rules, the places,
the pick rule, and every axis combination Unity can build (369, of which the places use 15).

## Many participants at once

The batch runner puts a queue of participants through the same pipeline, one after another:

```bash
python -m scene_orchestrator.batch queue.json --personas path/to/personas
python -m scene_orchestrator.batch queue.json --personas path/to/personas --sky-mode procedural
```

```json
[{"participant_id": "P01"},
 {"participant_id": "P02", "sky_mode": "procedural"},
 {"participant_id": "P07", "persona_file": "elsewhere/p07.json", "seed": 1234}]
```

A persona is `<personas>/<participant_id>.json` unless the entry names a `persona_file`. A persona
whose `user.alias` isn't the entry's `participant_id` is recorded as `failed`, so nobody gets
another participant's scene. One bad entry doesn't stop the rest.

It can be stopped and started again at any point:

- A scene already `ok` in the wanted sky mode is skipped.
- A skybox job that was submitted but never seen to finish (the runner was killed, or the SSH
  connection dropped) is recorded in the manifest as `slurm_job_id`. The next run waits for that
  job and fetches its image; it doesn't submit a second one.
- Everything else that isn't `ok`, fallbacks included, is redone.

If panoramas are wanted and the shared connection isn't open, nothing is run (exit status 2),
since otherwise every scene would be written as a fallback. The exit status is 0 when every scene
is `ok`, and 1 when any is `failed` or `fallback`.

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
still land in `/netscratch/demir/image-gen/out/`. For a scene, the job's id stays in the manifest,
and generating that scene again (same seed) picks the job up instead of submitting a new one.

## Configuration

[`configs/default.yaml`](configs/default.yaml) holds the output folder, the Pegasus login, where
image-gen lives there (repo, venv, weights, output), and the GPU job's partition, account and time
limit. Point `SCENE_ORCHESTRATOR_CONFIG` at another file to use different settings.

Its `auto_create` block names the folder of `<participant_id>.json` personas that scenes are built
from when Unity asks for them, and the sky mode of a scene nobody created yet. It points at the
fixtures; for the study, point it at the chatbot's output. Without the block, an unknown
participant is a 404 and no scene is built again.

Its `places` block says whether places not yet checked in the headset may be used
(`allow_candidates`). It is `true` while the places are being checked; set it to `false` for the
study, so every participant is at a verified place. Without the block it is `false`.

## Output

```text
out/manifest.json                   # one entry per scene; read this first
out/<scene_id>/scene_spec.json
out/<scene_id>/skybox_<seed>.png    # panorama scenes only
out/<scene_id>/skybox_<seed>.json   # image-gen's sidecar: prompt, seed, model + revision, steps, ...
```

The sidecar is image-gen's own, with `file` changed to the new name, plus `upstream_file` (where it
was made on Pegasus) and `slurm_job_id`. A failed skybox writes no files.

## Data contracts

[`models/`](src/scene_orchestrator/models/) holds the Pydantic models: `PersonaProfile` (Big Five
traits, 0..1) as input and `SceneSpec` (0.4) as output. Every numeric field in `SceneSpec` is
range-constrained, unknown fields and biomes are rejected, and so is a combination of terrain axes
Unity would change.

[`catalogue.py`](src/scene_orchestrator/catalogue.py) and [`axes.py`](src/scene_orchestrator/axes.py)
are hand copies of Unity's catalogue, settings and axis tables. `tests/test_catalogue.py` pins their
numbers, and compares them with the Unity project's own files when it finds the project (set
`UNITY_PROJECT` to the `UnityRelaxVR` folder; in WSL, its `/mnt/c/...` path).

Unity validates scene files against [`schema/scene_spec.schema.json`](schema/scene_spec.schema.json).
Regenerate it after changing `scene_spec.py`:

```bash
python -m scene_orchestrator.models.schema
```

A test fails while the committed schema is out of date.

## Endpoints

| Route | Purpose |
| --- | --- |
| `POST /scenes` | `{persona_file \| persona, seed?, sky_mode?}` → `202` + `{scene_id, status}`; the whole pipeline in the background |
| `GET /scenes/{participant_id}` | Everything for one participant: `queued` → `building_spec` → `generating_image` (panorama only) → `ready` \| `failed`, Slurm progress, persona, spec, files, error. With `auto_create`, builds the scene first if there is none, and [again](#when-a-scene-is-built-again) if its files are gone or its image can be made after all |
| `GET /scenes/{participant_id}/files/{name}` | Download one file listed in `files`, e.g. the skybox PNG. Unlisted names get 404 |
| `POST /skyboxes` | `{prompt, seed, scene_id}` → `202` + job; generation runs in the background |
| `GET /skyboxes/{id}` | `queued` → `generating_image` → `ready` \| `failed`, with progress, files or error |
| `GET /health` | Version, and whether the shared connection to Pegasus is open |
| `GET /docs` | Swagger UI; `/` redirects here |

Job state is in memory and lost on restart. The files in `out/` are the durable record.

## Test it with Unity

The Unity project (`UnityRelaxVR`) only downloads scenes. With `auto_create` in the config (the
default), asking for a participant whose persona is in `personas_dir` is enough: the server builds
the scene, and on plain Windows, where Pegasus can't be reached, it gets the procedural sky in the
same fetch. Steps 3 and 4 below are for a scene that already exists, or for choosing the sky
yourself.

### Procedural sky (no VPN, no cluster, plain Windows)

1. **Unity, once:** open `Assets/Scenes/ExperimentScene.unity`, click
   **Tools > Persona Scene > Set Up Experiment Scene**, save with Ctrl+S.
2. **Start the server** in a terminal at this repo's root, and leave it open:

   ```powershell
   .venv\Scripts\python -m uvicorn scene_orchestrator.api.app:app --host 127.0.0.1 --port 8001
   ```

   A `WARNING: Pegasus: ...` line is expected. `Uvicorn running on http://127.0.0.1:8001` means
   it's up.
3. **Create the scene** (optional, see above): open <http://127.0.0.1:8001/docs>, click **POST /scenes**, then
   **Try it out**, put this in the request body and click **Execute**:

   ```json
   {"persona_file": "fixtures/personas/chatbot/P01.json", "sky_mode": "procedural"}
   ```

   The answer is `{"scene_id": "P01", "status": "queued"}`. That's normal: the scene is built in
   the background.
4. **Check it's ready:** open <http://127.0.0.1:8001/scenes/P01>. It should show
   `"status": "ready"`, `"sky_mode": "procedural"` and `"files": ["scene_spec.json"]`.
5. **Fetch in Unity:** open `Assets/Scenes/BasicScene.unity`, press Play, type `P01`, press
   **Fetch**. The experiment scene loads by itself.
6. **Confirm:** the Console has a line starting `[SceneBuilder] P01: alpine_basin, sky: procedural`.
   The sky is drawn, not a photo: for P01 a mostly cloudy dawn, with a rose glow where the sun is
   about to rise and slowly drifting clouds, seen from the foot of the terraced mountain.

The fixtures between them show every kind of sky, at places in all five shelter bands:

| Participant | Sky | Place (shelter) | Biome |
| --- | --- | --- | --- |
| P01 | Mostly cloudy dawn, the sun just under the horizon | terraces_foot (4) | alpine basin |
| P02 | Sunny midday, a few clouds | ridge_lake_below (2) | wildflower meadow |
| P04 | Moonless night crowded with stars, the Milky Way | summit_top (1) | alpine basin |
| P05 | Cloudless midday | overlook_crest (1) | wildflower meadow |
| P06 | Rain at night, a full moon behind the cloud | hollow_pond (5) | birch grove |
| P07 | Snowfall at midday | snow_hollow (5) | snowy valley |
| P08 | Sunrise, scattered clouds | valley_lake_view (3) | lakeside |
| P09 | Dusk after sunset, a thin crescent and the first stars | rocky_shore_top (2) | rocky shore |
| P10 | Night, a half moon, scattered clouds, a few stars | meadow_lake (3) | lakeside |
| P11 | Clear sunset | ridge_lake_below (2) | wildflower meadow |
| P12 | Rainy morning | forest_lake (5) | forest clearing |
| P13 | Mostly cloudy afternoon | terraces_foot (4) | alpine basin |

P03 is missing a dimension on purpose: its scene fails, with the reason.

### Checking the places in the headset

Every place in [`places_v1.yaml`](src/scene_orchestrator/mapping/places_v1.yaml) starts as a
`candidate`. To look at one, build its review scene: the place forced, a neutral persona (every trait
0.5), seed 0, the procedural sky. It is registered like a participant's scene, as `place-<id>`.

```bash
python -m scene_orchestrator.places review                      # all of them
python -m scene_orchestrator.places review --place forest_lake  # one
```

In Unity, type `place-forest_lake` as the participant number and press **Fetch** (the server must be
running; it answers `GET /scenes/place-forest_lake` from `out/`). After looking:

```bash
python -m scene_orchestrator.places mark forest_lake verified --note "reads as a lake in the woods"
python -m scene_orchestrator.places mark summit_top rejected --note "the top's edge floats"
```

`mark` sets the place's `status`, `checked` (today) and `note`, and leaves the rest of the file as
it was. The server picks the change up when it restarts (or by itself, with
`--reload-include "*.yaml"`). For the study, set `places.allow_candidates: false` in the config, so
only verified places are used.

### Generated sky (needs the VPN and the cluster)

Run the server in WSL with the shared SSH connection open (see [Run](#run)), then press **Fetch**
(step 5). A participant with no scene yet, or with one that fell back to the procedural sky,
gets the image now: Unity waits, which takes minutes, and the Console line then says
`sky: image skybox_<seed>.png`. To watch it, open <http://127.0.0.1:8001/scenes/P01> until it says
`ready` and lists a `skybox_<seed>.png`.

A scene that was POSTed as `procedural` (step 3) stays procedural. Each POST replaces the
participant's previous scene, so switching sky is: POST again with the other `sky_mode`, then
Fetch again.

### If something looks wrong

| What you see | What it means |
| --- | --- |
| Unity: "No scene for participant P01" | No scene, and no `P01.json` in `auto_create.personas_dir` to build one from (ids are case-sensitive). Add the persona, or do step 3 |
| Unity: "Orchestrator not reachable. Using P01's files from an earlier fetch" | The server isn't running, so you're looking at old files. Do step 2 |
| Unity: "ready, with Unity's own sky: the 360 image could not be made" | The image was wanted but couldn't be made, so the scene fell back to procedural; the Console warning says why. Fix that (usually the SSH connection), wait a minute, and Fetch again: the image is tried again |
| Console: `sky: none, the scene's own sky kept` | An old spec from before sky modes. Create the scene again (step 3) |
| Pressing Play directly in `ExperimentScene` | Nothing is fetched; it shows the files of the last fetch |

## Tests

```bash
python -m pytest
```

No cluster needed: the tests simulate the Pegasus login node.
