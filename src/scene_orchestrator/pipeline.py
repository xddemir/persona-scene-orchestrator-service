"""One participant's scene, end to end.

Each participant gets exactly one scene, from their own persona, so the
scene_id is the participant id (e.g. "P01").

    raw persona -> ChatbotV2Adapter -> PersonaProfile
                -> RuleSpecMapper -> SceneSpec
                -> TemplatePromptBuilder -> spec.skybox.prompt
                -> Slurm job on Pegasus -> skybox PNG + sidecar   (panorama only)
                -> out/<scene_id>/scene_spec.json, out/manifest.json

The spec is built first and the prompt is derived from it, never the reverse.

sky_mode "procedural" stops before the Slurm job: the spec already describes
the whole scene, sky included. In "panorama" mode a skybox that cannot be made
does not fail the scene either. The spec is written with sky_mode flipped to
"procedural" and the manifest says "fallback", so Unity still renders a scene
matched to the persona, only without the panorama.

Every path ends in the manifest. A scene is entered as failed ("did not
finish") the moment it starts and overwritten when it ends, so even a process
killed halfway leaves Unity an entry to read rather than a missing one. That
entry also records the Slurm job once it is submitted, and the next run of the
scene picks the job up from there instead of submitting it again.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adapters import ChatbotV2Adapter, PersonaAdapter
from .clients.image_gen import SkyboxRequest, SlurmImageGenClient
from .manifest import ManifestStatus, read_manifest, update_manifest
from .mapping import PromptBuilder, RuleSpecMapper, SpecMapper, TemplatePromptBuilder, attach_prompt
from .models import PersonaProfile, SceneSpec, SkyMode
from .outputs import scene_spec_path, validate_scene_id, write_json

SCENE_SPEC_FILE = "scene_spec.json"
UNFINISHED = "generation started but did not finish"


def scene_id_for(participant_id: str) -> str:
    """The participant id, checked to be a safe folder name (else ValueError)."""
    return validate_scene_id(participant_id)


def derive_seed(participant_id: str) -> int:
    """A stable seed per participant, in 0 .. 2**32 - 1.

    SHA-256 rather than hash(): Python randomises str hashes per process, and
    the same participant must get the same scene on every run and machine.
    """
    digest = hashlib.sha256(participant_id.encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class SceneOutcome:
    status: ManifestStatus  # as written to the manifest
    spec: SceneSpec | None
    files: list[str]
    error: str | None = None
    persona: PersonaProfile | None = None

    @property
    def usable(self) -> bool:
        """Whether Unity has a scene to render: ok, or the procedural fallback."""
        return self.status != "failed"


@dataclass(frozen=True)
class _PendingJob:
    """A skybox job that was submitted and not seen to finish."""

    job_id: str
    seed: int | None

    @classmethod
    def from_entry(cls, entry: dict[str, Any] | None) -> _PendingJob | None:
        if not entry or entry.get("status") == "ok" or not entry.get("slurm_job_id"):
            return None
        return cls(str(entry["slurm_job_id"]), entry.get("seed"))


@dataclass(frozen=True)
class _Built:
    status: ManifestStatus
    sky_mode: SkyMode  # of the spec as written
    files: list[str]
    error: str | None = None
    pending: _PendingJob | None = None


class ScenePipeline:
    def __init__(
        self,
        image_gen: SlurmImageGenClient,
        out_dir: Path,
        *,
        adapter: PersonaAdapter | None = None,
        mapper: SpecMapper | None = None,
        prompts: PromptBuilder | None = None,
    ) -> None:
        self._image_gen = image_gen
        self._out_dir = Path(out_dir)
        self._adapter = adapter or ChatbotV2Adapter()
        self._mapper = mapper or RuleSpecMapper()
        self._prompts = prompts or TemplatePromptBuilder()

    def run(
        self,
        raw_persona: dict[str, Any],
        scene_id: str,
        seed: int | None = None,
        sky_mode: SkyMode = "panorama",
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[str], None] | None = None,
        on_built: Callable[[PersonaProfile, SceneSpec], None] | None = None,
    ) -> SceneOutcome:
        """Never raises: every outcome is returned and recorded in the manifest.

        on_built receives the persona and spec as soon as they exist, so a
        caller can show them while the (slow) image is still being made.
        """
        try:
            pending = _PendingJob.from_entry(read_manifest(self._out_dir).get(scene_id))
            # Carried over, so stopping again before the job is picked up
            # still doesn't lose it.
            self._unfinished(scene_id, sky_mode, pending)
        except Exception as exc:  # noqa: BLE001
            return SceneOutcome("failed", None, [], f"cannot write the manifest: {_describe(exc)}")

        seen: dict[str, Any] = {}

        def remember(persona: PersonaProfile, spec: SceneSpec) -> None:
            seen.update(persona=persona, spec=spec)
            if on_built is not None:
                on_built(persona, spec)

        try:
            built = self._build(
                raw_persona, scene_id, seed, sky_mode, pending,
                on_stage or (lambda _: None), on_progress, remember,
            )
        except Exception as exc:  # noqa: BLE001 - whatever it was, it goes in the manifest
            built = _Built("failed", sky_mode, [], _describe(exc))

        status, error = built.status, built.error
        try:
            update_manifest(
                self._out_dir, scene_id, status, built.sky_mode, built.files, error,
                slurm_job_id=built.pending.job_id if built.pending else None,
                seed=built.pending.seed if built.pending else None,
            )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = f"{error + '; ' if error else ''}manifest update failed: {_describe(exc)}"
        return SceneOutcome(status, seen.get("spec"), built.files, error, seen.get("persona"))

    def _build(
        self,
        raw_persona: dict[str, Any],
        scene_id: str,
        seed: int | None,
        sky_mode: SkyMode,
        pending: _PendingJob | None,
        on_stage: Callable[[str], None],
        on_progress: Callable[[str], None] | None,
        on_built: Callable[[PersonaProfile, SceneSpec], None],
    ) -> _Built:
        on_stage("building_spec")
        persona = self._adapter.adapt(raw_persona)
        if persona.participant_id != scene_id:
            raise ValueError(
                f"the persona is {persona.participant_id!r}'s, but the scene is {scene_id!r}: "
                "a participant's scene is built from their own persona"
            )
        if seed is None:
            seed = derive_seed(persona.participant_id)
        spec = self._mapper.map(persona, seed, scene_id).model_copy(update={"sky_mode": sky_mode})
        spec = attach_prompt(spec, self._prompts)
        on_built(persona, spec)

        files = [SCENE_SPEC_FILE]
        if sky_mode == "procedural":
            # No image to make: the spec's procedural_sky is the sky.
            self._write_spec(spec)
            return _Built("ok", "procedural", files)

        on_stage("generating_image")
        result = self._image_gen.generate_skybox(
            SkyboxRequest(prompt=spec.skybox.prompt, seed=spec.seed, scene_id=scene_id),
            self._out_dir,
            on_progress,
            # Only a job for this very request: another seed is another image.
            resume_job_id=pending.job_id if pending and pending.seed == spec.seed else None,
            on_submitted=lambda job_id: self._unfinished(
                scene_id, sky_mode, _PendingJob(job_id, spec.seed)
            ),
        )
        if result.ok:
            # A bare file name: Unity resolves it against the scene's folder.
            spec = spec.model_copy(
                update={"skybox": spec.skybox.model_copy(update={"uri": result.png.name})}
            )
            on_built(persona, spec)
            self._write_spec(spec)
            return _Built("ok", "panorama", [*files, result.png.name, result.sidecar.name])

        # No panorama, but the scene is not lost: procedural_sky is already
        # filled in from the same persona. The prompt stays on the spec, to
        # show what was attempted.
        spec = spec.model_copy(update={"sky_mode": "procedural"})
        on_built(persona, spec)
        self._write_spec(spec)
        still_running = (
            _PendingJob(result.pending_job_id, spec.seed) if result.pending_job_id else None
        )
        return _Built("fallback", "procedural", files, result.error, still_running)

    def _unfinished(self, scene_id: str, sky_mode: SkyMode, pending: _PendingJob | None) -> None:
        update_manifest(
            self._out_dir, scene_id, "failed", sky_mode, [], UNFINISHED,
            slurm_job_id=pending.job_id if pending else None,
            seed=pending.seed if pending else None,
        )

    def _write_spec(self, spec: SceneSpec) -> None:
        write_json(scene_spec_path(self._out_dir, spec.scene_id), spec.model_dump(mode="json"))


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
