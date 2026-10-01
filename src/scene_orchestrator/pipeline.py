"""One participant's scene, end to end.

Each participant gets exactly one scene, from their own persona, so the
scene_id is the participant id (e.g. "P01").

    raw persona -> ChatbotV2Adapter -> PersonaProfile
                -> RuleSpecMapper -> SceneSpec
                -> TemplatePromptBuilder -> spec.skybox.prompt
                -> Slurm job on Pegasus -> skybox PNG + sidecar
                -> out/<scene_id>/scene_spec.json, out/manifest.json

The spec is built first and the prompt is derived from it, never the reverse.

Every path ends in the manifest. A scene is entered as failed ("did not
finish") the moment it starts and overwritten when it ends, so even a process
killed halfway leaves Unity an entry to read rather than a missing one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adapters import ChatbotV2Adapter, PersonaAdapter
from .clients.image_gen import SkyboxRequest, SlurmImageGenClient
from .manifest import update_manifest
from .mapping import PromptBuilder, RuleSpecMapper, SpecMapper, TemplatePromptBuilder, attach_prompt
from .models import PersonaProfile, SceneSpec
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
    ok: bool
    spec: SceneSpec | None
    files: list[str]
    error: str | None = None
    persona: PersonaProfile | None = None


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
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[str], None] | None = None,
        on_built: Callable[[PersonaProfile, SceneSpec], None] | None = None,
    ) -> SceneOutcome:
        """Never raises: every outcome is returned and recorded in the manifest.

        on_built receives the persona and spec as soon as they exist, so a
        caller can show them while the (slow) image is still being made.
        """
        try:
            update_manifest(self._out_dir, scene_id, "failed", [], error=UNFINISHED)
        except Exception as exc:  # noqa: BLE001
            return SceneOutcome(False, None, [], f"cannot write the manifest: {_describe(exc)}")

        built: dict[str, Any] = {}

        def remember(persona: PersonaProfile, spec: SceneSpec) -> None:
            built.update(persona=persona, spec=spec)
            if on_built is not None:
                on_built(persona, spec)

        files: list[str] = []
        try:
            files, error = self._build(
                raw_persona, scene_id, seed,
                on_stage or (lambda _: None), on_progress, remember,
            )
        except Exception as exc:  # noqa: BLE001 - whatever it was, it goes in the manifest
            error = _describe(exc)

        try:
            update_manifest(
                self._out_dir, scene_id, "ok" if error is None else "failed", files, error
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{error + '; ' if error else ''}manifest update failed: {_describe(exc)}"
        return SceneOutcome(
            error is None, built.get("spec"), files, error, built.get("persona")
        )

    def _build(
        self,
        raw_persona: dict[str, Any],
        scene_id: str,
        seed: int | None,
        on_stage: Callable[[str], None],
        on_progress: Callable[[str], None] | None,
        on_built: Callable[[PersonaProfile, SceneSpec], None],
    ) -> tuple[list[str], str | None]:
        on_stage("building_spec")
        persona = self._adapter.adapt(raw_persona)
        if seed is None:
            seed = derive_seed(persona.participant_id)
        spec = attach_prompt(self._mapper.map(persona, seed, scene_id), self._prompts)
        on_built(persona, spec)

        on_stage("generating_image")
        result = self._image_gen.generate_skybox(
            SkyboxRequest(prompt=spec.skybox.prompt, seed=spec.seed, scene_id=scene_id),
            self._out_dir,
            on_progress,
        )
        image_files: list[str] = []
        if result.ok:
            # A bare file name: Unity resolves it against the scene's folder.
            spec = spec.model_copy(
                update={"skybox": spec.skybox.model_copy(update={"uri": result.png.name})}
            )
            image_files = [result.png.name, result.sidecar.name]
            on_built(persona, spec)

        # Written even when the image failed (with skybox.uri null): the
        # manifest says failed, and the spec shows what was attempted.
        write_json(scene_spec_path(self._out_dir, scene_id), spec.model_dump(mode="json"))
        return [SCENE_SPEC_FILE, *image_files], None if result.ok else result.error


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
