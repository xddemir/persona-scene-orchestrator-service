"""places_v1.yaml: the fixed list of places a scene can be at, loaded and checked.

A place is one Unity setting (archetype) with its axes set (landform,
position, water, water relation, vegetation, an optional season), plus a
biome and two numbers the pick rule reads: shelter_level and novelty. The
list is data, not code, so a place's status can be flipped after a headset
check (python -m scene_orchestrator.places mark). Which place a persona gets
is the mapper's rule (rule_mapper.pick_place); this module only reads the
list and refuses one Unity would not build as written.

The file's SHA-256 is kept with the places, so the manifest can record which
list each scene was made from.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..axes import Axes, axis_problems, resolve_axes
from ..models import Archetype, Biome, Landform, Position, Season, Vegetation
from ..models import WaterKind, WaterRelation

PLACES_FILE = Path(__file__).with_name("places_v1.yaml")

PlaceStatus = Literal["candidate", "verified", "rejected"]

# Review scenes are registered as "place-<id>", which must stay a scene id
# (at most 64 characters).
REVIEW_PREFIX = "place-"
PLACE_ID_PATTERN = r"^[a-z0-9][a-z0-9_]{0,57}$"

# The prop that only belongs at a lake. Unity's ground floats its own lilies
# on the water; a kit asking for them where there is no lake asks for nothing
# that fits.
LAKE_ONLY_PROPS = frozenset({"water_lily"})


class PlacesError(ValueError):
    """places_v1.yaml can't be used as written. The message says where and why."""


class Place(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=PLACE_ID_PATTERN)
    setting: Archetype  # the Unity setting: SceneSpec.archetype
    biome: Biome  # the kit: props, trees, sounds, the sky's hue
    landform: Landform
    position: Position
    water: WaterKind
    water_relation: WaterRelation
    vegetation: Vegetation
    season: Season | None = None  # None: the setting's
    shelter_level: int = Field(ge=1, le=5)  # 1 open .. 5 sheltered
    novelty: float = Field(ge=0.0, le=1.0)  # 0 familiar .. 1 unusual
    status: PlaceStatus
    checked: date | None  # the last headset check; the key is required, null allowed
    note: str

    @property
    def axes(self) -> Axes:
        """Its axes, resolved against its setting the way a spec's are."""
        return resolve_axes(
            self.setting,
            landform=self.landform,
            position=self.position,
            water=self.water,
            water_relation=self.water_relation,
            vegetation=self.vegetation,
            season=self.season,
        )

    @property
    def scene_id(self) -> str:
        """The scene a headset review loads it as."""
        return f"{REVIEW_PREFIX}{self.id}"


class _PlacesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    places: list[Place] = Field(min_length=1)


@dataclass(frozen=True)
class Places:
    """The places as loaded, with the file they came from and its hash."""

    places: tuple[Place, ...]
    path: Path
    sha256: str  # of the file's bytes as read: the list a scene was made from

    def get(self, place_id: str) -> Place:
        for place in self.places:
            if place.id == place_id:
                return place
        raise PlacesError(f"no place {place_id!r} in {self.path.name}")


def load_places(path: Path = PLACES_FILE, *, kits: Mapping[str, Collection[str]]) -> Places:
    """Read and check the places. `kits`: each biome's tree and props by name."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PlacesError(f"cannot read {path}: {exc}") from exc
    return parse_places(raw, kits=kits, path=path)


def parse_places(raw: bytes, *, kits: Mapping[str, Collection[str]], path: Path) -> Places:
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
        places = _PlacesFile.model_validate(data).places
    except (UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
        raise PlacesError(f"{path}: {exc}") from exc
    problems = check_places(places, kits)
    if problems:
        raise PlacesError(f"{path}:\n  " + "\n  ".join(problems))
    return Places(tuple(places), path, hashlib.sha256(raw).hexdigest())


def check_places(places: list[Place], kits: Mapping[str, Collection[str]]) -> list[str]:
    """Every reason a place can't be used, one line each, prefixed with its id."""
    problems = [
        f"{place_id}: listed {n} times" for place_id, n in Counter(p.id for p in places).items()
        if n > 1
    ]
    for place in places:
        if place.biome not in kits:
            problems.append(f"{place.id}: biome {place.biome} has no kit")
            continue
        problems += [f"{place.id}: {problem}" for problem in axis_problems(place.setting, place.axes)]
        lake_only = LAKE_ONLY_PROPS & set(kits[place.biome])
        if lake_only and place.water != "lake":
            problems.append(
                f"{place.id}: biome {place.biome}'s kit has {', '.join(sorted(lake_only))}, "
                f"which only goes with a lake, but the water is {place.water}"
            )
    return problems


# -- changing one place in the file -----------------------------------------


def set_place_fields(text: str, place_id: str, fields: Mapping[str, str]) -> str:
    """`text` with these keys of one place set to these (YAML) values, every
    other line as it was, comments included. Each key must already be there."""
    lines = text.splitlines(keepends=True)
    start, indent = None, 0
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)-\s+id:\s*['\"]?([^'\"\s#]+)['\"]?\s*(#.*)?$", line.rstrip("\r\n"))
        if match and match.group(2) == place_id:
            start, indent = i, len(match.group(1)) + 2
            break
    if start is None:
        raise PlacesError(f"no place {place_id!r}")

    # The place runs until the next line indented less than its keys.
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(lines[j]) - len(lines[j].lstrip()) < indent:
            end = j
            break

    for key, value in fields.items():
        pattern = re.compile(rf"^ {{{indent}}}{re.escape(key)}:")
        for j in range(start + 1, end):
            if pattern.match(lines[j]):
                body = lines[j].rstrip("\r\n")
                lines[j] = f"{' ' * indent}{key}: {value}{lines[j][len(body):]}"
                break
        else:
            raise PlacesError(f"place {place_id!r} has no {key!r} line to set")
    return "".join(lines)
