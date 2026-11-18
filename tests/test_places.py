"""The places (places_v1.yaml): loading, checking, the pick rule, the record
each scene keeps of its place, and the headset review tooling."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fakes import FakePegasus, make_client
from scene_orchestrator import places as places_cli
from scene_orchestrator.adapters import ChatbotV2Adapter
from scene_orchestrator.api.app import create_app
from scene_orchestrator.manifest import read_manifest
from scene_orchestrator.catalogue import CATALOGUE_V1
from scene_orchestrator.mapping.places import (
    MIN_PROP_KINDS,
    NOT_FOR_PLACES,
    PLACES_FILE,
    Place,
    PlacesError,
    parse_places,
    set_place_fields,
)
from scene_orchestrator.mapping.rule_mapper import (
    PLACES,
    NoUsablePlace,
    RuleSpecMapper,
    _novelty,
    _shelter,
    pick_place,
)
from scene_orchestrator.models import PersonaProfile, SceneSpec, Traits
from scene_orchestrator.pipeline import ScenePipeline

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "personas" / "chatbot"
TEXT = PLACES_FILE.read_text(encoding="utf-8")

# The starting list, as handed over on 2026-11-18. Change it here when the
# list is changed on purpose.
STARTING_LIST = [
    # id, setting, landform, position, water, relation, vegetation, season, biome, shelter, novelty
    ("forest_lake", "forest", "plain", "mid", "lake", "beside", "dense", None, "forest_clearing", 5, 0.2),
    ("hollow_pond", "hollow", "bowl", "bottom", "lake", "beside", "dense", None, "birch_grove", 5, 0.4),
    ("snow_hollow", "hollow", "bowl", "bottom", "none", "absent", "moderate", "winter", "snowy_valley", 5, 0.9),
    ("forest_river", "forest", "plain", "mid", "river", "beside", "moderate", None, "birch_grove", 4, 0.3),
    ("valley_stream", "hollow", "valley", "bottom", "river", "beside", "moderate", None, "forest_clearing", 4, 0.5),
    ("terraces_foot", "summit", "terraces", "bottom", "none", "absent", "dense", None, "alpine_basin", 4, 0.8),
    ("meadow_lake", "meadow", "plain", "mid", "lake", "beside", "moderate", None, "lakeside", 3, 0.1),
    ("valley_lake_view", "meadow", "valley", "mid", "lake", "distant", "moderate", None, "lakeside", 3, 0.5),
    ("beach_cove", "beach", "shore", "mid", "sea", "beside", "moderate", None, "coastal_pine", 3, 0.4),
    ("open_meadow", "meadow", "plain", "mid", "none", "absent", "sparse", None, "open_meadow", 2, 0.0),
    ("ridge_lake_below", "overlook", "ridge", "mid", "lake", "below", "sparse", None, "open_meadow", 2, 0.5),
    ("rocky_shore_top", "beach", "shore", "top", "sea", "beside", "sparse", None, "rocky_shore", 2, 0.8),
    ("overlook_crest", "overlook", "ridge", "top", "none", "absent", "sparse", None, "open_meadow", 1, 0.3),
    ("cliff_over_sea", "overlook", "ridge", "top", "sea", "below", "sparse", None, "coastal_pine", 1, 0.7),
    ("summit_top", "summit", "terraces", "top", "none", "absent", "sparse", None, "alpine_basin", 1, 0.9),
]


@pytest.fixture(autouse=True)
def _the_real_file_stays_as_it_was():
    """No test may change places_v1.yaml itself; put it back if one does."""
    before = PLACES_FILE.read_bytes()
    yield
    after = PLACES_FILE.read_bytes()
    if after != before:
        PLACES_FILE.write_bytes(before)
        pytest.fail("the test changed src/scene_orchestrator/mapping/places_v1.yaml")


def _persona(**traits) -> PersonaProfile:
    values = dict.fromkeys(Traits.model_fields, 0.5)
    values.update(traits)
    return PersonaProfile(participant_id="t", source="stub", traits=Traits(**values))


def _parse(text: str):
    return parse_places(text.encode("utf-8"), path=Path("places_test.yaml"))


FOREST_PROPS = [("pine_large", 1.0), ("bush", 0.8), ("pine_medium", 0.3), ("rock_small", 0.2),
                ("boulder", 0.15)]


def _with_place(**fields) -> str:
    """The real file with one more place: a forest lake, with these fields instead."""
    place = {
        "id": "extra", "setting": "forest", "biome": "forest_clearing", "landform": "plain",
        "position": "mid", "water": "lake", "water_relation": "beside", "vegetation": "dense",
        "shelter_level": 3, "novelty": 0.5, "focal": "pine_large", "framing": "pine_large",
        "props": FOREST_PROPS, "status": "candidate", "checked": None, "note": "",
        **fields,
    }
    lines = [f"  - id: {place.pop('id')}"]
    for key, value in place.items():
        if value == "<omit>":
            continue
        if key == "props" and isinstance(value, list):
            lines.append("    props:")
            lines += [f"      - {item}" if isinstance(item, str)  # a malformed line
                      else f"      - {item[0]}: {json.dumps(item[1])}" for item in value]
        else:
            lines.append(f"    {key}: {json.dumps(value)}")
    return TEXT + "\n" + "\n".join(lines) + "\n"


# -- loading ----------------------------------------------------------------


def test_the_places_load_with_the_hash_of_their_file():
    assert len(PLACES.places) == 15
    assert len({p.id for p in PLACES.places}) == 15
    assert PLACES.sha256 == hashlib.sha256(PLACES_FILE.read_bytes()).hexdigest()
    assert PLACES.path == PLACES_FILE


def test_the_starting_list_is_as_handed_over():
    got = [(p.id, p.setting, p.landform, p.position, p.water, p.water_relation, p.vegetation,
            p.season, p.biome, p.shelter_level, p.novelty) for p in PLACES.places]
    assert got == STARTING_LIST


def test_every_place_starts_as_an_unchecked_candidate():
    for place in PLACES.places:
        assert (place.status, place.checked) == ("candidate", None)


def test_review_scene_ids_fit_unity():
    for place in PLACES.places:
        assert place.scene_id == f"place-{place.id}"
        assert len(place.scene_id) <= 64


def test_axes_are_resolved_before_they_are_checked():
    """A place leaving the season out gets the setting's, as a spec would."""
    forest_lake = PLACES.get("forest_lake")
    assert forest_lake.season is None
    assert forest_lake.axes.season == "summer"  # the forest's
    assert PLACES.get("snow_hollow").axes.season == "winter"


def test_a_place_and_a_spec_are_checked_by_the_same_rules():
    """Every place gives a valid spec, and a combination one rejects the other does too."""
    for place in PLACES.places:
        RuleSpecMapper(place_id=place.id).map(_persona(), 0, "t")  # validates
    bad = {"landform": "plain", "water_relation": "below"}  # a forest with water below
    with pytest.raises(PlacesError, match=r"plain \+ water below: a plain has nothing to fall from"):
        _parse(_with_place(**bad))
    spec = RuleSpecMapper(place_id="forest_lake").map(_persona(), 0, "t").model_dump(mode="json")
    spec["terrain"]["water_relation"] = "below"
    with pytest.raises(ValidationError, match=r"plain \+ water below: a plain has nothing to fall from"):
        SceneSpec.model_validate(spec)


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        # landform x water relation (SceneAxes.WaterTable)
        ({"landform": "bowl", "water_relation": "below", "setting": "hollow",
          "position": "bottom"}, r"bowl \+ water below"),
        ({"setting": "summit", "landform": "terraces", "position": "top", "water": "none",
          "water_relation": "beside", "biome": "alpine_basin"}, "no water"),
        ({"setting": "beach", "landform": "shore", "water": "none", "water_relation": "absent",
          "biome": "rocky_shore"}, "a beach always has the sea"),
        # landform x position (SceneAxes.SpawnTable)
        ({"position": "top"}, r"plain \+ position top"),
        ({"setting": "hollow", "landform": "valley", "position": "top", "water": "river"},
         "the top of a valley is not a valley"),
        # the water against its relation
        ({"water": "none"}, "water_relation beside with no water"),
        ({"water_relation": "absent"}, "water lake with water_relation absent"),
        # the setting's May / Must / Never
        ({"setting": "summit", "landform": "ridge", "position": "top", "water": "lake",
          "water_relation": "below"}, "a summit never has a lake"),
        ({"setting": "meadow", "water": "sea"}, "a meadow never has the sea"),
        ({"setting": "beach", "landform": "shore", "water": "lake"}, "a beach never has a lake"),
        # the place's props
        ({"props": FOREST_PROPS[:4]}, "props: 4 kinds, needs at least 5"),
        ({"props": [*FOREST_PROPS, ("bush", 0.1)]}, "props: bush listed 2 times"),
        ({"props": [*FOREST_PROPS[:4], ("water_lily", 0.05)]},
         "props: water_lily: Unity lays a spec's props out on dry land"),
        ({"props": [*FOREST_PROPS[:4], ("fern", 0.5)]}, "props: fern places nothing in Unity"),
        ({"props": [*FOREST_PROPS[:4], ("birch_light", 0.5)]}, "birch_light places nothing"),
        ({"props": [*FOREST_PROPS[:4], ("mountain", 0.5)]}, "mountain places nothing"),
        ({"focal": "driftwood"}, "focal: driftwood places nothing"),
        ({"framing": "water_lily"}, "framing: water_lily: Unity lays"),
        ({"props": [*FOREST_PROPS[:4], ("boulder", 0)]}, "greater than 0"),
        ({"props": ["pine_large", "bush"]}, "one 'name: weight' line"),
        ({"focal": "<omit>"}, "focal"),
        # ids, names and fields
        ({"id": "forest_lake"}, "forest_lake: listed 2 times"),
        ({"biome": "volcano"}, "biome"),
        ({"setting": "castle"}, "setting"),
        ({"shelter_level": 6}, "shelter_level"),
        ({"novelty": 1.5}, "novelty"),
        ({"status": "maybe"}, "status"),
        ({"checked": "<omit>"}, "checked"),
        ({"colour": "green"}, "colour"),
        ({"id": "Forest Lake"}, "id"),
    ],
)
def test_a_place_unity_would_change_fails_the_load(fields, why):
    with pytest.raises(PlacesError, match=why):
        _parse(_with_place(**fields))


def test_every_place_has_its_own_props_most_typical_first():
    """At least five kinds with a model in Unity, all different; the first
    is always placed, openness takes up to five."""
    for place in PLACES.places:
        names = [name for name, _ in place.props]
        assert len(names) == len(set(names)) >= MIN_PROP_KINDS, place.id
        assert place.names <= CATALOGUE_V1.props - set(NOT_FOR_PLACES), place.id
        assert all(weight > 0 for _, weight in place.props)


def test_no_place_asks_for_water_lilies():
    """Unity's ground floats its own on a lake; a spec's would lie on dry land."""
    assert not any("water_lily" in place.names for place in PLACES.places)


def test_the_props_are_read_in_the_files_order():
    assert PLACES.get("forest_lake").props == (
        ("pine_large", 1.0), ("bush", 0.8), ("pine_medium", 0.3), ("rock_small", 0.2),
        ("boulder", 0.15),
    )
    assert (PLACES.get("forest_lake").focal, PLACES.get("forest_lake").framing) == (
        "pine_large", "pine_large"
    )


def test_a_valid_extra_place_loads():
    places = _parse(_with_place(landform="ridge", water_relation="below", position="top"))
    assert places.get("extra").water_relation == "below"


# -- the pick rule ----------------------------------------------------------


def _place(place_id: str, level: int, novelty: float, status: str = "verified") -> Place:
    return PLACES.get("forest_lake").model_copy(update={
        "id": place_id, "shelter_level": level, "novelty": novelty, "status": status,
    })


ONE_PER_BAND = [_place(f"band{level}", level, 0.5) for level in (1, 2, 3, 4, 5)]


@pytest.mark.parametrize(
    ("shelter", "band"),
    [(0.0, 1), (0.124, 1), (0.125, 2), (0.3, 2),
     (0.374, 2), (0.375, 3),  # 2.5 -> 3: Python's round() would give 2
     (0.5, 3), (0.624, 3), (0.625, 4),  # 3.5 -> 4
     (0.875, 5), (1.0, 5)],
)
def test_shelter_picks_the_band_rounding_half_up(shelter, band):
    place = pick_place(ONE_PER_BAND, shelter, 0.5, allow_candidates=False)
    assert place.shelter_level == band


def test_within_the_band_the_nearest_novelty_wins():
    places = [_place("a", 3, 0.1), _place("b", 3, 0.5), _place("c", 3, 0.9), _place("d", 2, 0.5)]
    assert pick_place(places, 0.5, 0.0, allow_candidates=False).id == "a"
    assert pick_place(places, 0.5, 0.45, allow_candidates=False).id == "b"
    assert pick_place(places, 0.5, 0.71, allow_candidates=False).id == "c"


def test_a_novelty_tie_goes_to_the_first_in_the_list():
    # 0.7 - 0.5 is 0.19999999999999996 in floats, 0.5 - 0.3 is 0.2: still a tie.
    places = [_place("first", 1, 0.3), _place("second", 1, 0.7)]
    assert pick_place(places, 0.0, 0.5, allow_candidates=False).id == "first"
    assert pick_place(places[::-1], 0.0, 0.5, allow_candidates=False).id == "second"
    # The real list: overlook_crest (0.3) before cliff_over_sea (0.7).
    assert pick_place(PLACES.places, 0.0, 0.5, allow_candidates=True).id == "overlook_crest"


def test_an_empty_band_falls_back_to_the_nearest_and_then_the_more_sheltered():
    places = [_place("open", 1, 0.5), _place("low", 2, 0.5), _place("high", 4, 0.5)]
    assert pick_place(places, 0.5, 0.5, allow_candidates=False).id == "high"  # 3: 2 and 4 tie
    assert pick_place(places, 1.0, 0.5, allow_candidates=False).id == "high"  # 5: 4 is nearest
    only_open = [_place("open", 1, 0.5)]
    assert pick_place(only_open, 1.0, 0.5, allow_candidates=False).id == "open"


def test_candidates_count_only_when_allowed_and_rejected_never():
    places = [_place("checked", 2, 0.5), _place("unchecked", 3, 0.5, "candidate"),
              _place("bad", 3, 0.5, "rejected")]
    assert pick_place(places, 0.5, 0.5, allow_candidates=False).id == "checked"
    assert pick_place(places, 0.5, 0.5, allow_candidates=True).id == "unchecked"
    with pytest.raises(NoUsablePlace, match="none is verified"):
        pick_place(places[2:], 0.5, 0.5, allow_candidates=True)


def test_without_a_usable_place_the_error_says_what_to_do():
    with pytest.raises(NoUsablePlace, match="places.allow_candidates is false"):
        pick_place(PLACES.places, 0.5, 0.5, allow_candidates=False)
    with pytest.raises(NoUsablePlace):
        RuleSpecMapper().map(_persona(), 0, "t")  # the study's default: verified only


def test_the_rule_is_pure():
    for shelter in (0.0, 0.33, 0.5, 0.71, 1.0):
        for novelty in (0.0, 0.5, 1.0):
            picks = {pick_place(PLACES.places, shelter, novelty, allow_candidates=True).id
                     for _ in range(5)}
            assert len(picks) == 1


def test_shelter_and_novelty_come_from_todays_traits():
    persona = _persona(neuroticism=0.8, openness=0.3)
    assert (_shelter(persona), _novelty(persona)) == (0.8, 0.3)


def test_the_place_follows_shelter_first():
    mapper = RuleSpecMapper(allow_candidates=True)
    levels = [mapper.place_for(_persona(neuroticism=n)).shelter_level for n in (0, 0.25, 0.5, 0.75, 1)]
    assert levels == [1, 2, 3, 4, 5]


# -- every scene records its place ------------------------------------------


def _raw(alias: str) -> dict:
    return json.loads((FIXTURES / f"{alias}.json").read_text(encoding="utf-8"))


def test_the_manifest_records_the_place_its_status_the_flag_and_the_list(out_dir):
    pipeline = ScenePipeline(
        make_client(FakePegasus()), out_dir, mapper=RuleSpecMapper(allow_candidates=True)
    )
    assert pipeline.run(_raw("P01"), "P01", sky_mode="procedural").status == "ok"
    assert read_manifest(out_dir)["P01"]["place"] == {
        "id": "terraces_foot",
        "status": "candidate",
        "allow_candidates": True,
        "places_sha256": hashlib.sha256(PLACES_FILE.read_bytes()).hexdigest(),
    }


def test_the_record_follows_the_list_the_scene_was_made_from(out_dir):
    """Another list (here: terraces_foot verified) and the study's flag."""
    verified = set_place_fields(TEXT, "terraces_foot", {"status": "verified"})
    places = _parse(verified)
    mapper = RuleSpecMapper(allow_candidates=False, places=places)
    pipeline = ScenePipeline(make_client(FakePegasus()), out_dir, mapper=mapper)
    assert pipeline.run(_raw("P01"), "P01", sky_mode="procedural").status == "ok"
    assert read_manifest(out_dir)["P01"]["place"] == {
        "id": "terraces_foot",
        "status": "verified",
        "allow_candidates": False,
        "places_sha256": hashlib.sha256(verified.encode()).hexdigest(),
    }


def test_a_scene_whose_persona_fails_records_no_place(out_dir):
    pipeline = ScenePipeline(
        make_client(FakePegasus()), out_dir, mapper=RuleSpecMapper(allow_candidates=True)
    )
    assert pipeline.run(_raw("P03"), "P03", sky_mode="procedural").status == "failed"
    assert "place" not in read_manifest(out_dir)["P03"]


def test_no_usable_place_fails_the_scene_with_the_reason(out_dir):
    pipeline = ScenePipeline(make_client(FakePegasus()), out_dir, mapper=RuleSpecMapper())
    outcome = pipeline.run(_raw("P01"), "P01", sky_mode="procedural")
    assert outcome.status == "failed"
    assert "NoUsablePlace" in read_manifest(out_dir)["P01"]["error"]


# -- headset review ---------------------------------------------------------


def test_review_registers_each_place_as_unity_fetches_a_participant(config, out_dir):
    outcomes = places_cli.review(list(PLACES.places), out_dir, allow_candidates=True)
    assert set(outcomes) == {p.scene_id for p in PLACES.places}
    assert {o.status for o in outcomes.values()} == {"ok"}

    manifest = read_manifest(out_dir)
    entry = manifest["place-forest_lake"]
    assert (entry["status"], entry["sky_mode"], entry["files"]) == ("ok", "procedural",
                                                                    ["scene_spec.json"])
    assert entry["place"] == {
        "id": "forest_lake", "status": "candidate", "allow_candidates": True,
        "places_sha256": PLACES.sha256, "forced": True,
    }

    with TestClient(create_app(config, image_gen=make_client(FakePegasus()))) as client:
        scene = client.get("/scenes/place-forest_lake").json()
        assert scene["status"] == "ready"
        assert scene["files"] == ["scene_spec.json"]
        spec = SceneSpec.model_validate(scene["spec"])
        file = client.get("/scenes/place-forest_lake/files/scene_spec.json")
        assert file.status_code == 200
        assert SceneSpec.model_validate(file.json()) == spec

    # The place, forced; a neutral persona; seed 0; no image.
    assert (spec.scene_id, spec.seed, spec.sky_mode) == ("place-forest_lake", 0, "procedural")
    assert (spec.archetype, spec.biome) == ("forest", "forest_clearing")
    assert (spec.terrain.landform, spec.terrain.position, spec.terrain.water.kind,
            spec.terrain.water_relation, spec.terrain.vegetation) == (
        "plain", "mid", "lake", "beside", "dense"
    )
    assert spec.spatial.enclosure == 0.475  # the rule's middle: 0.15 + 0.65 / 2


def test_review_forces_even_a_rejected_place(out_dir):
    rejected = PLACES.get("summit_top").model_copy(update={"status": "rejected"})
    outcome = places_cli.review([rejected], out_dir, allow_candidates=False)["place-summit_top"]
    assert outcome.status == "ok"
    assert outcome.spec.archetype == "summit"


def test_the_review_command_builds_one_place(tmp_path, out_dir, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"out_dir: {json.dumps(str(out_dir))}\nimage_gen:\n"
        + "".join(f"  {k}: {json.dumps(v)}\n" for k, v in {
            "login": "x@y", "repo": "/r", "venv": "/v", "hf_home": "/h", "remote_out": "/o",
            "account": "a"}.items()),
        encoding="utf-8",
    )
    assert places_cli.main(["review", "--place", "beach_cove", "--config", str(config)]) == 0
    assert "place-beach_cove: ok. beach, shore mid, sea beside" in capsys.readouterr().out
    assert set(read_manifest(out_dir)) == {"place-beach_cove"}
    assert read_manifest(out_dir)["place-beach_cove"]["place"]["allow_candidates"] is False

    assert places_cli.main(["review", "--place", "atlantis", "--config", str(config)]) == 2
    assert "no place 'atlantis'" in capsys.readouterr().err


# -- marking a place after the headset check --------------------------------


@pytest.fixture
def places_file(tmp_path: Path) -> Path:
    path = tmp_path / "places_v1.yaml"
    path.write_bytes(PLACES_FILE.read_bytes())
    return path


def test_mark_sets_status_date_and_note_and_nothing_else(places_file):
    place = places_cli.mark(
        "hollow_pond", "verified", 'reads as a pond: "quiet", even at dusk',
        path=places_file, today=date(2026, 11, 18),
    )
    assert (place.status, place.checked) == ("verified", date(2026, 11, 18))
    assert place.note == 'reads as a pond: "quiet", even at dusk'

    before = PLACES_FILE.read_text(encoding="utf-8").splitlines()
    after = places_file.read_text(encoding="utf-8").splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after)
    assert changed == [
        ("    status: candidate", "    status: verified"),
        ("    checked: null", "    checked: 2026-11-18"),
        ('    note: ""', '    note: "reads as a pond: \\"quiet\\", even at dusk"'),
    ]
    reloaded = parse_places(places_file.read_bytes(), path=places_file)
    assert reloaded.get("hollow_pond") == place
    assert reloaded.get("forest_lake") == PLACES.get("forest_lake")  # the one before it
    assert reloaded.get("snow_hollow") == PLACES.get("snow_hollow")  # and after


def test_mark_keeps_the_note_when_none_is_given(places_file):
    places_cli.mark("summit_top", "rejected", "edge floats", path=places_file,
                    today=date(2026, 11, 18))
    place = places_cli.mark("summit_top", "verified", path=places_file, today=date(2026, 11, 19))
    assert (place.status, place.checked, place.note) == ("verified", date(2026, 11, 19),
                                                         "edge floats")


def test_mark_refuses_an_unknown_place(places_file):
    with pytest.raises(PlacesError, match="no place 'atlantis'"):
        places_cli.mark("atlantis", "verified", path=places_file)
    assert places_file.read_bytes() == PLACES_FILE.read_bytes()


def test_the_mark_command_writes_the_file_it_is_pointed_at(places_file, monkeypatch, capsys):
    monkeypatch.setattr(places_cli, "PLACES_FILE", places_file)
    assert places_cli.main(["mark", "open_meadow", "rejected", "--note", "too flat"]) == 0
    assert "open_meadow: rejected, checked" in capsys.readouterr().out
    assert parse_places(places_file.read_bytes(), path=places_file).get(
        "open_meadow"
    ).status == "rejected"
    assert PLACES_FILE.read_text(encoding="utf-8") == TEXT  # the real file untouched


def test_a_marked_place_is_picked_without_candidates(places_file):
    places_cli.mark("valley_lake_view", "verified", path=places_file, today=date(2026, 11, 18))
    places = parse_places(places_file.read_bytes(), path=places_file)
    mapper = RuleSpecMapper(places=places)  # verified only
    for neuroticism in (0.0, 0.5, 1.0):  # the only usable place, whatever the shelter
        persona = _persona(neuroticism=neuroticism)
        assert mapper.place_for(persona).id == "valley_lake_view"
        assert mapper.provenance(persona)["place"]["status"] == "verified"


def test_the_fixture_adapter_still_drives_review_free_scenes():
    """A participant's persona, not the neutral one, outside review."""
    persona = ChatbotV2Adapter().adapt(_raw("P02"))
    assert RuleSpecMapper(allow_candidates=True).place_for(persona).id == "ridge_lake_below"
