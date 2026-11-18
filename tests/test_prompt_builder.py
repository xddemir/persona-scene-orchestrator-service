"""TemplatePromptBuilder: SceneSpec -> prompt text."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

from scene_orchestrator.adapters import ChatbotV2Adapter
from scene_orchestrator.mapping import (
    DEFAULT_NEGATIVE_PROMPT,
    PromptBuilder,
    RuleSpecMapper,
    TemplatePromptBuilder,
    attach_prompt,
)
from scene_orchestrator.axes import SPAWN_TABLE
from scene_orchestrator.mapping.prompt_builder import (
    CLOSED_ABOVE,
    MILKY_WAY_BANDS,
    MOON_BANDS,
    PLANT_WORDS,
    PLANTS_NAMED,
    RAIN_BANDS,
    SNOW_BANDS,
    STAR_BANDS,
    VIEW_PHRASES,
    WATER_PLACE_PHRASES,
    plant_phrase,
)
from scene_orchestrator.mapping.rule_mapper import PLACES
from scene_orchestrator.models import PersonaProfile, SceneSpec, Traits, Weather

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "personas" / "chatbot"
BUILDER: PromptBuilder = TemplatePromptBuilder()  # also checks it fits the protocol
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
MAPPER = RuleSpecMapper(allow_candidates=True)  # every place is still a candidate


def _spec(alias: str) -> SceneSpec:
    raw = json.loads((FIXTURES / f"{alias}.json").read_text(encoding="utf-8"))
    return MAPPER.map(ChatbotV2Adapter().adapt(raw), 1234, alias.lower())


def _persona(**traits) -> PersonaProfile:
    values = dict.fromkeys(TRAITS, 0.5)
    values.update(traits)
    return PersonaProfile(participant_id="t", source="stub", traits=Traits(**values))


def _all_specs():
    """Specs across the whole trait space, every biome tier, several seeds."""
    for values in itertools.product((0.0, 0.5, 1.0), repeat=5):
        persona = _persona(**dict(zip(TRAITS, values)))
        for seed in (0, 1, 2):
            yield MAPPER.map(persona, seed, "t")


# -- the fixtures -----------------------------------------------------------


def test_p01_prompt_is_sheltered_before_sunrise():
    spec = _spec("P01")  # at the foot of the terraced mountain (terraces_foot)
    prompt = BUILDER.build(spec)
    assert "a sheltered landscape at dawn before sunrise" in prompt
    assert "among tall pines, tufts of grass and young pines" in prompt  # its first props
    assert "mostly cloudy sky" in prompt
    assert "seen from the foot of a terraced mountain" in prompt
    # The summit caps the fog, so the horizon still shows.
    assert spec.archetype == "summit"
    assert "clear air" in prompt


def test_an_anxious_persona_in_a_hollow_is_misty():
    spec = MAPPER.map(_persona(openness=0.45, neuroticism=0.9), 1, "t")  # hollow_pond
    assert spec.archetype == "hollow"
    prompt = BUILDER.build(spec)
    assert "heavy drifting mist" in prompt
    assert "seen from a hollow among low hills, the shore of a lake close by" in prompt


def test_p02_prompt_is_open_bright_and_clear():
    prompt = BUILDER.build(_spec("P02"))
    assert "open" in prompt
    assert "midday sun" in prompt
    assert "a few clouds" in prompt
    assert "clear air" in prompt
    assert "bright cool light" in prompt


@pytest.mark.parametrize(
    ("alias", "phrases"),
    [
        ("P04", ["at night", "cloudless sky", "a sky crowded with stars", "the Milky Way",
                 "dim starlight"]),
        ("P06", ["at night", "overcast sky", "soft moonlight"]),
        ("P08", ["at sunrise", "scattered clouds", "warm dim light"]),
        ("P09", ["at dusk after sunset", "a few clouds"]),
        ("P10", ["at night", "scattered clouds", "a half moon", "a few faint stars",
                 "soft moonlight"]),
        ("P11", ["at sunset", "warm dim light"]),
        ("P12", ["in the morning,", "mostly cloudy sky", "soft grey light"]),
        ("P13", ["in the afternoon,", "mostly cloudy sky"]),
    ],
)
def test_the_fixtures_read_as_their_hour_and_weather(alias, phrases):
    prompt = BUILDER.build(_spec(alias))
    for phrase in phrases:
        assert phrase in prompt


@pytest.mark.parametrize(
    ("kind", "amount", "phrase"),
    [("rain", 0.83, "steady rain"), ("rain", 0.3, "light rain"),
     ("snow", 0.83, "steady snowfall"), ("snow", 0.3, "light snowfall")],
)
def test_what_falls_is_named_by_kind_and_amount(kind, amount, phrase):
    spec = _spec("P06")  # overcast, at night
    spec = spec.model_copy(update={"weather": Weather(precipitation=amount, kind=kind)})
    assert phrase in BUILDER.build(spec)


def test_a_moonless_night_names_no_moon():
    prompt = BUILDER.build(_spec("P04"))
    assert not any(phrase in prompt for _, _, phrase in MOON_BANDS if phrase)


@pytest.mark.parametrize("spec", list(_all_specs()))
def test_the_sky_named_is_one_that_can_be_seen(spec):
    prompt = BUILDER.build(spec)
    heavens = [phrase for bands in (MOON_BANDS, STAR_BANDS, MILKY_WAY_BANDS)
               for _, _, phrase in bands if phrase and phrase in prompt]
    falling = [phrase for bands in (RAIN_BANDS, SNOW_BANDS)
               for _, _, phrase in bands if phrase and phrase in prompt]
    closed = spec.procedural_sky.clouds.coverage > CLOSED_ABOVE

    if spec.lighting.time_of_day != "night" or closed:
        assert not heavens  # no moon or stars by day, or behind the cloud
    if closed:
        assert " sun" not in prompt.replace("sunrise", "").replace("sunset", "")
    else:
        assert not falling  # nothing falls from an open sky
    assert ("at night" in prompt) == (spec.lighting.time_of_day == "night")


# -- the view agrees with the land ------------------------------------------


def test_a_summit_is_seen_from_the_mountain_top():
    for seed in range(5):
        spec = MAPPER.map(_persona(openness=0.95, neuroticism=0.1), seed, "t")  # summit_top
        assert spec.archetype == "summit"
        assert "seen from a high mountain top" in BUILDER.build(spec)


@pytest.mark.parametrize(
    ("place_id", "phrase"),
    [
        ("rocky_shore_top", "seen from high above a rocky shore, "
                            "the open sea ahead, toward the setting sun"),
        ("cliff_over_sea", "seen from a ridge overlooking the land below, "
                           "the sea far below, toward the setting sun"),
        ("beach_cove", "seen from a beach, the open sea ahead, toward the setting sun"),
    ],
)
def test_the_sea_at_sunset_lies_toward_the_sun(place_id, phrase):
    persona = _persona(extraversion=0.42, conscientiousness=0.1, neuroticism=0.1)
    spec = RuleSpecMapper(allow_candidates=True, place_id=place_id).map(persona, 1, "t")
    assert spec.lighting.time_of_day == "sunset"
    assert phrase in BUILDER.build(spec)


def test_every_place_is_named_as_it_is_built():
    """The view and the water follow the place's axes, not its archetype."""
    for place in PLACES.places:
        spec = RuleSpecMapper(allow_candidates=True, place_id=place.id).map(_persona(), 1, "t")
        prompt = BUILDER.build(spec)
        if place.landform != "plain":
            assert VIEW_PHRASES[(place.landform, place.position)] in prompt
        if place.water != "none":
            assert WATER_PLACE_PHRASES[(place.water, place.water_relation)] in prompt


@pytest.mark.parametrize("spec", list(_all_specs()))
def test_the_plants_named_are_the_ones_the_spec_places(spec):
    """No birches over a pine grove: the words come from the place's own list."""
    prompt = BUILDER.build(spec)
    placed = {p.asset for p in spec.props} | {spec.focal.asset, spec.framing.asset}
    for asset, words in PLANT_WORDS.items():
        if words in prompt:
            assert asset in placed
    assert "birch" not in prompt
    assert plant_phrase(spec).count(",") + 1 <= PLANTS_NAMED


def test_a_snowy_place_is_named_snowy():
    spec = RuleSpecMapper(place_id="snow_hollow").map(_persona(), 1, "t")
    assert "a quiet snowy landscape" in BUILDER.build(spec)


def test_every_position_unity_builds_has_a_view():
    built = {(landform, position) for landform, row in SPAWN_TABLE.items()
             for position, fix in row.items() if fix is None}
    assert set(VIEW_PHRASES) == built


@pytest.mark.parametrize("spec", list(_all_specs()))
def test_the_prompt_names_only_the_water_the_land_has(spec):
    """No sea in a forest panorama, no lake where Unity lays a river."""
    prompt = BUILDER.build(spec)
    kind = spec.terrain.water.kind
    assert bool(re.search(r"\bsea\b", prompt)) == (kind == "sea")
    assert bool(re.search(r"\blake", prompt)) == (kind == "lake")
    assert bool(re.search(r"\briver", prompt)) == (kind == "river")


# -- shape of every prompt --------------------------------------------------


@pytest.mark.parametrize("spec", list(_all_specs()))
def test_every_prompt_is_well_formed(spec):
    prompt = BUILDER.build(spec)
    assert prompt.startswith("equirectangular 360 view, ")
    assert ",," not in prompt.replace(" ", "")
    assert not prompt.rstrip().endswith(",")
    assert "  " not in prompt
    assert prompt.endswith("photorealistic, serene")


def test_the_same_spec_gives_the_same_prompt():
    assert BUILDER.build(_spec("P01")) == BUILDER.build(_spec("P01"))


def test_slots_without_a_phrase_are_left_out():
    spec = _spec("P02")
    still = spec.model_copy(update={"motion": spec.motion.model_copy(update={"wind_strength": 0.1})})
    dry = still.model_copy(update={"terrain": still.terrain.model_copy(
        update={"water": still.terrain.water.model_copy(update={"present": False})})})
    prompt = BUILDER.build(dry)
    assert "breeze" not in prompt
    assert "water" not in prompt


# -- storing it on the spec -------------------------------------------------


def test_attach_prompt_fills_the_spec_and_leaves_the_original_alone():
    spec = _spec("P02")
    with_prompt = attach_prompt(spec)
    assert with_prompt.skybox.prompt == BUILDER.build(spec)
    assert spec.skybox.prompt is None
    SceneSpec.model_validate(json.loads(with_prompt.model_dump_json()))


def test_default_negative_prompt():
    assert DEFAULT_NEGATIVE_PROMPT == "people, text, watermark, buildings, distorted horizon"
