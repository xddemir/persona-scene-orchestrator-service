"""RuleSpecMapper: deterministic, persona-sensitive, and always a valid spec."""

from __future__ import annotations

import colorsys
import itertools
import json
from pathlib import Path
from typing import get_args

import pytest

from scene_orchestrator.adapters import ChatbotV2Adapter
from scene_orchestrator.mapping import RuleSpecMapper, SpecMapper, rules_markdown
from scene_orchestrator.mapping.rule_mapper import (
    BIOME_GROUND_COLOR,
    BIOME_SKY_HUE,
    BIOME_TIERS,
    BIOMES,
    HORIZON_ACCENT,
    HORIZON_VALUE,
    LINEAR_RULES,
    SKY_TINT_SATURATION,
    SKY_TINT_VALUE,
    _procedural_sky,
)
from scene_orchestrator.models import PersonaProfile, SceneSpec, Traits
from scene_orchestrator.models.scene_spec import Biome

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "personas" / "chatbot"
MAPPER: SpecMapper = RuleSpecMapper()  # also checks it fits the protocol


def _fixture(alias: str) -> PersonaProfile:
    raw = json.loads((FIXTURES / f"{alias}.json").read_text(encoding="utf-8"))
    return ChatbotV2Adapter().adapt(raw)


def _persona(**traits) -> PersonaProfile:
    values = dict.fromkeys(
        ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"), 0.5
    )
    values.update(traits)
    return PersonaProfile(participant_id="t", source="stub", traits=Traits(**values))


# -- determinism ------------------------------------------------------------


@pytest.mark.parametrize("alias", ["P01", "P02"])
def test_same_persona_and_seed_give_a_byte_identical_spec(alias):
    first = MAPPER.map(_fixture(alias), 1234, alias).model_dump_json()
    again = MAPPER.map(_fixture(alias), 1234, alias).model_dump_json()
    assert first.encode() == again.encode()


def test_the_seed_varies_placement_but_not_the_persona_rules():
    a = MAPPER.map(_fixture("P01"), 1, "p01")
    b = MAPPER.map(_fixture("P01"), 2, "p01")
    assert a.lighting.sun_azimuth_deg != b.lighting.sun_azimuth_deg
    assert a.spatial == b.spatial  # trait-driven, so seed-independent
    assert a.lighting.fog_density == b.lighting.fog_density


# -- P01 (closed, anxious) vs P02 (open, outgoing) --------------------------


@pytest.fixture(scope="module")
def p01() -> SceneSpec:
    return MAPPER.map(_fixture("P01"), 1234, "p01")


@pytest.fixture(scope="module")
def p02() -> SceneSpec:
    return MAPPER.map(_fixture("P02"), 1234, "p02")


def test_p01_and_p02_get_different_biomes(p01, p02):
    assert p01.biome != p02.biome


def test_anxious_p01_is_more_enclosed(p01, p02):
    assert p01.spatial.enclosure > p02.spatial.enclosure


def test_anxious_p01_sees_less_far(p01, p02):
    assert p01.spatial.sightline_distance_m < p02.spatial.sightline_distance_m


def test_anxious_p01_has_more_fog(p01, p02):
    assert p01.lighting.fog_density > p02.lighting.fog_density


def test_introverted_p01_has_a_lower_sun(p01, p02):
    assert p01.lighting.sun_elevation_deg < p02.lighting.sun_elevation_deg


def test_introverted_p01_has_warmer_light(p01, p02):
    assert p01.lighting.color_temperature_k < p02.lighting.color_temperature_k


def test_more_open_p01_gets_more_prop_types(p01, p02):
    assert len(p01.props) == 4  # 2 + round(3 * 0.78)
    assert len(p02.props) == 3  # 2 + round(3 * 0.45)


def _saturation(rgb: tuple[float, float, float]) -> float:
    return colorsys.rgb_to_hsv(*rgb)[1]


def test_anxious_p01_has_a_thicker_atmosphere(p01, p02):
    assert p01.procedural_sky.atmosphere_thickness > p02.procedural_sky.atmosphere_thickness
    assert p01.procedural_sky.atmosphere_thickness == 1.72  # 0.7 + (2.2 - 0.7) * 0.68


def test_introverted_p01_has_a_lower_exposure_and_a_smaller_sun(p01, p02):
    assert p01.procedural_sky.exposure < p02.procedural_sky.exposure
    assert p01.procedural_sky.exposure == 0.948  # 0.7 + (1.5 - 0.7) * 0.31
    assert p01.procedural_sky.sun_size < p02.procedural_sky.sun_size


def test_anxious_p01_has_a_less_saturated_sky(p01, p02):
    assert _saturation(p01.procedural_sky.sky_tint) < _saturation(p02.procedural_sky.sky_tint)


def test_anxious_p01_has_more_and_softer_cloud(p01, p02):
    assert p01.procedural_sky.clouds.coverage > p02.procedural_sky.clouds.coverage
    assert p01.procedural_sky.clouds.coverage == 0.54  # 0.20 + (0.70 - 0.20) * 0.68
    assert p01.procedural_sky.clouds.softness > p02.procedural_sky.clouds.softness


def test_outgoing_p02_has_brighter_clouds_and_a_stronger_sun_glow(p01, p02):
    assert p02.procedural_sky.clouds.brightness > p01.procedural_sky.clouds.brightness
    assert p02.procedural_sky.sun_halo > p01.procedural_sky.sun_halo
    assert p02.procedural_sky.sun_halo == pytest.approx(0.6175, abs=0.001)  # 0.15 + 0.55 * 0.85


def test_more_open_p01_has_more_intricate_clouds(p01, p02):
    assert p01.procedural_sky.clouds.detail > p02.procedural_sky.clouds.detail


def test_more_orderly_p02_has_more_regular_cloud_rows(p01, p02):
    assert p01.procedural_sky.clouds.banding == 0.55  # conscientiousness itself
    assert p02.procedural_sky.clouds.banding == 0.7


def test_the_procedural_sky_is_filled_in_for_a_panorama_scene(p01, p02):
    for spec in (p01, p02):
        assert spec.sky_mode == "panorama"
        assert spec.procedural_sky.ground_color == BIOME_GROUND_COLOR[spec.biome]
        assert spec.skybox.rotation_deg == 0


def test_p01_values_follow_the_table(p01):
    # neuroticism 0.68 -> 0.15 + (0.80 - 0.15) * 0.68
    assert p01.spatial.enclosure == 0.592
    assert p01.spatial.prospect_refuge_bias == "refuge"
    assert p01.props[0].distribution == "clustered"  # conscientiousness 0.55


# -- thresholds -------------------------------------------------------------


@pytest.mark.parametrize(
    ("neuroticism", "bias"),
    [(0.1, "prospect"), (0.39, "prospect"), (0.4, "balanced"), (0.6, "balanced"),
     (0.61, "refuge"), (0.95, "refuge")],
)
def test_prospect_refuge_follows_neuroticism(neuroticism, bias):
    spec = MAPPER.map(_persona(neuroticism=neuroticism), 1, "t")
    assert spec.spatial.prospect_refuge_bias == bias


@pytest.mark.parametrize(
    ("conscientiousness", "distribution"),
    [(0.2, "scattered"), (0.35, "scattered"), (0.36, "clustered"),
     (0.65, "clustered"), (0.66, "ringed")],
)
def test_distribution_follows_conscientiousness(conscientiousness, distribution):
    spec = MAPPER.map(_persona(conscientiousness=conscientiousness), 1, "t")
    assert {p.distribution for p in spec.props} == {distribution}


def test_openness_picks_the_biome_tier():
    seeds = range(50)
    low = {MAPPER.map(_persona(openness=0.1), s, "t").biome for s in seeds}
    high = {MAPPER.map(_persona(openness=0.9), s, "t").biome for s in seeds}
    assert low == set(BIOME_TIERS[0][1])
    assert high == set(BIOME_TIERS[-1][1])


def test_agreeableness_is_unmapped():
    a = MAPPER.map(_persona(agreeableness=0.0), 7, "t")
    b = MAPPER.map(_persona(agreeableness=1.0), 7, "t")
    assert a == b


def test_the_biome_sets_the_hue_and_neuroticism_the_saturation():
    calm = MAPPER.map(_persona(neuroticism=0.0), 3, "t")
    anxious = MAPPER.map(_persona(neuroticism=1.0), 3, "t")
    assert calm.biome == anxious.biome  # openness and seed are the same

    for spec, scale in ((calm, 1.0), (anxious, 0.45)):
        hue, saturation, value = colorsys.rgb_to_hsv(*spec.procedural_sky.sky_tint)
        assert hue * 360 == pytest.approx(BIOME_SKY_HUE[spec.biome], abs=1.0)
        assert saturation == pytest.approx(SKY_TINT_SATURATION * scale, abs=0.005)
        assert value == pytest.approx(SKY_TINT_VALUE, abs=0.005)


def test_the_horizon_is_a_paler_sky_until_openness_adds_a_second_colour():
    plain = MAPPER.map(_persona(openness=0.0), 3, "t").procedural_sky
    hue, saturation, value = colorsys.rgb_to_hsv(*plain.horizon_color)
    tint_hue, tint_saturation, _ = colorsys.rgb_to_hsv(*plain.sky_tint)
    assert hue == pytest.approx(tint_hue, abs=0.01)  # the same colour...
    assert saturation < tint_saturation  # ...paler...
    assert value == pytest.approx(HORIZON_VALUE, abs=0.005)  # ...and brighter

    # Compared within one biome: openness also picks the biome, and so the hue.
    def horizon(openness: float):
        traits = Traits(openness=openness, conscientiousness=0.5, extraversion=0.5,
                        agreeableness=0.5, neuroticism=0.5)
        return _procedural_sky("lakeside", traits, (0.0, 0.0)).horizon_color

    red, _, blue = horizon(0.0)
    open_red, _, open_blue = horizon(1.0)
    assert open_red > red and open_blue < blue  # towards HORIZON_ACCENT, a warm colour
    half_way = tuple(round((a + b) / 2, 3) for a, b in zip(horizon(0.0), HORIZON_ACCENT))
    assert horizon(1.0) == pytest.approx(half_way, abs=0.001)  # accent 0.5 at openness 1


def test_the_seed_moves_the_clouds_and_nothing_else_in_the_sky():
    a = MAPPER.map(_fixture("P01"), 1, "p01").procedural_sky
    b = MAPPER.map(_fixture("P01"), 2, "p01").procedural_sky
    assert a.clouds.offset != b.clouds.offset
    assert a.clouds.model_copy(update={"offset": b.clouds.offset}) == b.clouds
    assert (a.sun_halo, a.exposure, a.atmosphere_thickness) == (
        b.sun_halo, b.exposure, b.atmosphere_thickness
    )


def test_every_trait_but_agreeableness_shapes_the_sky():
    driving = {r.trait for r in LINEAR_RULES if r.parameter.startswith("procedural_sky.")}
    assert driving == {"neuroticism", "extraversion", "openness", "conscientiousness"}


def test_every_biome_has_a_sky_hue_and_a_ground_color():
    assert set(BIOME_SKY_HUE) == set(BIOME_GROUND_COLOR) == set(BIOMES) == set(get_args(Biome))


def test_more_enclosure_means_more_props_closer_in():
    open_ = MAPPER.map(_persona(neuroticism=0.0), 3, "t")
    closed = MAPPER.map(_persona(neuroticism=1.0), 3, "t")
    assert closed.props[0].count > open_.props[0].count
    assert closed.props[0].exclude_radius < open_.props[0].exclude_radius


# -- every spec is valid ----------------------------------------------------


@pytest.mark.parametrize("values", list(itertools.product((0.0, 0.5, 1.0), repeat=5)))
def test_every_produced_spec_validates(values):
    traits = dict(zip(
        ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"),
        values,
    ))
    for seed in (0, 1, 2**31):
        spec = MAPPER.map(_persona(**traits), seed, "t")
        SceneSpec.model_validate(json.loads(spec.model_dump_json()))
        assert {p.asset for p in spec.props} <= {a for a, _ in BIOMES[spec.biome].props}


def test_fixtures_produce_valid_specs(p01, p02):
    for spec in (p01, p02):
        assert SceneSpec.model_validate(json.loads(spec.model_dump_json())) == spec


# -- for the thesis ---------------------------------------------------------


def test_rules_print_as_a_table():
    table = rules_markdown()
    assert "| `spatial.enclosure` | neuroticism | 0.15 | 0.8 |" in table
    assert "agreeableness" not in table  # unmapped, so absent
