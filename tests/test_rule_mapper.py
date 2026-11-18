"""RuleSpecMapper: deterministic, persona-sensitive, and always a valid spec."""

from __future__ import annotations

import colorsys
import itertools
import json
import math
import re
from pathlib import Path
from typing import get_args

import pytest

from scene_orchestrator.adapters import ChatbotV2Adapter
from scene_orchestrator.mapping import RuleSpecMapper, SpecMapper, rules_markdown
from scene_orchestrator.mapping.rule_mapper import (
    BIOME_GROUND_COLOR,
    BIOME_SKY_HUE,
    BIOME_SOUNDS,
    CATALOGUE_V1,
    DAY_SUN_OFFSET_DEG,
    FOG_CAP,
    FOG_HORIZON_DEPTH,
    FRAMING_OFFSET_DEG,
    HORIZON_ACCENT,
    HORIZON_VALUE,
    LINEAR_RULES,
    MOON_AZIMUTH_SPREAD_DEG,
    MOON_COLOR_TEMPERATURE_K,
    PLACES,
    RESERVED_PROPS,
    SEASON_GRASS,
    SKY_TINT_SATURATION,
    SKY_TINT_VALUE,
    SUNRISE_GLOW,
    SUNSET_GLOW,
    _order,
    _procedural_sky,
    map_persona,
)
from scene_orchestrator.models import MAX_SEED, PersonaProfile, SceneSpec, TimeOfDay, Traits
from scene_orchestrator.models.scene_spec import Archetype, Biome
from scene_orchestrator.pipeline import derive_seed

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "personas" / "chatbot"
# Every place in places_v1.yaml is still a candidate.
MAPPER: SpecMapper = RuleSpecMapper(allow_candidates=True)  # also checks it fits the protocol
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def _fixture(alias: str) -> PersonaProfile:
    raw = json.loads((FIXTURES / f"{alias}.json").read_text(encoding="utf-8"))
    return ChatbotV2Adapter().adapt(raw)


def _persona(**traits) -> PersonaProfile:
    values = dict.fromkeys(TRAITS, 0.5)
    values.update(traits)
    return PersonaProfile(participant_id="t", source="stub", traits=Traits(**values))


def _place(persona: PersonaProfile) -> str:
    return MAPPER.place_for(persona).id


def _place_of(spec: SceneSpec):
    """The place a spec was built at: its setting, biome and axes are one place's."""
    terrain = spec.terrain
    key = (spec.archetype, spec.biome, terrain.landform, terrain.position, terrain.water.kind,
           terrain.water_relation, terrain.vegetation, terrain.season)
    (place,) = [p for p in PLACES.places
                if (p.setting, p.biome, p.landform, p.position, p.water, p.water_relation,
                    p.vegetation, p.season) == key]
    return place


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


def test_p01_and_p02_are_at_different_places(p01, p02):
    assert (_place(_fixture("P01")), _place(_fixture("P02"))) == (
        "terraces_foot", "ridge_lake_below"
    )
    assert p01.biome != p02.biome
    assert p01.archetype != p02.archetype


def test_anxious_p01_is_more_enclosed(p01, p02):
    assert p01.spatial.enclosure > p02.spatial.enclosure


def test_anxious_p01_sees_less_far(p01, p02):
    assert p01.spatial.sightline_distance_m < p02.spatial.sightline_distance_m


def test_anxious_p01_is_sheltered_and_p02_looks_out(p01, p02):
    assert (p01.spatial.prospect_refuge_bias, p02.spatial.prospect_refuge_bias) == (
        "refuge", "prospect"
    )


def test_introverted_p01_has_a_lower_sun(p01, p02):
    assert p01.lighting.sun_elevation_deg < p02.lighting.sun_elevation_deg
    assert p01.lighting.sun_elevation_deg == -2.1  # -30 + (60 + 30) * 0.31: just under the horizon
    assert (p01.lighting.time_of_day, p02.lighting.time_of_day) == ("sunrise", "midday")


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
    assert p01.procedural_sky.clouds.coverage == 0.662  # 0.05 + (0.95 - 0.05) * 0.68
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
    assert p01.props[0].distribution == "clustered"  # conscientiousness 0.55


# -- thresholds -------------------------------------------------------------


@pytest.mark.parametrize(
    ("conscientiousness", "distribution"),
    # scattered goes out as poisson: Unity doesn't build it yet
    [(0.2, "poisson"), (0.35, "poisson"), (0.36, "clustered"),
     (0.65, "clustered"), (0.66, "ringed"), (1.0, "ringed")],
)
def test_distribution_follows_conscientiousness(conscientiousness, distribution):
    spec = MAPPER.map(_persona(conscientiousness=conscientiousness), 1, "t")
    assert {p.distribution for p in spec.props} == {distribution}


@pytest.mark.parametrize(
    ("conscientiousness", "order"),
    [(0.0, None), (0.2, None), (0.35, None), (0.5, None), (0.65, None),
     (0.66, 0.029), (0.8, 0.429), (0.825, 0.5), (1.0, 1.0)],
)
def test_order_is_sent_only_with_ringed(conscientiousness, order):
    """Stretched over the ringed band: (C - 0.65) / 0.35, the whole 0..1 Unity reads."""
    spec = MAPPER.map(_persona(conscientiousness=conscientiousness), 1, "t")
    assert {p.order for p in spec.props} == {order}
    assert {p.distribution == "ringed" for p in spec.props} == {order is not None}


def test_order_covers_the_whole_range_inside_the_ringed_band():
    assert _order(0.65) == 0.0
    assert _order(1.0) == 1.0
    orders = [_order(c / 100) for c in range(66, 101)]
    assert orders == sorted(orders)  # more orderly, more exact rings


def test_order_is_never_sent_without_rings():
    for spec in ALL_SPECS:
        for prop in spec.props:
            assert (prop.order is not None) == (prop.distribution == "ringed")


@pytest.mark.parametrize(
    ("extraversion", "conscientiousness", "time_of_day"),
    [
        (0.0, 0.9, "night"),  # the sun 30 degrees under
        (0.26, 0.9, "night"),  # -6.6
        (0.27, 0.9, "sunrise"),  # -5.7: morning types see it come up
        (0.27, 0.5, "sunset"),  # and the others see it go down
        (0.44, 0.1, "sunset"),  # 9.6
        (0.45, 0.1, "afternoon"),  # 10.5
        (0.45, 0.51, "morning"),
        (0.77, 0.51, "morning"),  # 39.3
        (0.78, 0.51, "midday"),  # 40.2: midday has no halves
        (1.0, 0.1, "midday"),  # 60
    ],
)
def test_time_of_day_follows_extraversion_and_conscientiousness(
    extraversion, conscientiousness, time_of_day
):
    persona = _persona(extraversion=extraversion, conscientiousness=conscientiousness)
    assert MAPPER.map(persona, 1, "t").lighting.time_of_day == time_of_day


def test_the_key_light_is_the_sun_and_at_night_the_moon():
    night = MAPPER.map(_persona(extraversion=0.0, neuroticism=0.5), 1, "t").lighting
    dusk = MAPPER.map(_persona(extraversion=0.3), 1, "t").lighting
    noon = MAPPER.map(_persona(extraversion=1.0), 1, "t").lighting

    assert night.color_temperature_k == MOON_COLOR_TEMPERATURE_K  # cool
    assert night.intensity == 0.105  # half a moon: midway between 0.03 and 0.18
    assert dusk.color_temperature_k < 2500 < 5500 < noon.color_temperature_k  # warm, then neutral
    assert night.intensity < dusk.intensity < noon.intensity == 1.0


def test_an_anxious_night_is_never_quite_dark():
    calm = MAPPER.map(_persona(extraversion=0.0, neuroticism=0.0), 1, "t")
    anxious = MAPPER.map(_persona(extraversion=0.0, neuroticism=1.0), 1, "t")
    assert (calm.lighting.moon_phase, anxious.lighting.moon_phase) == (0.0, 1.0)  # new, full
    assert anxious.lighting.intensity > calm.lighting.intensity
    # ...and it is the calm one who sees the stars.
    assert calm.procedural_sky.stars.brightness > anxious.procedural_sky.stars.brightness


def test_the_moon_stands_as_far_from_the_sun_as_its_phase_says():
    def apart(neuroticism: float, seed: int) -> float:
        light = MAPPER.map(_persona(neuroticism=neuroticism), seed, "t").lighting
        return (light.moon_azimuth_deg - light.sun_azimuth_deg) % 360

    for seed in range(30):
        assert min(apart(0.0, seed), 360 - apart(0.0, seed)) <= MOON_AZIMUTH_SPREAD_DEG  # beside it
        assert abs(apart(1.0, seed) - 180) <= MOON_AZIMUTH_SPREAD_DEG  # opposite


def test_openness_fills_the_night_sky():
    plain = MAPPER.map(_persona(openness=0.0), 1, "t").procedural_sky.stars
    rich = MAPPER.map(_persona(openness=1.0), 1, "t").procedural_sky.stars
    assert (plain.density, rich.density) == (0.15, 1.0)
    assert (plain.milky_way, rich.milky_way) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("neuroticism", "precipitation"),
    [(0.0, 0.0), (0.7, 0.0), (0.85, 0.5), (1.0, 1.0)],
)
def test_rain_starts_where_the_cloud_cover_closes(neuroticism, precipitation):
    spec = MAPPER.map(_persona(neuroticism=neuroticism), 1, "t")
    assert spec.weather.precipitation == precipitation


def test_what_falls_is_snow_only_in_a_snowy_biome():
    assert any(s.biome == "snowy_valley" for s in ALL_SPECS)
    for spec in ALL_SPECS:
        assert spec.weather.kind == ("snow" if spec.biome == "snowy_valley" else "rain")


def test_the_low_sun_glows_rose_in_the_morning_and_amber_in_the_evening():
    def glow(**traits) -> tuple[float, float]:
        hue, saturation, value = colorsys.rgb_to_hsv(
            *MAPPER.map(_persona(**traits), 1, "t").procedural_sky.twilight_color
        )
        assert value == 1.0
        return hue * 360, saturation

    (rose, _), (amber, vivid) = glow(conscientiousness=0.9), glow(conscientiousness=0.1)
    assert rose == pytest.approx(SUNRISE_GLOW[0], abs=1.0)
    assert amber == pytest.approx(SUNSET_GLOW[0], abs=1.0)
    _, muted = glow(conscientiousness=0.1, neuroticism=1.0)
    assert muted < glow(conscientiousness=0.1, neuroticism=0.0)[1]  # like the rest of the sky
    assert vivid > muted


def test_the_fixtures_between_them_show_every_kind_of_sky():
    personas = {
        path.stem: _fixture(path.stem)
        for path in sorted(FIXTURES.glob("*.json"))
        if path.stem != "P03"  # incomplete on purpose
    }
    specs = [MAPPER.map(p, derive_seed(alias), alias) for alias, p in personas.items()]
    assert {s.lighting.time_of_day for s in specs} == set(get_args(TimeOfDay))
    # every shelter band, from the open crest to the sheltered hollow
    assert {MAPPER.place_for(p).shelter_level for p in personas.values()} == {1, 2, 3, 4, 5}
    falling = {(s.weather.kind, s.lighting.time_of_day) for s in specs if s.weather.precipitation}
    assert {kind for kind, _ in falling} == {"rain", "snow"}
    assert "night" in {time for _, time in falling}  # a rainy night among them
    nights = [s for s in specs if s.lighting.time_of_day == "night"]
    assert min(s.lighting.moon_phase for s in nights) < 0.1  # moonless
    assert max(s.lighting.moon_phase for s in nights) > 0.9  # and a full moon
    assert min(s.procedural_sky.clouds.coverage for s in specs) < 0.15  # clear
    assert max(s.procedural_sky.clouds.coverage for s in specs) > 0.85  # overcast


def test_agreeableness_is_unmapped():
    a = MAPPER.map(_persona(agreeableness=0.0), 7, "t")
    b = MAPPER.map(_persona(agreeableness=1.0), 7, "t")
    assert a == b


def test_the_biome_sets_the_hue_and_neuroticism_the_saturation():
    # Within one biome: neuroticism also picks the place, and so the biome.
    for neuroticism, scale in ((0.0, 1.0), (1.0, 0.45)):
        traits = _persona(neuroticism=neuroticism).traits
        for biome in BIOME_SOUNDS:
            sky = _procedural_sky(biome, traits, (0.0, 0.0))
            hue, saturation, value = colorsys.rgb_to_hsv(*sky.sky_tint)
            assert hue * 360 == pytest.approx(BIOME_SKY_HUE[biome], abs=1.0)
            assert saturation == pytest.approx(SKY_TINT_SATURATION * scale, abs=0.005)
            assert value == pytest.approx(SKY_TINT_VALUE, abs=0.005)


def test_the_horizon_is_a_paler_sky_until_openness_adds_a_second_colour():
    plain = MAPPER.map(_persona(openness=0.0), 3, "t").procedural_sky
    hue, saturation, value = colorsys.rgb_to_hsv(*plain.horizon_color)
    tint_hue, tint_saturation, _ = colorsys.rgb_to_hsv(*plain.sky_tint)
    assert hue == pytest.approx(tint_hue, abs=0.01)  # the same colour...
    assert saturation < tint_saturation  # ...paler...
    assert value == pytest.approx(HORIZON_VALUE, abs=0.005)  # ...and brighter

    # Compared within one biome: openness also picks the place, and so the hue.
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
    assert set(BIOME_SKY_HUE) == set(BIOME_GROUND_COLOR) == set(BIOME_SOUNDS) == set(get_args(Biome))


def test_more_enclosure_means_more_props_closer_in():
    open_ = MAPPER.map(_persona(neuroticism=0.0), 3, "t")
    closed = MAPPER.map(_persona(neuroticism=1.0), 3, "t")
    # Both places' first props weigh 1, so the count is enclosure's alone.
    assert _place_of(open_).props[0][1] == _place_of(closed).props[0][1] == 1.0
    assert closed.props[0].count > open_.props[0].count
    assert closed.props[0].exclude_radius < open_.props[0].exclude_radius


# -- the place (0.4) --------------------------------------------------------


def _all_specs() -> list[SceneSpec]:
    """Specs across the trait space and several seeds, the largest one included."""
    return [
        MAPPER.map(_persona(**dict(zip(TRAITS, values))), seed, "t")
        for values in itertools.product((0.0, 0.3, 0.5, 0.8, 1.0), repeat=5)
        if values[3] == 0.5  # agreeableness changes nothing
        for seed in (0, 1, 2, MAX_SEED)
    ]


ALL_SPECS = _all_specs()


def test_the_seed_draws_are_pinned():
    """A participant's sun, wind, clouds, moon and sides come from these
    draws only; changing their order changes every scene."""
    spec = MAPPER.map(_fixture("P01"), derive_seed("P01"), "P01")
    assert _place(_fixture("P01")) == "terraces_foot"
    assert [p.asset for p in spec.props] == ["pine_large", "grass_tuft", "pine_medium", "boulder"]
    assert spec.lighting.sun_azimuth_deg == 344.604
    assert spec.motion.wind_direction_deg == 330.891
    assert spec.procedural_sky.clouds.offset == (69.479, 87.185)
    assert (spec.lighting.moon_azimuth_deg, spec.lighting.moon_elevation_deg) == (104.204, 47.386)
    assert spec.terrain.spawn.facing_deg == 344.604  # sunrise: the sun is faced
    assert spec.framing.azimuth_deg == 59.032


def test_the_place_sets_the_archetype_biome_and_every_axis():
    for spec, persona in _SPECS_WITH_PERSONAS:
        place = MAPPER.place_for(persona)
        terrain = spec.terrain
        assert (spec.archetype, spec.biome) == (place.setting, place.biome)
        assert (terrain.landform, terrain.position, terrain.water_relation,
                terrain.vegetation, terrain.season) == (
            place.landform, place.position, place.water_relation, place.vegetation, place.season
        )
        assert terrain.water.kind == place.water
        assert terrain.water.present == (place.water != "none")


_SPECS_WITH_PERSONAS = [
    (MAPPER.map(persona, seed, "t"), persona)
    for persona in (_persona(neuroticism=n, openness=o)
                    for n in (0.0, 0.25, 0.5, 0.75, 1.0) for o in (0.0, 0.33, 0.66, 1.0))
    for seed in (0, 7)
]


def test_the_mapper_never_sends_a_terrain_number():
    """The place says which landscape; its numbers stay the setting's, tuned by hand."""
    for spec in ALL_SPECS:
        terrain, water = spec.terrain, spec.terrain.water
        assert (terrain.amplitude_m, terrain.feature_scale_m, terrain.roughness) == (None, None, None)
        assert (terrain.macro_shape, terrain.ground_material) == (None, None)
        assert (water.distance_m, water.azimuth_deg) == (None, None)
        assert spec.spatial.silhouette_distance_m is None


def test_position_is_sent_and_placement_is_not():
    """Unity reads the facing with the position, so the old placement name stays out."""
    for spec in ALL_SPECS:
        assert spec.terrain.position is not None
        assert spec.terrain.spawn.placement is None


@pytest.mark.parametrize("alias", ["P01", "P02", "P06", "P09", "P12"])
def test_the_seed_never_decides_what_the_persona_should(alias):
    """Same persona, other seeds: the same place, biome, prop kinds and axes.
    Only the azimuths and offsets move."""
    persona = _fixture(alias)
    specs = [MAPPER.map(persona, seed, alias) for seed in (0, 1, 2, 99, 12345, MAX_SEED)]

    def decided(spec: SceneSpec):
        terrain = spec.terrain.model_copy(update={"spawn": None})
        return (spec.archetype, spec.biome, terrain, [p.model_dump() for p in spec.props],
                spec.focal.asset, spec.framing.asset, spec.spatial, spec.lighting.fog_density,
                spec.weather, spec.audio)

    first = decided(specs[0])
    for spec in specs[1:]:
        assert decided(spec) == first
    # ...and what does move, does.
    assert len({s.lighting.sun_azimuth_deg for s in specs}) == len(specs)
    assert len({s.motion.wind_direction_deg for s in specs}) == len(specs)
    assert len({s.procedural_sky.clouds.offset for s in specs}) == len(specs)
    assert len({s.lighting.moon_elevation_deg for s in specs}) == len(specs)
    assert len({s.framing.azimuth_deg for s in specs}) > 1


def test_prop_kinds_are_the_front_of_the_places_list():
    for openness, n in ((0.0, 2), (0.16, 2), (0.17, 3), (0.5, 4), (0.83, 4), (0.84, 5), (1.0, 5)):
        persona = _persona(openness=openness)
        spec = MAPPER.map(persona, 1, "t")
        place = MAPPER.place_for(persona)
        assert [p.asset for p in spec.props] == [a for a, _ in place.props[:n]]


@pytest.mark.parametrize(
    ("neuroticism", "bias"),
    # the band is round half up(1 + 4n): <= 2 prospect, 3 balanced, >= 4 refuge
    [(0.0, "prospect"), (0.3, "prospect"), (0.374, "prospect"), (0.375, "balanced"),
     (0.6, "balanced"), (0.624, "balanced"), (0.625, "refuge"), (1.0, "refuge")],
)
def test_prospect_refuge_follows_the_places_shelter(neuroticism, bias):
    persona = _persona(neuroticism=neuroticism)
    level = MAPPER.place_for(persona).shelter_level
    spec = MAPPER.map(persona, 1, "t")
    assert spec.spatial.prospect_refuge_bias == bias
    assert bias == ("refuge" if level >= 4 else "prospect" if level <= 2 else "balanced")


def test_a_calm_open_persona_sits_on_the_summit():
    for seed in range(10):
        spec = MAPPER.map(_persona(openness=0.95, neuroticism=0.1), seed, "t")
        assert (spec.archetype, spec.biome) == ("summit", "alpine_basin")
        assert (spec.terrain.landform, spec.terrain.position) == ("terraces", "top")
        assert spec.terrain.water.kind == "none"


def test_every_name_sent_is_in_the_catalogue():
    for spec in ALL_SPECS:
        assert spec.catalogue_version == CATALOGUE_V1.version
        assert spec.archetype in CATALOGUE_V1.archetypes
        assets = {p.asset for p in spec.props} | {spec.focal.asset, spec.framing.asset}
        assert assets <= CATALOGUE_V1.props
        assert spec.terrain.ground_material is None  # the archetype's
        assert {p.distribution for p in spec.props} <= {"poisson", "clustered", "ringed"}


def test_the_catalogue_names_what_the_spec_can_carry():
    assert set(CATALOGUE_V1.archetypes) == set(get_args(Archetype))


def test_mountains_and_cliffs_are_left_to_unity():
    assert not RESERVED_PROPS & CATALOGUE_V1.props
    for spec in ALL_SPECS:
        assets = {p.asset for p in spec.props} | {spec.focal.asset, spec.framing.asset}
        assert not assets & RESERVED_PROPS


def test_the_props_sent_are_the_places_own():
    for spec in ALL_SPECS:
        place = _place_of(spec)
        assert [(p.asset, p.count > 0) for p in spec.props] == [
            (a, True) for a, _ in place.props[:len(spec.props)]
        ]


def test_the_water_is_heard_only_where_it_is():
    for spec in ALL_SPECS:
        layers = {layer.layer for layer in spec.audio}
        kind = spec.terrain.water.kind
        assert ("lake_lapping" in layers) == (kind == "lake")
        assert bool(layers & {"waves_gentle", "waves_rocky"}) == (kind == "sea")
        assert ("stream" in layers) == (kind == "river")


@pytest.mark.parametrize(
    ("extraversion", "time_of_day"),
    [(0.27, "sunrise"), (0.44, "sunset"), (0.3, "sunset")],
)
def test_a_low_sun_is_faced(extraversion, time_of_day):
    for seed in range(20):
        spec = MAPPER.map(_persona(extraversion=extraversion,
                                   conscientiousness=0.9 if time_of_day == "sunrise" else 0.1),
                          seed, "t")
        assert spec.lighting.time_of_day == time_of_day
        assert spec.terrain.spawn.facing_deg == spec.lighting.sun_azimuth_deg


def test_at_night_the_moon_is_faced():
    for seed in range(20):
        spec = MAPPER.map(_persona(extraversion=0.0), seed, "t")
        assert spec.lighting.time_of_day == "night"
        assert spec.terrain.spawn.facing_deg == spec.lighting.moon_azimuth_deg


def _apart(a: float, b: float) -> float:
    """The angle between two azimuths, 0..180."""
    return min((a - b) % 360, (b - a) % 360)


def test_by_day_the_sun_stands_to_one_side():
    sides = set()
    for seed in range(40):
        spec = MAPPER.map(_persona(extraversion=0.8), seed, "t")
        facing, sun = spec.terrain.spawn.facing_deg, spec.lighting.sun_azimuth_deg
        low, high = DAY_SUN_OFFSET_DEG
        assert low - 0.001 <= _apart(facing, sun) <= high + 0.001
        sides.add((facing - sun) % 360 < 180)
    assert sides == {True, False}  # left and right, from the seed


def test_the_focal_stands_where_the_participant_faces():
    for spec in ALL_SPECS:
        assert spec.focal.azimuth_deg == spec.terrain.spawn.facing_deg
        assert 25 <= spec.focal.distance_m <= 40
    calm = MAPPER.map(_persona(neuroticism=0.0), 1, "t").focal
    anxious = MAPPER.map(_persona(neuroticism=1.0), 1, "t").focal
    assert (calm.distance_m, anxious.distance_m) == (40, 25)


def test_the_focal_and_framing_are_the_places():
    for spec in ALL_SPECS:
        place = _place_of(spec)
        assert (spec.focal.asset, spec.framing.asset) == (place.focal, place.framing)


def test_the_framing_prop_is_close_and_at_the_edge_of_view():
    sides = set()
    for spec in ALL_SPECS:
        framing, facing = spec.framing, spec.terrain.spawn.facing_deg
        assert 3 <= framing.distance_m <= 5  # what Unity keeps it to
        low, high = FRAMING_OFFSET_DEG
        assert low - 0.001 <= _apart(framing.azimuth_deg, facing) <= high + 0.001
        sides.add((framing.azimuth_deg - facing) % 360 < 180)
    assert sides == {True, False}


# -- fog: a cap where the mountains should show, never under the floor ------


def test_the_fog_is_capped_only_where_mountains_stand_and_the_hollow_is_not():
    assert FOG_CAP == {"overlook": 0.0032, "summit": 0.0026, "meadow": 0.0039, "beach": 0.0042}
    for name, cap in FOG_CAP.items():
        setting = CATALOGUE_V1.archetypes[name]
        assert setting.has_mountains
        assert cap == pytest.approx(FOG_HORIZON_DEPTH / setting.horizon_m, abs=5e-5)
        visible = math.exp(-(cap * setting.horizon_m) ** 2)
        assert 0.28 <= visible <= 0.32  # about a third of the mountains still shows
    # The hollow's mountains are a must, but there the fog is the shelter:
    # capped, it would be 1.1 / 200.
    assert CATALOGUE_V1.archetypes["hollow"].has_mountains
    assert "hollow" not in FOG_CAP
    # The forest has no mountains, so nothing to keep in view.
    assert not CATALOGUE_V1.archetypes["forest"].has_mountains
    assert "forest" not in FOG_CAP


def test_fog_follows_its_cap_and_floor_in_every_spec():
    for spec in ALL_SPECS:
        setting = CATALOGUE_V1.archetypes[spec.archetype]
        fog = spec.lighting.fog_density
        assert fog >= setting.fog_floor
        if spec.archetype in FOG_CAP:
            assert fog <= FOG_CAP[spec.archetype]
    hollows = [s for s in ALL_SPECS if s.archetype == "hollow"]
    assert max(s.lighting.fog_density for s in hollows) == 0.045  # the rule's top, uncapped


@pytest.mark.parametrize(
    ("archetype", "neuroticism", "fog"),
    [
        ("forest", 0.0, 0.02),  # the rule says 0.005: Unity's floor
        ("forest", 0.5, 0.025),  # above the floor: the rule's own
        ("forest", 1.0, 0.045),  # no mountains, so no cap
        ("hollow", 0.0, 0.005),
        ("hollow", 1.0, 0.045),  # uncapped
        ("meadow", 1.0, 0.0039),
        ("beach", 1.0, 0.0042),
        ("overlook", 1.0, 0.0032),
        ("summit", 1.0, 0.0026),
        ("summit", 0.0, 0.0026),  # 0.005 by the rule, capped
    ],
)
def test_the_fog_at_each_setting(archetype, neuroticism, fog):
    place = next(p for p in PLACES.places if p.setting == archetype)
    spec = RuleSpecMapper(allow_candidates=True, place_id=place.id).map(
        _persona(neuroticism=neuroticism), 1, "t"
    )
    assert spec.lighting.fog_density == fog


# -- every spec is valid ----------------------------------------------------


@pytest.mark.parametrize("values", list(itertools.product((0.0, 0.5, 1.0), repeat=5)))
def test_every_produced_spec_validates(values):
    traits = dict(zip(TRAITS, values))
    for seed in (0, 1, 2**31, MAX_SEED):
        spec = MAPPER.map(_persona(**traits), seed, "t")
        SceneSpec.model_validate(json.loads(spec.model_dump_json()))
        assert {p.asset for p in spec.props} <= {a for a, _ in _place_of(spec).props}


def test_every_place_gives_a_valid_spec():
    for place in PLACES.places:
        mapper = RuleSpecMapper(allow_candidates=True, place_id=place.id)
        for values in itertools.product((0.0, 1.0), repeat=5):
            spec = mapper.map(_persona(**dict(zip(TRAITS, values))), 3, "t")
            assert SceneSpec.model_validate(json.loads(spec.model_dump_json())) == spec


def test_fixtures_produce_valid_specs(p01, p02):
    for spec in (p01, p02):
        assert SceneSpec.model_validate(json.loads(spec.model_dump_json())) == spec


# -- the grass, by season ---------------------------------------------------


def test_a_winter_place_sends_white_grass_and_no_flowers():
    spec = RuleSpecMapper(allow_candidates=True, place_id="snow_hollow").map(_persona(), 3, "t")
    grass = spec.terrain.grass
    assert spec.terrain.season == "winter"
    assert all(channel > 0.9 for channel in grass.healthy_color)
    assert grass.no_flowers is True
    assert grass.flower_density is None  # never 0: Unity would read that as "the season's"


def test_a_summer_place_and_a_place_without_a_season_send_no_grass():
    place = PLACES.get("forest_lake")
    assert place.season is None
    no_season = map_persona(_persona(), 3, "t", place=place)
    summer = map_persona(_persona(), 3, "t", place=place.model_copy(update={"season": "summer"}))
    assert (no_season.terrain.season, no_season.terrain.grass) == (None, None)
    assert (summer.terrain.season, summer.terrain.grass) == ("summer", None)


@pytest.mark.parametrize("season", ["spring", "summer", "autumn", "winter"])
def test_each_season_sends_its_row_of_the_table(season):
    place = PLACES.get("forest_lake").model_copy(update={"season": season})
    spec = map_persona(_persona(), 3, "t", place=place)
    assert spec.terrain.grass == SEASON_GRASS[season]
    assert SceneSpec.model_validate(json.loads(spec.model_dump_json())) == spec


def test_the_season_table_is_unitys():
    """A copy of GrassLook.OfSeason in the Unity project; the two must agree."""
    spring, autumn, winter = (SEASON_GRASS[s] for s in ("spring", "autumn", "winter"))
    assert SEASON_GRASS["summer"] is None
    assert spring.model_dump(exclude_none=True) == {
        "healthy_color": (0.55, 0.95, 0.40), "dry_color": (0.45, 0.80, 0.35),
        "height_scale": 0.85, "grass_density": 0.9, "flower_density": 1.8,
    }
    assert autumn.model_dump(exclude_none=True) == {
        "healthy_color": (0.85, 0.68, 0.32), "dry_color": (0.62, 0.42, 0.22),
        "grass_density": 0.8, "flower_density": 0.15,
    }
    assert winter.model_dump(exclude_none=True) == {
        "healthy_color": (0.93, 0.96, 1.00), "dry_color": (0.78, 0.82, 0.86),
        "width_scale": 0.9, "height_scale": 0.6, "grass_density": 0.45, "no_flowers": True,
    }


# -- against the JSON Schema Unity validates with ---------------------------

SCHEMA = json.loads(
    (Path(__file__).resolve().parent.parent / "schema" / "scene_spec.schema.json")
    .read_text(encoding="utf-8")
)
_JSON_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
    "integer": int, "number": (int, float),
}


def _schema_errors(value, node, where="$") -> list[str]:
    """Where `value` breaks `node` of the committed schema. Only the keywords
    that file uses; jsonschema isn't a dependency."""
    if "$ref" in node:
        node = SCHEMA["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    if "anyOf" in node:
        options = [_schema_errors(value, option, where) for option in node["anyOf"]]
        return [] if any(not errors for errors in options) else min(options, key=len)
    kind = node.get("type")
    if kind is not None:
        if isinstance(value, bool) and kind in ("integer", "number"):
            return [f"{where}: a boolean, not a {kind}"]
        if not isinstance(value, _JSON_TYPES[kind]):
            return [f"{where}: not a {kind}"]
    errors = []
    if "const" in node and value != node["const"]:
        errors.append(f"{where}: not {node['const']!r}")
    if "enum" in node and value not in node["enum"]:
        errors.append(f"{where}: {value!r} not one of {node['enum']}")
    if "minimum" in node and value < node["minimum"]:
        errors.append(f"{where}: below {node['minimum']}")
    if "maximum" in node and value > node["maximum"]:
        errors.append(f"{where}: above {node['maximum']}")
    if "exclusiveMinimum" in node and value <= node["exclusiveMinimum"]:
        errors.append(f"{where}: not above {node['exclusiveMinimum']}")
    if "pattern" in node and not re.search(node["pattern"], value):
        errors.append(f"{where}: doesn't match {node['pattern']}")
    if isinstance(value, list):
        if not node.get("minItems", 0) <= len(value) <= node.get("maxItems", len(value)):
            errors.append(f"{where}: {len(value)} items")
        prefix = node.get("prefixItems", [])
        for i, item in enumerate(value):
            item_node = prefix[i] if i < len(prefix) else node.get("items")
            if item_node is not None:
                errors += _schema_errors(item, item_node, f"{where}[{i}]")
    if isinstance(value, dict):
        properties = node.get("properties", {})
        errors += [f"{where}: no {name}" for name in node.get("required", []) if name not in value]
        for name, item in value.items():
            if name in properties:
                errors += _schema_errors(item, properties[name], f"{where}.{name}")
            elif node.get("additionalProperties") is False:
                errors.append(f"{where}: unknown field {name}")
    return errors


def test_the_schema_check_catches_bad_grass():
    """So the next test can't pass by checking nothing."""
    spec = json.loads(
        RuleSpecMapper(allow_candidates=True, place_id="snow_hollow")
        .map(_persona(), 3, "t").model_dump_json()
    )
    assert _schema_errors(spec, SCHEMA) == []
    for field, value in (("healthy_color", [1.0, 1.0]), ("dry_color", [0.5, 1.5, 0.5]),
                         ("width_scale", 0), ("flower_density", -1), ("blade_length", 1)):
        bad = json.loads(json.dumps(spec))
        bad["terrain"]["grass"][field] = value
        assert _schema_errors(bad, SCHEMA), field


def test_every_place_and_season_passes_the_schema():
    for place in PLACES.places:
        for season in (place.season, "spring", "summer", "autumn", "winter"):
            spec = map_persona(_persona(), 3, "t", place=place.model_copy(update={"season": season}))
            data = json.loads(spec.model_dump_json())
            assert _schema_errors(data, SCHEMA) == [], (place.id, season)
            assert SceneSpec.model_validate(data) == spec


# -- for the thesis ---------------------------------------------------------


def test_rules_print_as_a_table():
    table = rules_markdown()
    assert "| `spatial.enclosure` | neuroticism | 0.15 | 0.8 |" in table
    assert "agreeableness" not in table  # unmapped, so absent


def test_the_places_and_their_rule_print_too():
    tables = rules_markdown()
    assert "### Places (places_v1.yaml)" in tables
    assert ("| forest_lake | forest | plain | mid | lake beside | dense | (setting's) | "
            "forest_clearing | 5 | 0.2 | candidate |") in tables
    assert "| snow_hollow | hollow | bowl | bottom | none | moderate | winter |" in tables
    assert "### The rule that picks a place" in tables
    assert "rounded half up (2.5 → 3, 3.5 → 4)" in tables
    # plain 21, bowl 63, ridge 90, peak 90, shore 36, terraces 9, valley 60
    assert "Unity builds 369 combinations" in tables
    assert "uses 15 of them, in 15 places" in tables
    assert "| terraces | top | none | 3 | summit_top (no water, sparse) |" in tables


def test_the_0_4_settings_print_too():
    tables = rules_markdown()
    assert ("| summit | terraces | top | none | sparse | alpine_rock | 420 | 360° | 0 | 0.0026 |"
            in tables)
    assert ("| hollow | bowl | bottom | none | dense | moss_rock_mix | 200 | 360° | 0 | none |"
            in tables)
    assert ("| forest | plain | mid | lake beside | dense | moss_rock_mix | 90 | none | 0.02 | none |"
            in tables)
    assert "| > 0.65 | ringed | ringed | (C − 0.65) / 0.35: loose rings → exact rings |" in tables
    assert "| ≤ 0.35 | scattered | poisson | not sent |" in tables
    assert ("| forest_lake | pine_large | pine_large | pine_large 1, bush 0.8, pine_medium 0.3, "
            "rock_small 0.2, boulder 0.15 |") in tables
    assert "| lakeside | wind_reeds 0.2 | birdsong_wetland |" in tables
    assert "| `focal.distance_m` | neuroticism | 40 | 25 |" in tables


def test_the_grass_table_prints_too():
    tables = rules_markdown()
    assert "### Grass by season (`terrain.grass`)" in tables
    assert "| summer | not sent: FlipGameDev's billboards as they are |" in tables
    assert ("| winter | (0.93, 0.96, 1.00) | (0.78, 0.82, 0.86) | ×0.9 / ×0.6 | ×0.45 | none | – |"
            in tables)
    assert "| spring | (0.55, 0.95, 0.40) | (0.45, 0.80, 0.35) | ×1 / ×0.85 | ×0.9 | ×1.8 | – |" in tables
