"""RuleSpecMapper: deterministic, persona-sensitive, and always a valid spec."""

from __future__ import annotations

import colorsys
import itertools
import json
import math
from pathlib import Path
from typing import get_args

import pytest

from scene_orchestrator.adapters import ChatbotV2Adapter
from scene_orchestrator.mapping import RuleSpecMapper, SpecMapper, rules_markdown
from scene_orchestrator.mapping.rule_mapper import (
    ARCHETYPE_BY_BIOME,
    BIOME_GROUND_COLOR,
    BIOME_SKY_HUE,
    BIOME_TIERS,
    BIOMES,
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
    RESERVED_PROPS,
    SKY_TINT_SATURATION,
    SKY_TINT_VALUE,
    SUNRISE_GLOW,
    SUNSET_GLOW,
    _procedural_sky,
)
from scene_orchestrator.models import MAX_SEED, PersonaProfile, SceneSpec, TimeOfDay, Traits
from scene_orchestrator.models.scene_spec import Archetype, Biome, Placement, WaterKind
from scene_orchestrator.pipeline import derive_seed

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
    # scattered and ringed go out as poisson and clustered, the two Unity builds
    [(0.2, "poisson"), (0.35, "poisson"), (0.36, "clustered"),
     (0.65, "clustered"), (0.66, "clustered")],
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
    specs = [MAPPER.map(_persona(openness=0.9), seed, "t") for seed in range(30)]
    assert {s.biome for s in specs} == set(BIOME_TIERS[-1][1])
    for spec in specs:
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
    specs = [
        MAPPER.map(_fixture(path.stem), derive_seed(path.stem), path.stem)
        for path in sorted(FIXTURES.glob("*.json"))
        if path.stem != "P03"  # incomplete on purpose
    ]
    assert {s.lighting.time_of_day for s in specs} == set(get_args(TimeOfDay))
    assert {s.biome for s in specs} == set(get_args(Biome))
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


# -- the land and the view (0.3) --------------------------------------------

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def _all_specs() -> list[SceneSpec]:
    """Specs across the trait space and several seeds, the largest one included."""
    return [
        MAPPER.map(_persona(**dict(zip(TRAITS, values))), seed, "t")
        for values in itertools.product((0.0, 0.3, 0.5, 0.8, 1.0), repeat=5)
        if values[3] == 0.5  # agreeableness changes nothing
        for seed in (0, 1, 2, MAX_SEED)
    ]


ALL_SPECS = _all_specs()


def test_fields_from_before_0_3_keep_their_values():
    """The new draws come after the old ones, and the kits kept their five
    entries, so a participant's sun, wind, clouds and moon are as they were."""
    spec = MAPPER.map(_fixture("P01"), derive_seed("P01"), "P01")
    assert spec.biome == "snowy_valley"
    # was spruce_snowy, snow_drift, boulder_snowy, deadwood
    assert [p.asset for p in spec.props] == ["pine_large", "pine_medium", "boulder", "driftwood"]
    assert spec.lighting.sun_azimuth_deg == 132.879
    assert spec.motion.wind_direction_deg == 249.092
    assert spec.procedural_sky.clouds.offset == (48.878, 85.991)
    assert (spec.lighting.moon_azimuth_deg, spec.lighting.moon_elevation_deg) == (277.732, 29.69)


def test_every_name_sent_is_in_the_catalogue():
    for spec in ALL_SPECS:
        assert spec.catalogue_version == CATALOGUE_V1.version
        assert spec.archetype in CATALOGUE_V1.archetypes
        assets = {p.asset for p in spec.props} | {spec.focal.asset, spec.framing.asset}
        assert assets <= CATALOGUE_V1.props
        assert spec.terrain.ground_material is None  # the archetype's
        assert {p.distribution for p in spec.props} <= {"poisson", "clustered"}


def test_the_catalogue_names_what_the_spec_can_carry():
    assert set(CATALOGUE_V1.archetypes) == set(get_args(Archetype))
    for setting in CATALOGUE_V1.archetypes.values():
        assert setting.placement in get_args(Placement)
        assert setting.water in get_args(WaterKind)
        assert setting.ground in CATALOGUE_V1.grounds


def test_mountains_and_cliffs_are_left_to_unity():
    assert not RESERVED_PROPS & CATALOGUE_V1.props
    for spec in ALL_SPECS:
        assets = {p.asset for p in spec.props} | {spec.focal.asset, spec.framing.asset}
        assert not assets & {"mountain", "cliff"}


def test_every_kit_is_five_different_catalogue_props():
    for kit in BIOMES.values():
        assets = [asset for asset, _ in kit.props]
        assert len(assets) == len(set(assets)) == 5
        assert set(assets) <= CATALOGUE_V1.props
        assert kit.tree is None or kit.tree in CATALOGUE_V1.props


def test_every_biome_and_bias_has_an_archetype():
    assert set(ARCHETYPE_BY_BIOME) == set(get_args(Biome))
    for row in ARCHETYPE_BY_BIOME.values():
        assert set(row) == {"refuge", "balanced", "prospect"}
        assert set(row.values()) <= set(CATALOGUE_V1.archetypes)


def test_the_archetype_follows_the_biome_and_prospect_refuge():
    for spec in ALL_SPECS:
        assert spec.archetype == ARCHETYPE_BY_BIOME[spec.biome][spec.spatial.prospect_refuge_bias]


def test_a_calm_open_alpine_persona_sits_on_a_summit():
    summit = _persona(openness=0.95, neuroticism=0.1)
    specs = [MAPPER.map(summit, seed, "t") for seed in range(40)]
    alpine = [s for s in specs if s.biome == "alpine_basin"]
    assert alpine  # the seed picks the biome within openness' tier
    for spec in alpine:
        assert spec.archetype == "summit"
        assert spec.terrain.spawn.placement == "summit"
        assert spec.terrain.water.kind == "none"


def test_placement_and_water_are_the_archetypes_and_always_sent():
    for spec in ALL_SPECS:
        setting = CATALOGUE_V1.archetypes[spec.archetype]
        water = spec.terrain.water
        assert spec.terrain.spawn.placement == setting.placement
        assert water.kind == setting.water
        assert water.present == (water.kind != "none")
        assert (water.distance_m, water.azimuth_deg) == (None, None)  # placed by Unity
        if not water.present:
            assert water.motion == 0.0


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


def test_the_focal_is_the_biomes_tree_in_a_forest_or_hollow():
    for spec in ALL_SPECS:
        tree = BIOMES[spec.biome].tree
        if spec.archetype in ("forest", "hollow") and tree:
            assert spec.focal.asset == tree
        else:
            assert spec.focal.asset == "rock_cluster"


def test_the_framing_prop_is_close_and_at_the_edge_of_view():
    sides = set()
    for spec in ALL_SPECS:
        framing, facing = spec.framing, spec.terrain.spawn.facing_deg
        assert 3 <= framing.distance_m <= 5  # what Unity keeps it to
        low, high = FRAMING_OFFSET_DEG
        assert low - 0.001 <= _apart(framing.azimuth_deg, facing) <= high + 0.001
        assert framing.asset == (BIOMES[spec.biome].tree or "rock_cluster")
        sides.add((framing.azimuth_deg - facing) % 360 < 180)
    assert sides == {True, False}


def test_the_fog_leaves_a_third_of_the_horizon_except_in_a_hollow():
    for name, setting in CATALOGUE_V1.archetypes.items():
        if name in FOG_CAP:
            visible = math.exp(-(FOG_CAP[name] * setting.horizon_m) ** 2)
            assert 0.28 <= visible <= 0.32
            assert FOG_CAP[name] == pytest.approx(FOG_HORIZON_DEPTH / setting.horizon_m, abs=5e-5)
    assert "hollow" not in FOG_CAP
    for spec in ALL_SPECS:
        if spec.archetype in FOG_CAP:
            assert spec.lighting.fog_density <= FOG_CAP[spec.archetype]
    # In a hollow the fog is the shelter: it still thickens with neuroticism.
    hollows = [s for s in ALL_SPECS if s.archetype == "hollow"]
    assert max(s.lighting.fog_density for s in hollows) > 0.04


# -- every spec is valid ----------------------------------------------------


@pytest.mark.parametrize("values", list(itertools.product((0.0, 0.5, 1.0), repeat=5)))
def test_every_produced_spec_validates(values):
    traits = dict(zip(TRAITS, values))
    for seed in (0, 1, 2**31, MAX_SEED):
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


def test_the_0_3_rules_print_too():
    tables = rules_markdown()
    assert "| alpine_basin | hollow | overlook | summit |" in tables
    assert "| summit | terraces | summit | none | alpine_rock | 420 | 0.0026 |" in tables
    assert "| hollow | bowl | basin_floor | none | moss_rock_mix | 300 | none |" in tables
    assert "| > 0.65 | ringed | clustered |" in tables
    assert "| `focal.distance_m` | neuroticism | 40 | 25 |" in tables
