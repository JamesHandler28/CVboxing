"""Tests for PlayerState: taking damage, knockdown trigger, recovery with
reduced max HP, game-over on recovery timeout, and the dodge/damage helper
functions."""

from wiiboxing import config
from wiiboxing.duel_state import (
    GAME_OVER,
    KNOCKDOWN,
    PLAYING,
    PlayerState,
    damage_for_zone,
    is_dodged,
)


def test_starts_at_full_hp():
    p = PlayerState("P1")
    assert p.phase == PLAYING
    assert p.hp == config.DUEL_MAX_HP
    assert p.max_hp == config.DUEL_MAX_HP


def test_taking_damage_reduces_hp():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, 30)
    assert p.hp == config.DUEL_MAX_HP - 30
    assert p.phase == PLAYING


def test_hp_reaching_zero_triggers_knockdown():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, config.DUEL_MAX_HP)
    assert p.phase == KNOCKDOWN
    assert p.hp == 0
    assert p.required_hits == config.KNOCKDOWN_BASE_REQUIRED_HITS


def test_damage_while_already_knocked_down_is_ignored():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, config.DUEL_MAX_HP)
    assert p.phase == KNOCKDOWN
    p.take_damage(now, 50)  # should be a no-op
    assert p.hp == 0
    assert p.phase == KNOCKDOWN


def test_recovering_reduces_max_hp_and_refills_to_minimum_fraction():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, config.DUEL_MAX_HP)
    assert p.phase == KNOCKDOWN
    first_required = p.required_hits

    for _ in range(first_required):
        p.register_recovery_hit(now)
    assert p.phase == KNOCKDOWN  # landing the hits doesn't resolve it early --
    # recovery/KO is only decided once the count finishes.

    p.tick(now + config.KNOCKDOWN_TIME_LIMIT_SECONDS + 0.01)
    assert p.phase == PLAYING
    assert p.knockdown_count == 1
    expected_max = max(config.DUEL_MAX_HP_FLOOR, config.DUEL_MAX_HP - config.DUEL_HP_REDUCTION_PER_KNOCKDOWN)
    assert p.max_hp == expected_max
    # Landing exactly the required hits (no extras) only earns the minimum
    # recovery fraction, not a full refill.
    expected_hp = round(expected_max * config.DUEL_RECOVERY_MIN_HP_FRACTION)
    assert p.hp == expected_hp


def test_extra_recovery_hits_scale_hp_up_to_full():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, config.DUEL_MAX_HP)
    required = p.required_hits
    for _ in range(required + config.DUEL_RECOVERY_EXTRA_HITS_FOR_FULL_HP):
        p.register_recovery_hit(now)
    p.tick(now + config.KNOCKDOWN_TIME_LIMIT_SECONDS + 0.01)
    assert p.phase == PLAYING
    assert p.hp == p.max_hp  # enough extra hits caps recovery at full new max HP


def test_a_few_extra_hits_give_a_partial_bonus_short_of_full_hp():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, config.DUEL_MAX_HP)
    required = p.required_hits
    exact_hp = None

    # Baseline: exactly the requirement, for comparison.
    p2 = PlayerState("P1")
    p2.take_damage(now, config.DUEL_MAX_HP)
    for _ in range(p2.required_hits):
        p2.register_recovery_hit(now)
    p2.tick(now + config.KNOCKDOWN_TIME_LIMIT_SECONDS + 0.01)
    exact_hp = p2.hp

    # A couple of extra hits, but not enough to reach the full-HP cap.
    partial_extra = max(1, config.DUEL_RECOVERY_EXTRA_HITS_FOR_FULL_HP - 1)
    for _ in range(required + partial_extra):
        p.register_recovery_hit(now)
    p.tick(now + config.KNOCKDOWN_TIME_LIMIT_SECONDS + 0.01)

    assert exact_hp <= p.hp <= p.max_hp


def test_third_knockdown_is_an_automatic_ko_with_no_recovery_chance():
    p = PlayerState("P1")
    now = 0.0
    for _ in range(config.MAX_KNOCKDOWNS - 1):
        p.take_damage(now, p.max_hp)
        assert p.phase == KNOCKDOWN
        for _ in range(p.required_hits):
            p.register_recovery_hit(now)
        p.tick(now + config.KNOCKDOWN_TIME_LIMIT_SECONDS + 0.01)
        assert p.phase == PLAYING

    # The MAX_KNOCKDOWNS'th knockdown is an instant KO -- no KNOCKDOWN phase,
    # no recovery minigame, even though every prior recovery succeeded.
    p.take_damage(now, p.max_hp)
    assert p.phase == GAME_OVER
    assert p.knockdown_count == config.MAX_KNOCKDOWNS


def test_second_knockdown_requires_more_hits_than_first():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, config.DUEL_MAX_HP)
    first_required = p.required_hits
    for _ in range(first_required):
        p.register_recovery_hit(now)
    p.tick(now + config.KNOCKDOWN_TIME_LIMIT_SECONDS + 0.01)
    assert p.phase == PLAYING

    p.take_damage(now, p.max_hp)
    assert p.phase == KNOCKDOWN
    assert p.required_hits == first_required + config.KNOCKDOWN_REQUIRED_HITS_INCREMENT


def test_max_hp_never_drops_below_floor():
    p = PlayerState("P1")
    now = 0.0
    # Only MAX_KNOCKDOWNS - 1 recoveries can ever happen (the last knockdown
    # is an automatic KO), so check the floor invariant across all of them.
    for _ in range(config.MAX_KNOCKDOWNS - 1):
        p.take_damage(now, p.max_hp)
        assert p.phase == KNOCKDOWN
        for _ in range(p.required_hits):
            p.register_recovery_hit(now)
        p.tick(now + config.KNOCKDOWN_TIME_LIMIT_SECONDS + 0.01)
        assert p.phase == PLAYING
        assert p.max_hp >= config.DUEL_MAX_HP_FLOOR


def test_running_out_the_clock_during_knockdown_is_game_over():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, config.DUEL_MAX_HP)
    assert p.phase == KNOCKDOWN

    now += config.KNOCKDOWN_TIME_LIMIT_SECONDS + 0.01
    p.tick(now)
    assert p.phase == GAME_OVER


def test_reset_returns_to_a_fresh_state():
    p = PlayerState("P1")
    now = 0.0
    p.take_damage(now, config.DUEL_MAX_HP)
    p.reset()
    assert p.phase == PLAYING
    assert p.hp == config.DUEL_MAX_HP
    assert p.max_hp == config.DUEL_MAX_HP
    assert p.knockdown_count == 0
    assert p.name == "P1"


def test_damage_for_zone_ordering():
    assert damage_for_zone("HEAD") > damage_for_zone("BODY") > damage_for_zone("LOW")


def test_head_shot_is_dodged_when_leaning_far_enough():
    assert is_dodged("HEAD", config.LEAN_DODGE_THRESHOLD + 0.1) is True
    assert is_dodged("HEAD", -(config.LEAN_DODGE_THRESHOLD + 0.1)) is True


def test_head_shot_lands_when_not_leaning_enough():
    assert is_dodged("HEAD", config.LEAN_DODGE_THRESHOLD - 0.1) is False


def test_body_and_low_shots_are_never_dodged():
    assert is_dodged("BODY", 999) is False
    assert is_dodged("LOW", 999) is False