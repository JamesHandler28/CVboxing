"""
Duel mode per-player state machine (no cv2/mediapipe dependency -- pure
logic, so it's trivial to test).

Each player has HP. Getting hit by the OPPONENT's punch costs HP during
normal play (damage varies by zone -- see take_damage), unless it's blocked
or dodged (handled by the caller before take_damage is invoked; see
is_dodged below and glove_covers_zone in pose_utils.py). Hitting zero HP
knocks you down.

Knockdowns 1 and 2 are recoverable: you get KNOCKDOWN_TIME_LIMIT_SECONDS
(displayed by the caller as a rising boxing-style count toward 10) to land
enough recovery-target hits. Landing extra hits beyond the requirement
brings you back with more HP, scaling continuously up to a cap -- see
_recover(). Reaching MAX_KNOCKDOWNS is an automatic KO: no recovery chance
at all, regardless of how the first two recoveries went.
"""

from . import config

PLAYING = "PLAYING"
KNOCKDOWN = "KNOCKDOWN"
GAME_OVER = "GAME_OVER"


class PlayerState:
    def __init__(self, name):
        self.name = name
        self.max_hp = config.DUEL_MAX_HP
        self.hp = self.max_hp
        self.phase = PLAYING
        self.knockdown_count = 0  # total times knocked down so far (this knockdown included)
        self.knockdown_start_time = 0.0
        self.knockdown_hits = 0
        self.required_hits = 0

    def required_hits_for_current_knockdown(self):
        # knockdown_count is already incremented (1-indexed) by the time
        # this is called, so the 1st knockdown uses the base requirement
        # and each one after that gets a little harder.
        return (config.KNOCKDOWN_BASE_REQUIRED_HITS
                + (self.knockdown_count - 1) * config.KNOCKDOWN_REQUIRED_HITS_INCREMENT)

    def enter_knockdown(self, now):
        self.knockdown_count += 1

        if self.knockdown_count >= config.MAX_KNOCKDOWNS:
            self.phase = GAME_OVER
            print(f"{self.name} IS DOWN FOR THE {self.knockdown_count} TIME -- THAT'S A KO!")
            return

        self.phase = KNOCKDOWN
        self.knockdown_start_time = now
        self.knockdown_hits = 0
        self.required_hits = self.required_hits_for_current_knockdown()
        print(f"{self.name} KNOCKED DOWN! Needs {self.required_hits} hits in "
              f"{config.KNOCKDOWN_TIME_LIMIT_SECONDS:.0f}s to get back up.")

    def take_damage(self, now, amount):
        """Called when the OPPONENT's punch connects on this player. Only
        applies during normal play -- a player who's already down doesn't
        take further damage from punches thrown while they're recovering."""
        if self.phase != PLAYING:
            return
        self.hp = max(0, self.hp - amount)
        if self.hp <= 0:
            self.enter_knockdown(now)

    def register_recovery_hit(self, now):
        """Called when this (knocked-down) player lands a hit on their own
        recovery target during the knockdown minigame. Just tallies the hit
        -- success/failure and the resulting HP are resolved once the count
        finishes (see tick()), so landing extra hits after meeting the bare
        requirement is possible and pays off."""
        if self.phase != KNOCKDOWN:
            return
        self.knockdown_hits += 1
        print(f"  {self.name} recovery hit {self.knockdown_hits}/{self.required_hits}")

    def _recover(self):
        """Come back from a successful recovery. The max-HP ceiling still
        drops a bit each time (floored), same as before -- but how much of
        that new ceiling you actually come back WITH now scales
        continuously with how many extra hits you landed beyond the bare
        requirement, instead of always snapping straight to the new max."""
        self.max_hp = max(config.DUEL_MAX_HP_FLOOR, self.max_hp - config.DUEL_HP_REDUCTION_PER_KNOCKDOWN)

        extra_hits = self.knockdown_hits - self.required_hits
        progress = min(1.0, extra_hits / config.DUEL_RECOVERY_EXTRA_HITS_FOR_FULL_HP)
        fraction = (config.DUEL_RECOVERY_MIN_HP_FRACTION
                    + (1 - config.DUEL_RECOVERY_MIN_HP_FRACTION) * progress)

        self.hp = round(self.max_hp * fraction)
        self.phase = PLAYING
        print(f"{self.name} BACK UP with {self.hp}/{self.max_hp} HP "
              f"({self.knockdown_hits}/{self.required_hits} hits, {extra_hits} extra)")

    def tick(self, now):
        """Call once per frame. When the count finishes (the caller shows
        this as a rising count toward 10), resolve the knockdown: enough
        hits landed recovers (with HP scaled by how many extra hits you
        got in), not enough is a KO."""
        if self.phase != KNOCKDOWN:
            return
        if now - self.knockdown_start_time >= config.KNOCKDOWN_TIME_LIMIT_SECONDS:
            if self.knockdown_hits >= self.required_hits:
                self._recover()
            else:
                self.phase = GAME_OVER
                print(f"{self.name} COULDN'T BEAT THE COUNT. OUT.")

    def reset(self):
        self.__init__(self.name)


def damage_for_zone(zone):
    return {
        "HEAD": config.DUEL_HEAD_DAMAGE,
        "BODY": config.DUEL_BODY_DAMAGE,
        "LOW": config.DUEL_LOW_DAMAGE,
    }.get(zone, config.DUEL_BODY_DAMAGE)


def is_dodged(zone, defender_lean_offset):
    """Head shots are dodged if the defender is leaning far enough away.
    Body/low shots always land -- you slip your head, not your ribs."""
    return zone == "HEAD" and abs(defender_lean_offset) > config.LEAN_DODGE_THRESHOLD