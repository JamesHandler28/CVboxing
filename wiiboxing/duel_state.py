"""
Duel mode per-player state machine (no cv2/mediapipe dependency -- pure
logic, so it's trivial to test).

Each player has HP. Getting hit by the OPPONENT's punch costs HP during
normal play (damage varies by zone -- see take_damage). Hitting zero HP
knocks you down: you get KNOCKDOWN_TIME_LIMIT_SECONDS to land enough
recovery-target hits to get back up. Succeed, and you're back up, but with
your max HP reduced for next time (down to a floor). Fail the recovery
window, and you lose the match outright.
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
        self.knockdown_count = 0
        self.knockdown_start_time = 0.0
        self.knockdown_hits = 0
        self.required_hits = 0

    def required_hits_for_next_knockdown(self):
        return config.KNOCKDOWN_BASE_REQUIRED_HITS + self.knockdown_count * config.KNOCKDOWN_REQUIRED_HITS_INCREMENT

    def enter_knockdown(self, now):
        self.phase = KNOCKDOWN
        self.knockdown_start_time = now
        self.knockdown_hits = 0
        self.required_hits = self.required_hits_for_next_knockdown()
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
        recovery target during the knockdown minigame."""
        if self.phase != KNOCKDOWN:
            return
        self.knockdown_hits += 1
        print(f"  {self.name} recovery hit {self.knockdown_hits}/{self.required_hits}")
        if self.knockdown_hits >= self.required_hits:
            self.knockdown_count += 1
            self.max_hp = max(config.DUEL_MAX_HP_FLOOR, self.max_hp - config.DUEL_HP_REDUCTION_PER_KNOCKDOWN)
            self.hp = self.max_hp
            self.phase = PLAYING
            print(f"{self.name} BACK UP! (max HP now {self.max_hp})")

    def tick(self, now):
        """Call once per frame. Handles the knockdown timer running out."""
        if self.phase == KNOCKDOWN and now - self.knockdown_start_time >= config.KNOCKDOWN_TIME_LIMIT_SECONDS:
            self.phase = GAME_OVER
            print(f"{self.name} IS OUT.")

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
