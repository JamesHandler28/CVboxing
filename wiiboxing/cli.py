"""Console-script entry points, wired up in pyproject.toml."""


def detect():
    from .detector import run
    run()


def play():
    from .game import run
    run()


def duel():
    from .duel import run
    run()
