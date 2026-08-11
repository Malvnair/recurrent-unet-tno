"""Compatibility wrapper for the maintained CPU-first training entry point."""

from train_hand import train


if __name__ == "__main__":
    import runpy

    runpy.run_module("train_hand", run_name="__main__")
