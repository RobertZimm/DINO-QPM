import argparse

COMMANDS = ("train", "inference", "evaluate")

USAGE = """usage: python main.py [--log-level LEVEL] {train,inference,evaluate} [options]

commands:
  train       train a dense model and fine-tune it into DINO-QPM
  evaluate    evaluate a trained (or pretrained) model
  inference   predict and visualise single images

Run 'python main.py <command> --help' for the options of a command."""


def split_command(argv: list[str]) -> tuple[str | None, list[str]]:
    """Split the command name from its arguments.

    Returns ``(None, [])`` when no command or only ``-h``/``--help`` is given.
    """
    if not argv or argv[0] in {"-h", "--help"}:
        return None, []

    if argv[0] in COMMANDS:
        return argv[0], argv[1:]

    if argv[0].startswith("-"):
        # Backward-compatible path: legacy training flags without subcommand.
        return "train", argv

    raise ValueError(
        f"Unknown command '{argv[0]}'. Expected one of: {', '.join(COMMANDS)}"
    )


def parse_global_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse global CLI args and leave command-specific args untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Global logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser.parse_known_args(argv)
