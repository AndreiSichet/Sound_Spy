"""The ``soundspy`` command-line interface (development, data and training jobs).

One module per subcommand (``*_cmd.py``), each exposing ``register(subparsers)``;
``main.py`` wires them into a single argparse dispatcher.
"""


def not_implemented(step: str) -> int:
    """Print the placeholder message for a subcommand and return exit code 0."""
    print(f"not implemented yet ({step})")
    return 0
