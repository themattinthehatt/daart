"""Command-line interface for daart video model pre-training package."""

import logging
import sys
from argparse import ArgumentParser

from daart.cli import formatting
from daart.cli.commands import COMMANDS


def build_parser() -> ArgumentParser:
    """Build the main argument parser with all subcommands."""

    parser = formatting.ArgumentParser(
        prog='daart',
        description='Deep learning for animal action recognition toolbox.',
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help='Command to run',
        parser_class=formatting.SubArgumentParser,
    )

    # register all commands from the commands module
    for name, module in COMMANDS.items():
        module.register_parser(subparsers)

    return parser


def main():
    """Main CLI entry point."""

    # configure logging once at application startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s  %(name)s : %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            # logging.FileHandler('app.log')  # Optional: also log to file
        ]
    )

    parser = build_parser()

    # if no commands provided, display help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # parse arguments
    args = parser.parse_args()

    # get command handler
    command_handler = COMMANDS[args.command].handle

    # execute command
    command_handler(args)


if __name__ == '__main__':
    main()
