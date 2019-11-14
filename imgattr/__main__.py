import sys

from .cli import (
    command_parser,
    commands,
)

if __name__ == '__main__':
    command, command_parser = commands[command_parser.parse_args(sys.argv[1:2]).command]
    command(**vars(command_parser.parse_args(sys.argv[2:])))
