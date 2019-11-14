from argparse import ArgumentParser

from .prepare import filter_info

filter_info_parser = ArgumentParser(
    prog='filter_info',
    description='Filter the information given as a CSV file.',
)
filter_info_parser.add_argument(
    '-i', '--input',
    type=str,
    required=True,
    help='path to the raw CSV file',
    dest='input_path',
)
filter_info_parser.add_argument(
    '-o', '--output',
    type=str,
    default='data/info.csv',
    help='path to the output CSV file',
    dest='output_path',
)


class CLICommand:
    @staticmethod
    def filter_info(input_path, output_path):
        print(f'# Filtering info {input_path!r}…')
        info = filter_info(input_path)
        print(f'# Saving filtered info as {output_path!r}…')
        info.to_csv(output_path, index=False)


commands = {
    'filter_info': (CLICommand.filter_info, filter_info_parser),
}

command_parser = ArgumentParser(
    prog='python -m imgattr',
    description='Attribute image authorship, based on image features.',
)
command_parser.add_argument(
    'command',
    type=str,
    choices=commands,
    help='command to execute',
)
