import json
import argparse


class JsonConfiguration(object):

    ALLOWED_PREFIXES = ('', '--', '-',)

    def __init__(self, command_line_option='--config-file', prefix='--', 
                 parser_prefix='--'):
        if prefix not in JsonConfiguration.ALLOWED_PREFIXES:
            raise ValueError('prefix must be one of ', *JsonConfiguration.ALLOWED_PREFIXES)
        if parser_prefix not in JsonConfiguration.ALLOWED_PREFIXES:
            raise ValueError('parser_prefix must be one of ', *JsonConfiguration.ALLOWED_PREFIXES)
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(command_line_option, type=str)
        attr_name = command_line_option
        attr_name = attr_name if not attr_name.startswith(prefix)\
                    else attr_name[len(parser_prefix):]
        attr_name = attr_name.replace('-', '_')
        self.attr_name = attr_name
        self.prefix = parser_prefix

    def get_parser_options(self, parser):
        options = []
        for action in parser._actions:
            if isinstance(action, argparse._StoreAction):
                option_string = action.option_strings[0]
                if not option_string.startswith(self.prefix):
                    raise ValueError('This class accepts only parser with ' + self.prefix + ' options')
                options.append(option_string)
        return options

    def parse(self, argument_parser):
        options = self.get_parser_options(argument_parser)

        args = self.parser.parse_args()
        file_name = getattr(args, self.attr_name)
        with open(file_name) as fp:
            data = json.load(fp)
        
        values = []
        for key, val in data.items():    
            if not key.startswith(self.prefix):
                key = self.prefix + key
            if key in options:
                values.append(key)
                values.append(str(val))
        
        args = argument_parser.parse_args(values)
        return args
