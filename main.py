import yaml

import argparse



with open(r'config/defaults.yaml') as f:
    default_params = yaml.full_load(f)
print(default_params)

test_dict = {'asdf':39393, 'gringotts': 43}

parser = argparse.ArgumentParser()
# parser.add_argument("--verbose", help="increase output verbosity",default=False,
#                     action="store_true")
args = parser.parse_args()

vars(args).update(test_dict)

breakpoint()

