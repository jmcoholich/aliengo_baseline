import yaml

with open(r'config/defaults.yaml') as f:
    default_params = yaml.full_load(f)
print(default_params)

