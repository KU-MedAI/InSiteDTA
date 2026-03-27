import os
import inspect
import argparse


def print_args(args: argparse.Namespace):
    script_name = os.path.basename(inspect.stack()[1].filename)
    max_len = min(max([len(str(k) + str(v)) for k, v in vars(args).items()]) + 10, 80)

    print("+" + "-" * max_len + "+")
    print("| " + f"{script_name} configuration".ljust(max_len - 2) + " |")
    print("+" + "-" * max_len + "+")
    for k, v in vars(args).items():
        line = f"|  - {k}: {v}"
        print(line.ljust(max_len) + " |")
    print("+" + "-" * max_len + "+")


def parse_to_list(value, type) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        return [type(x.strip()) for x in value.strip("[]").split(",")]
    else:
        return [type(x) for x in value.split()]
