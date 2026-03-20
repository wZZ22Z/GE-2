import argparse
import sys

import gege as ge


def main():
    parser = argparse.ArgumentParser(description="Configuration file based evaluation", prog="eval")

    parser.add_argument(
        "config",
        metavar="config",
        type=str,
        help=(
            "Path to YAML configuration file that describes the evaluation process. See documentation"
            " docs/config_interface for more details."
        ),
    )

    args = parser.parse_args()
    config = ge.config.loadConfig(args.config, save=True)
    ge.manager.gege_eval(config)


if __name__ == "__main__":
    sys.exit(main())
