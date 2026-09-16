#!/usr/bin/env python3
"""Create immutable, separated CodeARC-Replay public/evaluator records."""
import argparse
import json
from pbpf.apbpf.codearc_materialize import materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--evaluator-root", required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--development-components", type=int, default=400)
    parser.add_argument("--primary-components", type=int, default=500)
    args = vars(parser.parse_args())
    result = materialize(**args)
    print(json.dumps({k: v for k, v in result.items() if k != "source_inventory"}, sort_keys=True))


if __name__ == "__main__":
    main()
