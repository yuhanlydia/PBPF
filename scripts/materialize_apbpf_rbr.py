#!/usr/bin/env python3
"""Materialize a new, explicitly exploratory RBR generated-candidate protocol."""
import argparse
import json
from pbpf.apbpf.rbr_materialize import materialize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'descriptions-archive', 'public-root', 'evaluator-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--development-components', type=int, default=400)
    p.add_argument('--primary-components', type=int, default=500)
    result = materialize(**vars(p.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != 'source_inventory'}))


if __name__ == '__main__':
    main()
