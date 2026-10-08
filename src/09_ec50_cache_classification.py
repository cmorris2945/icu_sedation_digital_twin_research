#!/usr/bin/env python3
"""Report aggregate EC50 boundary counts for a feature cache."""

import argparse
import pickle

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='cached_features.pkl')
    parser.add_argument('--tolerance', type=float, default=1e-3)
    args = parser.parse_args()

    with open(args.cache, 'rb') as cache_file:
        instances = pickle.load(cache_file)

    ec50 = instances.groupby('stay_id')['EC50_calibrated'].first()
    pinned_low = (ec50 - 0.5).abs() <= args.tolerance
    pinned_high = (ec50 - 10.0).abs() <= args.tolerance
    interior = ~(pinned_low | pinned_high)
    population_fallback = (ec50 - 3.08).abs() <= 1e-12

    print(f'cache={args.cache}')
    print(f'instances={len(instances)}')
    print(f'patients={len(ec50)}')
    print(f'pinned_low={int(pinned_low.sum())}')
    print(f'pinned_high={int(pinned_high.sum())}')
    print(f'interior={int(interior.sum())}')
    print(f'population_ec50={int(population_fallback.sum())}')


if __name__ == '__main__':
    main()