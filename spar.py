"""
Plays model.py against the sparring bots, matches in parallel, or plays the
sparring bots against each other. Not part of the submission.

    python spar.py                       # model:Model against everyone
    python spar.py --deals 1000 --seed 5
    python spar.py --model spar_v0:Model
    python spar.py --round-robin         # the field, to check it is balanced
"""

import argparse
import importlib
import itertools
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor

from arena.match import MatchConfig, repeat_match
from arena.players import player
from evals.score import score
from spar_strong import STRONG
from sparring import BOTS

FIELD = list(BOTS) + ["v0", "v1"] + list(STRONG)


def _write(name, obj, folder):
    path = os.path.join(folder, name + ".pkl")
    with open(path, "wb") as file:
        pickle.dump(obj, file)
    return path


def _field(folder):
    paths = {name: _write(name, cls(), folder) for name, cls in (BOTS | STRONG).items()}
    for name in ("v0", "v1"):
        paths[name] = _write(name, importlib.import_module("spar_" + name).Model(), folder)
    return paths


def _play(args):
    a, b, deals, seed, times = args
    config = MatchConfig(deals=deals, seed=seed, checkpoint_dir=None)
    result = score(repeat_match([player(a), player(b)], config, times))
    return (result.names, result.bb_per_100[0], result.ci95[0], result.faults,
            result.first_faults, max(result.max_move_seconds))


def _verdict(bb, ci):
    return "ahead" if bb - ci > 0 else "behind" if bb + ci < 0 else "unclear"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model:Model")
    parser.add_argument("--opponents", nargs="+", default=FIELD)
    parser.add_argument("--deals", type=int, default=500)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--round-robin", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as folder:
        field = _field(folder)
        if args.round_robin:
            pairs = list(itertools.combinations([field[n] for n in args.opponents], 2))
        else:
            module, _, name = args.model.partition(":")
            model = getattr(importlib.import_module(module), name or "Model")()
            you = _write("you", model, folder)
            pairs = [(you, field.get(name, name)) for name in args.opponents]
        jobs = [(a, b, args.deals, args.seed, args.times) for a, b in pairs]
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(_play, jobs))

    total = 0.0
    for names, bb, ci, faults, first, slowest in results:
        total += bb
        print(f"{names[0]:>8} v {names[1]:<8} {bb:>+8.1f} ± {ci:<6.1f} "
              f"{_verdict(bb, ci):>8}  faults {faults}  slowest {slowest:.2f}s")
        for fault in first:
            if fault:
                print(f"    first fault: {fault}")
    if not args.round_robin:
        print(f"mean bb/100 across the field: {total / len(results):+.1f}")


if __name__ == "__main__":
    main()
