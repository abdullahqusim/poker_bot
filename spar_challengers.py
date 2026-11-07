"""
Benchmark and sparring suite for the standalone challenger bots in `challengers.py`.

Examples:
    # Test model:Model against all 7 challenger bots (6 specialists + EnsembleBot)
    python spar_challengers.py

    # Test a specific challenger against all arena dummies
    python spar_challengers.py --vs-dummies --model challengers:EnsembleBot

    # Run a round-robin tournament among all 7 challenger bots + model:Model
    python spar_challengers.py --round-robin

    # Export .pkl files of the challenger bots to spar_trained/ so gauntlet.py
    # automatically includes them without modifying any existing files
    python spar_challengers.py --export-trained
"""

import argparse
import collections
import importlib
import itertools
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor

from arena.dummies import DUMMIES
from arena.match import MatchConfig, repeat_match
from arena.players import player
from challengers import CHALLENGERS
from evals.score import score


def _write_pkl(name: str, obj, folder: str) -> str:
    path = os.path.join(folder, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def _load_candidate(spec: str):
    module, _, attr = spec.partition(":")
    cls = getattr(importlib.import_module(module), attr or "Model")
    return cls()


def _play_pair(args):
    a, b, deals, seed, times, stack = args
    config = MatchConfig(deals=deals, seed=seed, stack=stack, checkpoint_dir=None)
    res = score(repeat_match([player(a), player(b)], config, times))
    return (
        res.names,
        res.bb_per_100[0],
        res.ci95[0],
        res.faults,
        res.first_faults,
        max(res.max_move_seconds),
    )


def _verdict(bb: float, ci: float) -> str:
    return "ahead" if bb - ci > 0 else ("behind" if bb + ci < 0 else "unclear")


def export_trained(folder: str = "spar_trained"):
    os.makedirs(folder, exist_ok=True)
    exported = []
    for name, cls in CHALLENGERS.items():
        path = _write_pkl(name, cls(), folder)
        exported.append(path)
    print(f"Exported {len(exported)} challenger bot pickles to {folder}/:")
    for p in exported:
        print(f"  - {p}")


def main():
    parser = argparse.ArgumentParser(description="Spar against the challenger bots in challengers.py")
    parser.add_argument("--model", default="model:Model", help="Model spec to test (default: model:Model)")
    parser.add_argument("--opponents", nargs="+", default=list(CHALLENGERS.keys()),
                        help="Challenger or dummy names to face")
    parser.add_argument("--vs-dummies", action="store_true",
                        help="Test all challenger bots (or --model) against all arena dummies")
    parser.add_argument("--round-robin", action="store_true",
                        help="Play a full round-robin among all challengers + --model")
    parser.add_argument("--export-trained", action="store_true",
                        help="Export challenger .pkl files into spar_trained/ for gauntlet.py")
    parser.add_argument("--deals", type=int, default=400)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--stack", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=7)
    args = parser.parse_args()

    if args.export_trained:
        export_trained()
        return

    with tempfile.TemporaryDirectory() as folder:
        challenger_paths = {
            name: _write_pkl(name, cls(), folder)
            for name, cls in CHALLENGERS.items()
        }

        if args.vs_dummies:
            # Test each challenger against the 6 arena dummies
            jobs = []
            for cname, cpath in challenger_paths.items():
                for dname in DUMMIES:
                    jobs.append((cpath, dname, args.deals, args.seed, args.times, args.stack))
            with ProcessPoolExecutor(args.jobs) as pool:
                results = list(pool.map(_play_pair, jobs))

            by_bot = collections.defaultdict(list)
            for names, bb, ci, faults, first, slowest in results:
                by_bot[names[0]].append((names[1], bb, ci, faults, first, slowest))

            for bname in CHALLENGERS:
                rows = by_bot[bname]
                mean_bb = sum(r[1] for r in rows) / len(rows)
                tot_faults = sum(r[3][0] for r in rows)
                print(f"\n=== {bname} vs Arena Dummies (mean: {mean_bb:+.1f} bb/100, faults: {tot_faults}) ===")
                for opp, bb, ci, faults, first, slowest in rows:
                    print(f"  {bname:>14} v {opp:<10} {bb:>+8.1f} ± {ci:<6.1f} {_verdict(bb, ci):>8}  slowest {slowest:.2f}s")
                    if first[0]:
                        print(f"    fault: {first[0]}")
            return

        if args.round_robin:
            you_path = _write_pkl("main_bot", _load_candidate(args.model), folder)
            all_entries = [("main_bot", you_path)] + list(challenger_paths.items())
            pairs = [
                (p1, p2)
                for (_, p1), (_, p2) in itertools.combinations(all_entries, 2)
            ]
            jobs = [(a, b, args.deals, args.seed, args.times, args.stack) for a, b in pairs]
            with ProcessPoolExecutor(args.jobs) as pool:
                results = list(pool.map(_play_pair, jobs))

            standings = collections.defaultdict(list)
            for names, bb, ci, faults, first, slowest in results:
                standings[names[0]].append(bb)
                standings[names[1]].append(-bb)
                print(f"{names[0]:>14} v {names[1]:<14} {bb:>+8.1f} ± {ci:<6.1f} "
                      f"{_verdict(bb, ci):>8}  faults {faults}  slowest {slowest:.2f}s")

            print("\n=== Round-Robin Leaderboard (Mean bb/100) ===")
            ranked = sorted(standings.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)
            for rank, (name, bbs) in enumerate(ranked, 1):
                avg = sum(bbs) / len(bbs)
                print(f"{rank:>2}. {name:<16} {avg:>+8.1f} bb/100 across {len(bbs)} matchups")
            return

        # Default: test --model against each challenger in CHALLENGERS
        you_path = _write_pkl("you", _load_candidate(args.model), folder)
        pairs = [(you_path, challenger_paths.get(opp, opp)) for opp in args.opponents]
        jobs = [(a, b, args.deals, args.seed, args.times, args.stack) for a, b in pairs]
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(_play_pair, jobs))

        total = 0.0
        for names, bb, ci, faults, first, slowest in results:
            total += bb
            print(f"{names[0]:>14} v {names[1]:<14} {bb:>+8.1f} ± {ci:<6.1f} "
                  f"{_verdict(bb, ci):>8}  faults {faults}  slowest {slowest:.2f}s")
            for fault in first:
                if fault:
                    print(f"    first fault: {fault}")
        print(f"mean bb/100 across {len(results)} challengers: {total / len(results):+.1f}")


if __name__ == "__main__":
    main()
