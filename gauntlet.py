"""
Random line-ups in random settings, so nothing gets tuned to one opponent.
Each run draws its own setting and opponents from everything available,
always able to include a copy of the bot under test. Every candidate plays
the same runs, so two candidates compare like for like. Not part of the
submission.

    python gauntlet.py --models model:Model model_next:Model --runs 60
    python gauntlet.py --settings event --runs 40    # only the event's format

Settings:
    event       heads-up, duplicate, 200 deals: arena.event's round
    short       the same with short stacks (15 big blinds)
    deep        the same with deep stacks (100 big blinds)
    table       3 to 6 players, duplicate
    tournament  2 to 4 players, stacks carry over, busted players leave
"""

import argparse
import collections
import glob
import importlib
import math
import os
import pickle
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor

from arena.dummies import DUMMIES
from arena.match import MatchConfig, play_match
from arena.players import player
from evals.score import score
from spar_strong import STRONG
from spar_weird import WEIRD
from sparring import BOTS

SETTINGS = {  # name: chance of being drawn
    "event": 0.5, "short": 0.1, "deep": 0.1, "table": 0.15, "tournament": 0.15,
}


def _write(name, obj, folder):
    path = os.path.join(folder, name + ".pkl")
    with open(path, "wb") as file:
        pickle.dump(obj, file)
    return path


def _candidate(spec):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name or "Model")()


def _pool(folder, teams):
    """Every opponent: name to player spec."""
    pool = {name: name for name in DUMMIES}
    for name, cls in (BOTS | STRONG | WEIRD).items():
        pool[name] = _write(name, cls(), folder)
    for name in ("v0", "v1"):
        pool[name] = _write(name, importlib.import_module("spar_" + name).Model(), folder)
    # One folder per team, so that each pickle finds its own model.py.
    for path in glob.glob(os.path.join(teams, "*", "*.pkl")) if teams else ():
        pool[os.path.splitext(os.path.basename(path))[0]] = path
    return pool


def _draw_runs(count, seed, names, only):
    rng = random.Random(seed)
    settings = [s for s in SETTINGS if not only or s in only]
    weights = [SETTINGS[s] for s in settings]
    runs = []
    for i in range(count):
        setting = rng.choices(settings, weights)[0]
        rivals = {"table": rng.randint(2, 5), "tournament": rng.randint(1, 3)}.get(setting, 1)
        # A copy of ourselves is in the draw: it is the "mirror" entry.
        line_up = rng.sample(names + ["mirror"], rivals)
        runs.append((i, setting, line_up, seed * 1000 + i))
    return runs


def _config(setting, run_seed):
    common = {"seed": run_seed, "checkpoint_dir": None}
    if setting == "tournament":
        return MatchConfig(deals=300, mode="tournament", **common)
    if setting == "table":
        return MatchConfig(deals=60, **common)
    stack = {"short": 300, "deep": 2000}.get(setting, 500)
    return MatchConfig(deals=200, stack=stack, **common)


def _play(job):
    """Our bb/100 in one run, and any fault of ours."""
    label, you, mirror, pool, (i, setting, line_up, run_seed) = job
    specs = [mirror if name == "mirror" else pool[name] for name in line_up]
    players = [player(you)] + [player(spec) for spec in specs]
    config = _config(setting, run_seed)
    log = play_match(players, config)
    if setting == "tournament":
        chips = log.final_stacks["you"] - config.stack
        bb, faults = 100 * chips / config.big_blind / max(1, len(log.hands)), 0
        faults = sum("you" in h.seats and h.seats.index("you") in h.faults for h in log.hands)
    else:
        result = score([log])
        bb, faults = result.bb_per_100[0], result.faults[0]
        if faults:
            print(f"FAULT {label} run {i} {line_up}: {faults} hands, first: "
                  f"{result.first_faults[0]}", flush=True)
    return label, i, setting, line_up, bb, faults


def _mean_ci(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("inf")
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(variance / n)


def _table(title, groups):
    print(f"  {title}")
    for key, values in sorted(groups.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        mean, ci = _mean_ci(values)
        print(f"    {key:<14} {mean:>+8.1f} ± {ci:<7.1f} over {len(values)} runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["model:Model"])
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS))
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--teams", help="folder of TEAM/TEAM.pkl to add to the pool")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as folder:
        pool = _pool(folder, args.teams)
        runs = _draw_runs(args.runs, args.seed, sorted(pool), args.settings)
        jobs = []
        for k, spec in enumerate(args.models):
            sub = os.path.join(folder, str(k))
            os.makedirs(sub)
            you = _write("you", _candidate(spec), sub)
            mirror = _write("mirror", _candidate(spec), sub)
            jobs += [(spec, you, mirror, pool, run) for run in runs]
        with ProcessPoolExecutor(args.jobs) as pool_of_workers:
            results = list(pool_of_workers.map(_play, jobs))

    for spec in args.models:
        rows = [r for r in results if r[0] == spec]
        mean, ci = _mean_ci([r[4] for r in rows])
        print(f"\n{spec}: {mean:+.1f} ± {ci:.1f} bb/100 over {len(rows)} runs, "
              f"faults {sum(r[5] for r in rows)}")
        by_setting = collections.defaultdict(list)
        by_rival = collections.defaultdict(list)
        for _, _, setting, line_up, bb, _ in rows:
            by_setting[setting].append(bb)
            if len(line_up) == 1 and setting != "tournament":
                by_rival[line_up[0]].append(bb)
        _table("by setting", by_setting)
        _table("heads-up, by opponent", by_rival)
    if len(args.models) > 1:
        first = {r[1]: r[4] for r in results if r[0] == args.models[0]}
        for spec in args.models[1:]:
            pairs = [r[4] - first[r[1]] for r in results if r[0] == spec]
            mean, ci = _mean_ci(pairs)
            print(f"\n{spec} minus {args.models[0]}, run by run: {mean:+.1f} ± {ci:.1f}")


if __name__ == "__main__":
    main()
