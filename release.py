"""
One command from model.py to a clean upload: run this every time the bot
changes, then upload everything in submission/. Not part of the submission.

    python release.py

It removes every old pickle, runs the validity test, builds the pickle with
submit, keeps exactly the files submit says to upload in submission/, and
checks that those are the ones in the repo root.
"""

import glob
import hashlib
import os
import pickle
import shutil
import subprocess
import sys

TEAM = "poker-i-barely-know-her"
OUT = "submission"


def sha(path):
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()[:12]


def main():
    for path in glob.glob("*.pkl"):
        os.remove(path)
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)

    test = [sys.executable, "-m", "pytest", "tests/test_model.py", "-q"]
    if subprocess.run(test).returncode:
        sys.exit("The validity test failed: nothing to upload.")

    build = [sys.executable, "-m", "submit", "--team", TEAM, "--out", OUT]
    done = subprocess.run(build, capture_output=True, text=True)
    print(done.stdout + done.stderr)
    if done.returncode:
        sys.exit("submit failed: nothing to upload.")

    listed = done.stdout.split("Upload these files in the form (all of them):")[1].split()
    for name in listed:
        if name.endswith(".py"):
            shutil.copy(name, OUT)
    if sorted(os.listdir(OUT)) != sorted(listed):
        sys.exit(f"submission/ holds {os.listdir(OUT)}, submit listed {listed}")
    for name in listed:
        if name.endswith(".py") and sha(name) != sha(os.path.join(OUT, name)):
            sys.exit(f"{name} differs from the copy in submission/")

    # The pickle must be today's code: same class, same attributes as a new one.
    sys.path.insert(0, os.getcwd())
    from model import Model

    with open(os.path.join(OUT, TEAM + ".pkl"), "rb") as file:
        built = pickle.load(file)
    if type(built) is not Model or set(vars(built)) != set(vars(Model())):
        sys.exit("The pickle does not match model.py")

    print(f"READY: upload these {len(listed)} files from {OUT}{os.sep}")
    for name in listed:
        print(f"  {name}  {sha(os.path.join(OUT, name))}")


if __name__ == "__main__":
    main()
