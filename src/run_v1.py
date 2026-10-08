"""
Quantisense Capital
V1 Master Orchestrator

Runs the currently completed V1 pipeline in chronological order.

Completed layers:
1. Classical market intelligence
2. Media interpretation
3. Hybrid intelligence
4. Quantum hybrid
5. Trained quantum OOS

Important:
This orchestrator does NOT claim that the media layer has
historical coverage yet. The current media dataset contains
only the observations actually available.
"""

from pathlib import Path
import subprocess
import sys
import time


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"


STEPS = [
    (
        "CLASSICAL MARKET INTELLIGENCE",
        "quantisense_engine.py",
    ),
    (
        "MEDIA INTERPRETATION",
        "media_interpretation.py",
    ),
    (
        "HYBRID INTELLIGENCE",
        "hybrid_intelligence.py",
    ),
    (
        "QUANTUM HYBRID",
        "quantum_hybrid.py",
    ),
    (
        "TRAINED QUANTUM OOS",
        "quantum_hybrid_oos.py",
    ),
]


def run_step(title, script):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Script: {script}")
    print()

    script_path = SRC_DIR / script

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
    )

    elapsed = time.time() - start

    print()
    print("-" * 70)
    print(f"Completed in {elapsed:.1f} seconds")

    if result.returncode != 0:
        print(f"FAILED: {script}")
        print(f"Return code: {result.returncode}")
        return False

    print(f"SUCCESS: {script}")
    return True


def main():
    print("=" * 70)
    print("QUANTISENSE CAPITAL — V1 MASTER PIPELINE")
    print("=" * 70)
    print()
    print(f"Project root: {BASE_DIR}")
    print(f"Python:       {sys.executable}")
    print()
    print("Running completed V1 components...")
    print()

    total_start = time.time()

    results = []

    for title, script in STEPS:
        success = run_step(title, script)
        results.append((script, success))

        if not success:
            print()
            print("=" * 70)
            print("V1 PIPELINE STOPPED")
            print("=" * 70)
            print()
            print("The failed component must be fixed before continuing.")
            sys.exit(1)

    total_elapsed = time.time() - total_start

    print()
    print("=" * 70)
    print("QUANTISENSE V1 PIPELINE COMPLETE")
    print("=" * 70)
    print()

    for script, success in results:
        status = "OK" if success else "FAILED"
        print(f"{status:8} {script}")

    print()
    print(f"Total runtime: {total_elapsed:.1f} seconds")
    print()

    print("V1 STATUS")
    print("-" * 70)
    print("Classical market intelligence:  COMPLETE")
    print("Media interpretation:           COMPLETE")
    print("Hybrid intelligence:            COMPLETE")
    print("Quantum hybrid:                 COMPLETE")
    print("Quantum OOS experiment:         COMPLETE")
    print("Historical media coverage:      LIMITED")
    print("Final unified evaluator:        NEXT")
    print("Multi-asset final evaluator:    NEXT")
    print("Risk/portfolio layer:            NEXT")
    print()

    print("IMPORTANT:")
    print("The current quantum result does not demonstrate")
    print("quantum advantage. It remains a benchmark.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()