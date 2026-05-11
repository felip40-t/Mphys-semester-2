"""
Generic LHE parser: reads events, applies Lorentz boosts, and writes
kinematic observables to disk in batches (never loading all events at once).

Output files written to ``output_dir``:
    theta1.npy    polar angle (rad) of the first decay product of boson 1
    phi1.npy      azimuthal angle (rad) of the first decay product of boson 1
    theta3.npy    polar angle (rad) of the first decay product of boson 2
    phi3.npy      azimuthal angle (rad) of the first decay product of boson 2
    cos_psi.npy   cosine of boson 1 scattering angle in the diboson CM frame
    inv_mass.npy  diboson invariant mass (GeV)

The file-name indices (1, 3) match the key convention in
``diboson.physics.coefficients``: theta_paths[1] / phi_paths[1] for boson 1,
theta_paths[3] / phi_paths[3] for boson 2.

Batching strategy
-----------------
During processing, each flush saves a numbered temp file
(e.g. ``_theta1_batch_0.npy``).  When all events are read, the temp files
are concatenated into the final ``{key}.npy`` and deleted.  This keeps peak
memory during the LHE loop to one batch at a time; the only moment all data
is in memory together is the final concatenation step, which is unavoidable
with the ``.npy`` format.

Typical usage
-------------
from diboson.io.parse_lhe import parse_lhe_file, ZZ_CONFIG, WW_CONFIG

parse_lhe_file("/path/to/events.lhe.gz", ZZ_CONFIG, "/path/to/output")
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pylhe

from diboson.physics.kinematics import (
    lorentz_boost,
    calc_inv_mass,
    calc_scattering_angle,
    calc_decay_angles,
)

from diboson import config
from diboson.event_gen.automate import ZZ_REGIONS, WW_REGIONS

# ---------------------------------------------------------------------------
# Process configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventConfig:
    """
    Identifies the PDG IDs of the two decay products of each boson.

    boson1_daughters : (id_v1, id_v2)
        v1 is the particle whose angles are saved as theta1/phi1.
    boson2_daughters : (id_v3, id_v4)
        v3 is the particle whose angles are saved as theta3/phi3.
    """
    boson1_daughters: tuple[int, int]
    boson2_daughters: tuple[int, int]


# pp -> ZZ -> e+e- mu+mu-
ZZ_CONFIG = EventConfig(
    boson1_daughters=(-11, 11),   # e+, e-
    boson2_daughters=(-13, 13),   # mu+, mu-
)

# pp -> WW -> e+ nu_e  mu- anti-nu_mu
WW_CONFIG = EventConfig(
    boson1_daughters=(-11, 12),   # e+, nu_e
    boson2_daughters=(13, -14),   # mu-, anti-nu_mu
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OUTPUT_KEYS = ("theta1", "phi1", "theta3", "phi3", "cos_psi", "inv_mass")


def _batch_path(output_dir: str, key: str, batch_idx: int) -> str:
    return os.path.join(output_dir, f"_{key}_batch_{batch_idx}.npy")


def _process_and_flush(
    raw: dict[str, list],
    output_dir: str,
    batch_idx: int,
) -> int:
    """Compute kinematics for accumulated raw 4-vectors, flush to a temp batch file, clear raw, return next batch_idx."""
    v1 = np.array(raw["v1"])
    v2 = np.array(raw["v2"])
    v3 = np.array(raw["v3"])
    v4 = np.array(raw["v4"])

    boson1  = v1 + v2
    diboson = boson1 + v3 + v4

    boson1_cm = lorentz_boost(boson1, diboson)
    cos_psi   = calc_scattering_angle(boson1_cm)
    inv_mass  = calc_inv_mass(diboson)
    phi1, phi3, theta1, theta3 = calc_decay_angles(v1, v2, v3, v4)

    results = {
        "theta1": theta1, "phi1": phi1,
        "theta3": theta3, "phi3": phi3,
        "cos_psi": cos_psi, "inv_mass": inv_mass,
    }
    for key, arr in results.items():
        np.save(_batch_path(output_dir, key, batch_idx), arr.astype(float))

    for key in raw:
        raw[key] = []
    return batch_idx + 1


def _finalize(output_dir: str, n_batches: int, append: bool) -> None:
    """
    Concatenate all temp batch files into the final .npy files, then delete
    the temp files.  If ``append`` is True and a final file already exists,
    its contents are prepended before the new batches.
    """
    for key in _OUTPUT_KEYS:
        final_path = os.path.join(output_dir, f"{key}.npy")
        arrays: list[np.ndarray] = []

        if append and os.path.exists(final_path):
            arrays.append(np.load(final_path))

        for i in range(n_batches):
            bp = _batch_path(output_dir, key, i)
            arrays.append(np.load(bp))
            os.remove(bp)

        if arrays:
            np.save(final_path, np.concatenate(arrays))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_lhe_file(
    lhe_file_path: str,
    config: EventConfig,
    output_dir: str,
    *,
    batch_size: int = 100_000,
    append: bool = False,
) -> int:
    """
    Parse a single LHE file, compute kinematic observables, and write them
    to ``output_dir`` as ``.npy`` files.

    Events are processed in batches of ``batch_size`` to limit peak memory.
    Each batch is flushed to a temp file; at the end all batches are
    concatenated into the final output files.

    Parameters
    ----------
    lhe_file_path : str
        Path to the ``.lhe`` or ``.lhe.gz`` file.
    config : EventConfig
        Particle-ID specification for the two bosons (use ``ZZ_CONFIG`` or
        ``WW_CONFIG``, or supply your own).
    output_dir : str
        Directory where output ``.npy`` files are written.
    batch_size : int
        Number of events accumulated in memory before flushing to a temp
        file.  Lower values reduce peak memory; 50 000 is a safe default.
    append : bool
        If ``True``, new events are concatenated onto any existing output
        files rather than overwriting them.

    Returns
    -------
    int
        Number of events successfully processed.
    """
    os.makedirs(output_dir, exist_ok=True)

    v1_id, v2_id = config.boson1_daughters
    v3_id, v4_id = config.boson2_daughters
    all_ids = {v1_id, v2_id, v3_id, v4_id}

    # Accumulate raw 4-vectors; compute kinematics in batch before each flush
    raw: dict[str, list] = {"v1": [], "v2": [], "v3": [], "v4": []}
    n_events = 0
    n_skipped = 0
    batch_idx = 0

    for event in pylhe.read_lhe_with_attributes(lhe_file_path):
        # Collect final-state 4-momenta keyed by PDG id
        momenta: dict[int, list[float]] = {}
        for particle in event.particles:
            if particle.status == 1 and particle.id in all_ids:
                momenta[particle.id] = [
                    particle.e, particle.px, particle.py, particle.pz
                ]

        # Skip incomplete events (should not happen in well-formed LHE files)
        if not all(pid in momenta for pid in all_ids):
            n_skipped += 1
            continue

        raw["v1"].append(momenta[v1_id])
        raw["v2"].append(momenta[v2_id])
        raw["v3"].append(momenta[v3_id])
        raw["v4"].append(momenta[v4_id])
        n_events += 1

        if n_events % batch_size == 0:
            batch_idx = _process_and_flush(raw, output_dir, batch_idx)
            print(f"  {n_events:,} events processed …")

    # Flush any remaining events in the last (partial) batch
    if raw["v1"]:
        batch_idx = _process_and_flush(raw, output_dir, batch_idx)

    _finalize(output_dir, batch_idx, append)

    if n_skipped:
        print(f"  Warning: {n_skipped} incomplete events skipped.")
    print(f"  Done. {n_events:,} events written to {output_dir}")
    return n_events


def parse_lhe_runs(
    events_dir: str,
    config: EventConfig,
    output_dir: str,
    *,
    regions: list | None = None,
    run_start: int = 1,
    run_end: int | None = None,
    batch_size: int = 100_000,
    append: bool = False,
) -> int:
    """
    Process one or more MadGraph ``run_NN`` directories sequentially.

    When ``regions`` is provided each run is saved to its own subdirectory of
    ``output_dir`` named ``cos_psi_<lo>_<hi>_inv_mass_<lo>_<hi>``, matching
    the convention expected by ``region_analysis._region_dir``.  Run
    ``run_start`` maps to ``regions[0]``, ``run_start+1`` to ``regions[1]``,
    and so on.

    When ``regions`` is ``None`` all runs are appended into the same flat
    ``output_dir`` (original behaviour, used for whole-phase-space runs).

    Parameters
    ----------
    events_dir : str
        Base directory containing ``run_01``, ``run_02``, ... sub-directories.
    config : EventConfig
        Process configuration (``ZZ_CONFIG`` or ``WW_CONFIG``).
    output_dir : str
        Root directory where output files are written.
    regions : list or None
        Ordered list of region specs ``[(cos_lo, cos_hi), (mass_lo, mass_hi)]``
        produced by ``automate._build_regions``.  Must cover at least as many
        entries as there are runs to process.
    run_start : int
        First run index to process (inclusive).
    run_end : int or None
        Last run index to process (inclusive).  If ``None``, all
        ``run_*`` directories found in ``events_dir`` are processed.
    batch_size : int
        Events accumulated in memory before each disk flush.
    append : bool
        If ``True`` and ``regions`` is ``None``, new events are concatenated
        onto any pre-existing output files rather than overwriting them.
        Within a single flat-mode call, runs after the first always append
        regardless of this flag.

    Returns
    -------
    int
        Total number of events processed across all runs.
    """
    if run_end is None:
        run_dirs = sorted(glob.glob(os.path.join(events_dir, "run_*")))
        if not run_dirs:
            raise FileNotFoundError(f"No run_* directories found in {events_dir}")
        run_numbers = sorted(
            n for n in (int(os.path.basename(d).split("_", 1)[1]) for d in run_dirs)
            if n >= run_start
        )
    else:
        run_numbers = list(range(run_start, run_end + 1))

    total = 0
    for run_num in run_numbers:
        run_dir = os.path.join(events_dir, f"run_{run_num:02d}")
        lhe_files = glob.glob(os.path.join(run_dir, "*.lhe.gz"))
        if not lhe_files:
            lhe_files = glob.glob(os.path.join(run_dir, "*.lhe"))
        if not lhe_files:
            print(f"Warning: no LHE file found in {run_dir}, skipping.")
            continue

        lhe_path = lhe_files[0]
        print(f"[run {run_num:02d}] {lhe_path}")

        if regions is not None:
            region = regions[run_num - run_start]
            (cos_lo, cos_hi), (mass_lo, mass_hi) = region
            region_name = f"cos_psi_{cos_lo}_{cos_hi}_inv_mass_{mass_lo}_{mass_hi}"
            run_output_dir = os.path.join(output_dir, region_name)
            run_append = False
        else:
            run_output_dir = output_dir
            run_append = append or total > 0

        n = parse_lhe_file(
            lhe_path,
            config,
            run_output_dir,
            batch_size=batch_size,
            append=run_append,
        )
        total += n

    print(f"Grand total: {total:,} events across {len(run_numbers)} run(s).")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    _CONFIGS = {"ZZ": ZZ_CONFIG, "WW": WW_CONFIG}
    _REGIONS = {"ZZ": ZZ_REGIONS, "WW": WW_REGIONS}
    _DEFAULT_EVENTS_DIRS = {
        "ZZ": str(config.ZZ_PROCESS_DIR / "Events"),
        "WW": str(config.WW_PROCESS_DIR / "Events"),
    }
    _DEFAULT_OUTPUT_DIRS = {
        "ZZ": str(config.ZZ_RAW_DIR),
        "WW": str(config.WW_RAW_DIR),
    }

    parser = argparse.ArgumentParser(
        description="Parse MadGraph LHE files and write kinematic observables to .npy files.",
    )
    parser.add_argument(
        "--process", choices=["ZZ", "WW"], required=True,
        help="Diboson process to parse.",
    )
    parser.add_argument(
        "--events-dir", default=None,
        help="Directory containing run_NN sub-directories. "
             "Defaults to the MadGraph process Events/ directory from config.py.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory where .npy output files are written. "
             "Defaults to the raw data directory from config.py.",
    )
    parser.add_argument(
        "--run-start", type=int, default=1,
        help="First run index to process (inclusive). Default: 1.",
    )
    parser.add_argument(
        "--run-end", type=int, default=None,
        help="Last run index to process (inclusive). Default: all runs found.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100_000,
        help="Events accumulated in memory before each disk flush. Default: 100,000.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append new events onto existing output files instead of overwriting. "
             "Only applies in --whole-phase-space mode.",
    )
    parser.add_argument(
        "--whole-phase-space", action="store_true",
        help="Write all runs into a single flat output directory instead of "
             "per-region subdirectories. Use for whole-phase-space validation runs.",
    )

    args = parser.parse_args()

    events_dir = args.events_dir or _DEFAULT_EVENTS_DIRS[args.process]
    output_dir = args.output_dir or _DEFAULT_OUTPUT_DIRS[args.process]
    regions = None if args.whole_phase_space else _REGIONS[args.process]

    print(f"Process  : {args.process}")
    print(f"Events   : {events_dir}")
    print(f"Output   : {output_dir}")
    print(f"Mode     : {'whole-phase-space' if args.whole_phase_space else 'binned'}")

    parse_lhe_runs(
        events_dir=events_dir,
        config=_CONFIGS[args.process],
        output_dir=output_dir,
        regions=regions,
        run_start=args.run_start,
        run_end=args.run_end,
        batch_size=args.batch_size,
        append=args.append,
    )


if __name__ == "__main__":
    main()
