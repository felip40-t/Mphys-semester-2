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

import glob
import os
from dataclasses import dataclass

import numpy as np
import pylhe

from diboson.physics.kinematics import (
    boostinvp,
    calc_inv_mass,
    calc_scattering_angle,
    phistar,
)

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


def _flush_batch(buffers: dict[str, list], output_dir: str, batch_idx: int) -> None:
    """Save each buffer to a numbered temp .npy file and clear the buffer."""
    for key, data in buffers.items():
        if data:
            np.save(_batch_path(output_dir, key, batch_idx), np.asarray(data, dtype=float))
            buffers[key] = []


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
    batch_size: int = 50_000,
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

    buffers: dict[str, list] = {key: [] for key in _OUTPUT_KEYS}
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

        v1 = np.array(momenta[v1_id])
        v2 = np.array(momenta[v2_id])
        v3 = np.array(momenta[v3_id])
        v4 = np.array(momenta[v4_id])

        # Diboson and boson 1 four-momenta
        boson1 = v1 + v2
        diboson = boson1 + v3 + v4

        # Scattering angle: boost boson 1 into the diboson CM frame
        boson1_cm = boostinvp(boson1, diboson)
        cos_psi = calc_scattering_angle(boson1_cm)
        inv_mass = calc_inv_mass(diboson)

        # Decay angles via the full boost + rotation procedure
        phi1, phi3, theta1, theta3 = phistar(v1, v2, v3, v4)

        buffers["theta1"].append(theta1)
        buffers["phi1"].append(phi1)
        buffers["theta3"].append(theta3)
        buffers["phi3"].append(phi3)
        buffers["cos_psi"].append(cos_psi)
        buffers["inv_mass"].append(inv_mass)
        n_events += 1

        if n_events % batch_size == 0:
            _flush_batch(buffers, output_dir, batch_idx)
            batch_idx += 1
            print(f"  {n_events:,} events processed …")

    # Flush any remaining events in the last (partial) batch
    if any(buffers[k] for k in _OUTPUT_KEYS):
        _flush_batch(buffers, output_dir, batch_idx)
        batch_idx += 1

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
    run_start: int = 1,
    run_end: int | None = None,
    batch_size: int = 50_000,
    append: bool = False,
) -> int:
    """
    Process one or more MadGraph ``run_NN`` directories sequentially,
    appending all events into the same output files.

    Parameters
    ----------
    events_dir : str
        Base directory containing ``run_01``, ``run_02``, ... sub-directories.
    config : EventConfig
        Process configuration (``ZZ_CONFIG`` or ``WW_CONFIG``).
    output_dir : str
        Directory where output files are written.
    run_start : int
        First run index to process (inclusive).
    run_end : int or None
        Last run index to process (inclusive).  If ``None``, all
        ``run_*`` directories found in ``events_dir`` are processed.
    batch_size : int
        Events accumulated in memory before each disk flush.
    append : bool
        If ``True``, new events are concatenated onto any pre-existing
        output files in ``output_dir`` rather than overwriting them.
        Within a single call, runs after the first always append
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
            int(os.path.basename(d).split("_", 1)[1]) for d in run_dirs
        )
    else:
        run_numbers = list(range(run_start, run_end + 1))

    total = 0
    for i, run_num in enumerate(run_numbers):
        run_dir = os.path.join(events_dir, f"run_{run_num:02d}")
        lhe_files = glob.glob(os.path.join(run_dir, "*.lhe.gz"))
        if not lhe_files:
            lhe_files = glob.glob(os.path.join(run_dir, "*.lhe"))
        if not lhe_files:
            print(f"Warning: no LHE file found in {run_dir}, skipping.")
            continue

        lhe_path = lhe_files[0]
        print(f"[run {run_num:02d}] {lhe_path}")
        n = parse_lhe_file(
            lhe_path,
            config,
            output_dir,
            batch_size=batch_size,
            append=(append or total > 0),
        )
        total += n

    print(f"Grand total: {total:,} events across {len(run_numbers)} run(s).")
    return total
