import re
import subprocess
import argparse

from diboson.config import ZZ_PROCESS_DIR, WW_PROCESS_DIR, COM_ENERGY, NEVENTS as _DEFAULT_NEVENTS
from diboson.main import _PROCESS_CONFIGS
from diboson.analysis.region_analysis import ProcessSpec

ZZ_REGIONS = _PROCESS_CONFIGS["ZZ"][0].build_regions()
WW_REGIONS = _PROCESS_CONFIGS["WW"][0].build_regions()

PROCESS_CONFIGS = {
    "ZZ": (ZZ_PROCESS_DIR, ZZ_REGIONS),
    "WW": (WW_PROCESS_DIR, WW_REGIONS),
}


_VANILLA_PATTERN = "      dummy_cuts=.true."
_VANILLA_BODY    = "      dummy_cuts=.true.\n\n      return\n      end"

# Matches the inserted cuts body (from the opening comment to the closing `end`),
# tolerating any values written by modify_fortran_file into the if-condition.
_CUTS_BODY_RE = re.compile(
    r"c\nc     Local variables for phase-space cuts\n.*?\n      end$",
    re.DOTALL | re.MULTILINE,
)

_DUMMY_CUTS_BODY = """\
c
c     Local variables for phase-space cuts
c
      real*8 PV1(0:3), PV2(0:3), PVV(0:3), V1b(0:3)
      real*8 inv_mass, cos_psi, e1_cm, boost_coeff

c     Reconstruct boson 4-momenta: V1 from particles 3+4, V2 from 5+6
      PV1(0) = P(0,3) + P(0,4)
      PV1(1) = P(1,3) + P(1,4)
      PV1(2) = P(2,3) + P(2,4)
      PV1(3) = P(3,3) + P(3,4)
      PV2(0) = P(0,5) + P(0,6)
      PV2(1) = P(1,5) + P(1,6)
      PV2(2) = P(2,5) + P(2,6)
      PV2(3) = P(3,5) + P(3,6)
      PVV(0) = PV1(0) + PV2(0)
      PVV(1) = PV1(1) + PV2(1)
      PVV(2) = PV1(2) + PV2(2)
      PVV(3) = PV1(3) + PV2(3)

c     Invariant mass of VV system (GeV)
      inv_mass = dsqrt(max(PVV(0)**2 - PVV(1)**2 - PVV(2)**2
     &               - PVV(3)**2, 0.0d0))

c     Boost V1 into VV centre-of-mass frame
c     Calculate energy of V1 in CM frame
      e1_cm = (PV1(0)*PVV(0) - PV1(1)*PVV(1) - PV1(2)*PVV(2)
     &       - PV1(3)*PVV(3)) / inv_mass
      boost_coeff = (e1_cm + PV1(0)) / (PVV(0) + inv_mass)
      V1b(0) = e1_cm
      V1b(1) = PV1(1) - boost_coeff * PVV(1)
      V1b(2) = PV1(2) - boost_coeff * PVV(2)
      V1b(3) = PV1(3) - boost_coeff * PVV(3)

c     cos(scattering angle) of V1 with beam (z) axis
      cos_psi = V1b(3) / dsqrt(V1b(1)**2 + V1b(2)**2 + V1b(3)**2)

      if (inv_mass.gt.(0.0d0) .and. inv_mass.lt.(9999.0d0) .and.
     &   cos_psi.lt.(1.0d0) .and. cos_psi.gt.(-1.0d0)) then
          dummy_cuts = .true.
      else
          dummy_cuts = .false.
      end if

      return
      end"""


def initialise_fortran_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    if _VANILLA_PATTERN not in content:
        return
    new_content = content.replace(
        _VANILLA_PATTERN + "\n\n      return\n      end",
        _DUMMY_CUTS_BODY,
        1,
    )
    with open(file_path, 'w') as f:
        f.write(new_content)
    print(f"Initialised {file_path} with phase-space cut body")


def reset_fortran_file(file_path):
    """Restore dummy_fct.f to the vanilla MadGraph state (dummy_cuts=.true., no body)."""
    with open(file_path, 'r') as f:
        content = f.read()
    if _VANILLA_BODY in content:
        return  # already vanilla
    new_content, count = _CUTS_BODY_RE.subn(_VANILLA_BODY, content)
    if count != 1:
        raise RuntimeError(f"Could not reset {file_path}: expected 1 match, got {count}")
    with open(file_path, 'w') as f:
        f.write(new_content)
    print(f"Reset {file_path} to vanilla state")


def modify_fortran_file(file_path, limits):
    with open(file_path, 'r') as f:
        content = f.read()

    pattern = (
        r"^\s+if \(inv_mass\.gt\.\([^)]*\) \.and\. inv_mass\.lt\.\([^)]*\) .and.\n"
        r"\s+&\s*cos_psi\.lt\.\([^)]*\) \.and\. cos_psi\.gt\.\([^)]*\)\) then"
    )
    replacement = (
        f"      if (inv_mass.gt.({limits[1][0]}d0) .and. inv_mass.lt.({limits[1][1]}d0) .and.\n"
        f"     &   cos_psi.lt.({limits[0][1]}d0) .and. cos_psi.gt.({limits[0][0]}d0)) then"
    )

    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 substitution in {file_path}, got {count}. Region: {limits}"
        )
    with open(file_path, 'w') as f:
        f.write(new_content)
    print(f"Updated {file_path} with limits {limits}")


def configure_run_card(process_dir, nevents=None):
    run_card = process_dir / "Cards" / "run_card.dat"
    with open(run_card, 'r') as f:
        content = f.read()

    n = nevents if nevents is not None else _DEFAULT_NEVENTS
    beam_energy = COM_ENERGY / 2

    content = re.sub(
        r'(\s*)\S+(\s*=\s*nevents\b)',
        lambda m: f'{m.group(1)}{n}{m.group(2)}',
        content,
    )
    content = re.sub(
        r'(\s*)\S+(\s*=\s*ebeam1\b)',
        lambda m: f'{m.group(1)}{beam_energy}{m.group(2)}',
        content,
    )
    content = re.sub(
        r'(\s*)\S+(\s*=\s*ebeam2\b)',
        lambda m: f'{m.group(1)}{beam_energy}{m.group(2)}',
        content,
    )

    with open(run_card, 'w') as f:
        f.write(content)
    print(f"Configured {run_card}: nevents={n}, ebeam1=ebeam2={beam_energy} GeV")


def generate_events(process_name, whole_phase_space=False, nevents=None):
    process_dir, regions = PROCESS_CONFIGS[process_name]
    configure_run_card(process_dir, nevents=nevents)
    fortran_dummy_fct = process_dir / "SubProcesses" / "dummy_fct.f"

    if whole_phase_space:
        reset_fortran_file(fortran_dummy_fct)
        subprocess.run(
            [process_dir / "bin" / "generate_events"],
            input="0\n0\n", text=True,
            cwd=process_dir, check=True,
        )
        print("Whole-phase-space run finished")
    else:
        initialise_fortran_file(fortran_dummy_fct)
        for region in regions:
            modify_fortran_file(fortran_dummy_fct, region)
            subprocess.run(
                [process_dir / "bin" / "generate_events"],
                input="0\n0\n", text=True,
                cwd=process_dir, check=True,
            )
            print(f"Region {region} finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MadGraph5 events for a diboson process.")
    parser.add_argument("--process", choices=["ZZ", "WW"], required=True,
                        help="Process to generate events for (ZZ or WW).")
    parser.add_argument("--whole-phase-space", action="store_true",
                        help="Generate one unbiased run over the full phase space "
                             "(dummy_fct.f has no cuts body). Default: binned mode.")
    parser.add_argument("--nevents", type=int, default=None,
                        help=f"Number of events per run. Default: NEVENTS from config.py ({_DEFAULT_NEVENTS}).")
    args = parser.parse_args()
    generate_events(args.process, whole_phase_space=args.whole_phase_space, nevents=args.nevents)
