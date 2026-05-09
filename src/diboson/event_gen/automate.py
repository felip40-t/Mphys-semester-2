import re
import subprocess
import argparse

from config import ZZ_PROCESS_DIR, WW_PROCESS_DIR, N_COS_BINS, ZZ_N_MASS_BINS, WW_N_MASS_BINS, MASS_BIN_MIN, COS_BIN_MIN, COM_ENERGY, NEVENTS


def _build_regions(n_cos, n_mass):
    return [
        [(COS_BIN_MIN + 0.1 * i, COS_BIN_MIN + 0.1 * i + 0.1), (MASS_BIN_MIN + 50.0 * j, MASS_BIN_MIN + 50.0 * j + 50.0)]
        for i in range(n_cos)
        for j in range(n_mass)
    ]


ZZ_REGIONS = _build_regions(N_COS_BINS, ZZ_N_MASS_BINS)

WW_REGIONS = _build_regions(N_COS_BINS, WW_N_MASS_BINS)

PROCESS_CONFIGS = {
    "ZZ": (ZZ_PROCESS_DIR, ZZ_REGIONS),
    "WW": (WW_PROCESS_DIR, WW_REGIONS),
}


_VANILLA_PATTERN = "      dummy_cuts=.true."

_DUMMY_CUTS_BODY = """\
c
c     Local variables for phase-space cuts
c
      real*8 PV1(0:3), PV2(0:3), PVV(0:3), V1b(0:3)
      real*8 M_squared, cos_psi, rmboost, aux, aaux

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

c     Invariant mass of VV system (GeV) -- also used as boost denominator
      rmboost = dsqrt(max(PVV(0)**2 - PVV(1)**2 - PVV(2)**2
     &               - PVV(3)**2, 0.0d0))
      M_squared = rmboost

c     Boost V1 into VV centre-of-mass frame (boostinvp algorithm)
      aux = (PV1(0)*PVV(0) - PV1(1)*PVV(1) - PV1(2)*PVV(2)
     &       - PV1(3)*PVV(3)) / rmboost
      aaux = (aux + PV1(0)) / (PVV(0) + rmboost)
      V1b(0) = aux
      V1b(1) = PV1(1) - aaux * PVV(1)
      V1b(2) = PV1(2) - aaux * PVV(2)
      V1b(3) = PV1(3) - aaux * PVV(3)

c     cos(scattering angle) of V1 with beam (z) axis
      cos_psi = V1b(3) / dsqrt(V1b(1)**2 + V1b(2)**2 + V1b(3)**2)

      if (M_squared.gt.(0.0d0) .and. M_squared.lt.(9999.0d0) .and.
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


def modify_fortran_file(file_path, limits):
    with open(file_path, 'r') as f:
        content = f.read()

    pattern = (
        r"^\s+if \(M_squared\.gt\.\([^)]*\) \.and\. M_squared\.lt\.\([^)]*\) .and.\n"
        r"\s+&\s*cos_psi\.lt\.\([^)]*\) \.and\. cos_psi\.gt\.\([^)]*\)\) then"
    )
    replacement = (
        f"      if (M_squared.gt.({limits[1][0]}d0) .and. M_squared.lt.({limits[1][1]}d0) .and.\n"
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


def configure_run_card(process_dir):
    run_card = process_dir / "Cards" / "run_card.dat"
    with open(run_card, 'r') as f:
        content = f.read()

    beam_energy = COM_ENERGY / 2

    content = re.sub(
        r'(\s*)\S+(\s*=\s*nevents\b)',
        lambda m: f'{m.group(1)}{NEVENTS}{m.group(2)}',
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
    print(f"Configured {run_card}: nevents={NEVENTS}, ebeam1=ebeam2={beam_energy} GeV")


def generate_events(process_name):
    process_dir, regions = PROCESS_CONFIGS[process_name]
    configure_run_card(process_dir)
    fortran_dummy_fct = process_dir / "SubProcesses" / "dummy_fct.f"
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
    parser.add_argument("--process", 
                choices=["ZZ", "WW"], 
                help="Process to generate events for (ZZ or WW)",
                required=True)
    args = parser.parse_args()
    generate_events(args.process)
