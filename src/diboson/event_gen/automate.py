import re
import subprocess
import argparse

from config import ZZ_PROCESS_DIR, WW_PROCESS_DIR


def _build_regions(n_cos, n_mass):
    return [
        [(0.1 * i, 0.1 * i + 0.1), (200.0 + 50.0 * j, 250.0 + 50.0 * j)]
        for i in range(n_cos)
        for j in range(n_mass)
    ]


ZZ_REGIONS = _build_regions(10, 18)

WW_REGIONS = _build_regions(10, 18)

PROCESS_CONFIGS = {
    "ZZ": (ZZ_PROCESS_DIR, ZZ_REGIONS),
    "WW": (WW_PROCESS_DIR, WW_REGIONS),
}


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


def generate_events(process_name):
    process_dir, regions = PROCESS_CONFIGS[process_name]
    fortran_dummy_fct = process_dir / "SubProcesses" / "dummy_fct.f"
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
