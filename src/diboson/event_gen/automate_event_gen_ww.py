
import re
import subprocess

from config import MG5_INSTALL_DIR

process_dir = MG5_INSTALL_DIR / "pp_WW_SM"
fortran_dummy_fct = process_dir / "SubProcesses" / "dummy_fct.f"

# Define regions of phase space for (0.0 to 1.0) and (200.0 to 900.0)
regions = [
    [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
    for cos_min in [0.0 + 0.1 * i for i in range(9)]
    for mass_min in [200.0 + 50.0 * j for j in range(14)]
]

regions += [
    [(0.9, 0.95), (mass_min, mass_min + 50.0)]
    for mass_min in [200.0 + 50.0 * i for i in range(14)]
]


def modify_fortran_file(file_path, limits):
    with open(file_path, 'r') as file:
        content = file.read()

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
    with open(file_path, 'w') as file:
        file.write(new_content)
    print(f"File {file_path} successfully updated with limits {limits}")


for region in regions:
    modify_fortran_file(fortran_dummy_fct, region)
    subprocess.run(
        [process_dir / 'bin' / 'generate_events'],
        input='0\n0\n', text=True,
        cwd=process_dir, check=True
    )
    print(f'Process for region {region} finished')
