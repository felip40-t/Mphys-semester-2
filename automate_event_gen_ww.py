
import re 
import os
import subprocess

mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "pp_WW_SM")

# Path to Fortran source file
fortran_dummy_fct = os.path.join(process_dir, "SubProcesses", "dummy_fct.f")

# Define regions of phase space for (0.0 to 1.0) and (200.0 to 1000.0)
regions = [
    [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
    for cos_min in [0.0 + 0.1 * i for i in range(9)]
    for mass_min in [200.0 + 50.0 * j for j in range(16)]
]

regions += [
    [(0.9, 0.95), (mass_min, mass_min + 50.0)]
    for mass_min in [200.0 + 50.0 * i for i in range(16)]
]


# Function to modify Fortran file
def modify_fortran_file(file_path, limits):
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Prepare the regex pattern.
    # This pattern assumes the original statement is formatted as in the Vim substitution.
    # Adjust the whitespace or pattern if needed.
    pattern = (
        r"^\s+if \(M_squared\.gt\.\([^)]*\) \.and\. M_squared\.lt\.\([^)]*\) .and.\n"
        r"\s+&\s*cos_psi\.lt\.\([^)]*\) \.and\. cos_psi\.gt\.\([^)]*\)\) then"
    )

    # Build the replacement string using the provided limits
    replacement = (
        f"      if (M_squared.gt.({limits[1][0]}d0) .and. M_squared.lt.({limits[1][1]}d0) .and.\n"
        f"     &   cos_psi.lt.({limits[0][1]}d0) .and. cos_psi.gt.({limits[0][0]}d0)) then"
    )

    # Perform the substitution; re.DOTALL makes '.' match newline characters.
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL | re.MULTILINE)

    if count == 0:
        print("Pattern not found. No changes made.")
    else:
        # Write the file back
        with open(file_path, 'w') as file:
            file.write(new_content)
        print(f"File {file_path} successfully updated with limits {limits}")


# Modify the Fortran file for each region and run the process
for region in regions:
    modify_fortran_file(fortran_dummy_fct, region)
    # Run the MadGraph process using subprocess, streamlined
    subprocess.run(
        [os.path.join(process_dir, 'bin', 'generate_events')],
        input='0\n0\n', text=True,
        cwd=process_dir, check=True
    )
    print(f'Process for region {region} finished')