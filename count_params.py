import re
import sys

def count_paramdeclarations(file_path):
    """Count ParamDeclaration instances in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        # Count lines that start with "ParamDeclaration(" 
        count = len(re.findall(r'^\s*ParamDeclaration\(', content, re.MULTILINE))
        return count
    except:
        return 0

files = {
    "radio._params.PARAMS": "src/tengri/components/radio/_params.py",
    "xray._params.PARAMS": "src/tengri/components/xray/_params.py",
    "agn._params.PARAMS": "src/tengri/components/agn/_params.py",
    "nebular._params.PARAMS": "src/tengri/components/nebular/_params.py",
    "nebular._params.CB19_PARAMS": "src/tengri/components/nebular/_params.py",
    "nebular._params.ELINE_PARAMS": "src/tengri/components/nebular/_params.py",
    "nebular._params.ELINE_BROAD_PARAMS": "src/tengri/components/nebular/_params.py",
    "nebular._params.CUE_IONSPEC_PARAMS": "src/tengri/components/nebular/_params.py",
    "nebular._params.CUE_GAS_EXTRA_PARAMS": "src/tengri/components/nebular/_params.py",
    "nebular._params.SHOCK_PARAMS": "src/tengri/components/nebular/_params.py",
    "dust._params.PARAMS": "src/tengri/components/dust/_params.py",
    "dust._params.ATTENUATION_PARAMS": "src/tengri/components/dust/_params.py",
    "dust._params.SINGLE_COMPONENT_PARAMS": "src/tengri/components/dust/_params.py",
    "igm._params.PATCHY_PARAMS": "src/tengri/components/igm/_params.py",
    "igm._params.DLA_PARAMS": "src/tengri/components/igm/_params.py",
    "stellar._params.ALPHA_FE_PARAMS": "src/tengri/components/stellar/_params.py",
    "stellar._params.EVOLVING_ALPHA_PARAMS": "src/tengri/components/stellar/_params.py",
    "_shared.PARAMS": "src/tengri/parameters/_shared.py",
}

total = 0
for source, path in files.items():
    count = count_paramdeclarations(path)
    if count > 0:
        total += count
        print(f"{source:45} {count:3} params")

# AGN extras
print(f"{'agn._params extras (neb_xid)':45}   1 param (in _AGN_EXTRAS)")
total += 1

print(f"\n{'TOTAL across all sources':45} {total:3} params")
