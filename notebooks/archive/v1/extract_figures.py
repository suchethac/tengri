"""Extract all PNG figures from executed Jupyter notebooks into notebook_figures/."""
import json
import base64
import os
import sys

OUT_DIR = "notebook_figures"
os.makedirs(OUT_DIR, exist_ok=True)

notebooks = sorted(f for f in os.listdir(".") if f.endswith(".ipynb")
                   and not f.startswith("00_quickstart_executed"))

for nb_path in notebooks:
    nb_name = nb_path.replace(".ipynb", "")
    with open(nb_path) as f:
        nb = json.load(f)

    fig_idx = 0
    has_outputs = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            has_outputs = True
            if out.get("output_type") == "display_data":
                img_data = out.get("data", {}).get("image/png")
                if img_data:
                    fig_idx += 1
                    fname = f"{nb_name}_fig{fig_idx:02d}.png"
                    fpath = os.path.join(OUT_DIR, fname)
                    with open(fpath, "wb") as img_f:
                        img_f.write(base64.b64decode(img_data))

    status = "EXECUTED" if has_outputs else "NOT EXECUTED"
    print(f"{nb_name}: {status}, {fig_idx} figures extracted")

print(f"\nTotal files in {OUT_DIR}/: {len(os.listdir(OUT_DIR))}")
