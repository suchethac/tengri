"""Add plt.savefig() before every plt.show() in notebook .py files.

Saves to notebook_figures/<notebook_name>_fig<NN>.png
"""
import re
import os

FIGDIR = "notebook_figures"
os.makedirs(FIGDIR, exist_ok=True)

notebooks = sorted(f for f in os.listdir(".") if f.endswith(".py")
                   and f[0].isdigit() and not f.startswith("extract")
                   and not f.startswith("add_"))

for nb_path in notebooks:
    nb_name = nb_path.replace(".py", "")
    with open(nb_path) as f:
        content = f.read()

    fig_idx = 0
    lines = content.split("\n")
    new_lines = []

    # Add mkdir at the top of the first code cell
    added_mkdir = False

    for line in lines:
        stripped = line.rstrip()

        # Insert os.makedirs once after setup_style() or first import
        if not added_mkdir and "setup_style()" in stripped:
            new_lines.append(line)
            new_lines.append(f'import os; os.makedirs("{FIGDIR}", exist_ok=True)')
            added_mkdir = True
            continue

        # Before plt.show(), insert savefig
        if stripped == "plt.show()":
            fig_idx += 1
            indent = len(line) - len(line.lstrip())
            pad = " " * indent
            fname = f"{nb_name}_fig{fig_idx:02d}.png"
            new_lines.append(
                f'{pad}plt.savefig("{FIGDIR}/{fname}", dpi=150, bbox_inches="tight")'
            )
            new_lines.append(line)
        else:
            new_lines.append(line)

    with open(nb_path, "w") as f:
        f.write("\n".join(new_lines))

    print(f"{nb_name}: {fig_idx} savefig calls added")
