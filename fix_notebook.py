"""Fix notebook kernelspec and cell source bugs."""
import json
from pathlib import Path

nb_path = Path("notebooks/CodeAlpha_SalesPrediction.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# 1. Fix kernelspec to Python 3.14
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3.14",
    "language": "python",
    "name": "python3"
}
nb["metadata"]["language_info"] = {
    "name": "python",
    "version": "3.14.6"
}

# 2. Fix the broken string concat in the feature importance cell (cell id "sec7_eval")
for cell in nb["cells"]:
    if cell.get("id") == "sec7_eval":
        # Find and fix the broken line that merged two source strings
        fixed_source = []
        for line in cell["source"]:
            # The bug: "        )\\nelse:\\n" should be two separate entries
            if '        )\\nelse:\\n' in line:
                idx = line.index('        )\\nelse:\\n')
                before = line[:idx + len('        )\\n')]
                after = '        )\\n'
                # Split into proper separate lines
                fixed_source.append(line.replace('        )\\nelse:\\n', '        )\\n'))
                fixed_source.append('else:\\n')
                continue
            fixed_source.append(line)
        cell["source"] = fixed_source
        print("Fixed sec7_eval cell source")
        break

# Write both locations
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
root_copy = Path("CodeAlpha_SalesPrediction.ipynb")
root_copy.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Saved {nb_path} and {root_copy}")
print("Kernelspec:", nb["metadata"]["kernelspec"])
