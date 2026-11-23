"""Markdown tables for docs/f1_study.md from round3_report.py outputs.

round3_tables.py "Released=r3_release" "New, balanced=r3_final+r3_det" ...
"""

import json
import sys

from common import OUT, TEST_SOURCES, macro

NAMES = {
    "smoking|smoking_site2": "Smoking: second CCTV estate, unseen cameras",
    "smoking|xrf55": "Smoking: indoor phoning/drinking/eating clips",
    "smoking|avenue": "Smoking: Avenue walkway",
    "smoking|meva": "Smoking: MEVA cameras",
    "smoking|virat": "Smoking: VIRAT car parks",
    "littering|littering_campus": "Littering: campus clips, unseen views",
    "littering|avenue": "Littering: Avenue walkway",
    "littering|meva": "Littering: MEVA cameras (unlabelled)",
    "littering|virat": "Littering: VIRAT car parks (unlabelled)",
    "loitering|avenue": "Loitering: Avenue walkway",
    "loitering|virat": "Loitering: VIRAT car parks (unlabelled)",
    "rough_sleeping|avenue": "Sleeping: Avenue walkway",
    "rough_sleeping|meva": "Sleeping: MEVA cameras",
    "rough_sleeping|virat": "Sleeping: VIRAT car parks",
    "unattended_items|avenue": "Unattended: Avenue walkway",
    "unattended_items|virat": "Unattended: VIRAT car parks",
}


def main():
    specs = dict(item.split("=", 1) for item in sys.argv[1:])
    data = {
        label: json.loads((OUT / "runs" / f"{tag}_site.json").read_text())
        for label, tag in specs.items()
    }
    labels = list(data)
    print(
        "| Held-out evidence | n (positive) | "
        + " | ".join(f"{k}: F1 / false alarms" for k in labels)
        + " | "
        + " | ".join(f"{k}: AP" for k in labels[:2])
        + " |"
    )
    print("|" + "---|" * (2 + len(labels) + 2))
    for name in TEST_SOURCES.values():
        first = data[labels[0]]["benchmark"][name]
        cells = [
            f"{data[k]['benchmark'][name]['f1']:.2f} / {data[k]['benchmark'][name]['fpr']:.0%}"
            for k in labels
        ]
        aps = [f"{data[k]['benchmark'][name]['ap']:.2f}" for k in labels[:2]]
        print(
            f"| {name} | {first['n']} ({first['positive']}) | "
            + " | ".join(cells + aps)
            + " |"
        )
    print(
        "| **Macro F1 / macro AP** | | "
        + " | ".join(f"**{macro(data[k]['benchmark']):.3f}**" for k in labels)
        + " | "
        + " | ".join(f"**{macro(data[k]['benchmark'], 'ap'):.3f}**" for k in labels[:2])
        + " |"
    )
    print()
    print("| New-site evidence (shift side) | Windows | " + " | ".join(labels) + " |")
    print("|" + "---|" * (2 + len(labels)))
    for key, name in NAMES.items():
        first = data[labels[0]]["site"][key]
        kind = "recall" if first["positive"] else "fpr"
        cells = [f"{data[k]['site'][key][kind]:.0%}" for k in labels]
        what = (
            "smokers found"
            if key == "smoking|smoking_site2"
            else "found"
            if first["positive"]
            else "flagged"
        )
        print(f"| {name} — {what} | {first['n']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
