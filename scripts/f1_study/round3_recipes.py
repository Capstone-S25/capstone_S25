"""Representation variants for round3.py (--variant)."""


def nodet(task, chosen):
    """Smoking heads without the detector features: the detector becomes its
    own channel instead of three numbers appended to a 2,000-number vector."""
    if task != "smoking":
        return chosen
    out = {}
    for member, recipes in chosen.items():
        seen, kept = [], []
        for name, recipe in recipes:
            plain = {k: v for k, v in recipe.items() if k != "smoke_detector"}
            if plain not in seen:
                seen.append(plain)
                kept.append((name.replace("/smokedet", "") + "/nodet", plain))
        out[member] = kept
    return out


VARIANTS = {"nodet": nodet}
