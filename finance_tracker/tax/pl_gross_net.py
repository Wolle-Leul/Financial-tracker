"""
Polish employment / B2B gross-to-net approximations (illustrative only).

Not tax/legal advice. Use for planning; allow users to override net in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetEstimate:
    net: float
    notes: str


def estimate_net_from_gross(gross: float, contract_type: str) -> NetEstimate:
    """
    Rough net from gross monthly amounts (PLN).

    - employment_pl: UoP-style — assume ~32% total deductions (social + health + PIT ballpark).
    - b2b_pl: simplified flat effective rate ~25% (varies wildly with ZUS variant).
    - other: midpoint between employment and B2B.
    """
    g = float(gross)
    if g <= 0:
        return NetEstimate(net=0.0, notes="Gross must be positive.")

    ct = (contract_type or "other").lower().replace("-", "_").replace(" ", "_")

    if ct in ("employment_pl", "uop", "umowa_o_prace"):
        eff = 0.68
        notes = "Approx. UoP: ~32% combined deductions (social/health/PIT); verify with payroll."
    elif ct in ("b2b_pl", "b2b", "dzialalnosc"):
        eff = 0.75
        notes = "Approx. B2B: ~25% effective (varies with ZUS/flat tax); verify with accountant."
    else:
        eff = 0.71
        notes = "Generic blend: ~29% effective; use manual net entry if you know exact net."

    return NetEstimate(net=round(g * eff, 2), notes=notes)
