"""Expand the non-dispatchable EGRM scientific design into exact cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import load_contract


SCHEMA_VERSION = "egrm-scientific-design-inventory-v1"


def expand_scientific_design(contract_path: str | Path) -> dict[str, Any]:
    """Return a deterministic cell inventory without executing any experiment."""

    contract, contract_hash = load_contract(contract_path)
    if contract["status"] != "design_draft_not_run":
        raise ValueError("scientific design expansion accepts design drafts only")

    arm_status = {arm["name"]: arm["implementation_status"] for arm in contract["arms"]}
    cells: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    for stage in contract["fixture"]["stages"]:
        stage_name = stage["name"]
        for unit in stage["unit_values"]:
            for arm in stage["arms"]:
                unit_text = str(unit)
                cell_id = (
                    f"{contract['study_id']}:{stage_name}:"
                    f"{stage['unit_axis']}={unit_text}:arm={arm}"
                )
                cells.append(
                    {
                        "cell_id": cell_id,
                        "stage": stage_name,
                        "unit_axis": stage["unit_axis"],
                        "unit_value": unit,
                        "arm": arm,
                        "arm_implementation_status": arm_status[arm],
                        "model": stage["model"],
                    }
                )
        observed = sum(cell["stage"] == stage_name for cell in cells)
        if observed != stage["registered_cells"]:
            raise ValueError(f"expanded cell count drifted for stage {stage_name}")
        stage_counts[stage_name] = observed

    cell_ids = [cell["cell_id"] for cell in cells]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("expanded scientific cell IDs are not unique")
    if len(cells) != contract["denominator"]["registered_cells"]:
        raise ValueError("expanded scientific denominator does not match the contract")

    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": contract["study_id"],
        "contract_hash": contract_hash,
        "contract_status": contract["status"],
        "scientific_evidence": False,
        "dispatchable": False,
        "budget_authorization": contract["fixture"]["budget_authorization"],
        "registered_cells": len(cells),
        "stage_counts": stage_counts,
        "cells": cells,
    }


def write_design_inventory(
    inventory: Mapping[str, Any], output_path: str | Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
