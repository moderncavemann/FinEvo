"""Outcome-blind, zero-provider authority and lineage import for V2.11.4.

V2.11.2 is an immutable terminal ``complete-with-no-go`` release.  Its two
fresh 2x12 long-context preflights completed and produced one valid global
post-gate authority.  The subsequent scientific cells did not reach provider
construction because the generic observed-p95 consumer lacked the V2.11.2
schema adapter.  Five deterministic candidate-admission cells completed, but
their outputs are deliberately excluded here.

This module exposes only three reusable, non-effect authorities:

* the frozen q-ref / Stage-0 calibration wrapper;
* the two capability/interface wrappers; and
* the two-model, 24-action/8-semantic observed-p95 dispatch reservations.

The complete V2.11.2 authority source and the terminal V2.11.3 zero-call no-go
lineage are audited before any child authority is emitted.  V2.11.3 contributes
only its terminal denominator and cumulative ledger debit: it is never treated
as a reusable authority or scientific result.  No A-D or cross-model artifact
is returned, no provider module is imported, and no provider client can be
constructed here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping

from .pilot_budget import ParentBudgetDebit
from .pilot_contract import (
    PilotContract,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
)
from .pilot_v2112_gate import (
    PilotV2112GateError,
    build_v2112_post_gate_authority,
    verified_v2112_gate_authority_binding,
    verify_v2112_gate_receipt,
)
from .pilot_v2112_parent_import import (
    PilotV2112ParentImportError,
    V2112_CALIBRATION_WRAPPER_SCHEMA_VERSION,
    V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION,
    calibration_wrapper_from_v2112_receipt,
    capability_wrappers_from_v2112_receipt,
    verified_v2112_inherited_capability_binding,
    verify_v2112_parent_import_receipt,
)
from .pilot_v27_stage0_import import (
    PilotV27Stage0ImportError,
    _atomic_exact_bytes_no_follow,
)
from .pilot_v29_stage0_import import _inventory


V2114_CONTRACT_ID = "finevo-pilot-v2.11.4"
V2114_SCIENCE_TAG = "pilot-v2.11.4-science"
V2114_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_4.yaml")
V2114_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.4/raw")
V2114_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_4_source_manifest.json"
)
V2114_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.4-source-manifest-v1"
V2114_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.11.4-parent-import-v1"
V2114_CALIBRATION_WRAPPER_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.4-imported-calibration-wrapper-v1"
)
V2114_CAPABILITY_WRAPPER_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.4-imported-capability-wrapper-v1"
)
V2114_PREFLIGHT_WRAPPER_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.4-imported-preflight-authority-v1"
)
V2114_DEFAULT_RECEIPT_PATH = V2114_RAW_ROOT / "parent-import/parent_import_receipt.json"

# Filled after the deterministic source manifest is rendered.  Keeping these
# as constants makes the child contract, import receipt, and runtime verifier
# all agree on one tracked file rather than trusting a caller-provided object.
V2114_SOURCE_MANIFEST_FILE_SHA256: str | None = (
    "fd37e5f7a6cfa0178fa0baec74fb0d18f058a361586296d50d4bcf611e13839d"
)
V2114_SOURCE_MANIFEST_CONTENT_SHA256: str | None = (
    "594b1a00910a1dbecd5e36fcac4397df5341e92b5a9802ce4ca781434b747760"
)

# Immutable V2.11.2 scientific release.
V2112_CONTRACT_ID = "finevo-pilot-v2.11.2"
V2112_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_2.yaml")
V2112_CONTRACT_FILE_SHA256 = (
    "130c890f7e6d5d61137b4aa189cfbcca39b8cd7aab455cc2a6f35aeddd8ee3a8"
)
V2112_CONTRACT_SHA256 = (
    "c04f7d4c5ae0962a4a64b0ac543d890a1475b6f184f516534eeb8ff026505a37"
)
V2112_SCIENCE_TAG = "pilot-v2.11.2-science"
V2112_SCIENCE_TAG_OBJECT = "1b9d9f163934e946255ec19aeebe2f121fba4cc3"
V2112_SCIENCE_COMMIT = "78870956b528946d415a9be5f5769b0893d16d74"
V2112_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.2/raw")
V2112_RAW_FILE_COUNT = 205
V2112_RAW_STORAGE_BYTES = 4_526_996
V2112_RAW_INVENTORY_SHA256 = (
    "e72bbfcba8ac7e9ba8daead6781cef30957186a0597effedb96aba1faf827473"
)
V2112_RUN_LEDGER_FILE_SHA256 = (
    "d8ec96c5aba434368b3d97a1bcc0d6a0116c62325161eae6ce13c9c581198625"
)
V2112_RUN_LEDGER_SHA256 = (
    "686d7f528268e0d9d6ac97ae27d483af9c2eb93be53bd329b4fd621c0ec2ae25"
)
V2112_RUN_LEDGER_EVENT_COUNT = 138
V2112_RUN_LEDGER_EVENT_HEAD = (
    "17d8cd912f96377ddf86faeb30e4a0f63f588af6479b46040d80b33c70b40b17"
)
V2112_BUDGET_LEDGER_FILE_SHA256 = (
    "0461fcbb6d4ca01677aa55410c28296e7aba6f2deff8cd044c459981e39c3ecf"
)
V2112_BUDGET_LEDGER_SHA256 = (
    "36dd9c62a56c7e87bb647feebeaa7f8d03b0a410d3c7d163834d5029f8da868b"
)
V2112_BUDGET_LEDGER_EVENT_COUNT = 174
V2112_BUDGET_LEDGER_EVENT_HEAD = (
    "e9d32d048db50bf861514498ec5831d4c5aee01c2357e6fb608681c716e032b6"
)
V2112_RELEASE_ATTESTATION_FILE_SHA256 = (
    "fa37dec1a2cf0ad5c7b6cebb3f022bc63640ab45b584e7e2b6b5a09a8867d589"
)
V2112_RELEASE_ATTESTATION_SHA256 = (
    "909b6c74e5fc62ff3dade7501107c0da21fc210f5d6b3ddb7098ddfdc1543358"
)
V2112_LAUNCH_INPUT_FILE_SHA256 = (
    "23211b9c539660dfa3e03ff26cc6fbaab0c5b3eb187ea18a6345868318add4c0"
)
V2112_LAUNCH_INPUT_SHA256 = (
    "071f22594890f14307c3b860403fb9ca453fdc600ed120f6b92ec9def7b5c80f"
)
V2112_PARENT_IMPORT_FILE_SHA256 = (
    "e73d12c47d8d773a732bd4d55679f46f2c0c69e510e3b4d815455f2208fa066d"
)
V2112_PARENT_IMPORT_CONTENT_SHA256 = (
    "7526789523225235b73808af13053b68bb7d39d278883f98c49dbe0394b57b19"
)
V2112_POST_GATE_FILE_SHA256 = (
    "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
)
V2112_POST_GATE_CONTENT_SHA256 = (
    "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
)

# Immutable V2.11.3 terminal zero-call no-go lineage.  Its V2.11.2-derived
# authority wrappers are not consumed here; the V2.11.2 source above is replayed
# directly.  These bindings exist solely to preserve the intervening release's
# ITT denominator, failure, and cumulative budget/storage provenance.
V2113_CONTRACT_ID = "finevo-pilot-v2.11.3"
V2113_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_3.yaml")
V2113_CONTRACT_FILE_SHA256 = (
    "a0e38d99e52a94c7434bf5e8c1befc9171988d5f722dccc544dc21b400550baf"
)
V2113_CONTRACT_SHA256 = (
    "84c818348fabfdd0ddd0ed503c0a5610faf10098f4973d1748b795e2e65b56f1"
)
V2113_SCIENCE_TAG = "pilot-v2.11.3-science"
V2113_SCIENCE_TAG_OBJECT = "87a1911284177b627755faf361ad4ea6c8213958"
V2113_SCIENCE_COMMIT = "65c613cdc9598dfffecbdf3a375cbf6113246782"
V2113_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.3/raw")
V2113_RAW_FILE_COUNT = 18
V2113_RAW_STORAGE_BYTES = 524_325
V2113_RAW_INVENTORY_SHA256 = (
    "691dd5f032d926d6c40cedf14ea380403163323c53bfbd9bc5c38d14e54decd3"
)
V2113_RUN_LEDGER_FILE_SHA256 = (
    "d544145fbcc9028401edd631c8506c0d2413a5e68a79fbf876847e086ad34e31"
)
V2113_RUN_LEDGER_SHA256 = (
    "97216e7b0a23b1b78a1e79d3ae166621147fab5582e5259434e1138c39946f40"
)
V2113_RUN_LEDGER_EVENT_COUNT = 138
V2113_RUN_LEDGER_EVENT_HEAD = (
    "012eca458ee6e6b86ad011f838632000e941c349fd5281338be12cd23e22cfbb"
)
V2113_BUDGET_LEDGER_FILE_SHA256 = (
    "d355c48f456d40f0e71b941d5b124cbbaa9959395db05afc2cf30da80cb66189"
)
V2113_BUDGET_LEDGER_SHA256 = (
    "366495f3cc4b8075e072c47fcf31c3eed40371996f0057efba64a1709ac5850a"
)
V2113_BUDGET_LEDGER_EVENT_COUNT = 12
V2113_BUDGET_LEDGER_EVENT_HEAD = (
    "e8e00716ddc602a559f8d436cb22b1b58583811c7a0d5dbe061eca56830e7e18"
)
V2113_RELEASE_ATTESTATION_FILE_SHA256 = (
    "4066dea1bc30edbbd70b5c6553890abae6b5633d4a35b376d96f808ab740da40"
)
V2113_RELEASE_ATTESTATION_SHA256 = (
    "d28ce40f62f7795810f0ac171573feb91f012b21edef2b3321bd86bf23fd4401"
)
V2113_LAUNCH_INPUT_FILE_SHA256 = (
    "b7a59b4cca02ad6aaf2a7d08c02c9f560ae64a3a467e566e45cdd53c16cf4536"
)
V2113_LAUNCH_INPUT_SHA256 = (
    "a3be56f20f9cffebeb906c48ca642048a153ebded5eefec8d5c37d08b711b111"
)
V2113_PARENT_IMPORT_FILE_SHA256 = (
    "f2339d7d6024c88cefde371c6e3101187df615f7efb088c80e79e17a558ca889"
)
V2113_PARENT_IMPORT_CONTENT_SHA256 = (
    "78cdd8dfe5da008c7b33740fafc3c6efe7a48f5e93b6f24a311539c233045d50"
)
V2113_PREFLIGHT_RECEIPT_FILE_SHA256 = (
    "41a9d8996c010f63b42178610a800c2a59ef6aa13d9666cae639c55ddaebab1a"
)
V2113_PREFLIGHT_RECEIPT_CONTENT_SHA256 = (
    "1044feb8cf050269c9aafb206bc5fc2c7b6f5b7c0d332d96663d4433ddf967ae"
)
V2113_ATTEMPT_COST_USD = 0.0
V2113_ATTEMPT_COMPLETIONS = 0
V2113_ATTEMPT_STORAGE_BYTES = 169_978
V2113_CUMULATIVE_COST_USD = 19.998220562500006
V2113_CUMULATIVE_COMPLETIONS = 1_004
V2113_CUMULATIVE_STORAGE_BYTES = 221_838_685
V2113_EXPECTED_STATUS_COUNTS = {"complete": 3, "integrity-stopped": 133}
V2113_EXPECTED_STAGE_STATUS_COUNTS: Mapping[str, Mapping[str, int]] = {
    "parent-import": {"complete": 1},
    "capability-gate": {"complete": 2},
    "long-context-preflight": {"integrity-stopped": 2},
    "experiment-c": {"integrity-stopped": 25},
    "experiment-a": {"integrity-stopped": 20},
    "experiment-d": {"integrity-stopped": 55},
    "experiment-b": {"integrity-stopped": 25},
    "cross-model": {"integrity-stopped": 6},
}
V2113_FAILURE_TYPE = "V2113PreflightAuthorityImportError"
V2113_FAILURE_MESSAGE = (
    "finevo-pilot-v2.11.3--long-context-preflight--gpt52_main--"
    "closed-loop-preflight--none--stage0-selected--s2010922376 "
    "V2.11.3 resealed authority drifted"
)

V2112_ATTEMPT_COST_USD = 1.41182075
V2112_ATTEMPT_COMPLETIONS = 64
V2112_ATTEMPT_STORAGE_BYTES = 3_830_082
V2112_CUMULATIVE_COST_USD = 19.998220562500006
V2112_CUMULATIVE_COMPLETIONS = 1_004
V2112_CUMULATIVE_STORAGE_BYTES = 221_668_707
V2113_INHERITED_V2112_DEBIT = ParentBudgetDebit(
    parent_contract_sha256=V2112_CONTRACT_SHA256,
    parent_run_ledger_sha256=V2112_RUN_LEDGER_SHA256,
    parent_budget_ledger_sha256=V2112_BUDGET_LEDGER_SHA256,
    stage_bucket="parent_v2112",
    cost_usd=V2112_CUMULATIVE_COST_USD,
    hosted_completions=V2112_CUMULATIVE_COMPLETIONS,
    storage_bytes=V2112_CUMULATIVE_STORAGE_BYTES,
    record_sha256="3ddc22970ff30d1ad9fc3b9efbffe5e4de1f641851bc9e3398aa2fd0977154a1",
)
V2114_CUMULATIVE_DEBIT = ParentBudgetDebit(
    parent_contract_sha256=V2113_CONTRACT_SHA256,
    parent_run_ledger_sha256=V2113_RUN_LEDGER_SHA256,
    parent_budget_ledger_sha256=V2113_BUDGET_LEDGER_SHA256,
    stage_bucket="parent_v2113",
    cost_usd=V2113_CUMULATIVE_COST_USD,
    hosted_completions=V2113_CUMULATIVE_COMPLETIONS,
    storage_bytes=V2113_CUMULATIVE_STORAGE_BYTES,
    record_sha256=("3f75623b4eb5b6c3c1c2e2a7e97687c215da025cbea309f94e861abee47f90ca"),
)

V2114_ALLOWED_MODELS = ("gpt52_main", "gpt56_diagnostic")
V2112_EXPECTED_STAGE_STATUS_COUNTS: Mapping[str, Mapping[str, int]] = {
    "parent-import": {"complete": 1},
    "capability-gate": {"complete": 2},
    "long-context-preflight": {"complete": 2},
    "experiment-c": {"complete": 5, "failed": 20},
    "experiment-a": {"failed": 20},
    "experiment-d": {"failed": 55},
    "experiment-b": {"failed": 25},
    "cross-model": {"failed": 6},
}
V2112_EXPECTED_STATUS_COUNTS = {"complete": 10, "failed": 126}
V2112_FAILURE_MESSAGE = (
    "source-backed observed p95 receipt verification failed: observed-p95 "
    "receipt top-level shape or schema drifted"
)
V2112_FAILURE_MESSAGE_SHA256 = (
    "39cb7f19f94e435d9eb4873df49beac2507703522f2ad9ffa7f688a5f6b92ef7"
)
V2112_OFFLINE_COMPLETE_ARM = "verified-error-candidate"

_STAGE_RECEIPTS: Mapping[str, Mapping[str, Any]] = {
    "parent-import": {
        "file_sha256": "778b41027dbd2db7d51b1ff2a729b96e296d2db02aeb3f75fb500ba4593472af",
        "content_sha256": "86e3efe0ddb7223a0632041d066e56e082c197df32d84655272a944b59e99627",
        "status": "complete",
        "go": True,
        "go_models": [],
    },
    "capability-gate": {
        "file_sha256": "4a8068793478da7ee35c7ca91bdd6e901ac758d066ed911e9848f3cfd170a6a5",
        "content_sha256": "2fed0195ae6f06f088ea219c11f30a5d0ba82368563e5e2fde4ce211672a7b64",
        "status": "complete",
        "go": True,
        "go_models": ["gpt52_main", "gpt56_diagnostic"],
    },
    "long-context-preflight": {
        "file_sha256": "17c8838edb1c5497311b5c4adba3e7715cd6e601811a33c3cdd59a97d2b359e1",
        "content_sha256": "1d06ad536faa0ad62f2ffec2e100d80f8463acd38c75606ac8b89a24e1568659",
        "status": "complete",
        "go": True,
        "go_models": ["gpt52_main", "gpt56_diagnostic"],
    },
    "experiment-c": {
        "file_sha256": "de003f869765771fe97e74242274e23cdd95b4b5f939f52f4897d51e50c4c7a0",
        "content_sha256": "cd7e6e496b01edc803386fda444ba1675a27d072104fb998560a817890b2aba9",
        "status": "complete-with-no-go",
        "go": False,
        "go_models": [],
    },
    "experiment-a": {
        "file_sha256": "d18734d4305ad8f536a5f0e83209e06688ea91b0e862d12875d31e4cc1f072ba",
        "content_sha256": "6227a63df3f3f89d1b164e1e5484ad2d008bac10a6eb9b05f7a99e58c05b2302",
        "status": "complete-with-no-go",
        "go": False,
        "go_models": [],
    },
    "experiment-d": {
        "file_sha256": "4ffe03c8a33f2d357d559d9ba59ba2ef3807f1ce2a713685801a40750e03eb1f",
        "content_sha256": "17e0a2019eb7839d191bf25040d5e0c8f5dd2eb28fea8e50fed8410297adb80e",
        "status": "complete-with-no-go",
        "go": False,
        "go_models": [],
    },
    "experiment-b": {
        "file_sha256": "ca5dbdf163947f168af1471b6c6a7139e2727fb3b6d658474d3f818ba66ce197",
        "content_sha256": "0fac20f96f7bb9ca0e406a4a536c1d5bde97ff841b1cbe79a86aa1aa68aa7610",
        "status": "complete-with-no-go",
        "go": False,
        "go_models": [],
    },
    "cross-model": {
        "file_sha256": "764b10d06ac494a4bb43710f69dfa31c965ab8d1960c183a49d7904a59b096c1",
        "content_sha256": "b8cb8df88298bc82e8c79277a3701e4b0b50c6204d2403f952cc6a96d45613da",
        "status": "complete-with-no-go",
        "go": False,
        "go_models": [],
    },
}

_V2113_STAGE_RECEIPTS: Mapping[str, Mapping[str, Any]] = {
    "parent-import": {
        "file_sha256": "22db6f79dca55099c9a81b4e022b942e33d65bc412881deb02e23c4ac6dc39a3",
        "content_sha256": "f97ba7bd6693a5221aacb4e071d9e0da352832995f0401c8c1515284d7f4fccd",
        "status": "complete",
        "go": True,
        "go_models": [],
    },
    "capability-gate": {
        "file_sha256": "143486f0f2e12891b04a76039e772e276fb814dc8d00243ff255681b3d4d7f82",
        "content_sha256": "189adec4e502169bfe5a137a1182cbe81d029f990c3bdaaebfb41599b4f6a4cc",
        "status": "complete",
        "go": True,
        "go_models": ["gpt52_main", "gpt56_diagnostic"],
    },
    "long-context-preflight": {
        "file_sha256": V2113_PREFLIGHT_RECEIPT_FILE_SHA256,
        "content_sha256": V2113_PREFLIGHT_RECEIPT_CONTENT_SHA256,
        "status": "complete-with-no-go",
        "go": False,
        "go_models": [],
    },
}

# Files semantically parsed during import.  Everything else in the frozen raw
# inventory is byte-hashed only.  In particular, no per-cell file below any
# scientific stage is decoded by this module.
V2114_PARSED_PARENT_SOURCE_ALLOWLIST = (
    "experiments/pilot_v2_11_2.yaml",
    "experiments/pilot_v2_11_2_source_manifest.json",
    "experiment_results/pilot-v2.11.2/raw/release_attestation.json",
    "experiment_results/pilot-v2.11.2/raw/scientific_launch_input.json",
    "experiment_results/pilot-v2.11.2/raw/run_ledger.json",
    "experiment_results/pilot-v2.11.2/raw/budget_ledger.json",
    "experiment_results/pilot-v2.11.2/raw/parent-import/parent_import_receipt.json",
    "experiment_results/pilot-v2.11.2/raw/parent-import/stage_receipt.json",
    "experiment_results/pilot-v2.11.2/raw/capability-gate/stage_receipt.json",
    "experiment_results/pilot-v2.11.2/raw/long-context-preflight/stage_receipt.json",
    "experiment_results/pilot-v2.11.2/raw/long-context-preflight/post_gate_authority.json",
    (
        "experiment_results/pilot-v2.11.2/raw/long-context-preflight/runs/"
        "finevo-pilot-v2.11.2--long-context-preflight--gpt52_main--"
        "closed-loop-preflight--none--stage0-selected--s2010922376/"
        "preflight_checkpoint.json"
    ),
    (
        "experiment_results/pilot-v2.11.2/raw/long-context-preflight/runs/"
        "finevo-pilot-v2.11.2--long-context-preflight--gpt52_main--"
        "closed-loop-preflight--none--stage0-selected--s2010922376/"
        "preflight_checkpoint_exactness.json"
    ),
    (
        "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
        "provider_call_journals/finevo-pilot-v2.11.2--long-context-preflight--"
        "gpt52_main--closed-loop-preflight--none--stage0-selected--"
        "s2010922376--preflight.json"
    ),
    (
        "experiment_results/pilot-v2.11.2/raw/long-context-preflight/runs/"
        "finevo-pilot-v2.11.2--long-context-preflight--gpt56_diagnostic--"
        "closed-loop-preflight--none--stage0-selected--s2010922376/"
        "preflight_checkpoint.json"
    ),
    (
        "experiment_results/pilot-v2.11.2/raw/long-context-preflight/runs/"
        "finevo-pilot-v2.11.2--long-context-preflight--gpt56_diagnostic--"
        "closed-loop-preflight--none--stage0-selected--s2010922376/"
        "preflight_checkpoint_exactness.json"
    ),
    (
        "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
        "provider_call_journals/finevo-pilot-v2.11.2--long-context-preflight--"
        "gpt56_diagnostic--closed-loop-preflight--none--stage0-selected--"
        "s2010922376--preflight.json"
    ),
    "experiment_results/pilot-v2.11.2/raw/experiment-c/stage_receipt.json",
    "experiment_results/pilot-v2.11.2/raw/experiment-a/stage_receipt.json",
    "experiment_results/pilot-v2.11.2/raw/experiment-d/stage_receipt.json",
    "experiment_results/pilot-v2.11.2/raw/experiment-b/stage_receipt.json",
    "experiment_results/pilot-v2.11.2/raw/cross-model/stage_receipt.json",
    "experiments/pilot_v2_11_3.yaml",
    "experiment_results/pilot-v2.11.3/raw/release_attestation.json",
    "experiment_results/pilot-v2.11.3/raw/scientific_launch_input.json",
    "experiment_results/pilot-v2.11.3/raw/run_ledger.json",
    "experiment_results/pilot-v2.11.3/raw/budget_ledger.json",
    "experiment_results/pilot-v2.11.3/raw/parent-import/parent_import_receipt.json",
    "experiment_results/pilot-v2.11.3/raw/parent-import/stage_receipt.json",
    "experiment_results/pilot-v2.11.3/raw/capability-gate/stage_receipt.json",
    "experiment_results/pilot-v2.11.3/raw/long-context-preflight/stage_receipt.json",
)
V2114_REUSABLE_AUTHORITY_KINDS = (
    "calibration-wrapper",
    "capability-wrapper:gpt52_main",
    "capability-wrapper:gpt56_diagnostic",
    "observed-p95:gpt52_main:action",
    "observed-p95:gpt52_main:semantic",
    "observed-p95:gpt56_diagnostic:action",
    "observed-p95:gpt56_diagnostic:semantic",
)
V2114_FORBIDDEN_IMPORT_PREFIXES = (
    "experiment_results/pilot-v2.11.2/raw/experiment-a/",
    "experiment_results/pilot-v2.11.2/raw/experiment-b/",
    "experiment_results/pilot-v2.11.2/raw/experiment-c/",
    "experiment_results/pilot-v2.11.2/raw/experiment-d/",
    "experiment_results/pilot-v2.11.2/raw/cross-model/",
    "experiment_results/pilot-v2.11.3/raw/experiment-a/",
    "experiment_results/pilot-v2.11.3/raw/experiment-b/",
    "experiment_results/pilot-v2.11.3/raw/experiment-c/",
    "experiment_results/pilot-v2.11.3/raw/experiment-d/",
    "experiment_results/pilot-v2.11.3/raw/cross-model/",
)

_EVIDENCE_BINDINGS: Mapping[str, str] = {
    "namespace": "evidence/current_v2/pilot-v2.11.2",
    "merge_commit": "b017a163911499313ba6b39a38d1226a644b95fa",
    "package_manifest_file_sha256": (
        "c8507e371e0d1deedbe6fad1e660be77d5a1b673c86a8403522247e199979988"
    ),
    "checksums_file_sha256": (
        "0038a758c6e9607826ba0a2d4e63f426c3898a31ecc89a51762774b823a9dc0c"
    ),
    "failure_ledger_file_sha256": (
        "3f508e662ea9d65d707461a13255ceae9d43eda9ba0da330bcd91ad1f26ee873"
    ),
    "reviewer_report_file_sha256": (
        "0e6cab12da8117300f405c1f9697d632d8cb6591f49f02043b565ae6f754d65b"
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ZERO_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}
_ZERO_PROVIDER_POLICY = {
    "provider_construction_during_import": False,
    "provider_calls_during_import": 0,
    "hosted_provider_calls_during_import": 0,
    "hosted_cost_usd_during_import": 0.0,
    "imported_effect_cells": 0,
    "imported_scientific_run_summaries": 0,
    "imported_scientific_outcome_artifacts": [],
    "decoded_completion_reuse": False,
}


class PilotV2114ParentImportError(RuntimeError):
    """Raised before any imported V2.11.4 authority can be consumed."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PilotV2114ParentImportError("value is not canonical JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise PilotV2114ParentImportError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PilotV2114ParentImportError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PilotV2114ParentImportError(f"{name} must contain one object")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(dict(value))
    result.pop("integrity", None)
    result["integrity"] = {"canonicalization": "json-sort-keys-utf8-v1"}
    result["integrity"]["content_sha256"] = canonical_sha256(result)
    return result


def _verify_seal(value: Mapping[str, Any], *, schema: str, name: str) -> str:
    candidate = _json_copy(dict(value))
    integrity = candidate.get("integrity")
    claimed = (
        integrity.pop("content_sha256", None) if isinstance(integrity, dict) else None
    )
    if (
        value.get("schema_version") != schema
        or not isinstance(integrity, Mapping)
        or set(value.get("integrity", {})) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or _SHA256_RE.fullmatch(str(claimed)) is None
        or claimed != canonical_sha256(candidate)
    ):
        raise PilotV2114ParentImportError(f"{name} seal drifted")
    return str(claimed)


def _verify_stage_receipt_seal(
    value: Mapping[str, Any], *, schema: str, name: str
) -> str:
    """Verify the stage-receipt-v2 seal, whose hash excludes integrity."""

    candidate = _json_copy(dict(value))
    integrity = candidate.pop("integrity", None)
    if (
        value.get("schema_version") != schema
        or not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or _SHA256_RE.fullmatch(str(integrity.get("content_sha256"))) is None
        or integrity.get("content_sha256") != canonical_sha256(candidate)
    ):
        raise PilotV2114ParentImportError(f"{name} seal drifted")
    return str(integrity["content_sha256"])


def _strict_root(value: str | Path, *, name: str) -> Path:
    root = Path(value).expanduser().absolute()
    try:
        info = root.lstat()
    except OSError as exc:
        raise PilotV2114ParentImportError(f"{name} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PilotV2114ParentImportError(
            f"{name} must be a real non-symlink directory"
        )
    return root


def _normalized_relative(
    value: str | PurePosixPath, *, top: str, name: str
) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or path.is_absolute()
        or path.parts[0] != top
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise PilotV2114ParentImportError(
            f"{name} must stay below {top}/ without path escape"
        )
    return path


def _read_regular(root: Path, relative: PurePosixPath, *, name: str) -> bytes:
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise PilotV2114ParentImportError(f"missing {name}: {relative}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise PilotV2114ParentImportError(f"{name} path is not guarded")
    path = current / relative.parts[-1]
    try:
        info = path.lstat()
    except OSError as exc:
        raise PilotV2114ParentImportError(f"missing {name}: {relative}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PilotV2114ParentImportError(f"{name} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        try:
            raw = os.read(fd, info.st_size + 1)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise PilotV2114ParentImportError(f"cannot read {name}") from exc
    if (
        len(raw) != info.st_size
        or after.st_dev != info.st_dev
        or after.st_ino != info.st_ino
        or after.st_mtime_ns != info.st_mtime_ns
    ):
        raise PilotV2114ParentImportError(f"{name} changed during guarded read")
    return raw


def _read_json(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular(root, relative, name=name)
    return raw, _strict_json(raw, name=name)


def _read_parent_json(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    if relative.as_posix() not in V2114_PARSED_PARENT_SOURCE_ALLOWLIST:
        raise PilotV2114ParentImportError(
            f"{name} is outside the frozen parsed-parent allowlist"
        )
    return _read_json(root, relative, name=name)


def _binding(
    relative: PurePosixPath, raw: bytes, value: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative.as_posix(),
        "byte_size": len(raw),
        "file_sha256": _sha256(raw),
    }
    integrity = value.get("integrity")
    if isinstance(integrity, Mapping) and isinstance(
        integrity.get("content_sha256"), str
    ):
        result["content_sha256"] = integrity["content_sha256"]
    elif isinstance(value.get("receipt_sha256"), str):
        result["content_sha256"] = value["receipt_sha256"]
    elif isinstance(value.get("attestation_sha256"), str):
        result["content_sha256"] = value["attestation_sha256"]
    elif isinstance(value.get("launch_input_sha256"), str):
        result["content_sha256"] = value["launch_input_sha256"]
    return result


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("/usr/bin/git", *args),
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotV2114ParentImportError("git release verification failed") from exc
    return completed.stdout.strip()


def _verify_parent_git(root: Path) -> dict[str, str]:
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    tag_object = _git(
        root, "rev-parse", "--verify", f"refs/tags/{V2112_SCIENCE_TAG}^{{tag}}"
    )
    tag_commit = _git(
        root, "rev-parse", "--verify", f"refs/tags/{V2112_SCIENCE_TAG}^{{commit}}"
    )
    tracked = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if (
        head != V2112_SCIENCE_COMMIT
        or tag_commit != V2112_SCIENCE_COMMIT
        or tag_object != V2112_SCIENCE_TAG_OBJECT
        or tracked
    ):
        raise PilotV2114ParentImportError(
            "V2.11.2 annotated tag, commit, or tracked source drifted"
        )
    return {
        "science_tag": V2112_SCIENCE_TAG,
        "science_tag_object": tag_object,
        "resolved_git_commit": tag_commit,
    }


def _verify_event_ledger(
    value: Mapping[str, Any],
    *,
    schema: str,
    contract_sha256: str,
    internal_sha256: str,
    event_count: int,
    event_head: str,
    run_count: int,
    name: str,
) -> None:
    candidate = _json_copy(dict(value))
    claimed = candidate.pop("ledger_sha256", None)
    events = value.get("events")
    runs = value.get("runs")
    if (
        value.get("schema_version") != schema
        or value.get("contract_hash") != contract_sha256
        or claimed != internal_sha256
        or canonical_sha256(candidate) != internal_sha256
        or not isinstance(events, list)
        or len(events) != event_count
        or not isinstance(runs, Mapping)
        or len(runs) != run_count
        or not events
        or events[-1].get("event_sha256") != event_head
    ):
        raise PilotV2114ParentImportError(f"{name} identity drifted")
    previous = "0" * 64
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise PilotV2114ParentImportError(f"{name} event is malformed")
        body = _json_copy(dict(event))
        digest = body.pop("event_sha256", None)
        if (
            event.get("event_index") != index
            or event.get("previous_event_sha256") != previous
            or digest != canonical_sha256(body)
        ):
            raise PilotV2114ParentImportError(f"{name} event chain drifted")
        previous = str(digest)


def _load_parent_contract(root: Path) -> tuple[PilotContract, dict[str, Any]]:
    raw, value = _read_parent_json(root, V2112_CONTRACT_PATH, name="V2.11.2 contract")
    if (
        _sha256(raw) != V2112_CONTRACT_FILE_SHA256
        or value.get("contract_id") != V2112_CONTRACT_ID
        or value.get("status") != "frozen"
        or value.get("implementation", {}).get("required_git_tag") != V2112_SCIENCE_TAG
        or value.get("integrity", {}).get("declared_sha256") != V2112_CONTRACT_SHA256
        or canonical_contract_sha256(value) != V2112_CONTRACT_SHA256
    ):
        raise PilotV2114ParentImportError("V2.11.2 contract identity drifted")
    try:
        contract = load_pilot_contract(root.joinpath(*V2112_CONTRACT_PATH.parts))
    except Exception as exc:
        raise PilotV2114ParentImportError("V2.11.2 contract failed to load") from exc
    if contract.canonical_hash != V2112_CONTRACT_SHA256:
        raise PilotV2114ParentImportError("V2.11.2 parsed contract hash drifted")
    return contract, _binding(V2112_CONTRACT_PATH, raw, value)


def _verify_raw_inventory(root: Path) -> dict[str, Any]:
    raw_root = root.joinpath(*V2112_RAW_ROOT.parts)
    try:
        rows, summary = _inventory(raw_root, declared_root=V2112_RAW_ROOT)
    except Exception as exc:
        raise PilotV2114ParentImportError(
            "V2.11.2 raw inventory verification failed"
        ) from exc
    expected = {
        "root": V2112_RAW_ROOT.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": V2112_RAW_FILE_COUNT,
        "storage_bytes": V2112_RAW_STORAGE_BYTES,
        "inventory_sha256": V2112_RAW_INVENTORY_SHA256,
    }
    if summary != expected:
        raise PilotV2114ParentImportError("V2.11.2 raw inventory drifted")
    return {**expected, "rows": rows}


def _verify_attestation_and_launch(root: Path) -> dict[str, Any]:
    attestation_path = V2112_RAW_ROOT / "release_attestation.json"
    launch_path = V2112_RAW_ROOT / "scientific_launch_input.json"
    att_raw, att = _read_parent_json(root, attestation_path, name="V2.11.2 attestation")
    launch_raw, launch = _read_parent_json(
        root, launch_path, name="V2.11.2 launch input"
    )
    if (
        _sha256(att_raw) != V2112_RELEASE_ATTESTATION_FILE_SHA256
        or att.get("schema_version") != "finevo-scientific-release-attestation-v2"
        or att.get("attestation_sha256") != V2112_RELEASE_ATTESTATION_SHA256
        or att.get("status") != "pass"
        or att.get("contract", {}).get("canonical_sha256") != V2112_CONTRACT_SHA256
        or att.get("head_commit") != V2112_SCIENCE_COMMIT
        or _sha256(launch_raw) != V2112_LAUNCH_INPUT_FILE_SHA256
        or launch.get("schema_version") != "finevo-scientific-launch-input-v1"
        or launch.get("launch_input_sha256") != V2112_LAUNCH_INPUT_SHA256
        or launch.get("contract_sha256") != V2112_CONTRACT_SHA256
    ):
        raise PilotV2114ParentImportError("V2.11.2 attestation or launch input drifted")
    return {
        "release_attestation": _binding(attestation_path, att_raw, att),
        "scientific_launch_input": _binding(launch_path, launch_raw, launch),
    }


def _verify_run_ledger(
    root: Path,
    contract: PilotContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = V2112_RAW_ROOT / "run_ledger.json"
    raw, ledger = _read_parent_json(root, relative, name="V2.11.2 run ledger")
    if _sha256(raw) != V2112_RUN_LEDGER_FILE_SHA256:
        raise PilotV2114ParentImportError("V2.11.2 run ledger bytes drifted")
    _verify_event_ledger(
        ledger,
        schema="finevo-pilot-run-ledger-v2",
        contract_sha256=V2112_CONTRACT_SHA256,
        internal_sha256=V2112_RUN_LEDGER_SHA256,
        event_count=V2112_RUN_LEDGER_EVENT_COUNT,
        event_head=V2112_RUN_LEDGER_EVENT_HEAD,
        run_count=136,
        name="V2.11.2 run ledger",
    )
    specs = {spec.run_id: spec.to_dict() for spec in contract.expand()}
    runs = ledger["runs"]
    if len(specs) != 136 or set(runs) != set(specs):
        raise PilotV2114ParentImportError("V2.11.2 ITT denominator drifted")
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    offline_complete: list[str] = []
    for run_id, row in runs.items():
        if not isinstance(row, Mapping) or row.get("spec") != specs[run_id]:
            raise PilotV2114ParentImportError("V2.11.2 run/spec binding drifted")
        status = str(row.get("status"))
        stage_id = str(specs[run_id]["stage_id"])
        totals[status] += 1
        by_stage[stage_id][status] += 1
        if stage_id in {
            "experiment-a",
            "experiment-b",
            "experiment-c",
            "experiment-d",
            "cross-model",
        }:
            if status == "complete":
                if (
                    stage_id != "experiment-c"
                    or specs[run_id].get("arm_id") != V2112_OFFLINE_COMPLETE_ARM
                    or specs[run_id].get("execution_mode")
                    != "offline_candidate_admission"
                ):
                    raise PilotV2114ParentImportError(
                        "V2.11.2 treatment outcome unexpectedly completed"
                    )
                offline_complete.append(run_id)
            elif (
                status != "failed"
                or row.get("failure", {}).get("message") != V2112_FAILURE_MESSAGE
                or row.get("failure", {}).get("message_sha256")
                != V2112_FAILURE_MESSAGE_SHA256
            ):
                raise PilotV2114ParentImportError(
                    "V2.11.2 science failure boundary drifted"
                )
    normalized_stages = {stage: dict(counts) for stage, counts in by_stage.items()}
    if (
        dict(totals) != V2112_EXPECTED_STATUS_COUNTS
        or normalized_stages
        != {
            stage: dict(counts)
            for stage, counts in V2112_EXPECTED_STAGE_STATUS_COUNTS.items()
        }
        or len(offline_complete) != 5
    ):
        raise PilotV2114ParentImportError("V2.11.2 terminal denominator drifted")
    return ledger, {
        **_binding(relative, raw, ledger),
        "internal_sha256": V2112_RUN_LEDGER_SHA256,
        "event_count": V2112_RUN_LEDGER_EVENT_COUNT,
        "event_head_sha256": V2112_RUN_LEDGER_EVENT_HEAD,
        "run_count": 136,
        "status_counts": dict(totals),
        "stage_status_counts": normalized_stages,
        "excluded_offline_complete_run_ids_sha256": canonical_sha256(
            sorted(offline_complete)
        ),
    }


def _verify_budget_ledger(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = V2112_RAW_ROOT / "budget_ledger.json"
    raw, ledger = _read_parent_json(root, relative, name="V2.11.2 budget ledger")
    if _sha256(raw) != V2112_BUDGET_LEDGER_FILE_SHA256:
        raise PilotV2114ParentImportError("V2.11.2 budget ledger bytes drifted")
    _verify_event_ledger(
        ledger,
        schema="finevo-pilot-budget-ledger-v2",
        contract_sha256=V2112_CONTRACT_SHA256,
        internal_sha256=V2112_BUDGET_LEDGER_SHA256,
        event_count=V2112_BUDGET_LEDGER_EVENT_COUNT,
        event_head=V2112_BUDGET_LEDGER_EVENT_HEAD,
        run_count=86,
        name="V2.11.2 budget ledger",
    )
    try:
        parent = ParentBudgetDebit.from_dict(ledger["parent_debit"])
    except Exception as exc:
        raise PilotV2114ParentImportError(
            "V2.11.2 inherited budget debit is malformed"
        ) from exc
    rows = ledger["runs"]
    cost = math.fsum(float(row["actual"]["cost_usd"]) for row in rows.values())
    completions = sum(int(row["actual"]["completions"]) for row in rows.values())
    storage = sum(int(row["actual"]["storage_bytes"]) for row in rows.values())
    science_calls = 0
    science_cost = 0.0
    for row in rows.values():
        run_id = str(row.get("reservation", {}).get("run_id", ""))
        if any(
            f"--{stage}--" in run_id
            for stage in (
                "experiment-a",
                "experiment-b",
                "experiment-c",
                "experiment-d",
                "cross-model",
            )
        ):
            science_calls += int(row["actual"]["completions"])
            science_cost += float(row["actual"]["cost_usd"])
    if (
        not math.isclose(cost, V2112_ATTEMPT_COST_USD, rel_tol=0, abs_tol=1e-12)
        or completions != V2112_ATTEMPT_COMPLETIONS
        or storage != V2112_ATTEMPT_STORAGE_BYTES
        or science_calls != 0
        or not math.isclose(science_cost, 0.0, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(
            parent.cost_usd + cost,
            V2112_CUMULATIVE_COST_USD,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or parent.hosted_completions + completions != V2112_CUMULATIVE_COMPLETIONS
        or parent.storage_bytes + storage != V2112_CUMULATIVE_STORAGE_BYTES
    ):
        raise PilotV2114ParentImportError("V2.11.2 attempt or cumulative debit drifted")
    return ledger, {
        **_binding(relative, raw, ledger),
        "internal_sha256": V2112_BUDGET_LEDGER_SHA256,
        "event_count": V2112_BUDGET_LEDGER_EVENT_COUNT,
        "event_head_sha256": V2112_BUDGET_LEDGER_EVENT_HEAD,
        "run_count": 86,
        "current_attempt": {
            "cost_usd": cost,
            "hosted_completions": completions,
            "storage_bytes": storage,
            "science_provider_calls": science_calls,
            "science_cost_usd": science_cost,
        },
    }


def _verify_stage_receipts(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage_id, expected in _STAGE_RECEIPTS.items():
        relative = V2112_RAW_ROOT / f"{stage_id}/stage_receipt.json"
        raw, receipt = _read_parent_json(
            root, relative, name=f"V2.11.2 {stage_id} receipt"
        )
        content = _verify_stage_receipt_seal(
            receipt,
            schema="finevo-pilot-stage-receipt-v2",
            name=f"V2.11.2 {stage_id} receipt",
        )
        if (
            _sha256(raw) != expected["file_sha256"]
            or content != expected["content_sha256"]
            or receipt.get("contract_id") != V2112_CONTRACT_ID
            or receipt.get("contract_sha256") != V2112_CONTRACT_SHA256
            or receipt.get("stage_id") != stage_id
            or receipt.get("status") != expected["status"]
            or receipt.get("status_counts")
            != dict(V2112_EXPECTED_STAGE_STATUS_COUNTS[stage_id])
            or receipt.get("terminal") is not True
            or receipt.get("denominator_terminal") is not True
            or receipt.get("go") is not expected["go"]
            or receipt.get("go_models") != expected["go_models"]
        ):
            raise PilotV2114ParentImportError(f"V2.11.2 {stage_id} receipt drifted")
        result[stage_id] = _binding(relative, raw, receipt)
    return result


def _verify_v2113_git(root: Path) -> dict[str, str]:
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    tag_object = _git(
        root, "rev-parse", "--verify", f"refs/tags/{V2113_SCIENCE_TAG}^{{tag}}"
    )
    tag_commit = _git(
        root, "rev-parse", "--verify", f"refs/tags/{V2113_SCIENCE_TAG}^{{commit}}"
    )
    tracked = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if (
        head != V2113_SCIENCE_COMMIT
        or tag_commit != V2113_SCIENCE_COMMIT
        or tag_object != V2113_SCIENCE_TAG_OBJECT
        or tracked
    ):
        raise PilotV2114ParentImportError(
            "V2.11.3 annotated tag, commit, or tracked source drifted"
        )
    return {
        "science_tag": V2113_SCIENCE_TAG,
        "science_tag_object": tag_object,
        "resolved_git_commit": tag_commit,
    }


def _load_v2113_contract(root: Path) -> tuple[PilotContract, dict[str, Any]]:
    raw, value = _read_parent_json(
        root, V2113_CONTRACT_PATH, name="V2.11.3 contract"
    )
    if (
        _sha256(raw) != V2113_CONTRACT_FILE_SHA256
        or value.get("contract_id") != V2113_CONTRACT_ID
        or value.get("status") != "frozen"
        or value.get("implementation", {}).get("required_git_tag")
        != V2113_SCIENCE_TAG
        or value.get("integrity", {}).get("declared_sha256")
        != V2113_CONTRACT_SHA256
        or canonical_contract_sha256(value) != V2113_CONTRACT_SHA256
    ):
        raise PilotV2114ParentImportError("V2.11.3 contract identity drifted")
    try:
        contract = load_pilot_contract(root.joinpath(*V2113_CONTRACT_PATH.parts))
    except Exception as exc:
        raise PilotV2114ParentImportError("V2.11.3 contract failed to load") from exc
    if contract.canonical_hash != V2113_CONTRACT_SHA256:
        raise PilotV2114ParentImportError("V2.11.3 parsed contract hash drifted")
    return contract, _binding(V2113_CONTRACT_PATH, raw, value)


def _verify_v2113_raw_inventory(root: Path) -> dict[str, Any]:
    raw_root = root.joinpath(*V2113_RAW_ROOT.parts)
    try:
        rows, summary = _inventory(raw_root, declared_root=V2113_RAW_ROOT)
    except Exception as exc:
        raise PilotV2114ParentImportError(
            "V2.11.3 raw inventory verification failed"
        ) from exc
    expected = {
        "root": V2113_RAW_ROOT.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": V2113_RAW_FILE_COUNT,
        "storage_bytes": V2113_RAW_STORAGE_BYTES,
        "inventory_sha256": V2113_RAW_INVENTORY_SHA256,
    }
    if summary != expected:
        raise PilotV2114ParentImportError("V2.11.3 raw inventory drifted")
    return {**expected, "rows": rows}


def _verify_v2113_attestation_and_launch(root: Path) -> dict[str, Any]:
    attestation_path = V2113_RAW_ROOT / "release_attestation.json"
    launch_path = V2113_RAW_ROOT / "scientific_launch_input.json"
    att_raw, att = _read_parent_json(
        root, attestation_path, name="V2.11.3 attestation"
    )
    launch_raw, launch = _read_parent_json(
        root, launch_path, name="V2.11.3 launch input"
    )
    local_tag = att.get("local_tag")
    if (
        _sha256(att_raw) != V2113_RELEASE_ATTESTATION_FILE_SHA256
        or att.get("schema_version") != "finevo-scientific-release-attestation-v2"
        or att.get("attestation_sha256") != V2113_RELEASE_ATTESTATION_SHA256
        or att.get("status") != "pass"
        or att.get("contract", {}).get("canonical_sha256")
        != V2113_CONTRACT_SHA256
        or att.get("head_commit") != V2113_SCIENCE_COMMIT
        or not isinstance(local_tag, Mapping)
        or local_tag.get("kind") != "annotated"
        or local_tag.get("name") != V2113_SCIENCE_TAG
        or local_tag.get("object_id") != V2113_SCIENCE_TAG_OBJECT
        or local_tag.get("peeled_commit") != V2113_SCIENCE_COMMIT
        or _sha256(launch_raw) != V2113_LAUNCH_INPUT_FILE_SHA256
        or launch.get("schema_version") != "finevo-scientific-launch-input-v1"
        or launch.get("launch_input_sha256") != V2113_LAUNCH_INPUT_SHA256
        or launch.get("contract_sha256") != V2113_CONTRACT_SHA256
    ):
        raise PilotV2114ParentImportError("V2.11.3 attestation or launch input drifted")
    return {
        "release_attestation": _binding(attestation_path, att_raw, att),
        "scientific_launch_input": _binding(launch_path, launch_raw, launch),
    }


def _verify_v2113_run_ledger(
    root: Path,
    contract: PilotContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = V2113_RAW_ROOT / "run_ledger.json"
    raw, ledger = _read_parent_json(root, relative, name="V2.11.3 run ledger")
    if _sha256(raw) != V2113_RUN_LEDGER_FILE_SHA256:
        raise PilotV2114ParentImportError("V2.11.3 run ledger bytes drifted")
    _verify_event_ledger(
        ledger,
        schema="finevo-pilot-run-ledger-v2",
        contract_sha256=V2113_CONTRACT_SHA256,
        internal_sha256=V2113_RUN_LEDGER_SHA256,
        event_count=V2113_RUN_LEDGER_EVENT_COUNT,
        event_head=V2113_RUN_LEDGER_EVENT_HEAD,
        run_count=136,
        name="V2.11.3 run ledger",
    )
    specs = {spec.run_id: spec.to_dict() for spec in contract.expand()}
    runs = ledger["runs"]
    if len(specs) != 136 or set(runs) != set(specs):
        raise PilotV2114ParentImportError("V2.11.3 ITT denominator drifted")
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for run_id, row in runs.items():
        if not isinstance(row, Mapping) or row.get("spec") != specs[run_id]:
            raise PilotV2114ParentImportError("V2.11.3 run/spec binding drifted")
        status = str(row.get("status"))
        stage_id = str(specs[run_id]["stage_id"])
        totals[status] += 1
        by_stage[stage_id][status] += 1
        if status == "complete":
            if stage_id not in {"parent-import", "capability-gate"}:
                raise PilotV2114ParentImportError(
                    "V2.11.3 scientific or preflight cell unexpectedly completed"
                )
            continue
        failure = row.get("failure")
        if (
            status != "integrity-stopped"
            or not isinstance(failure, Mapping)
            or failure.get("cause_type") != "PilotOrchestrationError"
            or failure.get("error_type") != V2113_FAILURE_TYPE
            or failure.get("message") != V2113_FAILURE_MESSAGE
            or failure.get("provider_construction") is not False
            or failure.get("provider_calls") != 0
        ):
            raise PilotV2114ParentImportError(
                "V2.11.3 zero-call terminal failure boundary drifted"
            )
    normalized_stages = {stage: dict(counts) for stage, counts in by_stage.items()}
    if (
        dict(totals) != V2113_EXPECTED_STATUS_COUNTS
        or normalized_stages
        != {
            stage: dict(counts)
            for stage, counts in V2113_EXPECTED_STAGE_STATUS_COUNTS.items()
        }
    ):
        raise PilotV2114ParentImportError("V2.11.3 terminal denominator drifted")
    return ledger, {
        **_binding(relative, raw, ledger),
        "internal_sha256": V2113_RUN_LEDGER_SHA256,
        "event_count": V2113_RUN_LEDGER_EVENT_COUNT,
        "event_head_sha256": V2113_RUN_LEDGER_EVENT_HEAD,
        "run_count": 136,
        "status_counts": dict(totals),
        "stage_status_counts": normalized_stages,
        "provider_calls_current_attempt": 0,
        "provider_construction_current_attempt": False,
    }


def _verify_v2113_budget_ledger(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = V2113_RAW_ROOT / "budget_ledger.json"
    raw, ledger = _read_parent_json(root, relative, name="V2.11.3 budget ledger")
    if _sha256(raw) != V2113_BUDGET_LEDGER_FILE_SHA256:
        raise PilotV2114ParentImportError("V2.11.3 budget ledger bytes drifted")
    _verify_event_ledger(
        ledger,
        schema="finevo-pilot-budget-ledger-v2",
        contract_sha256=V2113_CONTRACT_SHA256,
        internal_sha256=V2113_BUDGET_LEDGER_SHA256,
        event_count=V2113_BUDGET_LEDGER_EVENT_COUNT,
        event_head=V2113_BUDGET_LEDGER_EVENT_HEAD,
        run_count=5,
        name="V2.11.3 budget ledger",
    )
    try:
        inherited = ParentBudgetDebit.from_dict(ledger["parent_debit"])
    except Exception as exc:
        raise PilotV2114ParentImportError(
            "V2.11.3 inherited budget debit is malformed"
        ) from exc
    rows = ledger["runs"]
    cost = math.fsum(float(row["actual"]["cost_usd"]) for row in rows.values())
    completions = sum(int(row["actual"]["completions"]) for row in rows.values())
    storage = sum(int(row["actual"]["storage_bytes"]) for row in rows.values())
    statuses = Counter(str(row.get("status")) for row in rows.values())
    for row in rows.values():
        reservation = row.get("reservation")
        actual = row.get("actual")
        if (
            not isinstance(reservation, Mapping)
            or not isinstance(actual, Mapping)
            or reservation.get("cost_usd") != 0.0
            or reservation.get("completions") != 0
            or actual.get("cost_usd") != 0.0
            or actual.get("completions") != 0
        ):
            raise PilotV2114ParentImportError(
                "V2.11.3 budget ledger contains a current provider charge"
            )
    if (
        inherited != V2113_INHERITED_V2112_DEBIT
        or not math.isclose(cost, V2113_ATTEMPT_COST_USD, rel_tol=0, abs_tol=1e-12)
        or completions != V2113_ATTEMPT_COMPLETIONS
        or storage != V2113_ATTEMPT_STORAGE_BYTES
        or dict(statuses) != {"complete": 3, "integrity-stopped": 2}
        or not math.isclose(
            inherited.cost_usd + cost,
            V2113_CUMULATIVE_COST_USD,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or inherited.hosted_completions + completions
        != V2113_CUMULATIVE_COMPLETIONS
        or inherited.storage_bytes + storage != V2113_CUMULATIVE_STORAGE_BYTES
    ):
        raise PilotV2114ParentImportError(
            "V2.11.3 attempt or cumulative debit drifted"
        )
    return ledger, {
        **_binding(relative, raw, ledger),
        "internal_sha256": V2113_BUDGET_LEDGER_SHA256,
        "event_count": V2113_BUDGET_LEDGER_EVENT_COUNT,
        "event_head_sha256": V2113_BUDGET_LEDGER_EVENT_HEAD,
        "run_count": 5,
        "status_counts": dict(statuses),
        "current_attempt": {
            "cost_usd": cost,
            "hosted_completions": completions,
            "storage_bytes": storage,
            "provider_calls": 0,
        },
        "cumulative_debit": V2114_CUMULATIVE_DEBIT.to_dict(),
    }


def _verify_v2113_stage_receipts(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage_id, expected in _V2113_STAGE_RECEIPTS.items():
        relative = V2113_RAW_ROOT / f"{stage_id}/stage_receipt.json"
        raw, receipt = _read_parent_json(
            root, relative, name=f"V2.11.3 {stage_id} receipt"
        )
        content = _verify_stage_receipt_seal(
            receipt,
            schema="finevo-pilot-stage-receipt-v2",
            name=f"V2.11.3 {stage_id} receipt",
        )
        if (
            _sha256(raw) != expected["file_sha256"]
            or content != expected["content_sha256"]
            or receipt.get("contract_id") != V2113_CONTRACT_ID
            or receipt.get("contract_sha256") != V2113_CONTRACT_SHA256
            or receipt.get("stage_id") != stage_id
            or receipt.get("status") != expected["status"]
            or receipt.get("status_counts")
            != dict(V2113_EXPECTED_STAGE_STATUS_COUNTS[stage_id])
            or receipt.get("terminal") is not True
            or receipt.get("denominator_terminal") is not True
            or receipt.get("go") is not expected["go"]
            or receipt.get("go_models") != expected["go_models"]
        ):
            raise PilotV2114ParentImportError(f"V2.11.3 {stage_id} receipt drifted")
        artifacts = receipt.get("artifacts")
        if (
            not isinstance(artifacts, Mapping)
            or artifacts.get("imported_effect_cells") != 0
            or (
                stage_id == "long-context-preflight"
                and (
                    artifacts.get("provider_construction") is not False
                    or artifacts.get("provider_calls_current_attempt") != 0
                    or receipt.get("failure", {}).get("error_type")
                    != V2113_FAILURE_TYPE
                    or receipt.get("failure", {}).get("message")
                    != V2113_FAILURE_MESSAGE
                    or receipt.get("failure", {}).get("provider_construction")
                    is not False
                    or receipt.get("failure", {}).get("provider_calls") != 0
                )
            )
        ):
            raise PilotV2114ParentImportError(
                f"V2.11.3 {stage_id} zero-provider boundary drifted"
            )
        result[stage_id] = _binding(relative, raw, receipt)
    return result


def _verify_v2113_parent_import_lineage(root: Path) -> dict[str, Any]:
    relative = V2113_RAW_ROOT / "parent-import/parent_import_receipt.json"
    raw, receipt = _read_parent_json(
        root, relative, name="V2.11.3 parent import receipt"
    )
    content = _verify_seal(
        receipt,
        schema="finevo-pilot-v2.11.3-parent-import-v1",
        name="V2.11.3 parent import receipt",
    )
    policy = receipt.get("import_policy")
    if (
        _sha256(raw) != V2113_PARENT_IMPORT_FILE_SHA256
        or content != V2113_PARENT_IMPORT_CONTENT_SHA256
        or receipt.get("child_release", {}).get("contract_sha256")
        != V2113_CONTRACT_SHA256
        or receipt.get("child_release", {}).get("resolved_git_commit")
        != V2113_SCIENCE_COMMIT
        or receipt.get("child_release", {}).get("git_tag") != V2113_SCIENCE_TAG
        or not isinstance(policy, Mapping)
        or policy.get("provider_construction_during_import") is not False
        or policy.get("provider_calls_during_import") != 0
        or policy.get("hosted_provider_calls_during_import") != 0
        or policy.get("imported_effect_cells") != 0
        or policy.get("historical_scientific_cells_imported") != 0
        or receipt.get("scientific_evidence") is not False
    ):
        raise PilotV2114ParentImportError(
            "V2.11.3 parent import lineage drifted"
        )
    return _binding(relative, raw, receipt)


def _audit_v2113_terminal_lineage(
    *,
    lineage_repo_root: str | Path,
) -> dict[str, Any]:
    root = _strict_root(lineage_repo_root, name="V2.11.3 terminal lineage source")
    git = _verify_v2113_git(root)
    contract, contract_binding = _load_v2113_contract(root)
    inventory = _verify_v2113_raw_inventory(root)
    launch = _verify_v2113_attestation_and_launch(root)
    _, run_binding = _verify_v2113_run_ledger(root, contract)
    _, budget_binding = _verify_v2113_budget_ledger(root)
    stage_receipts = _verify_v2113_stage_receipts(root)
    parent_import = _verify_v2113_parent_import_lineage(root)
    return {
        "lineage_root": root,
        "contract": contract,
        "release": git,
        "contract_binding": contract_binding,
        "inventory": inventory,
        **launch,
        "run_ledger": run_binding,
        "budget_ledger": budget_binding,
        "stage_receipts": stage_receipts,
        "parent_import_receipt": parent_import,
        "terminal_denominator": {
            "registered_cells": 136,
            "terminal_cells": 136,
            "status_counts": dict(V2113_EXPECTED_STATUS_COUNTS),
            "stage_status_counts": {
                stage: dict(counts)
                for stage, counts in V2113_EXPECTED_STAGE_STATUS_COUNTS.items()
            },
            "scientific_complete": False,
            "scientific_matrix_complete": False,
            "scientific_claim_gates_supported": False,
            "implementation_root_cause": "resealed-authority-provenance-comparison-gap",
            "current_provider_calls": 0,
            "current_hosted_cost_usd": 0.0,
            "actor_performance_treatment_outcome_blind": True,
            "globally_a_d_outcome_blind": True,
        },
    }


def _preflight_run_id(model_id: str) -> str:
    return (
        "finevo-pilot-v2.11.2--long-context-preflight--"
        f"{model_id}--closed-loop-preflight--none--stage0-selected--"
        "s2010922376"
    )


def _deep_verify_post_gate(
    root: Path,
    *,
    contract: PilotContract,
    parent_receipt: Mapping[str, Any],
    run_ledger: Mapping[str, Any],
    budget_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    relative = V2112_RAW_ROOT / "long-context-preflight/post_gate_authority.json"
    raw, observed = _read_parent_json(
        root, relative, name="V2.11.2 post-gate authority"
    )
    if (
        _sha256(raw) != V2112_POST_GATE_FILE_SHA256
        or observed.get("receipt_sha256") != V2112_POST_GATE_CONTENT_SHA256
    ):
        raise PilotV2114ParentImportError("V2.11.2 post-gate bytes drifted")
    try:
        verified_observed = verify_v2112_gate_receipt(
            observed,
            expected_contract_sha256=V2112_CONTRACT_SHA256,
            expected_git_commit=V2112_SCIENCE_COMMIT,
        )
        inherited = {
            model_id: verified_v2112_inherited_capability_binding(
                parent_receipt,
                model_id=model_id,
                repo_root=root,
                child_contract_sha256=V2112_CONTRACT_SHA256,
                child_git_tag=V2112_SCIENCE_TAG,
                child_git_commit=V2112_SCIENCE_COMMIT,
            )
            for model_id in V2114_ALLOWED_MODELS
        }
    except (PilotV2112GateError, PilotV2112ParentImportError) as exc:
        raise PilotV2114ParentImportError(
            f"V2.11.2 post-gate parent/capability replay failed: {exc}"
        ) from exc

    fresh: dict[str, dict[str, Any]] = {}
    for model_id in V2114_ALLOWED_MODELS:
        run_id = _preflight_run_id(model_id)
        run_root = V2112_RAW_ROOT / f"long-context-preflight/runs/{run_id}"
        checkpoint_raw, checkpoint = _read_parent_json(
            root, run_root / "preflight_checkpoint.json", name="preflight checkpoint"
        )
        exactness_raw, exactness = _read_parent_json(
            root,
            run_root / "preflight_checkpoint_exactness.json",
            name="preflight exactness",
        )
        journal_relative = V2112_RAW_ROOT / (
            "long-context-preflight/provider_call_journals/" f"{run_id}--preflight.json"
        )
        journal_raw, journal = _read_parent_json(
            root, journal_relative, name="preflight provider journal"
        )
        checkpoint_run_id = checkpoint.get("run_config", {}).get("run_id")
        spec = next(
            (
                item
                for item in contract.expand(
                    stage="long-context-preflight", model=model_id
                )
                if item.run_id == run_id
            ),
            None,
        )
        if spec is None or not isinstance(checkpoint_run_id, str):
            raise PilotV2114ParentImportError(
                f"V2.11.2 {model_id} preflight identity drifted"
            )
        fresh[model_id] = {
            "ledger_run_id": run_id,
            "checkpoint_run_id": checkpoint_run_id,
            "run_spec_sha256": canonical_sha256(spec.to_dict()),
            "checkpoint_artifact_sha256": canonical_sha256(checkpoint),
            "checkpoint": checkpoint,
            "exactness_artifact_sha256": canonical_sha256(exactness),
            "exactness": exactness,
            "provider_call_journal_artifact_sha256": canonical_sha256(journal),
            "provider_call_journal": journal,
        }
        # File hashes are bound independently from canonical artifact hashes.
        if not checkpoint_raw or not exactness_raw or not journal_raw:
            raise PilotV2114ParentImportError("empty V2.11.2 preflight source")

    science_ids: dict[str, list[str]] = {
        model_id: [] for model_id in V2114_ALLOWED_MODELS
    }
    for stage_id in (
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
        "cross-model",
    ):
        for spec in contract.expand(stage=stage_id):
            science_ids[spec.model_id].append(spec.run_id)

    bound_head = observed.get("bindings", {}).get("ledger_event_chain_head")
    events = run_ledger.get("events")
    if (
        not isinstance(bound_head, str)
        or not isinstance(events, list)
        or bound_head
        not in {
            event.get("event_sha256") for event in events if isinstance(event, Mapping)
        }
    ):
        raise PilotV2114ParentImportError("V2.11.2 post-gate ledger prefix is absent")
    pre_science_ids = {
        spec.run_id
        for stage in ("parent-import", "capability-gate", "long-context-preflight")
        for spec in contract.expand(stage=stage)
    }
    budget_rows = budget_ledger.get("runs")
    if not isinstance(budget_rows, Mapping):
        raise PilotV2114ParentImportError("V2.11.2 budget rows are malformed")
    pre_science_storage = 0
    for run_id in pre_science_ids:
        row = budget_rows.get(run_id)
        actual = row.get("actual") if isinstance(row, Mapping) else None
        if not isinstance(actual, Mapping) or not isinstance(
            actual.get("storage_bytes"), int
        ):
            raise PilotV2114ParentImportError(
                "V2.11.2 pre-science storage actual is missing"
            )
        pre_science_storage += int(actual["storage_bytes"])
    statuses = {
        model_id: str(observed["model_decisions"][model_id]["terminal_status"])
        for model_id in V2114_ALLOWED_MODELS
    }
    try:
        rebuilt = build_v2112_post_gate_authority(
            contract_sha256=V2112_CONTRACT_SHA256,
            release_tag=V2112_SCIENCE_TAG,
            release_commit=V2112_SCIENCE_COMMIT,
            parent_import_receipt_binding=observed["bindings"]["parent_import_receipt"],
            parent_budget_debit=budget_ledger["parent_debit"],
            inherited_capability_bindings=inherited,
            fresh_preflight_artifacts=fresh,
            model_terminal_statuses=statuses,
            current_attempt_pre_science_storage_bytes=pre_science_storage,
            ledger_event_chain_head=bound_head,
            science_run_ids_by_model=science_ids,
            source_manifest_hashes={
                "file_sha256": observed["bindings"]["source_manifest"]["file_sha256"],
                "content_sha256": observed["bindings"]["source_manifest"][
                    "content_sha256"
                ],
            },
        )
        flat = verified_v2112_gate_authority_binding(
            relative.as_posix(),
            repo_root=root,
            expected_git_commit=V2112_SCIENCE_COMMIT,
            expected_contract_sha256=V2112_CONTRACT_SHA256,
        )
    except PilotV2112GateError as exc:
        raise PilotV2114ParentImportError(
            f"V2.11.2 deep gate replay failed: {exc}"
        ) from exc
    if (
        rebuilt != verified_observed
        or flat.get("receipt_file_sha256") != V2112_POST_GATE_FILE_SHA256
        or flat.get("receipt_content_sha256") != V2112_POST_GATE_CONTENT_SHA256
        or set(flat.get("reservations", {}))
        != {"openai/gpt-5.2-2025-12-11", "openai/gpt-5.6-sol"}
    ):
        raise PilotV2114ParentImportError(
            "V2.11.2 post-gate differs from its complete source replay"
        )
    return {
        "receipt": verified_observed,
        "binding": flat,
        "source_binding": _binding(relative, raw, observed),
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
    }


def _verify_evidence_bytes(evidence_repo_root: Path) -> dict[str, Any]:
    namespace = PurePosixPath(_EVIDENCE_BINDINGS["namespace"])
    result = {
        "namespace": namespace.as_posix(),
        "merge_commit": _EVIDENCE_BINDINGS["merge_commit"],
        "publication_status": "complete-with-no-go",
        "scientific_claim_gates_supported": False,
        "semantic_parse_during_import": False,
        "reclassification": "forbidden",
    }
    for name, filename in (
        ("package_manifest", "package_manifest.json"),
        ("checksums", "checksums.json"),
        ("failure_ledger", "failure_ledger.json"),
        ("reviewer_report", "reviewer_report.md"),
    ):
        relative = namespace / filename
        raw = _read_regular(
            evidence_repo_root,
            relative,
            name=f"V2.11.2 evidence {filename}",
        )
        expected = _EVIDENCE_BINDINGS[f"{name}_file_sha256"]
        if _sha256(raw) != expected:
            raise PilotV2114ParentImportError(
                f"V2.11.2 evidence {filename} hash drifted"
            )
        result[name] = {
            "path": relative.as_posix(),
            "file_sha256": expected,
        }
    return result


def _audit_parent_release(
    *,
    parent_science_root: str | Path,
    evidence_repo_root: str | Path,
) -> dict[str, Any]:
    root = _strict_root(parent_science_root, name="V2.11.2 science source")
    evidence_root = _strict_root(evidence_repo_root, name="V2.11.2 evidence source")
    git = _verify_parent_git(root)
    contract, contract_binding = _load_parent_contract(root)
    inventory = _verify_raw_inventory(root)
    launch = _verify_attestation_and_launch(root)
    run_ledger, run_binding = _verify_run_ledger(root, contract)
    budget_ledger, budget_binding = _verify_budget_ledger(root)
    stage_receipts = _verify_stage_receipts(root)

    parent_receipt_path = V2112_RAW_ROOT / "parent-import/parent_import_receipt.json"
    parent_raw, _ = _read_parent_json(
        root, parent_receipt_path, name="V2.11.2 parent import receipt"
    )
    if _sha256(parent_raw) != V2112_PARENT_IMPORT_FILE_SHA256:
        raise PilotV2114ParentImportError("V2.11.2 parent import bytes drifted")
    try:
        parent_receipt = verify_v2112_parent_import_receipt(
            root.joinpath(*parent_receipt_path.parts),
            repo_root=root,
            child_contract_sha256=V2112_CONTRACT_SHA256,
            child_git_tag=V2112_SCIENCE_TAG,
            child_git_commit=V2112_SCIENCE_COMMIT,
        )
        calibration = calibration_wrapper_from_v2112_receipt(parent_receipt)
        capabilities = capability_wrappers_from_v2112_receipt(parent_receipt)
    except PilotV2112ParentImportError as exc:
        raise PilotV2114ParentImportError(
            f"V2.11.2 parent import replay failed: {exc}"
        ) from exc
    if parent_receipt.get("integrity", {}).get(
        "content_sha256"
    ) != V2112_PARENT_IMPORT_CONTENT_SHA256 or set(capabilities) != set(
        V2114_ALLOWED_MODELS
    ):
        raise PilotV2114ParentImportError("V2.11.2 reusable wrappers drifted")
    gate = _deep_verify_post_gate(
        root,
        contract=contract,
        parent_receipt=parent_receipt,
        run_ledger=run_ledger,
        budget_ledger=budget_ledger,
    )
    evidence = _verify_evidence_bytes(evidence_root)
    return {
        "parent_root": root,
        "contract": contract,
        "release": git,
        "contract_binding": contract_binding,
        "inventory": inventory,
        **launch,
        "run_ledger": run_binding,
        "budget_ledger": budget_binding,
        "stage_receipts": stage_receipts,
        "parent_import_binding": {
            "path": parent_receipt_path.as_posix(),
            "file_sha256": V2112_PARENT_IMPORT_FILE_SHA256,
            "content_sha256": V2112_PARENT_IMPORT_CONTENT_SHA256,
        },
        "calibration_wrapper": calibration,
        "capability_wrappers": capabilities,
        "post_gate": gate,
        "evidence": evidence,
    }


def build_v2114_source_manifest(
    *,
    source_repo_root: str | Path,
    lineage_repo_root: str | Path,
    evidence_repo_root: str | Path,
) -> dict[str, Any]:
    """Build the tracked manifest from V2.11.2 authority and V2.11.3 lineage."""

    audit = _audit_parent_release(
        parent_science_root=source_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    lineage = _audit_v2113_terminal_lineage(
        lineage_repo_root=lineage_repo_root,
    )
    gate_binding = audit["post_gate"]["binding"]
    reservations = gate_binding["reservations"]
    models: dict[str, Any] = {}
    for model_id, runtime_model in (
        ("gpt52_main", "openai/gpt-5.2-2025-12-11"),
        ("gpt56_diagnostic", "openai/gpt-5.6-sol"),
    ):
        models[model_id] = {
            "runtime_model": runtime_model,
            "capability_wrapper_content_sha256": audit["capability_wrappers"][model_id][
                "integrity"
            ]["content_sha256"],
            "source_gate_receipt_file_sha256": gate_binding["receipt_file_sha256"],
            "source_gate_receipt_content_sha256": gate_binding[
                "receipt_content_sha256"
            ],
            "source_reservation_sha256": canonical_sha256(reservations[runtime_model]),
            "sample_counts": {"action": 24, "semantic": 8},
        }
    return _seal(
        {
            "schema_version": V2114_SOURCE_MANIFEST_SCHEMA_VERSION,
            "authority_parent_release": {
                "contract_id": V2112_CONTRACT_ID,
                "contract_sha256": V2112_CONTRACT_SHA256,
                **audit["release"],
                "publication_status": "immutable-terminal-no-go",
                "contract": audit["contract_binding"],
                "release_attestation": audit["release_attestation"],
                "scientific_launch_input": audit["scientific_launch_input"],
                "raw_inventory": {
                    key: value
                    for key, value in audit["inventory"].items()
                    if key != "rows"
                },
                "run_ledger": audit["run_ledger"],
                "budget_ledger": audit["budget_ledger"],
                "stage_receipts": audit["stage_receipts"],
                "parent_import_receipt": audit["parent_import_binding"],
                "post_gate_authority": audit["post_gate"]["source_binding"],
            },
            "terminal_lineage_release": {
                "contract_id": V2113_CONTRACT_ID,
                "contract_sha256": V2113_CONTRACT_SHA256,
                **lineage["release"],
                "publication_status": "immutable-terminal-zero-call-no-go",
                "contract": lineage["contract_binding"],
                "release_attestation": lineage["release_attestation"],
                "scientific_launch_input": lineage["scientific_launch_input"],
                "raw_inventory": {
                    key: value
                    for key, value in lineage["inventory"].items()
                    if key != "rows"
                },
                "run_ledger": lineage["run_ledger"],
                "budget_ledger": lineage["budget_ledger"],
                "stage_receipts": lineage["stage_receipts"],
                "parent_import_receipt": lineage["parent_import_receipt"],
                "preflight_no_go_receipt": lineage["stage_receipts"][
                    "long-context-preflight"
                ],
            },
            "published_parent_evidence": audit["evidence"],
            "authority_source_denominator": {
                "registered_cells": 136,
                "terminal_cells": 136,
                "status_counts": dict(V2112_EXPECTED_STATUS_COUNTS),
                "stage_status_counts": {
                    stage: dict(counts)
                    for stage, counts in V2112_EXPECTED_STAGE_STATUS_COUNTS.items()
                },
                "scientific_complete": False,
                "scientific_matrix_complete": False,
                "scientific_claim_gates_supported": False,
                "implementation_root_cause": (
                    "observed-p95-consumer-schema-dispatch-gap"
                ),
                "failed_scientific_cells": 126,
                "offline_candidate_admission_cells_observed": 5,
                "actor_performance_treatment_outcome_blind": True,
                "globally_a_d_outcome_blind": False,
            },
            "terminal_denominator": lineage["terminal_denominator"],
            "parsed_source_allowlist": list(V2114_PARSED_PARENT_SOURCE_ALLOWLIST),
            "reusable_authority_allowlist": list(V2114_REUSABLE_AUTHORITY_KINDS),
            "forbidden_import_prefixes": list(V2114_FORBIDDEN_IMPORT_PREFIXES),
            "calibration_source": {
                "schema_version": V2112_CALIBRATION_WRAPPER_SCHEMA_VERSION,
                "content_sha256": audit["calibration_wrapper"]["integrity"][
                    "content_sha256"
                ],
                "q_ref": 63.50397933257746,
                "selected_profile_id": "nu-0.5",
                "absolute_flow_utility_threshold": 0.05617208967516696,
                "scientific_evidence": False,
            },
            "model_authorities": models,
            "source_gate_binding": gate_binding,
            "cumulative_budget_debit": V2114_CUMULATIVE_DEBIT.to_dict(),
            "import_policy": {
                **_ZERO_PROVIDER_POLICY,
                "imported_calibration_wrappers": 1,
                "imported_capability_wrappers": 2,
                "imported_preflight_authorities": 2,
                "historical_capability_calls": 60,
                "historical_preflight_calls": 64,
                "historical_preflight_calls_already_in_parent_debit": 64,
                "historical_scientific_cells_imported": 0,
                "v2113_terminal_lineage_cells_audited": 136,
                "v2113_scientific_cells_imported": 0,
                "v2113_provider_calls_current_attempt": 0,
                "v2113_hosted_cost_usd_current_attempt": 0.0,
                "validation_before_provider_construction": True,
                "source_raw_tree_copied": False,
                "source_raw_trees_copied": False,
                "lineage_raw_tree_copied": False,
                "source_files_semantically_parsed_only_if_allowlisted": True,
            },
            "observation_boundary": {
                "v2_11_2_actor_performance_outcomes_generated": False,
                "v2_11_2_offline_candidate_metrics_inspected": True,
                "v2_11_2_offline_candidate_cells_must_be_freshly_rerun": 5,
                "v2_11_2_scientific_cell_reuse": "forbidden",
                "v2_11_3_scientific_cell_reuse": "forbidden",
                "v2_11_3_terminal_failure_is_lineage_only": True,
                "v2_11_3_actor_performance_outcomes_generated": False,
                "failed_seed_replacement": "forbidden",
                "matrix_shrink": "forbidden",
                "fresh_v2_11_4_scientific_cells": 131,
                "source_failure_is_treatment_effect_evidence": False,
                "source_gate_is_dispatch_budget_authority_only": True,
            },
        }
    )


def _load_tracked_manifest(repo_root: Path) -> dict[str, Any]:
    if (
        V2114_SOURCE_MANIFEST_FILE_SHA256 is None
        or V2114_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotV2114ParentImportError(
            "V2.11.4 source manifest hashes are not sealed"
        )
    raw, value = _read_json(
        repo_root,
        V2114_SOURCE_MANIFEST_PATH,
        name="tracked V2.11.4 source manifest",
    )
    content = _verify_seal(
        value,
        schema=V2114_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="tracked V2.11.4 source manifest",
    )
    canonical = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if (
        raw != canonical
        or _sha256(raw) != V2114_SOURCE_MANIFEST_FILE_SHA256
        or content != V2114_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV2114ParentImportError(
            "tracked V2.11.4 source manifest hash/canonicalization drifted"
        )
    return value


def write_v2114_source_manifest_draft(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    _verify_seal(
        manifest,
        schema=V2114_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.11.4 source manifest draft",
    )
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise PilotV2114ParentImportError(
            f"refusing to overwrite V2.11.4 source manifest: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "wb", closefd=True) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def load_v2114_source_manifest(*, repo_root: str | Path) -> dict[str, Any]:
    return _load_tracked_manifest(_strict_root(repo_root, name="V2.11.4 repository"))


def validate_v2114_source_manifest(
    value: Mapping[str, Any],
    *,
    source_repo_root: str | Path,
    lineage_repo_root: str | Path,
    evidence_repo_root: str | Path,
) -> dict[str, Any]:
    selected = _json_copy(dict(value))
    _verify_seal(
        selected,
        schema=V2114_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.11.4 source manifest",
    )
    expected = build_v2114_source_manifest(
        source_repo_root=source_repo_root,
        lineage_repo_root=lineage_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    if selected != expected:
        raise PilotV2114ParentImportError(
            "V2.11.4 source manifest differs from immutable authority/lineage replay"
        )
    return selected


def _validate_child(
    *,
    contract: PilotContract,
    child_git_commit: str,
    require_frozen: bool,
) -> dict[str, str]:
    if (
        getattr(contract, "contract_id", None) != V2114_CONTRACT_ID
        or getattr(contract, "implementation", {}).get("required_git_tag")
        != V2114_SCIENCE_TAG
        or _COMMIT_RE.fullmatch(child_git_commit) is None
        or (require_frozen and getattr(contract, "status", None) != "frozen")
        or (
            not require_frozen
            and getattr(contract, "status", None) not in {"draft", "frozen"}
        )
    ):
        raise PilotV2114ParentImportError(
            "V2.11.4 import requires its exact release-bound contract"
        )
    return {
        "contract_id": V2114_CONTRACT_ID,
        "contract_sha256": contract.canonical_hash,
        "git_tag": V2114_SCIENCE_TAG,
        "resolved_git_commit": child_git_commit,
    }


def _source_manifest_binding() -> dict[str, str]:
    if (
        V2114_SOURCE_MANIFEST_FILE_SHA256 is None
        or V2114_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotV2114ParentImportError(
            "V2.11.4 source manifest hashes are not sealed"
        )
    return {
        "path": V2114_SOURCE_MANIFEST_PATH.as_posix(),
        "file_sha256": V2114_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": V2114_SOURCE_MANIFEST_CONTENT_SHA256,
    }


def _parent_release_binding() -> dict[str, str]:
    return {
        "contract_id": V2112_CONTRACT_ID,
        "contract_sha256": V2112_CONTRACT_SHA256,
        "git_tag": V2112_SCIENCE_TAG,
        "git_tag_object": V2112_SCIENCE_TAG_OBJECT,
        "resolved_git_commit": V2112_SCIENCE_COMMIT,
    }


def _terminal_lineage_binding() -> dict[str, str]:
    return {
        "contract_id": V2113_CONTRACT_ID,
        "contract_sha256": V2113_CONTRACT_SHA256,
        "git_tag": V2113_SCIENCE_TAG,
        "git_tag_object": V2113_SCIENCE_TAG_OBJECT,
        "resolved_git_commit": V2113_SCIENCE_COMMIT,
    }


def _build_child_wrappers(
    *,
    audit: Mapping[str, Any],
    child: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    calibration_source = audit["calibration_wrapper"]
    calibration = _seal(
        {
            "schema_version": V2114_CALIBRATION_WRAPPER_SCHEMA_VERSION,
            "child_release": _json_copy(child),
            "parent_release": _parent_release_binding(),
            "source_manifest": _source_manifest_binding(),
            "source_wrapper": {
                "schema_version": V2112_CALIBRATION_WRAPPER_SCHEMA_VERSION,
                "content_sha256": calibration_source["integrity"]["content_sha256"],
                "parent_import_receipt": {
                    "path": (
                        V2112_RAW_ROOT / "parent-import/parent_import_receipt.json"
                    ).as_posix(),
                    "file_sha256": V2112_PARENT_IMPORT_FILE_SHA256,
                    "content_sha256": V2112_PARENT_IMPORT_CONTENT_SHA256,
                },
            },
            "calibration": _json_copy(calibration_source["calibration"]),
            **_ZERO_PROVIDER_POLICY,
            "scientific_evidence": False,
            "evidence_use": (
                "Outcome-blind q-ref, selected utility profile, and absolute "
                "flow-utility threshold only."
            ),
        }
    )
    capabilities: dict[str, Any] = {}
    preflights: dict[str, Any] = {}
    gate_binding = audit["post_gate"]["binding"]
    runtime_by_model = {
        "gpt52_main": "openai/gpt-5.2-2025-12-11",
        "gpt56_diagnostic": "openai/gpt-5.6-sol",
    }
    for model_id in V2114_ALLOWED_MODELS:
        source = audit["capability_wrappers"][model_id]
        capabilities[model_id] = _seal(
            {
                "schema_version": V2114_CAPABILITY_WRAPPER_SCHEMA_VERSION,
                "child_release": _json_copy(child),
                "parent_release": _parent_release_binding(),
                "source_manifest": _source_manifest_binding(),
                "source_wrapper": {
                    "schema_version": V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION,
                    "content_sha256": source["integrity"]["content_sha256"],
                },
                "capability": _json_copy(source["capability"]),
                "provider_construction_current_attempt": False,
                "provider_calls_current_attempt": 0,
                "hosted_provider_calls_current_attempt": 0,
                "current_attempt_usage": dict(_ZERO_USAGE),
                "imported_effect_cells": 0,
                "imported_preflight_samples": 0,
                "scientific_evidence": False,
                "evidence_scope": "preregistered_task_capability_gate",
            }
        )
        runtime_model = runtime_by_model[model_id]
        preflights[model_id] = _seal(
            {
                "schema_version": V2114_PREFLIGHT_WRAPPER_SCHEMA_VERSION,
                "child_release": _json_copy(child),
                "parent_release": _parent_release_binding(),
                "source_manifest": _source_manifest_binding(),
                "model_id": model_id,
                "runtime_model": runtime_model,
                "source_gate_receipt": {
                    "path": gate_binding["receipt_path"],
                    "file_sha256": gate_binding["receipt_file_sha256"],
                    "content_sha256": gate_binding["receipt_content_sha256"],
                    "git_commit": gate_binding["git_commit"],
                },
                "reservations": _json_copy(gate_binding["reservations"][runtime_model]),
                "source_reservation_sha256": canonical_sha256(
                    gate_binding["reservations"][runtime_model]
                ),
                "sample_counts": {"action": 24, "semantic": 8},
                "provider_construction_current_attempt": False,
                "provider_calls_current_attempt": 0,
                "hosted_provider_calls_current_attempt": 0,
                "historical_provider_calls": 32,
                "historical_calls_already_in_parent_debit": True,
                "imported_effect_cells": 0,
                "scientific_evidence": False,
                "evidence_use": (
                    "Prospective V2.11.4 dispatch-budget authority only; no "
                    "V2.11.2 scientific outcome is imported."
                ),
            }
        )
    return calibration, capabilities, preflights


def build_v2114_parent_import(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    child_git_commit: str,
    source_repo_root: str | Path,
    lineage_repo_root: str | Path,
    evidence_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact current-release receipt after exact source replay."""

    child_root = _strict_root(repo_root, name="V2.11.4 repository")
    child = _validate_child(
        contract=contract,
        child_git_commit=child_git_commit,
        require_frozen=True,
    )
    tracked = _load_tracked_manifest(child_root)
    evidence_root = child_root if evidence_repo_root is None else evidence_repo_root
    audit = _audit_parent_release(
        parent_science_root=source_repo_root,
        evidence_repo_root=evidence_root,
    )
    _audit_v2113_terminal_lineage(lineage_repo_root=lineage_repo_root)
    rebuilt_manifest = build_v2114_source_manifest(
        source_repo_root=source_repo_root,
        lineage_repo_root=lineage_repo_root,
        evidence_repo_root=evidence_root,
    )
    if tracked != rebuilt_manifest:
        raise PilotV2114ParentImportError(
            "tracked V2.11.4 source manifest differs from immutable replay"
        )
    calibration, capabilities, preflights = _build_child_wrappers(
        audit=audit,
        child=child,
    )
    return _seal(
        {
            "schema_version": V2114_PARENT_IMPORT_SCHEMA_VERSION,
            "child_release": child,
            "source_manifest": _source_manifest_binding(),
            "authority_parent_release": {
                **_parent_release_binding(),
                "release_attestation_sha256": V2112_RELEASE_ATTESTATION_SHA256,
                "run_ledger_sha256": V2112_RUN_LEDGER_SHA256,
                "budget_ledger_sha256": V2112_BUDGET_LEDGER_SHA256,
                "publication_status": "immutable-terminal-no-go",
            },
            "terminal_lineage_release": {
                **_terminal_lineage_binding(),
                "release_attestation_sha256": V2113_RELEASE_ATTESTATION_SHA256,
                "run_ledger_sha256": V2113_RUN_LEDGER_SHA256,
                "budget_ledger_sha256": V2113_BUDGET_LEDGER_SHA256,
                "preflight_receipt_content_sha256": (
                    V2113_PREFLIGHT_RECEIPT_CONTENT_SHA256
                ),
                "publication_status": "immutable-terminal-zero-call-no-go",
            },
            "terminal_parent_denominator": _json_copy(tracked["terminal_denominator"]),
            "authority_source_denominator": _json_copy(
                tracked["authority_source_denominator"]
            ),
            "calibration_wrapper": calibration,
            "capability_wrappers": capabilities,
            "preflight_authority_wrappers": preflights,
            "cumulative_parent_budget_debit": V2114_CUMULATIVE_DEBIT.to_dict(),
            "import_policy": _json_copy(tracked["import_policy"]),
            "scientific_evidence": False,
            "claim_boundary": (
                "V2.11.4 reuses only calibration, capability/interface, and "
                "fresh-preflight observed-p95 budget authority directly from "
                "V2.11.2. V2.11.3 contributes only terminal no-go and budget "
                "lineage. All 131 A-D/cross-model cells are new V2.11.4 cells; "
                "all earlier scientific outcomes and decoded completions are "
                "excluded."
            ),
        }
    )


def validate_v2114_parent_import_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: PilotContract,
    child_git_commit: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate a child-bound receipt without reopening the parent checkout."""

    root = _strict_root(repo_root, name="V2.11.4 repository")
    value = _json_copy(dict(receipt))
    _verify_seal(
        value,
        schema=V2114_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11.4 parent import receipt",
    )
    child = _validate_child(
        contract=contract,
        child_git_commit=child_git_commit,
        require_frozen=True,
    )
    manifest = _load_tracked_manifest(root)
    expected_top_level = {
        "schema_version",
        "child_release",
        "source_manifest",
        "authority_parent_release",
        "terminal_lineage_release",
        "terminal_parent_denominator",
        "authority_source_denominator",
        "calibration_wrapper",
        "capability_wrappers",
        "preflight_authority_wrappers",
        "cumulative_parent_budget_debit",
        "import_policy",
        "scientific_evidence",
        "claim_boundary",
        "integrity",
    }
    if (
        set(value) != expected_top_level
        or value.get("child_release") != child
        or value.get("source_manifest") != _source_manifest_binding()
        or value.get("authority_parent_release")
        != {
            **_parent_release_binding(),
            "release_attestation_sha256": V2112_RELEASE_ATTESTATION_SHA256,
            "run_ledger_sha256": V2112_RUN_LEDGER_SHA256,
            "budget_ledger_sha256": V2112_BUDGET_LEDGER_SHA256,
            "publication_status": "immutable-terminal-no-go",
        }
        or value.get("terminal_lineage_release")
        != {
            **_terminal_lineage_binding(),
            "release_attestation_sha256": V2113_RELEASE_ATTESTATION_SHA256,
            "run_ledger_sha256": V2113_RUN_LEDGER_SHA256,
            "budget_ledger_sha256": V2113_BUDGET_LEDGER_SHA256,
            "preflight_receipt_content_sha256": (
                V2113_PREFLIGHT_RECEIPT_CONTENT_SHA256
            ),
            "publication_status": "immutable-terminal-zero-call-no-go",
        }
        or value.get("terminal_parent_denominator") != manifest["terminal_denominator"]
        or value.get("authority_source_denominator")
        != manifest["authority_source_denominator"]
        or value.get("cumulative_parent_budget_debit")
        != V2114_CUMULATIVE_DEBIT.to_dict()
        or value.get("import_policy") != manifest["import_policy"]
        or value.get("scientific_evidence") is not False
    ):
        raise PilotV2114ParentImportError(
            "V2.11.4 parent import release/claim boundary drifted"
        )
    calibration = value.get("calibration_wrapper")
    capabilities = value.get("capability_wrappers")
    preflights = value.get("preflight_authority_wrappers")
    if (
        not isinstance(calibration, Mapping)
        or not isinstance(capabilities, Mapping)
        or set(capabilities) != set(V2114_ALLOWED_MODELS)
        or not isinstance(preflights, Mapping)
        or set(preflights) != set(V2114_ALLOWED_MODELS)
    ):
        raise PilotV2114ParentImportError("V2.11.4 wrapper denominator drifted")
    _verify_seal(
        calibration,
        schema=V2114_CALIBRATION_WRAPPER_SCHEMA_VERSION,
        name="V2.11.4 calibration wrapper",
    )
    if (
        calibration.get("child_release") != child
        or calibration.get("parent_release") != _parent_release_binding()
        or calibration.get("source_manifest") != _source_manifest_binding()
        or calibration.get("source_wrapper", {}).get("content_sha256")
        != manifest["calibration_source"]["content_sha256"]
        or calibration.get("scientific_evidence") is not False
    ):
        raise PilotV2114ParentImportError(
            "V2.11.4 calibration wrapper source boundary drifted"
        )
    for model_id in V2114_ALLOWED_MODELS:
        _verify_seal(
            capabilities[model_id],
            schema=V2114_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            name=f"V2.11.4 {model_id} capability wrapper",
        )
        _verify_seal(
            preflights[model_id],
            schema=V2114_PREFLIGHT_WRAPPER_SCHEMA_VERSION,
            name=f"V2.11.4 {model_id} preflight wrapper",
        )
        wrapper = preflights[model_id]
        capability = capabilities[model_id]
        expected_model = manifest["model_authorities"][model_id]
        runtime_model = expected_model["runtime_model"]
        source_gate = manifest["source_gate_binding"]
        if (
            capability.get("child_release") != child
            or capability.get("parent_release") != _parent_release_binding()
            or capability.get("source_manifest") != _source_manifest_binding()
            or capability.get("source_wrapper", {}).get("content_sha256")
            != expected_model["capability_wrapper_content_sha256"]
            or capability.get("provider_construction_current_attempt") is not False
            or capability.get("provider_calls_current_attempt") != 0
            or capability.get("imported_effect_cells") != 0
            or capability.get("scientific_evidence") is not False
            or wrapper.get("child_release") != child
            or wrapper.get("parent_release") != _parent_release_binding()
            or wrapper.get("source_manifest") != _source_manifest_binding()
            or wrapper.get("model_id") != model_id
            or wrapper.get("runtime_model") != runtime_model
            or wrapper.get("source_gate_receipt")
            != {
                "path": source_gate["receipt_path"],
                "file_sha256": source_gate["receipt_file_sha256"],
                "content_sha256": source_gate["receipt_content_sha256"],
                "git_commit": source_gate["git_commit"],
            }
            or wrapper.get("reservations") != source_gate["reservations"][runtime_model]
            or wrapper.get("source_reservation_sha256")
            != expected_model["source_reservation_sha256"]
            or wrapper.get("sample_counts") != {"action": 24, "semantic": 8}
            or wrapper.get("provider_construction_current_attempt") is not False
            or wrapper.get("provider_calls_current_attempt") != 0
            or wrapper.get("imported_effect_cells") != 0
            or wrapper.get("scientific_evidence") is not False
            or canonical_sha256(wrapper.get("reservations"))
            != wrapper.get("source_reservation_sha256")
        ):
            raise PilotV2114ParentImportError(
                f"V2.11.4 {model_id} preflight wrapper scope drifted"
            )
    return value


def verify_v2114_parent_import_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    contract: PilotContract,
    child_git_commit: str | None = None,
    source_repo_root: str | Path | None = None,
    lineage_repo_root: str | Path | None = None,
    evidence_repo_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Replay the immutable parent and require exact receipt equality."""

    root = _strict_root(repo_root, name="V2.11.4 repository")
    commit = child_git_commit or expected_git_commit
    if commit is None or (
        child_git_commit is not None
        and expected_git_commit is not None
        and child_git_commit != expected_git_commit
    ):
        raise PilotV2114ParentImportError(
            "V2.11.4 parent receipt requires one exact child commit"
        )
    if raw_root is not None:
        raw = Path(raw_root)
        if not raw.is_absolute():
            raw = root.joinpath(*PurePosixPath(str(raw_root)).parts)
        if raw.absolute() != root.joinpath(*V2114_RAW_ROOT.parts):
            raise PilotV2114ParentImportError(
                "V2.11.4 parent receipt raw namespace drifted"
            )
    if isinstance(receipt, Mapping):
        observed = _json_copy(dict(receipt))
    else:
        path = Path(receipt).absolute()
        expected = root.joinpath(*V2114_DEFAULT_RECEIPT_PATH.parts)
        if path != expected:
            raise PilotV2114ParentImportError(
                "V2.11.4 parent receipt path differs from raw namespace"
            )
        raw = _read_regular(
            root,
            V2114_DEFAULT_RECEIPT_PATH,
            name="V2.11.4 parent import receipt",
        )
        observed = _strict_json(raw, name="V2.11.4 parent import receipt")
    validate_v2114_parent_import_receipt(
        observed,
        contract=contract,
        child_git_commit=commit,
        repo_root=root,
    )
    if source_repo_root is None and lineage_repo_root is None:
        return observed
    if source_repo_root is None or lineage_repo_root is None:
        raise PilotV2114ParentImportError(
            "V2.11.4 exact replay requires both authority and lineage checkouts"
        )
    expected = build_v2114_parent_import(
        repo_root=root,
        contract=contract,
        child_git_commit=commit,
        source_repo_root=source_repo_root,
        lineage_repo_root=lineage_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    if observed != expected:
        raise PilotV2114ParentImportError(
            "V2.11.4 parent import receipt differs from exact source replay"
        )
    return observed


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    try:
        _atomic_exact_bytes_no_follow(
            repo_root=Path(repo_root),
            path=path,
            raw=raw,
        )
    except PilotV27Stage0ImportError as exc:
        raise PilotV2114ParentImportError(
            f"immutable V2.11.4 write failed: {exc}"
        ) from exc


def persist_v2114_parent_import(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    child_git_commit: str | None = None,
    evidence_repo_root: str | Path | None = None,
    destination: str | Path | None = None,
    raw_root: str | Path | None = None,
    git_commit: str | None = None,
    source_repo_root: str | Path | None = None,
    lineage_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist only the parent-import receipt, without constructing a provider.

    The two child-bound observed-p95 pairs are deliberately persisted by the
    two registered ``long-context-preflight`` authority-import cells.  Keeping
    that boundary explicit preserves the operational denominator and stage
    provenance.
    """

    root = _strict_root(repo_root, name="V2.11.4 repository")
    commit = child_git_commit or git_commit
    if (
        commit is None
        or source_repo_root is None
        or lineage_repo_root is None
        or (
            child_git_commit is not None
            and git_commit is not None
            and child_git_commit != git_commit
        )
    ):
        raise PilotV2114ParentImportError(
            "V2.11.4 parent import requires one exact commit plus authority and lineage checkouts"
        )
    if raw_root is not None:
        raw = Path(raw_root)
        if not raw.is_absolute():
            raw = root.joinpath(*PurePosixPath(str(raw_root)).parts)
        if raw.absolute() != root.joinpath(*V2114_RAW_ROOT.parts):
            raise PilotV2114ParentImportError(
                "V2.11.4 parent import raw namespace drifted"
            )
    receipt = build_v2114_parent_import(
        repo_root=root,
        contract=contract,
        child_git_commit=commit,
        source_repo_root=source_repo_root,
        lineage_repo_root=lineage_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    path = (
        root.joinpath(*V2114_DEFAULT_RECEIPT_PATH.parts)
        if destination is None
        else Path(destination).absolute()
    )
    if path != root.joinpath(*V2114_DEFAULT_RECEIPT_PATH.parts):
        raise PilotV2114ParentImportError(
            "V2.11.4 receipt destination differs from raw namespace"
        )
    _atomic_json(path, receipt, repo_root=root)
    raw = path.read_bytes()
    return {
        "receipt": str(path),
        "receipt_file_sha256": _sha256(raw),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        **_ZERO_PROVIDER_POLICY,
        "scientific_evidence": False,
    }


def parent_budget_debit_for_v2114(contract: Any = None) -> ParentBudgetDebit | None:
    if getattr(contract, "contract_id", None) != V2114_CONTRACT_ID:
        return None
    return V2114_CUMULATIVE_DEBIT


def calibration_wrapper_from_v2114_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: PilotContract | None = None,
    expected_git_commit: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    if contract is not None or expected_git_commit is not None or repo_root is not None:
        if contract is None or expected_git_commit is None or repo_root is None:
            raise PilotV2114ParentImportError(
                "V2.11.4 wrapper validation requires root, contract, and commit"
            )
        validate_v2114_parent_import_receipt(
            receipt,
            contract=contract,
            child_git_commit=expected_git_commit,
            repo_root=repo_root,
        )
    wrapper = receipt.get("calibration_wrapper")
    if not isinstance(wrapper, Mapping):
        raise PilotV2114ParentImportError("V2.11.4 calibration wrapper is absent")
    _verify_seal(
        wrapper,
        schema=V2114_CALIBRATION_WRAPPER_SCHEMA_VERSION,
        name="V2.11.4 calibration wrapper",
    )
    return _json_copy(wrapper)


def capability_wrappers_from_v2114_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: PilotContract | None = None,
    expected_git_commit: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    if contract is not None or expected_git_commit is not None or repo_root is not None:
        if contract is None or expected_git_commit is None or repo_root is None:
            raise PilotV2114ParentImportError(
                "V2.11.4 wrapper validation requires root, contract, and commit"
            )
        validate_v2114_parent_import_receipt(
            receipt,
            contract=contract,
            child_git_commit=expected_git_commit,
            repo_root=repo_root,
        )
    wrappers = receipt.get("capability_wrappers")
    if not isinstance(wrappers, Mapping) or set(wrappers) != set(V2114_ALLOWED_MODELS):
        raise PilotV2114ParentImportError("V2.11.4 capability wrappers drifted")
    result: dict[str, dict[str, Any]] = {}
    for model_id in V2114_ALLOWED_MODELS:
        wrapper = wrappers[model_id]
        if not isinstance(wrapper, Mapping):
            raise PilotV2114ParentImportError("capability wrapper is malformed")
        _verify_seal(
            wrapper,
            schema=V2114_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            name=f"V2.11.4 {model_id} capability wrapper",
        )
        result[model_id] = _json_copy(wrapper)
    return result


def preflight_wrappers_from_v2114_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: PilotContract | None = None,
    expected_git_commit: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    if contract is not None or expected_git_commit is not None or repo_root is not None:
        if contract is None or expected_git_commit is None or repo_root is None:
            raise PilotV2114ParentImportError(
                "V2.11.4 wrapper validation requires root, contract, and commit"
            )
        validate_v2114_parent_import_receipt(
            receipt,
            contract=contract,
            child_git_commit=expected_git_commit,
            repo_root=repo_root,
        )
    wrappers = receipt.get("preflight_authority_wrappers")
    if not isinstance(wrappers, Mapping) or set(wrappers) != set(V2114_ALLOWED_MODELS):
        raise PilotV2114ParentImportError("V2.11.4 preflight wrappers drifted")
    result: dict[str, dict[str, Any]] = {}
    for model_id in V2114_ALLOWED_MODELS:
        wrapper = wrappers[model_id]
        if not isinstance(wrapper, Mapping):
            raise PilotV2114ParentImportError("preflight wrapper is malformed")
        _verify_seal(
            wrapper,
            schema=V2114_PREFLIGHT_WRAPPER_SCHEMA_VERSION,
            name=f"V2.11.4 {model_id} preflight wrapper",
        )
        if canonical_sha256(wrapper.get("reservations")) != wrapper.get(
            "source_reservation_sha256"
        ):
            raise PilotV2114ParentImportError(
                f"V2.11.4 {model_id} preflight reservation hash drifted"
            )
        result[model_id] = _json_copy(wrapper)
    return result


__all__ = [
    "PilotV2114ParentImportError",
    "V2112_CONTRACT_SHA256",
    "V2112_SCIENCE_COMMIT",
    "V2112_SCIENCE_TAG",
    "V2114_ALLOWED_MODELS",
    "V2114_CALIBRATION_WRAPPER_SCHEMA_VERSION",
    "V2114_CAPABILITY_WRAPPER_SCHEMA_VERSION",
    "V2114_CONTRACT_ID",
    "V2114_CONTRACT_PATH",
    "V2114_CUMULATIVE_DEBIT",
    "V2114_PARENT_IMPORT_SCHEMA_VERSION",
    "V2114_PREFLIGHT_WRAPPER_SCHEMA_VERSION",
    "V2114_RAW_ROOT",
    "V2114_SCIENCE_TAG",
    "V2114_SOURCE_MANIFEST_CONTENT_SHA256",
    "V2114_SOURCE_MANIFEST_FILE_SHA256",
    "V2114_SOURCE_MANIFEST_PATH",
    "V2114_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v2114_parent_import",
    "build_v2114_source_manifest",
    "calibration_wrapper_from_v2114_receipt",
    "capability_wrappers_from_v2114_receipt",
    "load_v2114_source_manifest",
    "parent_budget_debit_for_v2114",
    "persist_v2114_parent_import",
    "preflight_wrappers_from_v2114_receipt",
    "validate_v2114_parent_import_receipt",
    "validate_v2114_source_manifest",
    "verify_v2114_parent_import_receipt",
    "write_v2114_source_manifest_draft",
]
