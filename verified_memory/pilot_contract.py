"""Frozen, hash-bound execution contract for the FinEvo mechanism pilot.

The contract is deliberately independent from the simulation runner.  It
validates the pre-registered model/request identities, seed registry, stage
matrix, interventions, budgets, and stop/go rules before a paid provider can
be constructed.  ``experiments/pilot_v1.yaml`` is JSON-compatible YAML and is
therefore parsed with the standard library rather than adding a configuration
dependency to the execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


PILOT_CONTRACT_SCHEMA_VERSION_V1 = "finevo-pilot-contract-v1"
PILOT_CONTRACT_SCHEMA_VERSION_V2 = "finevo-pilot-contract-v2"
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1 = (
    "finevo-pilot-contract-v2.1-amendment-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_2 = (
    "finevo-pilot-contract-v2.2-evaluator-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_3 = (
    "finevo-pilot-contract-v2.3-preflight-bootstrap-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_4 = (
    "finevo-pilot-contract-v2.4-matrix-amendment-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_5 = (
    "finevo-pilot-contract-v2.5-parent-import-retry-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_6 = (
    "finevo-pilot-contract-v2.6-p95-authority-retry-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_7 = (
    "finevo-pilot-contract-v2.7-stage0-evaluator-retry-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_8 = (
    "finevo-pilot-contract-v2.8-qref-identity-retry-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_9 = (
    "finevo-pilot-contract-v2.9-qref-summary-equivalence-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10 = (
    "finevo-pilot-contract-v2.10-p95-runner-binding-retry-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1 = (
    "finevo-pilot-contract-v2.10.1-qref-receipt-verifier-retry-overlay-v1"
)
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_2 = (
    "finevo-pilot-contract-v2.10.2-p95-consumer-adapter-retry-overlay-v1"
)
# Backward-compatible public name.  V1 artifacts remain immutable/readable;
# callers that need the science contract should use the explicit V2 constant.
PILOT_CONTRACT_SCHEMA_VERSION = PILOT_CONTRACT_SCHEMA_VERSION_V1
PILOT_CONTRACT_CANONICALIZATION = "json-sort-keys-utf8-v1"

PILOT_CONTRACT_ID_V2 = "finevo-pilot-v2"
PILOT_CONTRACT_ID_V2_1 = "finevo-pilot-v2.1"
PILOT_CONTRACT_ID_V2_2 = "finevo-pilot-v2.2"
PILOT_CONTRACT_ID_V2_3 = "finevo-pilot-v2.3"
PILOT_CONTRACT_ID_V2_4 = "finevo-pilot-v2.4"
PILOT_CONTRACT_ID_V2_5 = "finevo-pilot-v2.5"
PILOT_CONTRACT_ID_V2_6 = "finevo-pilot-v2.6"
PILOT_CONTRACT_ID_V2_7 = "finevo-pilot-v2.7"
PILOT_CONTRACT_ID_V2_8 = "finevo-pilot-v2.8"
PILOT_CONTRACT_ID_V2_9 = "finevo-pilot-v2.9"
PILOT_CONTRACT_ID_V2_10 = "finevo-pilot-v2.10"
PILOT_CONTRACT_ID_V2_10_1 = "finevo-pilot-v2.10.1"
PILOT_CONTRACT_ID_V2_10_2 = "finevo-pilot-v2.10.2"
PILOT_CONTRACT_ID_V2_11 = "finevo-pilot-v2.11"
PILOT_CONTRACT_ID_V2_11_1 = "finevo-pilot-v2.11.1"
PILOT_CONTRACT_ID_V2_11_2 = "finevo-pilot-v2.11.2"
PILOT_CONTRACT_ID_V2_11_3 = "finevo-pilot-v2.11.3"
PILOT_CONTRACT_ID_V2_11_4 = "finevo-pilot-v2.11.4"
PILOT_CONTRACT_ID_V2_11_5 = "finevo-pilot-v2.11.5"
PILOT_CONTRACT_ID_V2_11_6 = "finevo-pilot-v2.11.6"
PILOT_CONTRACT_ID_V2_11_7 = "finevo-pilot-v2.11.7"
PILOT_CONTRACT_ID_V2_11_8 = "finevo-pilot-v2.11.8"
PILOT_CONTRACT_ID_V2_11_9 = "finevo-pilot-v2.11.9"
PILOT_CONTRACT_ID_V2_11_10 = "finevo-pilot-v2.11.10"
PILOT_CONTRACT_ID_V2_11_11 = "finevo-pilot-v2.11.11"
PILOT_CONTRACT_TAG_V2 = "pilot-v2-science"
PILOT_CONTRACT_TAG_V2_1 = "pilot-v2.1-science"
PILOT_CONTRACT_TAG_V2_2 = "pilot-v2.2-science"
PILOT_CONTRACT_TAG_V2_3 = "pilot-v2.3-science"
PILOT_CONTRACT_TAG_V2_4 = "pilot-v2.4-science"
PILOT_CONTRACT_TAG_V2_5 = "pilot-v2.5-science"
PILOT_CONTRACT_TAG_V2_6 = "pilot-v2.6-science"
PILOT_CONTRACT_TAG_V2_7 = "pilot-v2.7-science"
PILOT_CONTRACT_TAG_V2_8 = "pilot-v2.8-science"
PILOT_CONTRACT_TAG_V2_9 = "pilot-v2.9-science"
PILOT_CONTRACT_TAG_V2_10 = "pilot-v2.10-science"
PILOT_CONTRACT_TAG_V2_10_1 = "pilot-v2.10.1-science"
PILOT_CONTRACT_TAG_V2_10_2 = "pilot-v2.10.2-science"
PILOT_CONTRACT_TAG_V2_11 = "pilot-v2.11-science"
PILOT_CONTRACT_TAG_V2_11_1 = "pilot-v2.11.1-science"
PILOT_CONTRACT_TAG_V2_11_2 = "pilot-v2.11.2-science"
PILOT_CONTRACT_TAG_V2_11_3 = "pilot-v2.11.3-science"
PILOT_CONTRACT_TAG_V2_11_4 = "pilot-v2.11.4-science"
PILOT_CONTRACT_TAG_V2_11_5 = "pilot-v2.11.5-science"
PILOT_CONTRACT_TAG_V2_11_6 = "pilot-v2.11.6-science"
PILOT_CONTRACT_TAG_V2_11_7 = "pilot-v2.11.7-science"
PILOT_CONTRACT_TAG_V2_11_8 = "pilot-v2.11.8-science"
PILOT_CONTRACT_TAG_V2_11_9 = "pilot-v2.11.9-science"
PILOT_CONTRACT_TAG_V2_11_10 = "pilot-v2.11.10-science"
PILOT_CONTRACT_TAG_V2_11_11 = "pilot-v2.11.11-science"
_PILOT_V2_4_AUTHORIZED_HARD_CAP_USD = 500.0
_PILOT_V2_4_HOSTED_STAGE_CAP_USD = 495.787229125
_PILOT_V2_4_HARD_CAP_STATUS = "authorized-explicit-user-2026-07-27"
PILOT_CONTRACT_V2_CANONICAL_SHA256 = (
    "980deddf2f82a762db7d73baa6ee0428c5e653298f4f275c5b3a5b23a95865c5"
)
PILOT_CONTRACT_V2_1_CANONICAL_SHA256 = (
    "ac1011e70f3fe85716c4f5c1497812e3c83b3112d7661d234fdaa913f58eadca"
)
PILOT_CONTRACT_V2_2_CANONICAL_SHA256 = (
    "72f9a4f7b687e6711d54d1bed45350963b2039e79565fffb39b03b8c6c66b493"
)
# Filled only after the V2.3 implementation, test inventory, and compact
# overlay are frozen.  Draft overlays are still self-hashed and fully
# validated; paid provenance additionally requires the frozen replacement.
PILOT_CONTRACT_V2_3_CANONICAL_SHA256 = (
    "10a76561ec59810e664d8415bff3a6aa89346a4cfd67b6e7f8aa1257d015c424"
)
# Frozen only after the implementation inventory, explicit $500 hard-cap
# authorization, and Linux/macOS CI identities were recorded.
PILOT_CONTRACT_V2_4_CANONICAL_SHA256: Optional[str] = (
    "12d27b165a36fd0645ddedbf06e8d10a1f178b895272da6f4ea929b73d8506c3"
)
# V2.5 is frozen only after its independent implementation inventory and
# release identity were recorded.  It may not reuse the V2.4 release receipt.
PILOT_CONTRACT_V2_5_CANONICAL_SHA256: Optional[str] = (
    "1f9809062684a1a2afb96b7342b88a06810e0e87ac883aa63a858a65a81d188d"
)
# V2.6 is frozen only after its terminal V2.5 source package and independent
# implementation inventory were recorded.  It may not reuse the V2.5 release
# receipt.
PILOT_CONTRACT_V2_6_CANONICAL_SHA256: Optional[str] = (
    "bb6b12d71227c423e5a67452dc496f26843dec74e359b9b04bf096dc17d0c509"
)
# V2.7 is frozen against the immutable V2.6 terminal source package and its
# independent implementation inventory.  Paid provenance still requires the
# matching merged commit, independent Linux/macOS CI, and annotated tag.
PILOT_CONTRACT_V2_7_CANONICAL_SHA256: Optional[str] = (
    "938627d42ec8ec78e8424793797593736b79936b00813b81259af54e6df6779f"
)
# V2.8 is frozen against the immutable V2.7 terminal source package, its
# nested V2.6 Stage-0 source, and an independent implementation inventory.
# Paid provenance still requires the matching merged commit, independent
# Linux/macOS CI, and annotated tag.
PILOT_CONTRACT_V2_8_CANONICAL_SHA256: Optional[str] = (
    "948eac04516dd2c292d68beb732f97532b13e667a180e8c2db16fbb927f92f19"
)
# V2.9 is frozen against the terminal V2.8 evidence package, immutable source
# manifest, and the final implementation/test inventories. Paid provenance
# still requires the matching merged commit, independent Linux/macOS CI, and
# annotated tag before any dispatch.
PILOT_CONTRACT_V2_9_CANONICAL_SHA256: Optional[str] = (
    "0b07881aaceeb020dc5943ede647a665f9e9bf786a1cac109ab720e05d81d361"
)
PILOT_V2_9_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "9e228bcf3f6a1d636bca7aeef5ab0daab1ce3b3667d0b91f6722388876676eb4"
)
PILOT_V2_9_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "d0e0aaf9da4c2d7fb64521250305eec2433b8866792d273d0c9b5678041b8760"
)
# V2.10 is frozen only after binding the immutable V2.9 source manifest,
# independent implementation inventory, and canonical expanded contract.
# The loader still fails closed if any of these identities drift.
PILOT_CONTRACT_V2_10_CANONICAL_SHA256: Optional[str] = (
    "d1b54c14d016c2b157db9e334d054ab9c7e86371d3fb9662a95fb94e50ce964b"
)
PILOT_V2_10_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "8540bde06f364aa9ccf2a6937b78dec1f0d3b2c66b9e4943f9a3d2e20e4b19a7"
)
PILOT_V2_10_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "fc781697a9260fa63d0535eafa24b87a8386a76dca55f3ce95ba59e12ceb4224"
)
# V2.10.1 is frozen against its independent implementation/CI inventory,
# exact source manifest, and immutable V2.10 terminal publication.  The loader
# fails closed if any binding or the expanded contract drifts.
PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256: Optional[str] = (
    "1f9c642c155d5256815cb14a68335b65a25497523c14210c36f89070b3c8d996"
)
PILOT_V2_10_1_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "e9360d9754cd054386ff03264c331091404555379457a59a7b01344f4a8f2d8f"
)
PILOT_V2_10_1_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "11447dd0c231140102411eb231b8716c8f1581d0fa1533e98ccc51c3afb31426"
)
# V2.10.2 is frozen against its independent implementation/test inventories,
# exact V2.10.1 terminal lineage, and immutable source manifest.
PILOT_CONTRACT_V2_10_2_CANONICAL_SHA256: Optional[str] = (
    "b8de8cfb2560d894dad65d68df8ae9126527d12d3807bef045fa52f5e9d4159e"
)
PILOT_CONTRACT_V2_11_CANONICAL_SHA256: Optional[str] = (
    "d65a60ccab684654979fce598f013b72c813f36f5d40d6063b81ac87557c2c36"
)
# V2.11.1 is frozen against its independent implementation/test inventories,
# exact immutable V2.11 no-go lineage, and source manifest.  Paid execution
# additionally requires the matching Linux/macOS CI and annotated release tag.
PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256: Optional[str] = (
    "818607de5cd512cee60ece06c3f81612e6945cf7ff6d1e48ca643d2109cd7410"
)
PILOT_V2_11_1_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "78f7910ddbd5aa1207b869fc68d45650576e2e370af0f06cc53d0bc7226b71c5"
)
PILOT_V2_11_1_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "7cf33945cff145fa5ca4cf6aae521acec8c96bf521abac076982ffc2e88b7812"
)
# V2.11.2 remains draft until its V2.11.1 terminal source manifest, complete
# implementation/test inventories, canonical contract, merge commit, CI and
# annotated tag are independently frozen.  The draft renderer remains
# independently testable, while every tracked frozen contract must equal this
# pinned canonical identity before provenance validation can begin.
PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256: Optional[str] = (
    "c04f7d4c5ae0962a4a64b0ac543d890a1475b6f184f516534eeb8ff026505a37"
)
PILOT_V2_11_2_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "f38fb442b04ab9a0a85a246954b486f11e6c6571434336d3650b89833c70e90f"
)
PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "4143655b99feba414c319b951578dd652d4bcae550391ea83acdd4d74c00c9d3"
)
# V2.11.3 freezes the independently rendered 136-cell contract, including its
# five exact CI inventory values.  Merge-commit CI and the annotated science
# tag remain separate runtime provenance gates; this pin prevents any later
# contract-byte change from being treated as the same release.
PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256: Optional[str] = (
    "84c818348fabfdd0ddd0ed503c0a5610faf10098f4973d1748b795e2e65b56f1"
)
# This independent hash freezes every science-design field (including all 131
# scientific cells) without depending on release-only CI inventory metadata.
PILOT_CONTRACT_V2_11_3_SCIENCE_DESIGN_SHA256 = (
    "7aca2f627854bb3c7c34e4e7b11a772000bc3b92eb01358ba14bfc9c9e676e7d"
)
PILOT_V2_11_3_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "f05dbac4951e99476c06883e3c1b792e7ccb459c16eb4d78ac15ddf7905598de"
)
PILOT_V2_11_3_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "5c8e554d1a00803b81deb4f31b4a87ddf54a272861a7c750985cd72b18a95f00"
)
# V2.11.4 starts as an explicitly non-dispatchable draft.  These release-only
# identities are populated only after the source manifest, implementation/test
# inventories, and canonical expanded contract have independently stabilized.
PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256: Optional[str] = (
    "e898fe49935dae9ae7f0d7ac577dae943192953c1da581d70c334f8c64924e46"
)
# Unlike the release-only identities above, this design pin is computed from
# the deterministic draft.  The renderer separately proves exact equality of
# all 131 normalized scientific specs against V2.11.3; this hash then freezes
# the complete V2.11.4 design fieldset and is never guessed from a release ID.
PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256: Optional[str] = (
    "8cf696da20de1ee703ff5248a8e081eee0d331f28ab35676264609087b3f3658"
)
PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "fd37e5f7a6cfa0178fa0baec74fb0d18f058a361586296d50d4bcf611e13839d"
)
PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "594b1a00910a1dbecd5e36fcac4397df5341e92b5a9802ce4ca781434b747760"
)
# V2.11.5 is bootstrapped in three fail-closed passes: source-manifest,
# science-design, then expanded-contract identity.  ``None`` is accepted only
# by the explicit draft/frozen-candidate renderer paths.
PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256: Optional[str] = (
    "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
)
PILOT_CONTRACT_V2_11_5_SCIENCE_DESIGN_SHA256: Optional[str] = (
    "044055e33576ec87aa26b3e146d7ca6306b5ccf6e0b3e1dbb99d4585cbcb7b51"
)
PILOT_V2_11_5_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "fea5a276fb64fdd5bf0539014687ea39a891e9d305205b1d2046a2c15a892d16"
)
PILOT_V2_11_5_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "be84d33f561a5ab8927f13e0753f5109b5f018dc790ae180d5e0e6e0228af559"
)
# V2.11.6 is a continuation overlay, not a replacement denominator.  Its
# three release-only identities are populated in separate bootstrap passes
# after the 87-cell continuation contract and source manifest stabilize.
PILOT_CONTRACT_V2_11_6_CANONICAL_SHA256: Optional[str] = (
    "879359813cf733e1aced869b28adcbeaffdb4dd4333226224601e82fa36f0fac"
)
PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256: Optional[str] = (
    "af846dbd5697c2dfbd09b860162f6bcbec929e8d7a5ad5a09350bc45a091ca87"
)
PILOT_V2_11_6_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "710db4414471005d088cd64fb1e1a7c4a46fd99f8852b05f3f17f2acaead240d"
)
PILOT_V2_11_6_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "c510941c565d1120604199139d193990948d6b65be15a823ba1d4850968f2ce0"
)
# V2.11.7 is bootstrapped only after the immutable V2.11.6 zero-provider
# no-go and the corrected V2.11.5 budget decomposition are both bound.  These
# values remain ``None`` during the explicit draft/source/design freeze passes.
PILOT_CONTRACT_V2_11_7_CANONICAL_SHA256: Optional[str] = (
    "376c41f7b2793d4039bae43a652d6ba73759cce7b9b3f04fc665c41a23659e3b"
)
PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256: Optional[str] = (
    "86e97e9d4454dd75eb27b142131159b6c55e95127d16672714d3d0b92c28e2d1"
)
PILOT_V2_11_7_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "dd124c09359d0bd08411add3486cc43887cbee207fdbb6f9bc929e5c1eb81ef9"
)
PILOT_V2_11_7_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "64be1bf836d131d8ec0542e68388dbc328314af7e891549600f5871f8f61f2b0"
)
# V2.11.8 is a prospective recovery release.  Source, design, and expanded
# contract identities remain unpinned until the independent freeze passes.
PILOT_CONTRACT_V2_11_8_CANONICAL_SHA256: Optional[str] = (
    "25d43667520633c5dfa299a693fd4a42736524c2737c2acf6422e2d32f0106c8"
)
PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256: Optional[str] = (
    "9ace9d3796940b6cf32c28c80b2f1cd2f583bf934f8e87a73be2f75a02893024"
)
PILOT_V2_11_8_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "acfc9dc6c751e8ab9f314133de856bae7a0a4021c067f693ed8ebff938b230a6"
)
PILOT_V2_11_8_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "104b63db289234820aebf14f42808c26cd01d9f8a19029fef793887bfff47cd3"
)
# V2.11.9 is the frozen release-binding recovery.  Its design, source, and
# canonical identities were sealed in that order; frozen parsing fails closed
# if any bound value drifts.
PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256: Optional[str] = (
    "ec16563bf906b8f6c1492d2a30f291d2c849cd639c2f314e7a1c8ac619e3fa3f"
)
PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256: Optional[str] = (
    "ad2609dbc1b2d736560bcfc874d2af5899f7a048a0b6aeadbe2e350f91244e01"
)
PILOT_V2_11_9_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "609adf9d12543b4caa7adb0cbddb8c8a9073a10f689adf52a8670608d16e9cb1"
)
PILOT_V2_11_9_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "36a790fe5edd6269218d6010046ec9293c3c418d8bc58a4dd5d89a6a70a547d6"
)
# V2.11.10 is bootstrapped only after the immutable V2.11.9 terminal no-go
# and its independent V2.11.5 dispatch authority are bound.  The source,
# science-design, and canonical pins are sealed in that order.
PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256: Optional[str] = (
    "76a03b7781b1bea317855010d9a3b34b49fcfba3f27cc344954daf19abcd2b1f"
)
PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256: Optional[str] = (
    "54e1833e1cea3cc9243646da7a3e50919e5e0e0e6f47d5a1cd06a647b660dc57"
)
PILOT_V2_11_10_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "a64f98052a43aed76dc1c3e1fd5ef3f0383278bf0f867099c7dbfa79484b6928"
)
PILOT_V2_11_10_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "5632d997905b755678907841ef89825791ef89824e5bcd4f989d4bf5ba1678f3"
)
# V2.11.11 is a fresh-seed cohort after V2.11.10 terminated.  Its source and
# design identities are sealed before the canonical contract identity.
PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256: Optional[str] = (
    "67f35bc6f912706c3287c7f841c2da96da40a57fb09026dc5691631e018143ca"
)
PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256: Optional[str] = (
    "9d79bc14863f4f29ab55ca4a753d71ec18807029c1f214dc8010c4dd7dc2448b"
)
PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "1d5a2c02962e2fbccfb11a8fdf6aa02d4404c6c30ac91a6095c4b7c21379a146"
)
PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "e91fb31f221e18edbaa3cd6967bcc3b9f08cbb31dd4c919af0ac98504a9aa09f"
)
PILOT_V2_10_2_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "f1d953f5b39ab1032ffeb37b73db7c80d54296fba046eddf7e2485e4dc1cc2bd"
)
PILOT_V2_10_2_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "cafbc2cef89c3d605b7242327b9e7aa418ef26ce14eb37c3c89cf2996600f130"
)
PILOT_V2_10_1_EVIDENCE_COMMIT: Optional[str] = (
    "b7001a0174d1a420b592cd68976a3ca8388cb748"
)
PILOT_V2_10_1_EVIDENCE_MERGE_COMMIT: Optional[str] = (
    "a730d0d97118a6d5cf79df66cb97cb1a32c510d9"
)
PILOT_V2_10_1_EVIDENCE_PACKAGE_FILE_SHA256: Optional[str] = (
    "f471a6e3f7a5cd024ac0c34ff9ef1cd42e333e5f141ef01db67f23f088fd3590"
)
PILOT_V2_10_1_EVIDENCE_CHECKSUMS_FILE_SHA256: Optional[str] = (
    "092af0afc03bb88d1c4ccc45e7dad41f5df6d57c9fa4b535b0c69ae3e7b21b54"
)
PILOT_V2_10_1_RAW_FILE_COUNT: Optional[int] = 966
PILOT_V2_10_1_RAW_STORAGE_BYTES: Optional[int] = 23_559_957
PILOT_V2_10_1_RAW_INVENTORY_SHA256: Optional[str] = (
    "63385589f81342822f705c47fe09ce10629a1ccc667ec13e47e7de36cec31413"
)
PILOT_V2_10_1_RUN_LEDGER_INTERNAL_SHA256: Optional[str] = (
    "75e91445745ec5480577327053a8d7eaefc4352cb6f3f176693460cc712d22b6"
)
PILOT_V2_10_1_BUDGET_LEDGER_INTERNAL_SHA256: Optional[str] = (
    "87d313e4f96766f3137c5c0175b0adb6e8a24d4c7697e556e2e0e46f00525161"
)
PILOT_V2_10_EVIDENCE_COMMIT: Optional[str] = "1e96373fa847b44e3418a777c1ed74165ecf2bac"
PILOT_V2_10_EVIDENCE_MERGE_COMMIT: Optional[str] = (
    "2c4f4750d02c9c6b90051cfaa4f16b8ab16aa637"
)
PILOT_V2_10_EVIDENCE_PACKAGE_FILE_SHA256: Optional[str] = (
    "9aa7d07d1d813a5acdea39401e017d5cefe9d85f9127917b119d2453ff972806"
)
PILOT_V2_10_EVIDENCE_CHECKSUMS_FILE_SHA256: Optional[str] = (
    "b117c3e9d2555af9582c22de08b6e39f1366876d9bc0c6a84b37728533748695"
)
PILOT_V2_10_RAW_FILE_COUNT: Optional[int] = 637
PILOT_V2_10_RAW_STORAGE_BYTES: Optional[int] = 20_126_496
PILOT_V2_10_RAW_INVENTORY_SHA256: Optional[str] = (
    "d8964a15abed0d77598d2c2cf80136e438b67559796cc93f8566dca17e584baa"
)
PILOT_V2_10_RUN_LEDGER_INTERNAL_SHA256: Optional[str] = (
    "ef2a7a1d003e4b876749cf87e7a49bd5080e1096b7d6beedb797d6adde149db6"
)
PILOT_V2_10_BUDGET_LEDGER_INTERNAL_SHA256: Optional[str] = (
    "a03b87a18aae4ddcbcc2f546bf142745969a6225922d7b2da327c7f9730d3f6a"
)
# Immutable V2.9 terminal-evidence identities.  Draft V2.10 overlays carry
# null bindings; the frozen amendment will bind these exact publication
# artifacts after the V2.10 source manifest has been generated and sealed.
PILOT_V2_9_EVIDENCE_COMMIT: Optional[str] = "51525614e138e5b7ac498d15b409048d5110b753"
PILOT_V2_9_EVIDENCE_MERGE_COMMIT: Optional[str] = (
    "08fcbc0dd9319fcc86c3f4e812c3db504a0c5a17"
)
PILOT_V2_9_EVIDENCE_PACKAGE_FILE_SHA256: Optional[str] = (
    "6d006ba59c5af6a1e0dd3931466b90d4599edc0ded47e2de3ea4f8ecd6c4831a"
)
PILOT_V2_9_EVIDENCE_CHECKSUMS_FILE_SHA256: Optional[str] = (
    "b0de7185c710b69736ddfe1d331b7f6308165a9f03bb0c616f14ec1fd7a515db"
)
# Filled only after the V2.8 no-go package is committed and merged.  A draft
# V2.9 amendment may carry null evidence bindings but cannot be frozen.
PILOT_V2_8_EVIDENCE_COMMIT: Optional[str] = "00cc7142ae7af603f7989804a43c4d509456bad2"
PILOT_V2_8_EVIDENCE_MERGE_COMMIT: Optional[str] = (
    "981e2af20372c0413600f2bbd1b732f2d643593e"
)
PILOT_V2_8_EVIDENCE_PACKAGE_FILE_SHA256: Optional[str] = (
    "90580f8471b02ad4d156a6e39ce09676e5cccbeadb6a4d21ad54ff88a3867ef6"
)
PILOT_V2_8_EVIDENCE_CHECKSUMS_FILE_SHA256: Optional[str] = (
    "1cc99291ba0bc9582c36414fce2bdc815d3cad0e753bdbb440140ad9f61127a9"
)
PILOT_V2_8_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "1e95025be3466faa38936a3c4617ace0c625fa198eb506f1431a0b6401c4e1f8"
)
PILOT_V2_8_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "f1cc3e150ae506e933233d42d7e8725c4c91e05a46c48e71c74e098eabeed3b4"
)
PILOT_V2_7_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "ee0ef62f5dcde9fc820aef6d23d1ce5a8c5bca7b9f20486bf42233f18763a1c8"
)
PILOT_V2_7_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "f195661d01d0aa6742d9e2f2658b6b1acb38715ddbd43e4e5fd375309d78dbe4"
)
PILOT_V2_6_SOURCE_MANIFEST_FILE_SHA256: Optional[str] = (
    "f84778ed279b8ca98b9b61e26619669fade54b95d0c3e4f17874733acbc84efe"
)
PILOT_V2_6_SOURCE_MANIFEST_CONTENT_SHA256: Optional[str] = (
    "78d42a49f16cbbee4fc5e76de17ff26c501a5dcb04a5eb1f79cbe080d2b1b669"
)
PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256 = (
    "ac11b024435d6d6b03a68b59e5f59f28d92a822ddd3712b1b4c612b668a20586"
)

# Immutable terminal V2.3 identities used by the prospective V2.4 parent
# import.  V2.4 does not resume or rewrite the 174-cell V2.3 denominator.
PILOT_V2_4_PARENT_RELEASE_COMMIT = "ab32e3c9dcf581a40f3093652e144b56f853c782"
PILOT_V2_4_PARENT_CONTRACT_FILE_SHA256 = (
    "98d507a4094ed1d5266b123b4d79a4e386bd1a49e1441f1b1ee3f3362ac54699"
)
PILOT_V2_4_PARENT_RELEASE_ATTESTATION_FILE_SHA256 = (
    "812b497a22a752591f2b3aab8a7afbfd95a9b9832a86543d0aa480abb52b824e"
)
PILOT_V2_4_PARENT_RUN_LEDGER_FILE_SHA256 = (
    "0cc360c23c6400286c6df1c7b3a3ff004af5aeda4f90bcd2140c44fc29c386dc"
)
PILOT_V2_4_PARENT_RUN_LEDGER_INTERNAL_SHA256 = (
    "7c9baf0c9c4f9a83f81bef2c5213b59d2e44c43fdabce03efa86affef7fefc27"
)
PILOT_V2_4_PARENT_RUN_LEDGER_EVENT_HEAD = (
    "c1ea08426d1a5d3b8bfd8f2a0a54c02316d532618310920b9f96c1b594568226"
)
PILOT_V2_4_PARENT_BUDGET_LEDGER_FILE_SHA256 = (
    "d3bc35b703e3e0a8e201d301c2e0154d2e43ad59b468340f9e887c98ff0a01bf"
)
PILOT_V2_4_PARENT_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "7e84abf9f79690bbf8293596142b464685d94fce0fbb9e6a58568aa319ffd265"
)
PILOT_V2_4_PARENT_BUDGET_LEDGER_EVENT_HEAD = (
    "9d02e301d6b2b4a9933fe4a70ebd180ecb200d1e737106bfc0546d3898d46407"
)

# Immutable pilot-v2.2 release/failure identities for the single V2.3
# closed-loop preflight bootstrap correction.  V2.2 reached the scientific
# runner but failed before either provider dispatched because the runner
# required the very p95 observation that the preflight was meant to produce.
PILOT_V2_3_PARENT_RELEASE_COMMIT = "25bca84ab7e50a5bfbd48646fa954b73c8f0e2b4"
PILOT_V2_3_PARENT_LAUNCH_INPUT_SHA256 = (
    "ad49daca5cf3b087c18cd7693a0eb9f9f21cff7ac7949758e7bc849f9d9eb529"
)
PILOT_V2_3_PARENT_LAUNCH_INPUT_FILE_SHA256 = (
    "c5c184e98e2745a418784b7245447044e169bd7bf5627c9baf4c58ae9fe99487"
)
PILOT_V2_3_PARENT_RELEASE_ATTESTATION_SHA256 = (
    "cfc544ab6cf3c22824f23b60d62b25a2ac5447c5af800f51a5376ef287d42ffd"
)
PILOT_V2_3_PARENT_RELEASE_ATTESTATION_FILE_SHA256 = (
    "700e3acf33aadae8c21da78751be63cffb277d75402fb1da8cd14ebac5032498"
)
PILOT_V2_3_PARENT_RUN_LEDGER_FILE_SHA256 = (
    "8667a0de51f6281fb51bd3087f5256f912209ca07857aea524d249952e5bf3fb"
)
PILOT_V2_3_PARENT_RUN_LEDGER_INTERNAL_SHA256 = (
    "19c89a56e6b2317bf97eccd631a472fd2772fd37deede0117d2723b827ed9d42"
)
PILOT_V2_3_PARENT_RUN_LEDGER_EVENT_HEAD = (
    "ca077ef0bbcacdb75e238abf9e166c8382351e2661597c19aec1cf5de8787b2e"
)
PILOT_V2_3_PARENT_BUDGET_LEDGER_FILE_SHA256 = (
    "0ca717fd2dea5d5292a372434e94ee850542274d32f951b5a71cf5bb8ff2b388"
)
PILOT_V2_3_PARENT_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "021d451e5b06a893d466848fd6313555dc24c13269376de06e398ff47b3bd998"
)
PILOT_V2_3_PARENT_BUDGET_LEDGER_EVENT_HEAD = (
    "cfdf85abc6b843ab775ff06bbce4afa2c26a3af2f02f97a4bddc3f0693dbf8a8"
)
PILOT_V2_3_PARENT_CAPABILITY_STAGE_RECEIPT_FILE_SHA256 = (
    "6eae7ea2132e5a332defec22db56ee78a2718964967e591ea6f566e120aef72e"
)
PILOT_V2_3_PARENT_CAPABILITY_STAGE_RECEIPT_CONTENT_SHA256 = (
    "f96a514b481e861a5102d5bd7af8e839d2e1eab70a436c92fcd32c57de8d7180"
)
PILOT_V2_3_PARENT_PREFLIGHT_STAGE_RECEIPT_FILE_SHA256 = (
    "d38829ebf569c7af5fb7ec8157fd9375bcb4300db873ed3d428edcc85723a411"
)
PILOT_V2_3_PARENT_PREFLIGHT_STAGE_RECEIPT_CONTENT_SHA256 = (
    "c6b4a899a85f7e0edff3221df6a62d8884961955e5321aad8394b4273d5119e3"
)

# Exact immutable source identities for the one evaluator-only V2.2
# correction.  Keeping these values named and local to the contract validator
# makes the overlay fail closed before an imported capability artifact can be
# considered by the orchestrator.
PILOT_V2_2_PARENT_RELEASE_COMMIT = "5d604eea3b14113a55599806b31c2f8f09089bbd"
PILOT_V2_2_PARENT_LAUNCH_INPUT_SHA256 = (
    "0598ec40701a660fd2d32709ab8f6a8dd5ccae8b965cff76f1992a5d00021716"
)
PILOT_V2_2_PARENT_RELEASE_ATTESTATION_SHA256 = (
    "3a23899160eadb37135384e5dbbdbf034e485ab0bf51ba5521dab6c56abd65bc"
)
PILOT_V2_2_PARENT_RUN_LEDGER_FILE_SHA256 = (
    "73b4e8572730ea91a9ab3838ea36b65b6d17625797333ee2eee53c2186ac736e"
)
PILOT_V2_2_PARENT_RUN_LEDGER_INTERNAL_SHA256 = (
    "ad3e5deb3a01f0ab4b5d079657830755170f9cbe88ab8b8d5cbdf21625738de9"
)
PILOT_V2_2_PARENT_BUDGET_LEDGER_FILE_SHA256 = (
    "405a818a1ee0dd4562e3a5452104f6b3411834f1d9d5800a81dcbc515b0144f2"
)
PILOT_V2_2_PARENT_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "8a871a52014a49d50ac8c8701022fea74b57df61aaa2e6a9f4d18c009200d2e1"
)
PILOT_V2_2_PARENT_STAGE_RECEIPT_FILE_SHA256 = (
    "5aa7374c1e2cc942cfa3a86c01256a8b59b954de2129aa3c855c2d411281704f"
)
PILOT_V2_2_PARENT_STAGE_RECEIPT_CONTENT_SHA256 = (
    "0e98cddf636a6912692ff081fa2e152a49c119608d43fa8ae5debfb1906d9ac8"
)

PILOT_V2_2_GPT52_CAPABILITY_SHA256 = (
    "bd64ace07e04b13755ed46f9b2599b9c11a7fadf902500f57b9ab020cba4448d"
)
PILOT_V2_2_GPT52_GATE_SHA256 = (
    "01c61c25e7d25577975dbe3aae8a408f464d685210b071aa612f4bd46bb78eda"
)
PILOT_V2_2_GPT52_TERMINAL_SHA256 = (
    "b653b8bf4f702632fb9781e884c63ecabd285eef548df64ed8ddd3247b64fb2b"
)
PILOT_V2_2_GPT52_TERMINAL_CONTENT_SHA256 = (
    "1e1122c419a49728c796cd682670c9e3f314183ac0a86bc6a72a58082695d39f"
)
PILOT_V2_2_LLAMA33_CAPABILITY_SHA256 = (
    "4c4c864733f32166c286e22b446dc3849df624a267ad083426ee4a89e79052ca"
)
PILOT_V2_2_LLAMA33_GATE_SHA256 = PILOT_V2_2_GPT52_GATE_SHA256
PILOT_V2_2_LLAMA33_TERMINAL_SHA256 = (
    "544b409e6ce8538958ec6278f5311429f14cf24591f8479bff287512d02e7380"
)
PILOT_V2_2_LLAMA33_TERMINAL_CONTENT_SHA256 = (
    "16a5f772eeb952aba55dcb2f655642fa7a9db1c3c9453a82b40617278f660070"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SEED_CAPABILITIES = frozenset({"best_effort", "unsupported", "deterministic"})
_TRANSPORTS = frozenset({"openai", "openrouter", "ollama", "diagnostic"})
_JSON_MODES = frozenset({"json_object", "prompt_only"})
_REASONING_MODES = frozenset({"fixed", "omitted"})
_REASONING_EFFORTS = frozenset(
    {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
)
_DISPATCH_MODES = frozenset({"explicit_supported", "documented_unsupported_omitted"})
_DECODING_FIELDS = frozenset(
    {"temperature", "top_p", "seed", "reasoning", "response_format"}
)
_MODEL_ROLES = frozenset(
    {
        "primary",
        "controlled_second",
        "secondary_diagnostic",
        "capability_no_go",
        "calibration_only",
    }
)
_SCIENCE_TASK_CAPS = {
    "capability-choice": (2048, 512),
    "capability-proposal": (4096, 4096),
    "actor-action": (2048, 1024),
    "semantic-proposal": (4096, 4096),
}
_SCIENCE_TASK_CAPS_V2_11 = {
    "capability-choice": (4096, 512),
    "capability-proposal": (4096, 4096),
    "actor-action": (4096, 1024),
    "semantic-proposal": (4096, 4096),
}
_SCIENCE_TASK_CAPS_V2_11_11 = {
    "capability-choice": (4096, 512),
    "capability-proposal": (4096, 4096),
    "actor-action": (8192, 1024),
    "semantic-proposal": (4096, 4096),
}

PILOT_V1_ACTION_GRID = {
    "labor_step_hours": 8.0,
    "consumption_step": 0.02,
}
PILOT_V1_NARRATIVE_FIXTURES = {
    "none": "",
    "aligned": (
        "Savings now earn a high interest rate. Preserving cash and reducing "
        "current consumption is directionally consistent with this environment."
    ),
    "paraphrase": (
        "Returns on saved funds are elevated, so conserving available cash and "
        "lowering near-term spending points in the same direction."
    ),
    "opposite": (
        "A high interest rate makes immediate spending more attractive; increase "
        "current consumption rather than preserving cash."
    ),
}
PILOT_V1_SENSITIVITY_WEIGHTS = (0.25, 0.50, 0.75)
PILOT_V1_SENSITIVITY_OUTCOMES = (
    "utility_advantage_positive",
    "absolute_flow_utility",
    "three_period_cumulative_advantage_positive",
)


class PilotContractError(ValueError):
    """Raised when a pilot contract or runtime binding is not exact."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical UTF-8 JSON."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _freeze_json(value: Any) -> Any:
    copied = _json_copy(value)

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType(
                {str(key): freeze(val) for key, val in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(val) for val in item)
        return item

    return freeze(copied)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotContractError(f"{name} must be an object")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: frozenset[str] = frozenset(),
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise PilotContractError(f"invalid {name} keys: {', '.join(details)}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotContractError(f"{name} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PilotContractError(f"{name} must be boolean")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotContractError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise PilotContractError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise PilotContractError(f"{name} must be <= {maximum}")
    return result


def _optional_number(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotContractError(f"{name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PilotContractError(f"{name} must be finite and nonnegative")
    return result


def _string_tuple(
    value: Any, name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PilotContractError(f"{name} must be an array")
    result = tuple(_text(item, f"{name} item") for item in value)
    if not result and not allow_empty:
        raise PilotContractError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise PilotContractError(f"{name} contains duplicates")
    return result


def _sha256(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if not _SHA256_RE.fullmatch(result):
        raise PilotContractError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _git_commit(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if not _GIT_COMMIT_RE.fullmatch(result):
        raise PilotContractError(f"{name} must be a lowercase 40-character commit")
    return result


@dataclass(frozen=True, slots=True)
class ReasoningProfile:
    """Frozen reasoning request semantics."""

    mode: str
    effort: Optional[str] = None
    exclude: bool = True

    def __post_init__(self) -> None:
        mode = _text(self.mode, "reasoning.mode")
        if mode not in _REASONING_MODES:
            raise PilotContractError(f"unsupported reasoning mode: {mode}")
        object.__setattr__(self, "mode", mode)
        _boolean(self.exclude, "reasoning.exclude")
        if mode == "fixed":
            effort = _text(self.effort, "reasoning.effort")
            if effort not in _REASONING_EFFORTS:
                raise PilotContractError(f"unsupported reasoning effort: {effort}")
            object.__setattr__(self, "effort", effort)
        elif self.effort is not None:
            raise PilotContractError("omitted reasoning cannot declare an effort")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReasoningProfile":
        value = _mapping(value, "reasoning")
        _strict_keys(
            value,
            required={"mode", "effort", "exclude"},
            name="reasoning",
        )
        return cls(
            mode=value["mode"],
            effort=value["effort"],
            exclude=value["exclude"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "effort": self.effort,
            "exclude": self.exclude,
        }

    def openrouter_payload(self) -> Optional[dict[str, Any]]:
        if self.mode == "omitted":
            return None
        return {"effort": self.effort, "exclude": self.exclude}


@dataclass(frozen=True, slots=True)
class DecodingFieldDispatch:
    """Per-profile disposition of one potentially unsupported request field."""

    requested_value: Any
    dispatch_mode: str
    catalog_evidence_required: bool

    def __post_init__(self) -> None:
        mode = _text(self.dispatch_mode, "decoding field dispatch_mode")
        if mode not in _DISPATCH_MODES:
            raise PilotContractError(f"unsupported dispatch mode: {mode}")
        object.__setattr__(self, "dispatch_mode", mode)
        _boolean(
            self.catalog_evidence_required,
            "decoding field catalog_evidence_required",
        )
        object.__setattr__(self, "requested_value", _freeze_json(self.requested_value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecodingFieldDispatch":
        value = _mapping(value, "decoding field")
        _strict_keys(
            value,
            required={
                "requested_value",
                "dispatch_mode",
                "catalog_evidence_required",
            },
            name="decoding field",
        )
        return cls(
            requested_value=value["requested_value"],
            dispatch_mode=value["dispatch_mode"],
            catalog_evidence_required=value["catalog_evidence_required"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_value": _thaw_json(self.requested_value),
            "dispatch_mode": self.dispatch_mode,
            "catalog_evidence_required": self.catalog_evidence_required,
        }


@dataclass(frozen=True, slots=True)
class ParameterDispatchPolicy:
    """Uniform fail-closed policy applied to every V2 model profile."""

    policy_id: str
    fields: tuple[str, ...]
    allowed_modes: tuple[str, ...]
    unsupported_field_action: str
    unknown_support_action: str
    omission_receipt_status: str
    uniform_across_profiles: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        fields = _string_tuple(self.fields, "parameter dispatch fields")
        if frozenset(fields) != _DECODING_FIELDS:
            raise PilotContractError(
                "parameter dispatch policy must cover the five frozen decoding fields"
            )
        object.__setattr__(self, "fields", fields)
        modes = _string_tuple(self.allowed_modes, "parameter dispatch modes")
        if frozenset(modes) != _DISPATCH_MODES:
            raise PilotContractError(
                "parameter dispatch policy modes differ from the frozen V2 policy"
            )
        object.__setattr__(self, "allowed_modes", modes)
        if self.unsupported_field_action != "omit-before-dispatch":
            raise PilotContractError(
                "unsupported request fields must be omitted before dispatch"
            )
        if self.unknown_support_action != "stop-before-dispatch":
            raise PilotContractError(
                "unknown parameter support must stop before dispatch"
            )
        if self.omission_receipt_status != "omitted_unsupported":
            raise PilotContractError(
                "unsupported omissions must be recorded as omitted_unsupported"
            )
        if not _boolean(
            self.uniform_across_profiles, "parameter dispatch uniform_across_profiles"
        ):
            raise PilotContractError(
                "parameter dispatch policy must be uniform across profiles"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterDispatchPolicy":
        value = _mapping(value, "parameter_dispatch_policy")
        _strict_keys(
            value,
            required={
                "policy_id",
                "fields",
                "allowed_modes",
                "unsupported_field_action",
                "unknown_support_action",
                "omission_receipt_status",
                "uniform_across_profiles",
            },
            name="parameter_dispatch_policy",
        )
        return cls(
            policy_id=value["policy_id"],
            fields=_string_tuple(value["fields"], "parameter dispatch fields"),
            allowed_modes=_string_tuple(
                value["allowed_modes"], "parameter dispatch modes"
            ),
            unsupported_field_action=value["unsupported_field_action"],
            unknown_support_action=value["unknown_support_action"],
            omission_receipt_status=value["omission_receipt_status"],
            uniform_across_profiles=value["uniform_across_profiles"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "fields": list(self.fields),
            "allowed_modes": list(self.allowed_modes),
            "unsupported_field_action": self.unsupported_field_action,
            "unknown_support_action": self.unknown_support_action,
            "omission_receipt_status": self.omission_receipt_status,
            "uniform_across_profiles": self.uniform_across_profiles,
        }


@dataclass(frozen=True, slots=True)
class TaskOutputContract:
    """Model-independent output cap and parser contract for one call role."""

    task_id: str
    max_completion_tokens: int
    max_visible_json_bytes: int
    visible_token_count_required: bool
    reasoning_token_count_required: bool
    science_parse_mode: str
    report_recovery_modes: tuple[str, ...]
    recovered_output_scientific_success: bool
    required_finish_reason: str

    def __post_init__(self) -> None:
        task_id = _text(self.task_id, "task output contract task_id")
        object.__setattr__(self, "task_id", task_id)
        if task_id not in _SCIENCE_TASK_CAPS:
            raise PilotContractError(f"unknown science task output contract: {task_id}")
        _integer(
            self.max_completion_tokens,
            f"{task_id}.max_completion_tokens",
            minimum=1,
        )
        _integer(
            self.max_visible_json_bytes,
            f"{task_id}.max_visible_json_bytes",
            minimum=2,
        )
        if not _boolean(
            self.visible_token_count_required,
            f"{task_id}.visible_token_count_required",
        ):
            raise PilotContractError("visible token counts must be recorded")
        if not _boolean(
            self.reasoning_token_count_required,
            f"{task_id}.reasoning_token_count_required",
        ):
            raise PilotContractError("reasoning token counts must be recorded")
        if self.science_parse_mode != "exact_json_only":
            raise PilotContractError(
                "scientific success requires exact JSON without parser recovery"
            )
        recovery = _string_tuple(
            self.report_recovery_modes,
            f"{task_id}.report_recovery_modes",
        )
        if recovery != ("fenced_json", "substring_json"):
            raise PilotContractError(
                "V2 recovery reporting must cover fenced_json and substring_json"
            )
        object.__setattr__(self, "report_recovery_modes", recovery)
        if _boolean(
            self.recovered_output_scientific_success,
            f"{task_id}.recovered_output_scientific_success",
        ):
            raise PilotContractError(
                "recovered JSON cannot count as V2 scientific parse success"
            )
        if self.required_finish_reason != "stop":
            raise PilotContractError("V2 task outputs require finish_reason=stop")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskOutputContract":
        value = _mapping(value, "task output contract")
        fields = {
            "task_id",
            "max_completion_tokens",
            "max_visible_json_bytes",
            "visible_token_count_required",
            "reasoning_token_count_required",
            "science_parse_mode",
            "report_recovery_modes",
            "recovered_output_scientific_success",
            "required_finish_reason",
        }
        _strict_keys(value, required=fields, name="task output contract")
        return cls(
            task_id=value["task_id"],
            max_completion_tokens=value["max_completion_tokens"],
            max_visible_json_bytes=value["max_visible_json_bytes"],
            visible_token_count_required=value["visible_token_count_required"],
            reasoning_token_count_required=value["reasoning_token_count_required"],
            science_parse_mode=value["science_parse_mode"],
            report_recovery_modes=_string_tuple(
                value["report_recovery_modes"], "report recovery modes"
            ),
            recovered_output_scientific_success=value[
                "recovered_output_scientific_success"
            ],
            required_finish_reason=value["required_finish_reason"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "max_completion_tokens": self.max_completion_tokens,
            "max_visible_json_bytes": self.max_visible_json_bytes,
            "visible_token_count_required": self.visible_token_count_required,
            "reasoning_token_count_required": self.reasoning_token_count_required,
            "science_parse_mode": self.science_parse_mode,
            "report_recovery_modes": list(self.report_recovery_modes),
            "recovered_output_scientific_success": (
                self.recovered_output_scientific_success
            ),
            "required_finish_reason": self.required_finish_reason,
        }


@dataclass(frozen=True, slots=True)
class ModelRolePolicy:
    """Frozen scientific role and dispatch surface for one model profile."""

    profile_id: str
    role: str
    dispatch_eligible: bool
    ineligibility_reason: Optional[str]
    allowed_stages: tuple[str, ...]
    allowed_call_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(self.profile_id, "profile_id"))
        role = _text(self.role, "model role")
        if role not in _MODEL_ROLES:
            raise PilotContractError(f"unsupported model role: {role}")
        object.__setattr__(self, "role", role)
        eligible = _boolean(self.dispatch_eligible, "model role dispatch_eligible")
        reason = self.ineligibility_reason
        if eligible:
            if reason is not None:
                raise PilotContractError(
                    "dispatch-eligible model role cannot have an ineligibility reason"
                )
        else:
            reason = _text(reason, "model role ineligibility_reason")
            if self.allowed_stages or self.allowed_call_roles:
                raise PilotContractError(
                    "dispatch-ineligible model role cannot allow stages or call roles"
                )
        object.__setattr__(self, "ineligibility_reason", reason)
        object.__setattr__(
            self,
            "allowed_stages",
            _string_tuple(
                self.allowed_stages,
                "model role allowed_stages",
                allow_empty=not eligible,
            ),
        )
        object.__setattr__(
            self,
            "allowed_call_roles",
            _string_tuple(
                self.allowed_call_roles,
                "model role allowed_call_roles",
                allow_empty=not eligible,
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelRolePolicy":
        value = _mapping(value, "model role")
        _strict_keys(
            value,
            required={
                "profile_id",
                "role",
                "dispatch_eligible",
                "ineligibility_reason",
                "allowed_stages",
                "allowed_call_roles",
            },
            name="model role",
        )
        return cls(
            profile_id=value["profile_id"],
            role=value["role"],
            dispatch_eligible=value["dispatch_eligible"],
            ineligibility_reason=value["ineligibility_reason"],
            allowed_stages=_string_tuple(
                value["allowed_stages"],
                "model role allowed_stages",
                allow_empty=True,
            ),
            allowed_call_roles=_string_tuple(
                value["allowed_call_roles"],
                "model role allowed_call_roles",
                allow_empty=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "role": self.role,
            "dispatch_eligible": self.dispatch_eligible,
            "ineligibility_reason": self.ineligibility_reason,
            "allowed_stages": list(self.allowed_stages),
            "allowed_call_roles": list(self.allowed_call_roles),
        }


@dataclass(frozen=True, slots=True)
class DenominatorPolicy:
    """Typed ITT and inference-unit contract for the mechanism micro-pilot."""

    policy_id: str
    registered_cells_are_itt: bool
    parse_failure_outcome: str
    provider_budget_integrity_failure_outcome: str
    failed_seed_replacement: str
    seed_inference_unit: str
    rule_inference_unit: str
    checkpoint_inference_unit: str
    core_complete_pairs_min: int
    core_registered_pairs: int
    cross_model_complete_pairs_min: int
    cross_model_registered_pairs: int
    raw_paired_deltas_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "denominator policy")
        )
        if not _boolean(
            self.registered_cells_are_itt, "denominator registered_cells_are_itt"
        ):
            raise PilotContractError("all preregistered cells must remain in ITT")
        if self.parse_failure_outcome != "candidate_not_activated":
            raise PilotContractError(
                "parse failures must be counted as candidate_not_activated"
            )
        if (
            self.provider_budget_integrity_failure_outcome
            != "terminate_run_keep_denominator"
        ):
            raise PilotContractError(
                "provider/budget/integrity failures must terminate but remain in ITT"
            )
        if self.failed_seed_replacement != "forbidden":
            raise PilotContractError("failed pilot seeds cannot be replaced")
        expected_units = (
            (self.seed_inference_unit, "seed"),
            (self.rule_inference_unit, "seed-agent-family"),
            (self.checkpoint_inference_unit, "seed-checkpoint"),
        )
        if any(actual != expected for actual, expected in expected_units):
            raise PilotContractError("V2 inference units differ from preregistration")
        if (
            _integer(self.core_complete_pairs_min, "core_complete_pairs_min") != 4
            or _integer(self.core_registered_pairs, "core_registered_pairs") != 5
            or _integer(
                self.cross_model_complete_pairs_min,
                "cross_model_complete_pairs_min",
            )
            != 2
            or _integer(
                self.cross_model_registered_pairs, "cross_model_registered_pairs"
            )
            != 3
        ):
            raise PilotContractError("V2 denominator pair counts drifted")
        if not _boolean(self.raw_paired_deltas_required, "raw_paired_deltas_required"):
            raise PilotContractError("raw paired deltas are required")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DenominatorPolicy":
        value = _mapping(value, "denominator_policy")
        fields = {
            "policy_id",
            "registered_cells_are_itt",
            "parse_failure_outcome",
            "provider_budget_integrity_failure_outcome",
            "failed_seed_replacement",
            "seed_inference_unit",
            "rule_inference_unit",
            "checkpoint_inference_unit",
            "core_complete_pairs_min",
            "core_registered_pairs",
            "cross_model_complete_pairs_min",
            "cross_model_registered_pairs",
            "raw_paired_deltas_required",
        }
        _strict_keys(value, required=fields, name="denominator_policy")
        return cls(**{key: value[key] for key in fields})

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "registered_cells_are_itt": self.registered_cells_are_itt,
            "parse_failure_outcome": self.parse_failure_outcome,
            "provider_budget_integrity_failure_outcome": (
                self.provider_budget_integrity_failure_outcome
            ),
            "failed_seed_replacement": self.failed_seed_replacement,
            "seed_inference_unit": self.seed_inference_unit,
            "rule_inference_unit": self.rule_inference_unit,
            "checkpoint_inference_unit": self.checkpoint_inference_unit,
            "core_complete_pairs_min": self.core_complete_pairs_min,
            "core_registered_pairs": self.core_registered_pairs,
            "cross_model_complete_pairs_min": self.cross_model_complete_pairs_min,
            "cross_model_registered_pairs": self.cross_model_registered_pairs,
            "raw_paired_deltas_required": self.raw_paired_deltas_required,
        }


@dataclass(frozen=True, slots=True)
class ReleaseRequirements:
    """Static CI identity and expected freeze values for a later attestor."""

    remote: str
    branch: str
    tag: str
    workflow_file: str
    workflow_name: str
    required_job_names: tuple[str, ...]
    expected_ci: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.remote != "origin":
            raise PilotContractError("V2 release remote must be origin")
        if self.branch != "main":
            raise PilotContractError("V2 release branch must be main")
        if self.tag not in {
            PILOT_CONTRACT_TAG_V2,
            PILOT_CONTRACT_TAG_V2_1,
            PILOT_CONTRACT_TAG_V2_2,
            PILOT_CONTRACT_TAG_V2_3,
            PILOT_CONTRACT_TAG_V2_4,
            PILOT_CONTRACT_TAG_V2_5,
            PILOT_CONTRACT_TAG_V2_6,
            PILOT_CONTRACT_TAG_V2_7,
            PILOT_CONTRACT_TAG_V2_8,
            PILOT_CONTRACT_TAG_V2_9,
            PILOT_CONTRACT_TAG_V2_10,
            PILOT_CONTRACT_TAG_V2_10_1,
            PILOT_CONTRACT_TAG_V2_10_2,
            PILOT_CONTRACT_TAG_V2_11,
            PILOT_CONTRACT_TAG_V2_11_1,
            PILOT_CONTRACT_TAG_V2_11_2,
            PILOT_CONTRACT_TAG_V2_11_3,
            PILOT_CONTRACT_TAG_V2_11_4,
            PILOT_CONTRACT_TAG_V2_11_5,
            PILOT_CONTRACT_TAG_V2_11_6,
            PILOT_CONTRACT_TAG_V2_11_7,
            PILOT_CONTRACT_TAG_V2_11_8,
            PILOT_CONTRACT_TAG_V2_11_9,
            PILOT_CONTRACT_TAG_V2_11_10,
            PILOT_CONTRACT_TAG_V2_11_11,
        }:
            raise PilotContractError(
                "V2 release tag must be a registered annotated science tag"
            )
        if self.workflow_file != ".github/workflows/verified-memory-ci.yml":
            raise PilotContractError("V2 release workflow file drifted")
        if self.workflow_name != "Verified memory CI":
            raise PilotContractError("V2 release workflow name drifted")
        jobs = _string_tuple(self.required_job_names, "required_job_names")
        if jobs != (
            "Python 3.12.7 / ubuntu-24.04",
            "Python 3.12.7 / macos-14",
        ):
            raise PilotContractError("V2 release requires the frozen Linux/macOS jobs")
        object.__setattr__(self, "required_job_names", jobs)
        expected_ci = _mapping(self.expected_ci, "release expected_ci")
        expected_fields = {
            "test_count",
            "test_collection_sha256",
            "compiled_source_count",
            "compiled_source_inventory_sha256",
            "sealed_manifest_inventory_sha256",
        }
        _strict_keys(
            expected_ci,
            required=expected_fields,
            name="release expected_ci",
        )
        for name in ("test_count", "compiled_source_count"):
            value = expected_ci[name]
            if value is not None:
                _integer(value, name, minimum=1)
        for name in (
            "test_collection_sha256",
            "compiled_source_inventory_sha256",
            "sealed_manifest_inventory_sha256",
        ):
            value = expected_ci[name]
            if value is not None:
                _sha256(value, name)
        object.__setattr__(self, "expected_ci", _freeze_json(expected_ci))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseRequirements":
        value = _mapping(value, "release_requirements")
        fields = {
            "remote",
            "branch",
            "tag",
            "workflow_file",
            "workflow_name",
            "required_job_names",
            "expected_ci",
        }
        _strict_keys(value, required=fields, name="release_requirements")
        return cls(
            remote=value["remote"],
            branch=value["branch"],
            tag=value["tag"],
            workflow_file=value["workflow_file"],
            workflow_name=value["workflow_name"],
            required_job_names=_string_tuple(
                value["required_job_names"], "required_job_names"
            ),
            expected_ci=_mapping(value["expected_ci"], "release expected_ci"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "remote": self.remote,
            "branch": self.branch,
            "tag": self.tag,
            "workflow_file": self.workflow_file,
            "workflow_name": self.workflow_name,
            "required_job_names": list(self.required_job_names),
            "expected_ci": _thaw_json(self.expected_ci),
        }


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    """Frozen catalog and dispatch-endpoint prices in USD per million tokens."""

    captured_at: str
    source: str
    currency: str
    unit: str
    dispatch_basis: str
    catalog_input: Optional[float]
    catalog_output: Optional[float]
    catalog_cached_input: Optional[float]
    endpoint_input: Optional[float]
    endpoint_output: Optional[float]
    endpoint_cached_input: Optional[float]
    model_reference: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("captured_at", "source", "currency", "unit", "dispatch_basis"):
            object.__setattr__(self, name, _text(getattr(self, name), f"price.{name}"))
        if self.currency != "USD":
            raise PilotContractError("price currency must be USD")
        if self.unit != "per_million_tokens":
            raise PilotContractError("price unit must be per_million_tokens")
        if self.dispatch_basis not in {"catalog", "endpoint"}:
            raise PilotContractError("price dispatch_basis must be catalog or endpoint")
        for name in (
            "catalog_input",
            "catalog_output",
            "catalog_cached_input",
            "endpoint_input",
            "endpoint_output",
            "endpoint_cached_input",
        ):
            object.__setattr__(
                self, name, _optional_number(getattr(self, name), f"price.{name}")
            )
        if self.model_reference is not None:
            object.__setattr__(
                self,
                "model_reference",
                _text(self.model_reference, "price.model_reference"),
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PriceSnapshot":
        value = _mapping(value, "price_snapshot")
        fields = {
            "captured_at",
            "source",
            "currency",
            "unit",
            "dispatch_basis",
            "catalog_input",
            "catalog_output",
            "catalog_cached_input",
            "endpoint_input",
            "endpoint_output",
            "endpoint_cached_input",
        }
        _strict_keys(
            value,
            required=fields,
            optional=frozenset({"model_reference"}),
            name="price_snapshot",
        )
        return cls(
            **{field: value[field] for field in fields},
            model_reference=value.get("model_reference"),
        )

    @property
    def dispatch_input(self) -> Optional[float]:
        return (
            self.endpoint_input
            if self.dispatch_basis == "endpoint"
            else self.catalog_input
        )

    @property
    def dispatch_output(self) -> Optional[float]:
        return (
            self.endpoint_output
            if self.dispatch_basis == "endpoint"
            else self.catalog_output
        )

    @property
    def dispatch_cached_input(self) -> Optional[float]:
        return (
            self.endpoint_cached_input
            if self.dispatch_basis == "endpoint"
            else self.catalog_cached_input
        )

    @property
    def known_for_dispatch(self) -> bool:
        return self.dispatch_input is not None and self.dispatch_output is not None

    def assert_known_for_dispatch(self) -> None:
        if not self.known_for_dispatch:
            raise PilotContractError(
                "provider price is unknown for the frozen dispatch endpoint"
            )

    def assert_positive_for_hosted_dispatch(self) -> None:
        """Require a conservative, nonzero frozen price for hosted dispatch."""

        self.assert_known_for_dispatch()
        if float(self.dispatch_input) <= 0.0 or float(self.dispatch_output) <= 0.0:
            raise PilotContractError(
                "hosted provider dispatch input/output prices must be finite "
                "and positive"
            )

    def costs_per_1k(self) -> dict[str, float]:
        self.assert_known_for_dispatch()
        prompt = float(self.dispatch_input) / 1000.0
        cached = self.dispatch_cached_input
        return {
            "prompt": prompt,
            "cached_prompt": prompt if cached is None else float(cached) / 1000.0,
            "completion": float(self.dispatch_output) / 1000.0,
        }

    def to_dict(self) -> dict[str, Any]:
        result = {
            "captured_at": self.captured_at,
            "source": self.source,
            "currency": self.currency,
            "unit": self.unit,
            "dispatch_basis": self.dispatch_basis,
            "catalog_input": self.catalog_input,
            "catalog_output": self.catalog_output,
            "catalog_cached_input": self.catalog_cached_input,
            "endpoint_input": self.endpoint_input,
            "endpoint_output": self.endpoint_output,
            "endpoint_cached_input": self.endpoint_cached_input,
        }
        if self.model_reference is not None:
            result["model_reference"] = self.model_reference
        return result


@dataclass(frozen=True, slots=True)
class ProviderRequestProfile:
    """Exact provider/model request identity used by a pilot matrix cell."""

    profile_id: str
    transport: str
    requested_model: str
    served_model: str
    provider_pin: tuple[str, ...]
    routing_mode: str
    seed_capability: str
    reasoning: ReasoningProfile
    json_mode: str
    price_snapshot: PriceSnapshot
    max_attempts: int = 1
    allow_fallbacks: bool = False
    require_parameters: bool = True
    service_tier: Optional[str] = None
    short_context_prompt_token_ceiling: Optional[int] = None
    artifact_identity: tuple[tuple[str, str], ...] = ()
    decoding_fields: tuple[tuple[str, DecodingFieldDispatch], ...] = ()
    dispatch_eligible: bool = True
    ineligibility_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "transport",
            "requested_model",
            "served_model",
            "routing_mode",
            "seed_capability",
            "json_mode",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.transport not in _TRANSPORTS:
            raise PilotContractError(f"unsupported transport: {self.transport}")
        if self.seed_capability not in _SEED_CAPABILITIES:
            raise PilotContractError(
                f"unsupported seed capability: {self.seed_capability}"
            )
        if self.json_mode not in _JSON_MODES:
            raise PilotContractError(f"unsupported JSON mode: {self.json_mode}")
        if not isinstance(self.reasoning, ReasoningProfile):
            raise PilotContractError("reasoning must be a ReasoningProfile")
        if not isinstance(self.price_snapshot, PriceSnapshot):
            raise PilotContractError("price_snapshot must be a PriceSnapshot")
        _integer(self.max_attempts, "max_attempts", minimum=1, maximum=1)
        _boolean(self.allow_fallbacks, "allow_fallbacks")
        _boolean(self.require_parameters, "require_parameters")
        service_tier = self.service_tier
        if service_tier is not None:
            service_tier = _text(service_tier, "service_tier")
            if self.transport != "openai" or service_tier != "default":
                raise PilotContractError(
                    "service_tier is supported only as default for direct OpenAI"
                )
        object.__setattr__(self, "service_tier", service_tier)
        if self.short_context_prompt_token_ceiling is not None:
            ceiling = _integer(
                self.short_context_prompt_token_ceiling,
                "short_context_prompt_token_ceiling",
                minimum=1,
            )
            if self.transport != "openai":
                raise PilotContractError(
                    "short-context prompt ceiling is supported only for OpenAI"
                )
            object.__setattr__(self, "short_context_prompt_token_ceiling", ceiling)
        pins = tuple(_text(item, "provider_pin item") for item in self.provider_pin)
        if len(pins) != len(set(pins)):
            raise PilotContractError("provider_pin contains duplicates")
        object.__setattr__(self, "provider_pin", pins)
        identity = tuple(
            sorted(
                (
                    _text(key, "artifact_identity key"),
                    _text(value, f"artifact_identity[{key}]"),
                )
                for key, value in self.artifact_identity
            )
        )
        if len(identity) != len({key for key, _ in identity}):
            raise PilotContractError("artifact_identity contains duplicate keys")
        object.__setattr__(self, "artifact_identity", identity)
        decoding = tuple(sorted(self.decoding_fields, key=lambda item: item[0]))
        if decoding:
            decoding_keys = tuple(
                _text(key, "decoding_fields key") for key, _ in decoding
            )
            if len(decoding_keys) != len(set(decoding_keys)):
                raise PilotContractError("decoding_fields contains duplicate keys")
            if frozenset(decoding_keys) != _DECODING_FIELDS:
                raise PilotContractError(
                    "V2 profile decoding_fields must cover the five frozen fields"
                )
            if any(not isinstance(item, DecodingFieldDispatch) for _, item in decoding):
                raise PilotContractError(
                    "decoding_fields values must be DecodingFieldDispatch objects"
                )
        object.__setattr__(self, "decoding_fields", decoding)
        eligible = _boolean(self.dispatch_eligible, "dispatch_eligible")
        reason = self.ineligibility_reason
        if eligible:
            if reason is not None:
                raise PilotContractError(
                    "dispatch-eligible profile cannot declare ineligibility_reason"
                )
        else:
            reason = _text(reason, "ineligibility_reason")
        object.__setattr__(self, "ineligibility_reason", reason)

        if self.transport == "openrouter":
            if not self.provider_pin:
                raise PilotContractError("OpenRouter profiles require a provider pin")
            if self.allow_fallbacks:
                raise PilotContractError("OpenRouter pilot profiles forbid fallbacks")
            if not self.require_parameters:
                raise PilotContractError(
                    "OpenRouter pilot profiles require parameter support"
                )
            if self.json_mode != "json_object":
                raise PilotContractError("OpenRouter pilot profiles require JSON mode")
            if self.routing_mode != "standard":
                raise PilotContractError(
                    "OpenRouter pilot profiles require standard non-fast routing"
                )
            if self.requested_model.endswith((":nitro", ":floor")):
                raise PilotContractError(
                    "OpenRouter fast/floor aliases are not permitted in the pilot"
                )
        elif self.routing_mode not in {"direct", "local", "diagnostic"}:
            raise PilotContractError("non-OpenRouter routing mode is invalid")

        if self.transport == "ollama":
            keys = dict(self.artifact_identity)
            v1_keys = {"manifest_sha256", "model_layer_digest"}
            v2_keys = {
                "manifest_sha256",
                "model_layer_digest",
                "model_layer_size_bytes",
                "ollama_version",
                "adapter",
                "base_url",
            }
            key_set = frozenset(keys)
            if key_set not in {frozenset(v1_keys), frozenset(v2_keys)}:
                raise PilotContractError(
                    "local model profile requires the frozen V1 or V2 artifact identity"
                )
            _sha256(keys["manifest_sha256"], "local manifest_sha256")
            layer = keys["model_layer_digest"]
            if not layer.startswith("sha256:"):
                raise PilotContractError("local model layer digest must use sha256:")
            _sha256(layer.split(":", 1)[1], "local model layer digest")
            if key_set == frozenset(v2_keys):
                try:
                    layer_size = int(keys["model_layer_size_bytes"])
                except ValueError as exc:
                    raise PilotContractError(
                        "local model_layer_size_bytes must be an integer"
                    ) from exc
                _integer(
                    layer_size,
                    "local model_layer_size_bytes",
                    minimum=1,
                )
                _text(keys["ollama_version"], "local ollama_version")
                if keys["adapter"] != "ollama-python":
                    raise PilotContractError("local adapter must be ollama-python")
                if keys["base_url"] not in {
                    "http://127.0.0.1:11434",
                    "http://localhost:11434",
                }:
                    raise PilotContractError("local Ollama endpoint must be loopback")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProviderRequestProfile":
        value = _mapping(value, "provider request profile")
        fields = {
            "profile_id",
            "transport",
            "requested_model",
            "served_model",
            "provider_pin",
            "routing_mode",
            "seed_capability",
            "reasoning",
            "json_mode",
            "price_snapshot",
            "max_attempts",
            "allow_fallbacks",
            "require_parameters",
            "artifact_identity",
        }
        v2_fields = {
            "decoding_fields",
            "dispatch_eligible",
            "ineligibility_reason",
        }
        present_v2 = bool(set(value) & v2_fields)
        _strict_keys(
            value,
            required=fields | (v2_fields if present_v2 else set()),
            optional=frozenset({"service_tier", "short_context_prompt_token_ceiling"}),
            name="provider request profile",
        )
        artifact = _mapping(value["artifact_identity"], "artifact_identity")
        decoding: tuple[tuple[str, DecodingFieldDispatch], ...] = ()
        if present_v2:
            raw_decoding = _mapping(value["decoding_fields"], "decoding_fields")
            decoding = tuple(
                (
                    str(key),
                    DecodingFieldDispatch.from_dict(
                        _mapping(item, f"decoding_fields.{key}")
                    ),
                )
                for key, item in raw_decoding.items()
            )
        return cls(
            profile_id=value["profile_id"],
            transport=value["transport"],
            requested_model=value["requested_model"],
            served_model=value["served_model"],
            provider_pin=_string_tuple(
                value["provider_pin"], "provider_pin", allow_empty=True
            ),
            routing_mode=value["routing_mode"],
            seed_capability=value["seed_capability"],
            reasoning=ReasoningProfile.from_dict(value["reasoning"]),
            json_mode=value["json_mode"],
            price_snapshot=PriceSnapshot.from_dict(value["price_snapshot"]),
            max_attempts=value["max_attempts"],
            allow_fallbacks=value["allow_fallbacks"],
            require_parameters=value["require_parameters"],
            service_tier=value.get("service_tier"),
            short_context_prompt_token_ceiling=value.get(
                "short_context_prompt_token_ceiling"
            ),
            artifact_identity=tuple(
                (str(key), str(item)) for key, item in artifact.items()
            ),
            decoding_fields=decoding,
            dispatch_eligible=(value["dispatch_eligible"] if present_v2 else True),
            ineligibility_reason=(
                value["ineligibility_reason"] if present_v2 else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "profile_id": self.profile_id,
            "transport": self.transport,
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "provider_pin": list(self.provider_pin),
            "routing_mode": self.routing_mode,
            "seed_capability": self.seed_capability,
            "reasoning": self.reasoning.to_dict(),
            "json_mode": self.json_mode,
            "price_snapshot": self.price_snapshot.to_dict(),
            "max_attempts": self.max_attempts,
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "artifact_identity": dict(self.artifact_identity),
        }
        if self.service_tier is not None:
            result["service_tier"] = self.service_tier
        if self.short_context_prompt_token_ceiling is not None:
            result["short_context_prompt_token_ceiling"] = (
                self.short_context_prompt_token_ceiling
            )
        if self.decoding_fields:
            result.update(
                {
                    "decoding_fields": {
                        key: item.to_dict() for key, item in self.decoding_fields
                    },
                    "dispatch_eligible": self.dispatch_eligible,
                    "ineligibility_reason": self.ineligibility_reason,
                }
            )
        return result

    def validate_provider_configuration(
        self,
        *,
        transport: str,
        model: str,
        max_attempts: int,
    ) -> None:
        if not self.dispatch_eligible:
            raise PilotContractError(
                f"profile {self.profile_id} is not dispatch eligible: "
                f"{self.ineligibility_reason}"
            )
        if transport != self.transport:
            raise PilotContractError(
                f"profile {self.profile_id} requires transport {self.transport}, "
                f"not {transport}"
            )
        if model != self.requested_model:
            raise PilotContractError(
                f"profile {self.profile_id} requested-model mismatch"
            )
        if max_attempts != self.max_attempts:
            raise PilotContractError(
                f"profile {self.profile_id} requires exactly one provider attempt"
            )
        if self.transport in {"openai", "openrouter"}:
            self.price_snapshot.assert_positive_for_hosted_dispatch()
        else:
            self.price_snapshot.assert_known_for_dispatch()

    def validate_dispatch(
        self,
        *,
        transport: str,
        model: str,
        seed: Optional[int],
        max_attempts: int,
    ) -> None:
        self.validate_provider_configuration(
            transport=transport,
            model=model,
            max_attempts=max_attempts,
        )
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise PilotContractError("decoding seed must be an integer or null")
        if self.decoding_fields:
            seed_dispatch = dict(self.decoding_fields)["seed"].dispatch_mode
            if seed_dispatch == "documented_unsupported_omitted" and seed is not None:
                raise PilotContractError(
                    f"profile {self.profile_id} must omit decoding seed on the wire"
                )
            if seed_dispatch == "explicit_supported" and seed is None:
                raise PilotContractError(
                    f"profile {self.profile_id} requires the frozen decoding seed"
                )
        else:
            # Immutable V1 compatibility: the historical schema used the coarse
            # model capability field as the wire-dispatch decision.
            if self.seed_capability == "unsupported" and seed is not None:
                raise PilotContractError(
                    f"profile {self.profile_id} does not support a decoding seed"
                )
            if self.seed_capability != "unsupported" and seed is None:
                raise PilotContractError(
                    f"profile {self.profile_id} requires the frozen decoding seed"
                )

    def validate_served_model(self, served_model: Any) -> str:
        actual = _text(served_model, "served model")
        if actual != self.served_model:
            raise PilotContractError(
                f"served model {actual!r} does not match frozen "
                f"{self.served_model!r}"
            )
        return actual

    def openrouter_request_options(self) -> dict[str, Any]:
        if self.transport != "openrouter":
            raise PilotContractError("profile is not an OpenRouter request")
        provider = {
            "order": list(self.provider_pin),
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        extra_body: dict[str, Any] = {"provider": provider}
        reasoning = self.reasoning.openrouter_payload()
        if reasoning is not None:
            extra_body["reasoning"] = reasoning
        return {
            "response_format": {"type": "json_object"},
            "extra_body": extra_body,
        }

    def openai_request_options(self) -> dict[str, Any]:
        if self.transport != "openai":
            raise PilotContractError("profile is not a direct OpenAI request")
        result: dict[str, Any] = {}
        if self.json_mode == "json_object":
            result["response_format"] = {"type": "json_object"}
        if self.reasoning.mode == "fixed":
            result["reasoning_effort"] = self.reasoning.effort
        if self.service_tier is not None:
            result["service_tier"] = self.service_tier
        return result


@dataclass(frozen=True, slots=True)
class PilotStageCell:
    models: tuple[str, ...]
    arms: tuple[str, ...]
    narratives: tuple[str, ...]
    execution_mode: str = "actor_run"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotStageCell":
        value = _mapping(value, "stage cell")
        _strict_keys(
            value,
            required={"models", "arms", "narratives", "execution_mode"},
            name="stage cell",
        )
        return cls(
            models=_string_tuple(value["models"], "stage cell models"),
            arms=_string_tuple(value["arms"], "stage cell arms"),
            narratives=_string_tuple(value["narratives"], "stage cell narratives"),
            execution_mode=_text(value["execution_mode"], "execution_mode"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "arms": list(self.arms),
            "narratives": list(self.narratives),
            "execution_mode": self.execution_mode,
        }


@dataclass(frozen=True, slots=True)
class PilotStage:
    stage_id: str
    enabled: bool
    budget_bucket: str
    num_agents: int
    episode_length: int
    seed_set: str
    utility_profiles: tuple[str, ...]
    shock_id: str
    cells: tuple[PilotStageCell, ...]
    prerequisites: tuple[str, ...] = ()
    reuse: tuple[str, ...] = ()
    call_roles: tuple[str, ...] = ()
    evidence_class: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("stage_id", "budget_bucket", "seed_set", "shock_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        _boolean(self.enabled, "stage.enabled")
        _integer(self.num_agents, "stage.num_agents", minimum=2)
        _integer(self.episode_length, "stage.episode_length", minimum=1)
        if not self.utility_profiles:
            raise PilotContractError("stage utility_profiles must not be empty")
        if not self.cells:
            raise PilotContractError("stage cells must not be empty")
        object.__setattr__(
            self,
            "call_roles",
            _string_tuple(
                self.call_roles,
                "stage call_roles",
                allow_empty=True,
            ),
        )
        if self.evidence_class is not None:
            normalized_evidence_class = _text(
                self.evidence_class,
                "stage.evidence_class",
            )
            if normalized_evidence_class not in {"operational", "scientific"}:
                raise PilotContractError(
                    "stage.evidence_class must be operational or scientific"
                )
            object.__setattr__(
                self,
                "evidence_class",
                normalized_evidence_class,
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotStage":
        value = _mapping(value, "stage")
        fields = {
            "stage_id",
            "enabled",
            "budget_bucket",
            "num_agents",
            "episode_length",
            "seed_set",
            "utility_profiles",
            "shock_id",
            "cells",
            "prerequisites",
            "reuse",
        }
        _strict_keys(
            value,
            required=fields,
            optional=frozenset({"call_roles", "evidence_class"}),
            name="stage",
        )
        cells = value["cells"]
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence):
            raise PilotContractError("stage cells must be an array")
        return cls(
            stage_id=value["stage_id"],
            enabled=value["enabled"],
            budget_bucket=value["budget_bucket"],
            num_agents=value["num_agents"],
            episode_length=value["episode_length"],
            seed_set=value["seed_set"],
            utility_profiles=_string_tuple(
                value["utility_profiles"], "stage utility_profiles"
            ),
            shock_id=value["shock_id"],
            cells=tuple(PilotStageCell.from_dict(cell) for cell in cells),
            prerequisites=_string_tuple(
                value["prerequisites"], "stage prerequisites", allow_empty=True
            ),
            reuse=_string_tuple(value["reuse"], "stage reuse", allow_empty=True),
            call_roles=_string_tuple(
                value.get("call_roles", ()),
                "stage call_roles",
                allow_empty=True,
            ),
            evidence_class=value.get("evidence_class"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "stage_id": self.stage_id,
            "enabled": self.enabled,
            "budget_bucket": self.budget_bucket,
            "num_agents": self.num_agents,
            "episode_length": self.episode_length,
            "seed_set": self.seed_set,
            "utility_profiles": list(self.utility_profiles),
            "shock_id": self.shock_id,
            "cells": [cell.to_dict() for cell in self.cells],
            "prerequisites": list(self.prerequisites),
            "reuse": list(self.reuse),
        }
        if self.call_roles:
            result["call_roles"] = list(self.call_roles)
        if self.evidence_class is not None:
            result["evidence_class"] = self.evidence_class
        return result


@dataclass(frozen=True, slots=True)
class PilotRunSpec:
    contract_id: str
    stage_id: str
    model_id: str
    requested_model: str
    arm_id: str
    narrative_id: str
    environment_seed: int
    decoding_seed: Optional[int]
    utility_profile_id: str
    shock_id: str
    budget_bucket: str
    num_agents: int
    episode_length: int
    execution_mode: str

    @property
    def run_id(self) -> str:
        fields = (
            self.contract_id,
            self.stage_id,
            self.model_id,
            self.arm_id,
            self.narrative_id,
            self.utility_profile_id,
            f"s{self.environment_seed}",
        )
        return "--".join(field.replace("/", "_").replace(":", "_") for field in fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "contract_id": self.contract_id,
            "stage_id": self.stage_id,
            "model_id": self.model_id,
            "requested_model": self.requested_model,
            "arm_id": self.arm_id,
            "narrative_id": self.narrative_id,
            "environment_seed": self.environment_seed,
            "decoding_seed": self.decoding_seed,
            "utility_profile_id": self.utility_profile_id,
            "shock_id": self.shock_id,
            "budget_bucket": self.budget_bucket,
            "num_agents": self.num_agents,
            "episode_length": self.episode_length,
            "execution_mode": self.execution_mode,
        }


def _contract_hash_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_copy(value)
    integrity = _mapping(payload.get("integrity"), "integrity")
    integrity = dict(integrity)
    integrity.pop("declared_sha256", None)
    payload["integrity"] = integrity
    return payload


def canonical_contract_sha256(value: Mapping[str, Any]) -> str:
    """Hash a contract while excluding its self-declared digest field."""

    return canonical_sha256(_contract_hash_payload(value))


_V2_1_SCIENCE_DESIGN_FIELDS = (
    "seeds",
    "provider_profiles",
    "arms",
    "narratives",
    "shocks",
    "utility",
    "stop_go",
    "stages",
    "parameter_dispatch_policy",
    "task_output_contracts",
    "model_roles",
    "non_claims",
)
PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256 = (
    "f3ea82bf587079dc5b999df71cd8bb748db7d56aa20bf759b370fd03bec82168"
)


def science_design_sha256(value: Mapping[str, Any]) -> str:
    """Hash the frozen experiment design, excluding operational budget caps."""

    payload = {field: _json_copy(value[field]) for field in _V2_1_SCIENCE_DESIGN_FIELDS}
    denominator = _json_copy(value["denominator_policy"])
    denominator.pop("policy_id")
    payload["denominator_policy"] = denominator
    return canonical_sha256(payload)


_V2_1_EXPECTED_CI_FIELDS = {
    "test_count",
    "test_collection_sha256",
    "compiled_source_count",
    "compiled_source_inventory_sha256",
    "sealed_manifest_inventory_sha256",
}


def _validate_v2_1_expected_ci_state(
    value: Any,
    *,
    status: str,
    name: str,
) -> Mapping[str, Any]:
    """Require an all-null draft or an all-concrete frozen CI identity."""

    expected_ci = _mapping(value, name)
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name=name,
    )
    null_fields = {
        field for field in _V2_1_EXPECTED_CI_FIELDS if expected_ci[field] is None
    }
    if status == "draft":
        if null_fields != _V2_1_EXPECTED_CI_FIELDS:
            raise PilotContractError("V2.1 draft expected_ci must be exactly all-null")
    elif status == "frozen":
        if null_fields:
            raise PilotContractError(
                "V2.1 frozen expected_ci must be exactly all-concrete"
            )
        _integer(expected_ci["test_count"], "test_count", minimum=1)
        _integer(
            expected_ci["compiled_source_count"],
            "compiled_source_count",
            minimum=1,
        )
        for field in (
            "test_collection_sha256",
            "compiled_source_inventory_sha256",
            "sealed_manifest_inventory_sha256",
        ):
            _sha256(expected_ci[field], field)
    else:
        raise PilotContractError("V2.1 status must be draft or frozen")
    return expected_ci


def _validate_v2_1_operational_amendment(
    value: Any,
) -> Mapping[str, Any]:
    """Validate the one authorized operational retry and its parent receipts."""

    amendment = _mapping(value, "operational_amendment")
    _strict_keys(
        amendment,
        required={
            "schema_version",
            "amendment_id",
            "parent",
            "failure",
            "inherited_results",
            "retry_policy",
            "budget_carry_forward",
        },
        name="operational_amendment",
    )
    if (
        amendment["schema_version"] != "finevo-pilot-operational-amendment-v1"
        or amendment["amendment_id"] != "finevo-pilot-v2.1-operational-retry-1"
    ):
        raise PilotContractError("V2.1 operational amendment identity drifted")

    parent = _mapping(amendment["parent"], "operational_amendment.parent")
    _strict_keys(
        parent,
        required={
            "contract_id",
            "contract_sha256",
            "release_tag",
            "release_commit",
            "launch_input_sha256",
            "release_attestation_sha256",
            "run_ledger_file_sha256",
            "run_ledger_internal_sha256",
            "budget_ledger_file_sha256",
            "budget_ledger_internal_sha256",
        },
        name="operational_amendment.parent",
    )
    expected_parent = {
        "contract_id": PILOT_CONTRACT_ID_V2,
        "contract_sha256": PILOT_CONTRACT_V2_CANONICAL_SHA256,
        "release_tag": PILOT_CONTRACT_TAG_V2,
        "release_commit": "3664778727813e5e8328b4b17b91a28c8122f87c",
        "launch_input_sha256": (
            "6516ce8660d588aaf13381353f67d9cacd991d5236a7d3ae8c41ef1c0a88d357"
        ),
        "release_attestation_sha256": (
            "54a11dc86df139a3656934ff81920ae1f10c9425afa44815e30c9befda583895"
        ),
        "run_ledger_file_sha256": (
            "34b1a763f4f1c5824249e4acaaa83334f2b254eb899b4def14a4c3365eefd60f"
        ),
        "run_ledger_internal_sha256": (
            "9d54ac1f22a56bafbe59164c7074d87bf914d290bc1282c631d50a4529f41fff"
        ),
        "budget_ledger_file_sha256": (
            "f1318f47977ad956e206dfb53ff8a14350338c691b17f44280121504a99d2882"
        ),
        "budget_ledger_internal_sha256": (
            "d9ec2c1bdfcc407aeb555ba71ee9d5e274d924e9a98791c5839ec749f3b1a0f2"
        ),
    }
    if _json_copy(parent) != expected_parent:
        raise PilotContractError("V2.1 parent release binding drifted")

    failure = _mapping(amendment["failure"], "operational_amendment.failure")
    _strict_keys(
        failure,
        required={
            "affected_run_id",
            "error_type",
            "failure_count",
            "capability_sha256",
            "gate_sha256",
            "terminal_sha256",
            "served_model_observation",
            "capability_status",
            "parent_terminal_status",
            "root_cause_codes",
            "secret_rotation_required",
        },
        name="operational_amendment.failure",
    )
    expected_failure = {
        "affected_run_id": (
            "finevo-pilot-v2--capability-gate--gpt52_main--capability-probe--"
            "none--provider-preflight-default--s2010922376"
        ),
        "error_type": "APIConnectionError",
        "failure_count": 30,
        "capability_sha256": (
            "da9076389db58fd682d213ccb932d66bb767f73423e5476abea788eb1f8fd294"
        ),
        "gate_sha256": (
            "176547171d88dad5e757dc1795cef749bea57ea7e7291191a240c8cd92c57997"
        ),
        "terminal_sha256": (
            "10b5ff7c78b4697b9754c809bed0e7d14380729a640632585085ad7f886704c6"
        ),
        "served_model_observation": "null-pre-response",
        "capability_status": "not_evaluable",
        "parent_terminal_status": "capability-no-go",
        "root_cause_codes": [
            "credential-header-trailing-whitespace",
            "capability-reader-nullability-drift",
        ],
        "secret_rotation_required": True,
    }
    if _json_copy(failure) != expected_failure:
        raise PilotContractError("V2.1 retry failure binding drifted")

    inherited = amendment["inherited_results"]
    if (
        isinstance(inherited, (str, bytes))
        or not isinstance(inherited, Sequence)
        or len(inherited) != 1
    ):
        raise PilotContractError(
            "V2.1 must inherit exactly one parent capability result"
        )
    inherited_result = _mapping(
        inherited[0],
        "operational_amendment.inherited_results[0]",
    )
    _strict_keys(
        inherited_result,
        required={
            "model_id",
            "run_id",
            "status",
            "capability_sha256",
            "gate_sha256",
            "terminal_sha256",
            "scores",
        },
        name="operational_amendment.inherited_results[0]",
    )
    scores = _mapping(
        inherited_result["scores"],
        "operational_amendment.inherited_results[0].scores",
    )
    _strict_keys(
        scores,
        required={
            "utility_ranking",
            "rule_application",
            "rule_proposal",
        },
        name="operational_amendment.inherited_results[0].scores",
    )
    for score_name in (
        "utility_ranking",
        "rule_application",
        "rule_proposal",
    ):
        _strict_keys(
            _mapping(
                scores[score_name],
                ("operational_amendment.inherited_results[0].scores." f"{score_name}"),
            ),
            required={"correct", "denominator"},
            name=("operational_amendment.inherited_results[0].scores." f"{score_name}"),
        )
    expected_inherited = {
        "model_id": "llama33_local_controlled",
        "run_id": (
            "finevo-pilot-v2--capability-gate--llama33_local_controlled--"
            "capability-probe--none--provider-preflight-default--s2010922376"
        ),
        "status": "capability-no-go",
        "capability_sha256": (
            "4c4c864733f32166c286e22b446dc3849df624a267ad083426ee4a89e79052ca"
        ),
        "gate_sha256": (
            "01c61c25e7d25577975dbe3aae8a408f464d685210b071aa612f4bd46bb78eda"
        ),
        "terminal_sha256": (
            "544b409e6ce8538958ec6278f5311429f14cf24591f8479bff287512d02e7380"
        ),
        "scores": {
            "utility_ranking": {"correct": 12, "denominator": 12},
            "rule_application": {"correct": 10, "denominator": 12},
            "rule_proposal": {"correct": 0, "denominator": 6},
        },
    }
    if _json_copy(inherited_result) != expected_inherited:
        raise PilotContractError("V2.1 inherited capability result drifted")

    retry = _mapping(
        amendment["retry_policy"],
        "operational_amendment.retry_policy",
    )
    _strict_keys(
        retry,
        required={
            "eligible_model_ids",
            "ineligible_parent_terminal_model_ids",
            "preserve_parent_denominator",
            "retry_is_operational_amendment",
            "unchanged_science_fields",
            "failed_seed_replacement",
            "outcome_inspected_for_retry",
        },
        name="operational_amendment.retry_policy",
    )
    expected_retry = {
        "eligible_model_ids": ["gpt52_main"],
        "ineligible_parent_terminal_model_ids": ["llama33_local_controlled"],
        "preserve_parent_denominator": True,
        "retry_is_operational_amendment": True,
        "unchanged_science_fields": "science-critical-v2-fieldset",
        "failed_seed_replacement": "forbidden",
        "outcome_inspected_for_retry": False,
    }
    if _json_copy(retry) != expected_retry:
        raise PilotContractError("V2.1 operational retry policy drifted")

    carry = _mapping(
        amendment["budget_carry_forward"],
        "operational_amendment.budget_carry_forward",
    )
    _strict_keys(
        carry,
        required={
            "source_stage_bucket",
            "cost_usd",
            "hosted_completions",
            "storage_bytes",
        },
        name="operational_amendment.budget_carry_forward",
    )
    expected_carry = {
        "source_stage_bucket": "capability",
        "cost_usd": 1.0701145,
        "hosted_completions": 30,
        "storage_bytes": 479367,
    }
    if _json_copy(carry) != expected_carry:
        raise PilotContractError("V2.1 parent budget carry-forward drifted")
    return _freeze_json(amendment)


def _v2_2_expected_evaluator_amendment() -> dict[str, Any]:
    """Return the exact one-off evaluator correction authorized by V2.2."""

    gpt52_run_id = (
        "finevo-pilot-v2.1--capability-gate--gpt52_main--capability-probe--"
        "none--provider-preflight-default--s2010922376"
    )
    llama33_run_id = (
        "finevo-pilot-v2--capability-gate--llama33_local_controlled--"
        "capability-probe--none--provider-preflight-default--s2010922376"
    )
    return {
        "schema_version": "finevo-pilot-evaluator-amendment-v1",
        "amendment_id": ("finevo-pilot-v2.2-capability-admission-correction-1"),
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_1,
            "contract_sha256": PILOT_CONTRACT_V2_1_CANONICAL_SHA256,
            "release_tag": PILOT_CONTRACT_TAG_V2_1,
            "release_commit": PILOT_V2_2_PARENT_RELEASE_COMMIT,
            "launch_input_sha256": PILOT_V2_2_PARENT_LAUNCH_INPUT_SHA256,
            "release_attestation_sha256": (
                PILOT_V2_2_PARENT_RELEASE_ATTESTATION_SHA256
            ),
            "run_ledger_file_sha256": (PILOT_V2_2_PARENT_RUN_LEDGER_FILE_SHA256),
            "run_ledger_internal_sha256": (
                PILOT_V2_2_PARENT_RUN_LEDGER_INTERNAL_SHA256
            ),
            "budget_ledger_file_sha256": (PILOT_V2_2_PARENT_BUDGET_LEDGER_FILE_SHA256),
            "budget_ledger_internal_sha256": (
                PILOT_V2_2_PARENT_BUDGET_LEDGER_INTERNAL_SHA256
            ),
            "capability_stage_receipt_file_sha256": (
                PILOT_V2_2_PARENT_STAGE_RECEIPT_FILE_SHA256
            ),
            "capability_stage_receipt_content_sha256": (
                PILOT_V2_2_PARENT_STAGE_RECEIPT_CONTENT_SHA256
            ),
        },
        "source_attempts": [
            {
                "model_id": "gpt52_main",
                "contract_id": PILOT_CONTRACT_ID_V2_1,
                "contract_sha256": PILOT_CONTRACT_V2_1_CANONICAL_SHA256,
                "release_tag": PILOT_CONTRACT_TAG_V2_1,
                "release_commit": PILOT_V2_2_PARENT_RELEASE_COMMIT,
                "run_id": gpt52_run_id,
                "capability_sha256": PILOT_V2_2_GPT52_CAPABILITY_SHA256,
                "gate_sha256": PILOT_V2_2_GPT52_GATE_SHA256,
                "terminal_sha256": PILOT_V2_2_GPT52_TERMINAL_SHA256,
                "terminal_content_sha256": (PILOT_V2_2_GPT52_TERMINAL_CONTENT_SHA256),
                "old_scores": {
                    "utility_ranking": {"correct": 12, "denominator": 12},
                    "rule_application": {"correct": 12, "denominator": 12},
                    "rule_proposal": {"correct": 0, "denominator": 6},
                },
                "old_status": "capability-no-go",
            },
            {
                "model_id": "llama33_local_controlled",
                "contract_id": PILOT_CONTRACT_ID_V2,
                "contract_sha256": PILOT_CONTRACT_V2_CANONICAL_SHA256,
                "release_tag": PILOT_CONTRACT_TAG_V2,
                "release_commit": ("3664778727813e5e8328b4b17b91a28c8122f87c"),
                "run_id": llama33_run_id,
                "capability_sha256": PILOT_V2_2_LLAMA33_CAPABILITY_SHA256,
                "gate_sha256": PILOT_V2_2_LLAMA33_GATE_SHA256,
                "terminal_sha256": PILOT_V2_2_LLAMA33_TERMINAL_SHA256,
                "terminal_content_sha256": (PILOT_V2_2_LLAMA33_TERMINAL_CONTENT_SHA256),
                "old_scores": {
                    "utility_ranking": {"correct": 12, "denominator": 12},
                    "rule_application": {"correct": 10, "denominator": 12},
                    "rule_proposal": {"correct": 0, "denominator": 6},
                },
                "old_status": "capability-no-go",
            },
        ],
        "defect": {
            "code": "hidden-exact-expected-match-not-preregistered",
            "affected_capability_schema": "finevo-capability-gate-v3",
            "contract_gate_path": (
                "stop_go.capability.semantic_candidate_acceptance_required"
            ),
            "contract_gate_value": True,
            "old_proposal_gate": ("semantic_candidate_accepted && semantic_match"),
            "corrected_proposal_gate": "semantic_candidate_accepted",
            "semantic_match_disposition": "diagnostic-only",
            "source_candidate_payload_retained": False,
            "independent_candidate_replay_available": False,
        },
        "rescore_policy": {
            "eligible_source_model_ids": [
                "gpt52_main",
                "llama33_local_controlled",
            ],
            "apply_uniformly_to_all_source_models": True,
            "interface_valid_required": True,
            "strict_parse_required": True,
            "strict_schema_required": True,
            "accepted_parse_mode_required": True,
            "semantic_candidate_acceptance_required": True,
            "candidate_status_required": "provisional",
            "minimum_unique_support_ids": 2,
            "support_ids_must_be_allowed": True,
            "semantic_match_required": False,
            "source_denominator_preserved": True,
            "provider_redispatch": "forbidden",
            "failed_seed_replacement": "forbidden",
            "provider_calls": 0,
            "old_scores_retained": True,
            "capability_outcomes_inspected": True,
            "scientific_effect_outcomes_inspected": False,
            "rescore_is_scientific_evidence": False,
        },
        "corrected_results": [
            {
                "model_id": "gpt52_main",
                "source_run_id": gpt52_run_id,
                "status": "complete",
                "scores": {
                    "utility_ranking": {"correct": 12, "denominator": 12},
                    "rule_application": {"correct": 12, "denominator": 12},
                    "rule_proposal": {"correct": 6, "denominator": 6},
                },
                "provider_calls": 0,
            },
            {
                "model_id": "llama33_local_controlled",
                "source_run_id": llama33_run_id,
                "status": "complete",
                "scores": {
                    "utility_ranking": {"correct": 12, "denominator": 12},
                    "rule_application": {"correct": 10, "denominator": 12},
                    "rule_proposal": {"correct": 6, "denominator": 6},
                },
                "provider_calls": 0,
            },
        ],
        "budget_carry_forward": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_1,
            "source_contract_sha256": PILOT_CONTRACT_V2_1_CANONICAL_SHA256,
            "source_stage_bucket": "capability",
            "cost_usd": 1.53775475,
            "hosted_completions": 60,
            "storage_bytes": 715_860,
        },
    }


def _validate_v2_2_evaluator_amendment(
    value: Any,
) -> Mapping[str, Any]:
    """Fail closed unless V2.2 carries the exact audited evaluator correction."""

    amendment = _mapping(value, "evaluator_amendment")
    _strict_keys(
        amendment,
        required={
            "schema_version",
            "amendment_id",
            "parent",
            "source_attempts",
            "defect",
            "rescore_policy",
            "corrected_results",
            "budget_carry_forward",
        },
        name="evaluator_amendment",
    )
    expected = _v2_2_expected_evaluator_amendment()
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.2 evaluator amendment drifted")
    return _freeze_json(amendment)


def _v2_3_expected_preflight_bootstrap_amendment() -> dict[str, Any]:
    """Return the exact zero-dispatch V2.2 preflight correction."""

    return {
        "schema_version": "finevo-pilot-preflight-bootstrap-amendment-v1",
        "amendment_id": ("finevo-pilot-v2.3-closed-loop-preflight-bootstrap-1"),
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_2,
            "contract_sha256": PILOT_CONTRACT_V2_2_CANONICAL_SHA256,
            "release_tag": PILOT_CONTRACT_TAG_V2_2,
            "release_commit": PILOT_V2_3_PARENT_RELEASE_COMMIT,
            "launch_input_sha256": PILOT_V2_3_PARENT_LAUNCH_INPUT_SHA256,
            "launch_input_file_sha256": (PILOT_V2_3_PARENT_LAUNCH_INPUT_FILE_SHA256),
            "release_attestation_sha256": (
                PILOT_V2_3_PARENT_RELEASE_ATTESTATION_SHA256
            ),
            "release_attestation_file_sha256": (
                PILOT_V2_3_PARENT_RELEASE_ATTESTATION_FILE_SHA256
            ),
            "run_ledger_file_sha256": (PILOT_V2_3_PARENT_RUN_LEDGER_FILE_SHA256),
            "run_ledger_internal_sha256": (
                PILOT_V2_3_PARENT_RUN_LEDGER_INTERNAL_SHA256
            ),
            "run_ledger_event_head": (PILOT_V2_3_PARENT_RUN_LEDGER_EVENT_HEAD),
            "run_ledger_event_count": 151,
            "budget_ledger_file_sha256": (PILOT_V2_3_PARENT_BUDGET_LEDGER_FILE_SHA256),
            "budget_ledger_internal_sha256": (
                PILOT_V2_3_PARENT_BUDGET_LEDGER_INTERNAL_SHA256
            ),
            "budget_ledger_event_head": (PILOT_V2_3_PARENT_BUDGET_LEDGER_EVENT_HEAD),
            "budget_ledger_event_count": 6,
            "capability_stage_receipt_file_sha256": (
                PILOT_V2_3_PARENT_CAPABILITY_STAGE_RECEIPT_FILE_SHA256
            ),
            "capability_stage_receipt_content_sha256": (
                PILOT_V2_3_PARENT_CAPABILITY_STAGE_RECEIPT_CONTENT_SHA256
            ),
            "preflight_stage_receipt_file_sha256": (
                PILOT_V2_3_PARENT_PREFLIGHT_STAGE_RECEIPT_FILE_SHA256
            ),
            "preflight_stage_receipt_content_sha256": (
                PILOT_V2_3_PARENT_PREFLIGHT_STAGE_RECEIPT_CONTENT_SHA256
            ),
        },
        "failure_audits": [
            {
                "model_id": "gpt52_main",
                "run_id": (
                    "finevo-pilot-v2.2--closed-loop-preflight--gpt52_main--"
                    "closed-loop-preflight--none--provider-preflight-default--"
                    "s2010922376"
                ),
                "provider_transport": "openai",
                "failure_manifest_file_sha256": (
                    "c9cbef1ef06afa3e731cb25f7371ba7227a78feeef94acc8cc111a0850b5e63e"
                ),
                "failure_payload_sha256": (
                    "6a1566b8e9db8f9ac10e8875ba09b070f15a64b910890ee9a75c05bc2f3d70b9"
                ),
                "failure_payload_size_bytes": 17370,
                "error_type": "VerifiedRunError",
                "error_message": (
                    "scientific dispatch lacks an exact observed+25% preflight "
                    "p95 reservation for openai/gpt-5.2-2025-12-11::action"
                ),
                "completed_provider_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "partial_streams_persisted": False,
                "failure_artifact_storage_bytes": 17663,
            },
            {
                "model_id": "llama33_local_controlled",
                "run_id": (
                    "finevo-pilot-v2.2--closed-loop-preflight--"
                    "llama33_local_controlled--closed-loop-preflight--none--"
                    "provider-preflight-default--s2010922376"
                ),
                "provider_transport": "ollama",
                "failure_manifest_file_sha256": (
                    "e8296a010529aa28e2440fb5f18a474d7cd6bd7963aecbd00fc4d7270b390a7d"
                ),
                "failure_payload_sha256": (
                    "efad5d73e6c23cf097dc7a0db8da022d8a4a37d6b9627cecce3beceee05eff6a"
                ),
                "failure_payload_size_bytes": 17621,
                "error_type": "VerifiedRunError",
                "error_message": (
                    "scientific dispatch lacks an exact observed+25% preflight "
                    "p95 reservation for "
                    "ollama/llama3.3:70b-instruct-q4_K_M::action"
                ),
                "completed_provider_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "partial_streams_persisted": False,
                "failure_artifact_storage_bytes": 17914,
            },
        ],
        "defect": {
            "code": "closed-loop-preflight-p95-bootstrap-cycle",
            "affected_path": ("verified_memory.pilot_orchestrator._preflight_config"),
            "guard_path": (
                "verified_memory.runner.validate_preflight_p95_reservations"
            ),
            "required_observation": "closed-loop model-by-call-kind p95",
            "missing_bootstrap_source": "capability usage rows",
            "failure_before_first_provider_dispatch": True,
            "provider_key_failure": False,
            "model_output_failure": False,
            "scientific_effect_outcomes_available": False,
        },
        "retry_policy": {
            "eligible_stage_ids": ["closed-loop-preflight"],
            "eligible_model_ids": [
                "gpt52_main",
                "llama33_local_controlled",
            ],
            "preserve_parent_denominator": True,
            "retry_is_operational_amendment": True,
            "provider_redispatch": "allowed-once-after-zero-dispatch-parent",
            "same_environment_seed_required": 2010922376,
            "failed_seed_replacement": "forbidden",
            "capability_import_reused": True,
            "provider_calls_in_parent_attempt": 0,
            "parent_actual_cost_usd": 0.0,
            "implementation_failure_inspected": True,
            "model_outputs_inspected": False,
            "scientific_effect_outcomes_inspected": False,
        },
        "bootstrap_policy": {
            "allowed_execution_mode": "closed_loop_preflight",
            "applies_to_all_registered_closed_loop_preflights": True,
            "same_model_validated_capability_source_required": True,
            "source": "validated-capability-usage-projection-rows",
            "source_output_contract_map": {
                "actor-action": "action",
                "semantic-proposal": "semantic",
            },
            "required_sample_counts": {
                "action": 24,
                "semantic": 6,
            },
            "target_dispatch_call_counts": {
                "action": 12,
                "semantic": 4,
            },
            "p95_method": "nearest-rank-with-observed-maximum-floor",
            "reserve_multiplier": 1.25,
            "runtime_model_prefix_required": True,
            "unknown_price_policy": "stop-before-dispatch",
            "missing_or_malformed_source_policy": "stop-before-dispatch",
            "normal_scientific_dispatch_reservation_source": (
                "sealed-closed-loop-preflight-projection-only"
            ),
            "normal_scientific_dispatch_policy_unchanged": True,
        },
        "budget_carry_forward": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_2,
            "source_contract_sha256": PILOT_CONTRACT_V2_2_CANONICAL_SHA256,
            "source_stage_bucket": "capability",
            "cost_usd": 1.53775475,
            "hosted_completions": 60,
            "storage_bytes": 751437,
        },
        "limitations": [
            (
                "Only the two zero-dispatch V2.2 primary preflight cells are "
                "operational retries; the same bootstrap mechanism applies "
                "to every registered V2.3 closed-loop preflight after its "
                "same-model capability source passes. No science arms, seeds, "
                "shocks, utility candidates, or stop/go thresholds change."
            ),
            (
                "Capability observations bootstrap the measurement run only; "
                "all later scientific calls still require the sealed "
                "closed-loop observed-plus-25-percent projection."
            ),
            (
                "The immutable V2.2 no-go remains an implementation failure "
                "receipt and is not reclassified as model or scientific "
                "evidence."
            ),
        ],
    }


def _validate_v2_3_preflight_bootstrap_amendment(
    value: Any,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "preflight_bootstrap_amendment")
    expected = _v2_3_expected_preflight_bootstrap_amendment()
    _strict_keys(
        amendment,
        required=set(expected),
        name="preflight_bootstrap_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.3 preflight bootstrap amendment drifted")
    return _freeze_json(amendment)


def _v2_4_expected_parent_import_arm() -> dict[str, Any]:
    return {
        "arm_id": "parent-import",
        "family": "parent-authority-import",
        "execution_mode": "parent_authority_import",
        "parameters": {
            "provider_calls": 0,
            "scientific_evidence": False,
            "parent_denominator_reused": False,
            "parent_artifacts_read_only": True,
        },
    }


def _v2_4_expected_model_roles() -> dict[str, Any]:
    return {
        "gpt52_main": {
            "profile_id": "gpt52_main",
            "role": "primary",
            "dispatch_eligible": True,
            "ineligibility_reason": None,
            "allowed_stages": [
                "experiment-c",
                "experiment-a",
                "experiment-d",
                "experiment-b",
            ],
            "allowed_call_roles": [
                "actor-action",
                "semantic-proposal",
                "offline-verifier",
                "checkpoint-branch",
            ],
        },
        "llama33_local_controlled": {
            "profile_id": "llama33_local_controlled",
            "role": "controlled_second",
            "dispatch_eligible": True,
            "ineligibility_reason": None,
            "allowed_stages": [
                "stage0-calibration",
                "local-experiment-c",
                "local-experiment-a",
                "local-experiment-d",
                "local-experiment-b",
            ],
            "allowed_call_roles": [
                "actor-action",
                "semantic-proposal",
                "offline-verifier",
                "checkpoint-branch",
            ],
        },
        "qref_scripted": {
            "profile_id": "qref_scripted",
            "role": "calibration_only",
            "dispatch_eligible": True,
            "ineligibility_reason": None,
            "allowed_stages": [
                "parent-import",
                "q-ref-resolution",
            ],
            "allowed_call_roles": [
                "parent-authority-import",
                "qref-scripted",
            ],
        },
    }


def _v2_4_stage(
    stage_id: str,
    *,
    budget_bucket: str,
    model_id: str,
    arms: Sequence[str],
    execution_mode: str,
    seed_set: str,
    shock_id: str,
    utility_profiles: Sequence[str],
    prerequisites: Sequence[str],
    call_roles: Sequence[str],
    num_agents: int = 4,
    episode_length: int = 12,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "enabled": True,
        "budget_bucket": budget_bucket,
        "num_agents": num_agents,
        "episode_length": episode_length,
        "seed_set": seed_set,
        "utility_profiles": list(utility_profiles),
        "shock_id": shock_id,
        "cells": [
            {
                "models": [model_id],
                "arms": list(arms),
                "narratives": ["none"],
                "execution_mode": execution_mode,
            }
        ],
        "prerequisites": list(prerequisites),
        "reuse": [],
        "call_roles": list(call_roles),
    }


def _v2_4_expected_stages() -> list[dict[str, Any]]:
    calibration_profiles = [
        "center",
        "psi-1",
        "psi-4",
        "nu-0.5",
        "nu-2",
        "q0-0.5x",
        "q0-2x",
    ]
    local = "llama33_local_controlled"
    hosted = "gpt52_main"
    return [
        _v2_4_stage(
            "parent-import",
            budget_bucket="parent_v23",
            model_id="qref_scripted",
            arms=["parent-import"],
            execution_mode="parent_authority_import",
            seed_set="preflight",
            shock_id="baseline-3pct",
            utility_profiles=["provider-preflight-default"],
            prerequisites=[],
            call_roles=["parent-authority-import"],
            num_agents=2,
            episode_length=1,
        ),
        _v2_4_stage(
            "q-ref-resolution",
            budget_bucket="local",
            model_id="qref_scripted",
            arms=["qref-scripted"],
            execution_mode="q_ref_resolution",
            seed_set="q-ref",
            shock_id="baseline-3pct",
            utility_profiles=["provider-preflight-default"],
            prerequisites=["parent-import"],
            call_roles=["qref-scripted"],
        ),
        _v2_4_stage(
            "stage0-calibration",
            budget_bucket="local",
            model_id=local,
            arms=["stage0-no-memory-no-context"],
            execution_mode="actor_run",
            seed_set="calibration",
            shock_id="baseline-3pct",
            utility_profiles=calibration_profiles,
            prerequisites=["parent-import", "q-ref-resolution"],
            call_roles=["actor-action"],
        ),
        _v2_4_stage(
            "local-experiment-c",
            budget_bucket="local",
            model_id=local,
            arms=[
                "full",
                "unverified-dual",
                "verified-error-candidate",
                "verified-error-forced",
                "unverified-error-forced",
            ],
            execution_mode="actor_run",
            seed_set="main",
            shock_id="registered-rate-shock",
            utility_profiles=["stage0-selected"],
            prerequisites=["stage0-calibration"],
            call_roles=[
                "actor-action",
                "semantic-proposal",
                "offline-verifier",
            ],
        ),
        _v2_4_stage(
            "local-experiment-a",
            budget_bucket="local",
            model_id=local,
            arms=["no-context", "prompt-only", "retrieval-only", "full"],
            execution_mode="actor_run",
            seed_set="main",
            shock_id="registered-rate-shock",
            utility_profiles=["stage0-selected"],
            prerequisites=["local-experiment-c"],
            call_roles=["actor-action", "semantic-proposal"],
        ),
        _v2_4_stage(
            "local-experiment-d",
            budget_bucket="local",
            model_id=local,
            arms=[
                "matched-a",
                "matched-b",
                "no-memory",
                "shuffled-episodic",
                "wrong-context",
                "error-verified",
                "error-unverified",
            ],
            execution_mode="checkpoint_continuation",
            seed_set="main",
            shock_id="registered-rate-shock",
            utility_profiles=["stage0-selected"],
            prerequisites=["local-experiment-c", "local-experiment-a"],
            call_roles=["actor-action", "checkpoint-branch"],
        ),
        _v2_4_stage(
            "local-experiment-b",
            budget_bucket="local",
            model_id=local,
            arms=[
                "no-memory",
                "episodic-only",
                "semantic-only",
                "unverified-dual",
                "full",
            ],
            execution_mode="actor_run",
            seed_set="main",
            shock_id="registered-rate-shock",
            utility_profiles=["stage0-selected"],
            prerequisites=["local-experiment-d"],
            call_roles=["actor-action", "semantic-proposal"],
        ),
        _v2_4_stage(
            "experiment-c",
            budget_bucket="hosted_confirmatory",
            model_id=hosted,
            arms=[
                "full",
                "unverified-dual",
                "verified-error-candidate",
                "verified-error-forced",
                "unverified-error-forced",
            ],
            execution_mode="actor_run",
            seed_set="main",
            shock_id="registered-rate-shock",
            utility_profiles=["stage0-selected"],
            prerequisites=["local-experiment-b"],
            call_roles=[
                "actor-action",
                "semantic-proposal",
                "offline-verifier",
            ],
        ),
        _v2_4_stage(
            "experiment-a",
            budget_bucket="hosted_confirmatory",
            model_id=hosted,
            arms=["no-context", "prompt-only", "retrieval-only", "full"],
            execution_mode="actor_run",
            seed_set="main",
            shock_id="registered-rate-shock",
            utility_profiles=["stage0-selected"],
            prerequisites=["experiment-c"],
            call_roles=["actor-action", "semantic-proposal"],
        ),
        _v2_4_stage(
            "experiment-d",
            budget_bucket="hosted_confirmatory",
            model_id=hosted,
            arms=[
                "matched-a",
                "matched-b",
                "no-memory",
                "wrong-context",
                "error-verified",
                "error-unverified",
            ],
            execution_mode="checkpoint_continuation",
            seed_set="main",
            shock_id="registered-rate-shock",
            utility_profiles=["stage0-selected"],
            prerequisites=["experiment-c", "experiment-a"],
            call_roles=["actor-action", "checkpoint-branch"],
        ),
        _v2_4_stage(
            "experiment-b",
            budget_bucket="hosted_confirmatory",
            model_id=hosted,
            arms=["full", "episodic-only", "no-memory"],
            execution_mode="actor_run",
            seed_set="main",
            shock_id="registered-rate-shock",
            utility_profiles=["stage0-selected"],
            prerequisites=["experiment-d"],
            call_roles=["actor-action", "semantic-proposal"],
        ),
    ]


def _v2_4_expected_non_claims() -> list[str]:
    return [
        (
            "V2.3 remains an immutable 174-cell complete-with-no-go budget "
            "receipt and is not resumed, rewritten, or counted in V2.4 effects."
        ),
        (
            "The V2.4 parent-import cell performs zero provider calls and "
            "imports only hash-bound operational authorities, not science outcomes."
        ),
        (
            "Narrative interventions and new cross-model provider calls are "
            "deferred and are not registered V2.4 cells."
        ),
        (
            "GPT-5.6, Gemini-3.5-Flash, Llama-4-Maverick, and Opus-4.8 retain "
            "their V2.3 boundary statuses without V2.4 redispatch."
        ),
        (
            "The local Llama and GPT-5.2 matrices are environment-seed paired "
            "but do not reuse decoded completions."
        ),
        (
            "The 4-agent by 12-month mechanism micro-pilot does not establish "
            "backbone independence, full-scale validity, or real-economy validity."
        ),
        ("No V2.4 paid dispatch is authorized while the contract status is draft."),
    ]


def _v2_4_expected_matrix_amendment() -> dict[str, Any]:
    return {
        "schema_version": "finevo-pilot-matrix-amendment-v1",
        "amendment_id": "finevo-pilot-v2.4-local-first-confirmatory-1",
        "prospective_registration": {
            "outcome_blind": True,
            "scientific_outcomes_observed_before_amendment": False,
            "scientific_effect_outcomes_observed": False,
            "parent_terminal_status_inspected": True,
            "parent_science_outputs_inspected": False,
            "parent_artifacts_modified": False,
            "parent_runs_resumed": False,
            "failed_seed_replacement": "forbidden",
        },
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_3,
            "contract_sha256": PILOT_CONTRACT_V2_3_CANONICAL_SHA256,
            "contract_file_sha256": PILOT_V2_4_PARENT_CONTRACT_FILE_SHA256,
            "release_tag": PILOT_CONTRACT_TAG_V2_3,
            "release_commit": PILOT_V2_4_PARENT_RELEASE_COMMIT,
            "release_attestation_file_sha256": (
                PILOT_V2_4_PARENT_RELEASE_ATTESTATION_FILE_SHA256
            ),
            "run_ledger_file_sha256": PILOT_V2_4_PARENT_RUN_LEDGER_FILE_SHA256,
            "run_ledger_internal_sha256": (
                PILOT_V2_4_PARENT_RUN_LEDGER_INTERNAL_SHA256
            ),
            "run_ledger_event_count": 176,
            "run_ledger_event_head": PILOT_V2_4_PARENT_RUN_LEDGER_EVENT_HEAD,
            "budget_ledger_file_sha256": (PILOT_V2_4_PARENT_BUDGET_LEDGER_FILE_SHA256),
            "budget_ledger_internal_sha256": (
                PILOT_V2_4_PARENT_BUDGET_LEDGER_INTERNAL_SHA256
            ),
            "budget_ledger_event_count": 22,
            "budget_ledger_event_head": (PILOT_V2_4_PARENT_BUDGET_LEDGER_EVENT_HEAD),
            "registered_cells": 174,
            "terminal_cells": 174,
            "terminal_status": "complete-with-no-go",
            "terminal_reason": "budget-no-go",
            "terminal_status_counts": {
                "complete": 8,
                "budget-stopped": 151,
                "capability-no-go": 14,
                "failed": 1,
            },
        },
        "budget_carry_forward": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_3,
            "source_contract_sha256": PILOT_CONTRACT_V2_3_CANONICAL_SHA256,
            "source_stage_bucket": "parent_v23",
            "cost_usd": 3.212770875,
            "hosted_completions": 184,
            "storage_bytes": 4_196_087,
            "debit_before_new_dispatch": True,
            "parent_import_cell_additional_cost_usd": 0.0,
            "parent_import_cell_additional_hosted_completions": 0,
        },
        "parent_source_manifest": {
            "path": "experiments/pilot_v2_4_parent_source_manifest.json",
            "schema_version": "finevo-pilot-v2.4-parent-source-manifest-v1",
            "file_sha256": (
                "d6a867cd7add43818127af7778a447d579ac1ab31ed6d053bcd29d69b3cf0f33"
            ),
            "content_sha256": (
                "7ae427fe6eac5aa6e04eddd3efa9e63405e128c782013ed3f67c35808be3cec5"
            ),
        },
        "parent_authority_import": {
            "execution_mode": "parent_authority_import",
            "provider_calls": 0,
            "authority_remains_parent_labeled": True,
            "import_is_scientific_evidence": False,
            "exact_file_and_content_hash_required": True,
            "same_runtime_and_served_model_required": True,
            "missing_or_malformed_source_policy": "stop-before-dispatch",
            "allowed_use": (
                "source-backed-observed-p95-reservation-authority-for-v2.4"
            ),
            "capability_stage_receipt": {
                "file_sha256": (
                    "28fb188c09ab69fd2623818a15d2edcdcf329925dee5ef1f222b34240535c475"
                ),
                "content_sha256": (
                    "d15fce84b8911450ab41b07cb7e2dd99e7c37752047a7b81ce230ba3fbb37b88"
                ),
            },
            "profiles": {
                "gpt52_main": {
                    "runtime_model": "openai/gpt-5.2-2025-12-11",
                    "served_model": "gpt-5.2-2025-12-11",
                    "authority_receipt_file_sha256": (
                        "1edce6fa30f530e2f83c16b64488da394d5aca04565276d482f61262fe522289"
                    ),
                    "authority_receipt_content_sha256": (
                        "3406c3bdd532f3fe529d85072d207ff3a97b4bf3a8ed3afc78397df972fdf3f0"
                    ),
                    "source_run_spec_sha256": (
                        "b64434c45890fa8a13ec7f4e2570b37a2e065374b6fbaac4f39e5b81a1269934"
                    ),
                    "source_projection_file_sha256": (
                        "87b5f31e91b1ad7d75b2dbd9f3558a6893a23225b6e5f867a43cd04f1a7a12e3"
                    ),
                    "source_projection_content_sha256": (
                        "88a964e42cfff6cdd7b1aa1f6082abe5165f854d297414444ea3a0c1ab9d3a7b"
                    ),
                },
                "llama33_local_controlled": {
                    "runtime_model": "ollama/llama3.3:70b-instruct-q4_K_M",
                    "served_model": "llama3.3:70b-instruct-q4_K_M",
                    "authority_receipt_file_sha256": (
                        "26d7360577d6729e5e9a5ec4a8b2dea529a69b14770775ab1e96a3666dcc3aba"
                    ),
                    "authority_receipt_content_sha256": (
                        "fc857c65af5995847d2b8166ef9785309cdeb250df75d48d0aa608741b0033fa"
                    ),
                    "source_run_spec_sha256": (
                        "e27609907a940f38e73490f1b8d9e3902a18800203c29c78036b212050d68dcd"
                    ),
                    "source_projection_file_sha256": (
                        "8d6c9867ed7ac5a2968069eeb2df0bb1205ed5fce48b9640c5619fff72ea7ba0"
                    ),
                    "source_projection_content_sha256": (
                        "6ed39de17f7a5f4f17e58cf0e4a29581201c6dfd1c13b21e964371df3f35c09d"
                    ),
                },
            },
        },
        "matrix": {
            "stage_order": [
                "parent-import",
                "q-ref-resolution",
                "stage0-calibration",
                "local-experiment-c",
                "local-experiment-a",
                "local-experiment-d",
                "local-experiment-b",
                "experiment-c",
                "experiment-a",
                "experiment-d",
                "experiment-b",
            ],
            "active_provider_profile_ids": [
                "gpt52_main",
                "llama33_local_controlled",
                "qref_scripted",
            ],
            "main_seeds": [
                1099057501,
                1421875452,
                1769977770,
                959809858,
                617806385,
            ],
            "registered_cells": 211,
            "scientific_cells": 209,
            "parent_import_cells": 1,
            "q_ref_cells": 1,
            "local_model_id": "llama33_local_controlled",
            "hosted_confirmatory_model_id": "gpt52_main",
            "stage_priority": "C-A-D-B",
            "narrative_registration": "deferred-unregistered",
            "new_cross_model_calls": "deferred-unregistered",
            "local_arms": {
                "C": [
                    "full",
                    "unverified-dual",
                    "verified-error-candidate",
                    "verified-error-forced",
                    "unverified-error-forced",
                ],
                "A": ["no-context", "prompt-only", "retrieval-only", "full"],
                "D": [
                    "matched-a",
                    "matched-b",
                    "no-memory",
                    "shuffled-episodic",
                    "wrong-context",
                    "error-verified",
                    "error-unverified",
                ],
                "B": [
                    "no-memory",
                    "episodic-only",
                    "semantic-only",
                    "unverified-dual",
                    "full",
                ],
            },
            "hosted_arms": {
                "C": [
                    "full",
                    "unverified-dual",
                    "verified-error-candidate",
                    "verified-error-forced",
                    "unverified-error-forced",
                ],
                "A": ["no-context", "prompt-only", "retrieval-only", "full"],
                "D": [
                    "matched-a",
                    "matched-b",
                    "no-memory",
                    "wrong-context",
                    "error-verified",
                    "error-unverified",
                ],
                "B": ["full", "episodic-only", "no-memory"],
            },
        },
        "budget_projection": {
            "hard_cap_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "hard_cap_status": _PILOT_V2_4_HARD_CAP_STATUS,
            "automatic_reserve_usd": 1.0,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "new_hosted_core_completions": 4240,
            "new_local_logical_calls": 5672,
            "sealed_observed_p95_plus_25pct_core_cost_usd": 143.6043,
            "paid_dispatch_allowed_while_draft": False,
            "matrix_shrink_on_projection_failure": False,
        },
        "preserved_model_boundaries": {
            "gpt56_diagnostic": "secondary-diagnostic-no-v2.4-redispatch",
            "gemini35_flash_diagnostic": (
                "capability-pass-preflight-failed-no-v2.4-redispatch"
            ),
            "llama4_maverick_diagnostic": ("capability-no-go-no-v2.4-redispatch"),
            "opus48_no_go": "omitted-no-v2.4-redispatch",
        },
    }


def _validate_v2_4_matrix_amendment(value: Any) -> Mapping[str, Any]:
    amendment = _mapping(value, "matrix_amendment")
    expected = _v2_4_expected_matrix_amendment()
    _strict_keys(
        amendment,
        required=set(expected),
        name="matrix_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.4 matrix amendment drifted")
    return _freeze_json(amendment)


def _v2_5_expected_parent_import_retry_amendment() -> dict[str, Any]:
    """Return the only outcome-blind retry amendment accepted for V2.5."""

    return {
        "schema_version": "finevo-pilot-parent-import-retry-amendment-v1",
        "amendment_id": "finevo-pilot-v2.5-parent-import-integrity-retry-1",
        "source_manifest": {
            "path": "experiments/pilot_v2_5_source_manifest.json",
            "schema_version": "finevo-pilot-v2.5-source-manifest-v1",
            "file_sha256": (
                "62b01d74d93ae249dab8711ab7748fdd3bcfaad1ee8c5fb39f71c4bd492b81a4"
            ),
            "content_sha256": (
                "4f21ee44514d29e5b7fcbd4ca48daad046dc674fd0dda5f4b2c502c87f346c91"
            ),
        },
        "failure_classification": {
            "parent_contract_id": PILOT_CONTRACT_ID_V2_4,
            "parent_contract_sha256": PILOT_CONTRACT_V2_4_CANONICAL_SHA256,
            "failed_stage_id": "parent-import",
            "terminal_status": "integrity-stopped",
            "propagated_terminal_cells": 211,
            "provider_calls": 0,
            "cost_usd": 0.0,
            "scientific_effect_outcomes_available": False,
            "scientific_effect_outcomes_inspected": False,
            "root_cause_code": (
                "parent-checkpoint-code-binding-evaluated-against-child-import-root"
            ),
        },
        "correction_policy": {
            "parent_artifact_root_role": "v2.3-raw-source-only",
            "parent_code_tag": PILOT_CONTRACT_TAG_V2_3,
            "parent_code_tag_object": ("e985abd6749471363db6b27bda66485c0b578bb3"),
            "parent_code_commit": PILOT_V2_4_PARENT_RELEASE_COMMIT,
            "historical_code_binding_verification": (
                "annotated-tag-peeled-commit-git-tree"
            ),
            "historical_source_hashes_and_binding_hash_required": True,
            "child_code_as_parent_binding_authority": False,
            "unconditional_strict_disable_forbidden": True,
            "current_compatibility_replay_after_historical_gate": True,
            "current_recomputed_exactness_must_equal_frozen": True,
            "parent_checkpoint_rewrite_forbidden": True,
            "parent_authority_resigning_forbidden": True,
            "provider_construction_during_import": False,
        },
        "retry_policy": {
            "eligible_stage_ids": ["parent-import"],
            "retry_count": 1,
            "new_contract_required": True,
            "preserve_parent_denominator": True,
            "v2_4_raw_resume": "forbidden",
            "v2_4_failed_cell_reclassification": "forbidden",
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "provider_redispatch_before_import_success": "forbidden",
            "downstream_dispatch_requires_import_success": True,
            "retry_is_operational_amendment": True,
            "outcome_blind": True,
        },
        "budget_carry_forward": {
            "total_cap_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "v2_3": {
                "cost_usd": 3.212770875,
                "hosted_completions": 184,
                "storage_bytes": 4_196_087,
            },
            "v2_4_incremental": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "storage_bytes": 518_235,
            },
            "budget_reset": False,
            "debit_before_new_dispatch": True,
        },
        "science_design_invariance": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_4,
            "source_science_design_sha256": (PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256),
            "registered_cells": 211,
            "scientific_cells": 209,
            "matrix_profile_id": (
                "local-llama-full-ad-gpt52-fixed-confirmatory-cadb-v1"
            ),
            "seeds_arms_models_stage_order_unchanged": True,
            "provider_profiles_unchanged": True,
            "shock_utility_stop_go_unchanged": True,
            "budget_envelope_unchanged": True,
        },
        "raw_namespace": {
            "parent": "experiment_results/pilot-v2.4/raw",
            "child": "experiment_results/pilot-v2.5/raw",
            "shared": False,
        },
    }


def _validate_v2_5_parent_import_retry_amendment(
    value: Any,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "parent_import_retry_amendment")
    expected = _v2_5_expected_parent_import_retry_amendment()
    _strict_keys(
        amendment,
        required=set(expected),
        name="parent_import_retry_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.5 parent-import retry amendment drifted")
    return _freeze_json(amendment)


def _v2_6_expected_p95_authority_retry_amendment(
    *,
    status: str,
) -> dict[str, Any]:
    """Return the only outcome-blind V2.6 authority-interface retry."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.6 status must be draft or frozen")
    source_file_sha256 = (
        None if status == "draft" else PILOT_V2_6_SOURCE_MANIFEST_FILE_SHA256
    )
    source_content_sha256 = (
        None if status == "draft" else PILOT_V2_6_SOURCE_MANIFEST_CONTENT_SHA256
    )
    if status == "frozen" and (
        source_file_sha256 is None or source_content_sha256 is None
    ):
        raise PilotContractError(
            "V2.6 cannot be frozen before its source-manifest hashes"
        )

    return {
        "schema_version": "finevo-pilot-p95-authority-retry-amendment-v1",
        "amendment_id": "finevo-pilot-v2.6-p95-authority-interface-retry-1",
        "source_manifest": {
            "path": "experiments/pilot_v2_6_source_manifest.json",
            "schema_version": "finevo-pilot-v2.6-source-manifest-v1",
            "file_sha256": source_file_sha256,
            "content_sha256": source_content_sha256,
        },
        "failure_classification": {
            "parent_contract_id": PILOT_CONTRACT_ID_V2_5,
            "parent_contract_sha256": PILOT_CONTRACT_V2_5_CANONICAL_SHA256,
            "failed_stage_id": "stage0-calibration",
            "terminal_status": "complete-with-no-go",
            "registered_cells": 211,
            "scientific_cells": 209,
            "status_counts": {
                "complete": 2,
                "failed": 14,
                "integrity-stopped": 195,
            },
            "provider_calls": 0,
            "incremental_cost_usd": 0.0,
            "incremental_hosted_completions": 0,
            "scientific_effect_outcomes_available": False,
            "scientific_effect_outcomes_inspected": False,
            "root_cause_code": (
                "v2.5-inherited-p95-schema-not-dispatched-by-runner-verifier"
            ),
            "failure_message_sha256": (
                "39cb7f19f94e435d9eb4873df49beac2507703522f2ad9ffa7f688a5f6b92ef7"
            ),
            "run_ledger_internal_sha256": (
                "7d223ddc2cc46b022f051217b9f6767bf9264fb66212b1a63a3498fb6447220f"
            ),
            "budget_ledger_internal_sha256": (
                "7b448a0ebc002b932150c68f2c4e552e940ce186ea5e58afed8673af627d9162"
            ),
        },
        "correction_policy": {
            "parent_authority_contract_id": PILOT_CONTRACT_ID_V2_5,
            "parent_authority_tag": PILOT_CONTRACT_TAG_V2_5,
            "parent_authority_commit": "a3ec8d96162b50e41e7d4700e0534ce33c1958c3",
            "parent_receipt_schema": (
                "finevo-pilot-v2.5-inherited-observed-p95-authority-v1"
            ),
            "child_receipt_schema": (
                "finevo-pilot-v2.6-inherited-observed-p95-authority-v1"
            ),
            "parent_receipt_is_source_only": True,
            "parent_receipt_and_projection_bytes_immutable": True,
            "parent_receipt_reverified_before_child_reseal": True,
            "child_receipt_resealed_to_current_contract_tag_and_head": True,
            "generic_reader_exact_schema_dispatch_required": True,
            "unknown_inherited_schema_fails_closed": True,
            "current_head_equals_annotated_tag_commit_required": True,
            "parent_authority_resigning_forbidden": True,
            "provider_construction_during_import": False,
        },
        "retry_policy": {
            "new_contract_required": True,
            "new_211_cell_denominator_required": True,
            "preserve_parent_denominator": True,
            "v2_5_raw_resume": "forbidden",
            "v2_5_failed_cell_reclassification": "forbidden",
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "provider_redispatch_before_import_success": "forbidden",
            "downstream_dispatch_requires_import_success": True,
            "retry_is_operational_amendment": True,
            "outcome_blind": True,
        },
        "budget_carry_forward": {
            "total_cap_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "cumulative_prior": {
                "stage_bucket": "parent_v23",
                "parent_contract_sha256": (
                    "1f9809062684a1a2afb96b7342b88a06810e0e87ac883aa63a858a65a81d188d"
                ),
                "parent_run_ledger_sha256": (
                    "7d223ddc2cc46b022f051217b9f6767bf9264fb66212b1a63a3498fb6447220f"
                ),
                "parent_budget_ledger_sha256": (
                    "7b448a0ebc002b932150c68f2c4e552e940ce186ea5e58afed8673af627d9162"
                ),
                "cost_usd": 3.212770875,
                "hosted_completions": 184,
                "storage_bytes": 6_303_635,
                "record_sha256": (
                    "4f445491738ea756280fca0b8c5c82823f4cefe7574cd368ed0c2c51c6a48802"
                ),
            },
            "v2_5_incremental": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "raw_file_count": 61,
                "storage_bytes": 1_589_313,
            },
            "budget_reset": False,
            "debit_before_new_dispatch": True,
            "manual_reserve_automatic_use": False,
        },
        "science_design_invariance": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_5,
            "source_science_design_sha256": PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
            "registered_cells": 211,
            "scientific_cells": 209,
            "matrix_profile_id": (
                "local-llama-full-ad-gpt52-fixed-confirmatory-cadb-v1"
            ),
            "seeds_arms_models_stage_order_unchanged": True,
            "provider_profiles_unchanged": True,
            "shock_utility_stop_go_unchanged": True,
            "budget_envelope_unchanged": True,
        },
        "raw_namespace": {
            "parent": "experiment_results/pilot-v2.5/raw",
            "child": "experiment_results/pilot-v2.6/raw",
            "shared": False,
        },
    }


def _validate_v2_6_p95_authority_retry_amendment(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "p95_authority_retry_amendment")
    expected = _v2_6_expected_p95_authority_retry_amendment(status=status)
    _strict_keys(
        amendment,
        required=set(expected),
        name="p95_authority_retry_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.6 p95-authority retry amendment drifted")
    return _freeze_json(amendment)


def _v2_7_expected_stage0_evaluator_retry_amendment(
    *,
    status: str,
) -> dict[str, Any]:
    """Return the only accepted V2.7 Stage-0 evaluator/import amendment."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.7 status must be draft or frozen")
    source_file_sha256 = (
        None if status == "draft" else PILOT_V2_7_SOURCE_MANIFEST_FILE_SHA256
    )
    source_content_sha256 = (
        None if status == "draft" else PILOT_V2_7_SOURCE_MANIFEST_CONTENT_SHA256
    )
    if status == "frozen" and (
        source_file_sha256 is None or source_content_sha256 is None
    ):
        raise PilotContractError(
            "V2.7 cannot be frozen before its source-manifest hashes"
        )

    return {
        "schema_version": "finevo-pilot-stage0-evaluator-retry-amendment-v1",
        "amendment_id": "finevo-pilot-v2.7-stage0-baseline-evaluator-retry-1",
        "source_manifest": {
            "path": "experiments/pilot_v2_7_source_manifest.json",
            "schema_version": "finevo-pilot-v2.7-source-manifest-v1",
            "file_sha256": source_file_sha256,
            "content_sha256": source_content_sha256,
        },
        "failure_classification": {
            "parent_contract_id": PILOT_CONTRACT_ID_V2_6,
            "parent_contract_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
            "parent_release_tag": PILOT_CONTRACT_TAG_V2_6,
            "parent_release_commit": ("0f59a15bc2cc3cce68f64de1dc1be78f7d74e214"),
            "failed_stage_id": "stage0-calibration",
            "terminal_status": "complete-with-no-go",
            "registered_cells": 211,
            "scientific_cells": 209,
            "status_counts": {
                "complete": 16,
                "integrity-stopped": 195,
            },
            "completed_cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            },
            "local_model_calls": 672,
            "hosted_provider_calls": 0,
            "incremental_cost_usd": 0.0,
            "incremental_hosted_completions": 0,
            "stage0_calibration_artifacts_available": True,
            "stage0_calibration_selection_observed_before_amendment": True,
            "a_d_treatment_effect_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_available": False,
            "a_d_treatment_effect_outcomes_inspected": False,
            "root_cause_code": (
                "baseline-only-stage0-routed-through-shock-recovery-summary"
            ),
            "root_cause_message": ("run has no pre-shock utility observations"),
            "run_ledger_internal_sha256": (
                "cca42a01c6685994fa8b22e0fc7c7fb2067e4b1973fd37acf7c47be4591337d4"
            ),
            "budget_ledger_internal_sha256": (
                "73f218feb368c5770908a4107d78037e11871e1c68e1fdd8461b94953549cdba"
            ),
        },
        "observation_boundary": {
            "stage0_calibration_selection_observed_before_amendment": True,
            "stage0_guardrail_outputs_may_have_been_inspected": True,
            "stage0_candidate_winner_may_have_been_observed": True,
            "a_d_treatment_effect_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_observed": False,
            "amendment_is_outcome_blind_with_respect_to_a_d_effects": True,
            "calibration_thresholds_unchanged": True,
            "calibration_tiebreak_order_unchanged": True,
            "calibration_seed_set_unchanged": True,
            "calibration_candidate_profiles_unchanged": True,
            "calibration_model_and_actions_unchanged": True,
            "failed_seed_replacement": "forbidden",
        },
        "artifact_import": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_6,
            "source_contract_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
            "source_release_tag": PILOT_CONTRACT_TAG_V2_6,
            "source_raw_namespace": "experiment_results/pilot-v2.6/raw",
            "child_raw_namespace": "experiment_results/pilot-v2.7/raw",
            "shared_namespace": False,
            "exact_source_file_inventory_required": True,
            "exact_file_and_content_hashes_required": True,
            "imported_complete_cells": 16,
            "imported_cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            },
            "parent_authority_reverification_required": True,
            "q_ref_reverification_required": True,
            "stage0_manifest_reverification_required": True,
            "child_artifacts_resealed_to_current_contract_tag_and_head": True,
            "provider_construction_during_import": False,
            "provider_redispatch_for_imported_cells": "forbidden",
            "decoded_completion_reuse_beyond_imported_stage0": "forbidden",
            "missing_or_malformed_source_policy": "stop-before-dispatch",
        },
        "stage0_reader_correction": {
            "reader_schema": "finevo-pilot-stage0-analysis-v1",
            "reader_scope": "stage0-baseline-calibration",
            "baseline_only_schedule": True,
            "phase_agnostic": True,
            "pre_shock_phase_required": False,
            "shock_phase_required": False,
            "recovery_phase_required": False,
            "shock_recovery_effect_metrics_computed": False,
            "allowed_record_streams": [
                "actions",
                "utility_ledger",
                "errors",
            ],
            "allowed_selector_inputs": [
                "max_abs_budget_residual",
                "clipping_count",
                "ceiling_labor_rate",
                "zero_labor_rate",
                "interior_labor_rate",
                "interior_consumption_rate",
                "median_labor_disutility_to_consumption_utility",
            ],
            "inherited_selector_method": ("guardrail-then-registered-tiebreak-v1"),
            "inherited_tiebreak_order": [
                "maximize mean interior action coverage",
                "minimize component-balance log distance from one",
                "minimize normalized center distance",
                "declaration order only for an exact remaining tie",
            ],
            "inherited_candidate_profiles": [
                "center",
                "psi-1",
                "psi-4",
                "nu-0.5",
                "nu-2",
                "q0-0.5x",
                "q0-2x",
            ],
            "inherited_calibration_seeds": [1942013315, 760687867],
            "future_treatment_information_allowed": False,
        },
        "retry_policy": {
            "new_contract_required": True,
            "new_211_cell_denominator_required": True,
            "preserve_parent_denominator": True,
            "v2_6_raw_resume": "forbidden",
            "v2_6_terminal_cell_reclassification": "forbidden",
            "v2_6_status_counts_rewrite": "forbidden",
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "provider_redispatch_for_imported_cells": "forbidden",
            "downstream_dispatch_requires_import_and_selection_success": True,
            "retry_is_evaluator_and_import_amendment": True,
            "a_d_outcome_blind": True,
        },
        "budget_carry_forward": {
            "total_cap_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "cumulative_prior": {
                "stage_bucket": "parent_v23",
                "cost_usd": 3.212770875,
                "hosted_completions": 184,
                "parent_contract_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
                "parent_run_ledger_sha256": (
                    "cca42a01c6685994fa8b22e0fc7c7fb2067e4b1973fd37acf7c47be4591337d4"
                ),
                "parent_budget_ledger_sha256": (
                    "73f218feb368c5770908a4107d78037e11871e1c68e1fdd8461b94953549cdba"
                ),
                "storage_bytes": 19_181_432,
                "record_sha256": (
                    "6d5a9461485122a3770e9855229dfc120728ab6da1f4f9074c5150515a62285e"
                ),
            },
            "v2_6_incremental": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "local_model_calls": 672,
            },
            "budget_reset": False,
            "debit_before_new_dispatch": True,
            "manual_reserve_automatic_use": False,
        },
        "science_design_invariance": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_6,
            "source_science_design_sha256": (PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256),
            "registered_cells": 211,
            "scientific_cells": 209,
            "matrix_profile_id": (
                "local-llama-full-ad-gpt52-fixed-confirmatory-cadb-v1"
            ),
            "seeds_arms_models_stage_order_unchanged": True,
            "provider_profiles_unchanged": True,
            "shock_utility_stop_go_unchanged": True,
            "budget_envelope_unchanged": True,
        },
    }


def _validate_v2_7_stage0_evaluator_retry_amendment(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "stage0_evaluator_retry_amendment")
    expected = _v2_7_expected_stage0_evaluator_retry_amendment(status=status)
    _strict_keys(
        amendment,
        required=set(expected),
        name="stage0_evaluator_retry_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.7 Stage-0 evaluator retry amendment drifted")
    return _freeze_json(amendment)


def _v2_8_expected_qref_identity_retry_amendment(
    *,
    status: str,
) -> dict[str, Any]:
    """Return the only accepted V2.8 q-ref identity retry amendment."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.8 status must be draft or frozen")
    source_file_sha256 = (
        None if status == "draft" else PILOT_V2_8_SOURCE_MANIFEST_FILE_SHA256
    )
    source_content_sha256 = (
        None if status == "draft" else PILOT_V2_8_SOURCE_MANIFEST_CONTENT_SHA256
    )
    if status == "frozen" and (
        source_file_sha256 is None or source_content_sha256 is None
    ):
        raise PilotContractError(
            "V2.8 cannot be frozen before its source-manifest hashes"
        )

    return {
        "schema_version": "finevo-pilot-qref-identity-retry-amendment-v1",
        "amendment_id": "finevo-pilot-v2.8-qref-identity-retry-1",
        "source_manifest": {
            "path": "experiments/pilot_v2_8_source_manifest.json",
            "schema_version": "finevo-pilot-v2.8-source-manifest-v1",
            "file_sha256": source_file_sha256,
            "content_sha256": source_content_sha256,
        },
        "failure_classification": {
            "parent_contract_id": PILOT_CONTRACT_ID_V2_7,
            "parent_contract_sha256": PILOT_CONTRACT_V2_7_CANONICAL_SHA256,
            "parent_release_tag": PILOT_CONTRACT_TAG_V2_7,
            "parent_release_commit": ("60566410f38f7842169e93ae9822f180235b60b6"),
            "parent_evidence_commit": ("f15a26418264b5de31f53dbe7c46c1949761fcb6"),
            "parent_evidence_merge_commit": (
                "e951aa865186a7c2e841316fc6bb08a716aeaf80"
            ),
            "failed_stage_id": "q-ref-resolution",
            "terminal_status": "complete-with-no-go",
            "registered_cells": 211,
            "scientific_cells": 209,
            "status_counts": {
                "complete": 1,
                "integrity-stopped": 210,
            },
            "completed_cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 0,
                "stage0-calibration": 0,
            },
            "hosted_provider_calls": 0,
            "incremental_cost_usd": 0.0,
            "incremental_hosted_completions": 0,
            "q_ref_resolution_artifact_available": False,
            "stage0_calibration_artifacts_available_in_v2_7": False,
            "a_d_treatment_effect_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_available": False,
            "a_d_treatment_effect_outcomes_inspected": False,
            "root_cause_code": (
                "qref-contract-cell-id-conflated-with-runner-execution-id"
            ),
            "root_cause_message": ("imported source run identity is malformed"),
            "run_ledger_internal_sha256": (
                "ab532bb56232efbc42d6e5f48c9f80c451f461a732cf2607774f6055de9deb4a"
            ),
            "budget_ledger_internal_sha256": (
                "70ff3f40bbebaea766c6403fc1f2879af9002faff287a112a39c2ce405d92170"
            ),
            "q_ref_failure_stage_receipt_file_sha256": (
                "45ad6725749333852902cfade0b0858bfbe9a85af7648556343f7d350510201d"
            ),
            "evidence_package_manifest_file_sha256": (
                "1b44a8984b61f00cbae4851a599674fb3e0479ca60d3259961460f99519e23bb"
            ),
        },
        "observation_boundary": {
            "q_ref_identity_failure_observed_before_amendment": True,
            "stage0_calibration_selection_observed_before_amendment": True,
            "stage0_guardrail_outputs_may_have_been_inspected": True,
            "stage0_candidate_winner_may_have_been_observed": True,
            "a_d_treatment_effect_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_observed": False,
            "amendment_is_outcome_blind_with_respect_to_a_d_effects": True,
            "calibration_thresholds_unchanged": True,
            "calibration_tiebreak_order_unchanged": True,
            "calibration_seed_set_unchanged": True,
            "calibration_candidate_profiles_unchanged": True,
            "calibration_model_and_actions_unchanged": True,
            "failed_seed_replacement": "forbidden",
        },
        "source_lineage": {
            "amendment_parent_contract_id": PILOT_CONTRACT_ID_V2_7,
            "amendment_parent_contract_sha256": (PILOT_CONTRACT_V2_7_CANONICAL_SHA256),
            "amendment_parent_raw_namespace": ("experiment_results/pilot-v2.7/raw"),
            "nested_stage0_source_contract_id": PILOT_CONTRACT_ID_V2_6,
            "nested_stage0_source_contract_sha256": (
                PILOT_CONTRACT_V2_6_CANONICAL_SHA256
            ),
            "nested_stage0_snapshot_namespace": (
                "experiment_results/pilot-v2.7/raw/parent-import/" "v2_6_raw_snapshot"
            ),
            "child_raw_namespace": "experiment_results/pilot-v2.8/raw",
            "shared_namespace": False,
            "exact_parent_inventory_required": True,
            "exact_nested_source_inventory_required": True,
            "source_artifacts_rewritten": False,
            "parent_terminal_no_go_preserved": True,
            "parent_denominator_reclassified": False,
            "parent_evidence_package_required": True,
        },
        "q_ref_regeneration": {
            "execution_mode": "q_ref_resolution",
            "source_result_reuse": "forbidden",
            "fresh_zero_hosted_provider_regeneration": True,
            "hosted_provider_construction_during_regeneration": False,
            "scripted_diagnostic_provider_required": True,
            "hosted_provider_calls": 0,
            "scripted_diagnostic_calls": 48,
            "hosted_cost_usd": 0.0,
            "config_run_id_policy": ("must-equal-current-contract-cell-run-id"),
            "contract_cell_run_id_policy": ("fresh-finevo-pilot-v2.8-itt-identity"),
            "scripted_action_schedule_unchanged": True,
            "baseline_shock_schedule_unchanged": True,
            "environment_source_hash_reverification_required": True,
            "q_ref_scalar_reverification_required": True,
            "full_source_core_validation_required": True,
        },
        "stage0_import": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_6,
            "source_contract_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
            "source_via_v2_7_nested_snapshot": True,
            "imported_complete_cells": 14,
            "imported_cell_breakdown": {
                "stage0-calibration": 14,
            },
            "exact_file_and_content_hashes_required": True,
            "stage0_manifest_reverification_required": True,
            "provider_construction_during_import": False,
            "provider_redispatch_for_imported_cells": "forbidden",
            "decoded_completion_reuse_beyond_imported_stage0": "forbidden",
            "child_artifacts_resealed_to_current_contract_tag_and_head": True,
            "missing_or_malformed_source_policy": "stop-before-dispatch",
        },
        "fresh_science_dispatch": {
            "a_d_cells": 195,
            "a_d_provider_dispatch": "fresh-only",
            "imported_a_d_completions": 0,
            "decoded_completion_reuse": "forbidden",
            "downstream_dispatch_requires_q_ref_and_stage0_success": True,
        },
        "retry_policy": {
            "new_contract_required": True,
            "new_211_cell_denominator_required": True,
            "preserve_parent_denominator": True,
            "v2_7_raw_resume": "forbidden",
            "v2_7_terminal_cell_reclassification": "forbidden",
            "v2_7_status_counts_rewrite": "forbidden",
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "q_ref_provider_redispatch": "forbidden",
            "stage0_provider_redispatch": "forbidden",
            "a_d_outcome_blind": True,
        },
        "budget_carry_forward": {
            "total_cap_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "cumulative_prior": {
                "stage_bucket": "parent_v23",
                "cost_usd": 3.212770875,
                "hosted_completions": 184,
                "parent_contract_sha256": (PILOT_CONTRACT_V2_7_CANONICAL_SHA256),
                "parent_run_ledger_sha256": (
                    "ab532bb56232efbc42d6e5f48c9f80c451f461a732cf2607774f6055de9deb4a"
                ),
                "parent_budget_ledger_sha256": (
                    "70ff3f40bbebaea766c6403fc1f2879af9002faff287a112a39c2ce405d92170"
                ),
                "record_sha256": (
                    "a5caad9515eb797a035c26d32d0a0cf7bfd7f0df210e7362bd3b93da18ff3ff7"
                ),
                "storage_bytes": 32_158_175,
            },
            "v2_7_incremental": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "local_model_calls": 0,
            },
            "budget_reset": False,
            "debit_before_new_dispatch": True,
            "manual_reserve_automatic_use": False,
            "whole_remaining_matrix_projection_required": True,
            "projection_reserve_multiplier": 1.25,
            "unknown_price_policy": "stop-before-dispatch",
        },
        "science_design_invariance": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_7,
            "source_science_design_sha256": (PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256),
            "registered_cells": 211,
            "scientific_cells": 209,
            "matrix_profile_id": (
                "local-llama-full-ad-gpt52-fixed-confirmatory-cadb-v1"
            ),
            "seeds_arms_models_stage_order_unchanged": True,
            "provider_profiles_unchanged": True,
            "shock_utility_stop_go_unchanged": True,
            "p95_authority_basis_unchanged": True,
            "budget_envelope_unchanged": True,
        },
        "evidence_lineage": {
            "parent_evidence_status": "complete-with-no-go",
            "parent_evidence_commit": ("f15a26418264b5de31f53dbe7c46c1949761fcb6"),
            "parent_evidence_merge_commit": (
                "e951aa865186a7c2e841316fc6bb08a716aeaf80"
            ),
            "parent_evidence_namespace": ("evidence/current_v2/pilot-v2.7"),
            "parent_evidence_rewrite": "forbidden",
            "parent_claim_reclassification": "forbidden",
            "v2_8_effect_aggregation_uses_only_v2_8_a_d_cells": True,
            "q_ref_and_stage0_are_prerequisites_not_effect_evidence": True,
        },
    }


def _validate_v2_8_qref_identity_retry_amendment(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "qref_identity_retry_amendment")
    expected = _v2_8_expected_qref_identity_retry_amendment(status=status)
    _strict_keys(
        amendment,
        required=set(expected),
        name="qref_identity_retry_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.8 q-ref identity retry amendment drifted")
    return _freeze_json(amendment)


def _v2_9_expected_qref_summary_equivalence_amendment(
    *,
    status: str,
) -> dict[str, Any]:
    """Return the only accepted V2.9 deterministic-summary amendment."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.9 status must be draft or frozen")
    source_file_sha256 = (
        None if status == "draft" else PILOT_V2_9_SOURCE_MANIFEST_FILE_SHA256
    )
    source_content_sha256 = (
        None if status == "draft" else PILOT_V2_9_SOURCE_MANIFEST_CONTENT_SHA256
    )
    evidence_commit = None if status == "draft" else PILOT_V2_8_EVIDENCE_COMMIT
    evidence_merge_commit = (
        None if status == "draft" else PILOT_V2_8_EVIDENCE_MERGE_COMMIT
    )
    evidence_package_sha256 = (
        None if status == "draft" else PILOT_V2_8_EVIDENCE_PACKAGE_FILE_SHA256
    )
    evidence_checksums_sha256 = (
        None if status == "draft" else PILOT_V2_8_EVIDENCE_CHECKSUMS_FILE_SHA256
    )
    if status == "frozen" and any(
        item is None
        for item in (
            source_file_sha256,
            source_content_sha256,
            evidence_commit,
            evidence_merge_commit,
            evidence_package_sha256,
            evidence_checksums_sha256,
        )
    ):
        raise PilotContractError(
            "V2.9 cannot be frozen before its source and parent-evidence " "bindings"
        )

    return {
        "schema_version": ("finevo-pilot-qref-summary-equivalence-retry-amendment-v1"),
        "amendment_id": ("finevo-pilot-v2.9-qref-summary-equivalence-retry-1"),
        "source_manifest": {
            "path": "experiments/pilot_v2_9_source_manifest.json",
            "schema_version": "finevo-pilot-v2.9-source-manifest-v1",
            "file_sha256": source_file_sha256,
            "content_sha256": source_content_sha256,
        },
        "failure_classification": {
            "parent_contract_id": PILOT_CONTRACT_ID_V2_8,
            "parent_contract_sha256": PILOT_CONTRACT_V2_8_CANONICAL_SHA256,
            "parent_release_tag": PILOT_CONTRACT_TAG_V2_8,
            "parent_release_commit": ("1988f10b5a06c3b9b3093b969c99593676721a09"),
            "parent_evidence_commit": evidence_commit,
            "parent_evidence_merge_commit": evidence_merge_commit,
            "failed_stage_id": "q-ref-resolution",
            "terminal_status": "complete-with-no-go",
            "registered_cells": 211,
            "scientific_cells": 209,
            "status_counts": {
                "complete": 1,
                "failed": 1,
                "integrity-stopped": 209,
            },
            "completed_cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 0,
                "stage0-calibration": 0,
            },
            "scripted_diagnostic_calls": 48,
            "scripted_diagnostic_total_tokens": 15_905,
            "hosted_provider_calls": 0,
            "incremental_cost_usd": 0.0,
            "incremental_hosted_completions": 0,
            "q_ref_verified_run_artifacts_available": True,
            "q_ref_resolution_artifact_available": False,
            "stage0_calibration_artifacts_available_in_v2_8": False,
            "a_d_treatment_effect_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_available": False,
            "a_d_treatment_effect_outcomes_inspected": False,
            "root_cause_code": (
                "qref-raw-summary-equivalence-included-identity-and-" "monotonic-time"
            ),
            "root_cause_message": (
                "V2.8 fresh q-ref differs from its audit reference: "
                "['run_summary_exact']"
            ),
            "failure_message_sha256": (
                "713b1d429fd939e74b2007d78d3c3789ce10376ee0ba970e1cfe1359503c246a"
            ),
            "run_ledger_internal_sha256": (
                "9b5f4bd1acdc5a525fb58b04b02ba29e31b05b594bfc411863e7baf3eb11f0d9"
            ),
            "budget_ledger_internal_sha256": (
                "07c936d61a7c38e6a7877ffaeeaf6c8ecb7fd4f495dbe8ed012a9a2861004b8f"
            ),
            "q_ref_failure_stage_receipt_file_sha256": (
                "7f386e0423866f03f8eedf9106200c1c24dd262c9263853625d7a1eaa6b69d72"
            ),
            "q_ref_failure_manifest_file_sha256": (
                "332eb6390c62a417d6ed1dc7c7c335a75ad732bea090418d6a08d8a6bcf2b92e"
            ),
            "evidence_package_manifest_file_sha256": (evidence_package_sha256),
            "evidence_checksums_file_sha256": evidence_checksums_sha256,
        },
        "observation_boundary": {
            "q_ref_summary_equivalence_failure_observed_before_amendment": True,
            "q_ref_action_utility_and_shock_streams_observed": True,
            "q_ref_is_prerequisite_not_treatment_effect_evidence": True,
            "stage0_calibration_selection_observed_before_amendment": True,
            "stage0_guardrail_outputs_may_have_been_inspected": True,
            "stage0_candidate_winner_may_have_been_observed": True,
            "a_d_treatment_effect_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_observed": False,
            "amendment_is_outcome_blind_with_respect_to_a_d_effects": True,
            "calibration_thresholds_unchanged": True,
            "calibration_tiebreak_order_unchanged": True,
            "calibration_seed_set_unchanged": True,
            "calibration_candidate_profiles_unchanged": True,
            "calibration_model_and_actions_unchanged": True,
            "failed_seed_replacement": "forbidden",
        },
        "source_lineage": {
            "amendment_parent_contract_id": PILOT_CONTRACT_ID_V2_8,
            "amendment_parent_contract_sha256": (PILOT_CONTRACT_V2_8_CANONICAL_SHA256),
            "amendment_parent_raw_namespace": ("experiment_results/pilot-v2.8/raw"),
            "nested_stage0_source_contract_id": PILOT_CONTRACT_ID_V2_6,
            "nested_stage0_source_contract_sha256": (
                PILOT_CONTRACT_V2_6_CANONICAL_SHA256
            ),
            "nested_stage0_snapshot_namespace": (
                "experiment_results/pilot-v2.8/raw/parent-import/"
                "v2_7_raw_snapshot/parent-import/v2_6_raw_snapshot"
            ),
            "child_raw_namespace": "experiment_results/pilot-v2.9/raw",
            "shared_namespace": False,
            "exact_parent_inventory_required": True,
            "exact_nested_source_inventory_required": True,
            "source_artifacts_rewritten": False,
            "parent_terminal_no_go_preserved": True,
            "parent_denominator_reclassified": False,
            "parent_evidence_package_required": True,
        },
        "q_ref_regeneration": {
            "execution_mode": "q_ref_resolution",
            "source_result_reuse": "forbidden",
            "fresh_zero_hosted_provider_regeneration": True,
            "hosted_provider_construction_during_regeneration": False,
            "scripted_diagnostic_provider_required": True,
            "scripted_diagnostic_calls": 48,
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "config_run_id_policy": ("must-equal-current-contract-cell-run-id"),
            "contract_cell_run_id_policy": ("fresh-finevo-pilot-v2.9-itt-identity"),
            "scripted_action_schedule_unchanged": True,
            "baseline_shock_schedule_unchanged": True,
            "environment_source_hash_reverification_required": True,
            "q_ref_scalar_reverification_required": True,
            "full_source_core_validation_required": True,
            "audit_reference_contract_id": PILOT_CONTRACT_ID_V2_8,
            "audit_reference_is_failed_prerequisite_artifact": True,
        },
        "run_summary_equivalence": {
            "policy_id": "finevo-pilot-v2.9-qref-run-summary-projection-v1",
            "comparison_mode": ("identity-bound-allowlist-normalization-then-exact"),
            "raw_summary_hashes_required": True,
            "projected_summary_hashes_required": True,
            "projected_summary_exact_match_required": True,
            "unexpected_difference_policy": "stop-before-stage0",
            "identity_paths": [
                "$.run_id",
                "$.api.budget_id",
                "$.api.completions[*].budget_id",
            ],
            "monotonic_time_paths": [
                "$.api.elapsed_seconds",
                "$.api.completions[*].started_elapsed_seconds",
                "$.api.completions[*].finished_elapsed_seconds",
                "$.api.completions[*].elapsed_seconds",
            ],
            "expected_completion_rows": 48,
            "expected_observed_scalar_paths": 1002,
            "expected_allowed_difference_paths": 195,
            "identity_relation_validation_required": True,
            "finite_nonnegative_timer_validation_required": True,
            "completion_interval_validation_required": True,
            "tokens_cost_models_labels_tags_retained_exactly": True,
            "actions_utility_shocks_remain_separate_exact_checks": True,
        },
        "stage0_import": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_6,
            "source_contract_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
            "source_via_v2_8_nested_snapshot": True,
            "imported_complete_cells": 14,
            "imported_cell_breakdown": {
                "stage0-calibration": 14,
            },
            "exact_file_and_content_hashes_required": True,
            "stage0_manifest_reverification_required": True,
            "provider_construction_during_import": False,
            "provider_redispatch_for_imported_cells": "forbidden",
            "decoded_completion_reuse_beyond_imported_stage0": "forbidden",
            "child_artifacts_resealed_to_current_contract_tag_and_head": True,
            "missing_or_malformed_source_policy": "stop-before-dispatch",
        },
        "fresh_science_dispatch": {
            "a_d_cells": 195,
            "a_d_provider_dispatch": "fresh-only",
            "imported_a_d_completions": 0,
            "decoded_completion_reuse": "forbidden",
            "downstream_dispatch_requires_q_ref_and_stage0_success": True,
        },
        "retry_policy": {
            "new_contract_required": True,
            "new_211_cell_denominator_required": True,
            "preserve_parent_denominator": True,
            "v2_8_raw_resume": "forbidden",
            "v2_8_terminal_cell_reclassification": "forbidden",
            "v2_8_status_counts_rewrite": "forbidden",
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "q_ref_provider_redispatch": "forbidden",
            "stage0_provider_redispatch": "forbidden",
            "a_d_outcome_blind": True,
        },
        "budget_carry_forward": {
            "total_cap_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "cumulative_prior": {
                "stage_bucket": "parent_v23",
                "cost_usd": 3.212770875,
                "hosted_completions": 184,
                "parent_contract_sha256": (PILOT_CONTRACT_V2_8_CANONICAL_SHA256),
                "parent_run_ledger_sha256": (
                    "9b5f4bd1acdc5a525fb58b04b02ba29e31b05b594bfc411863e7baf3eb11f0d9"
                ),
                "parent_budget_ledger_sha256": (
                    "07c936d61a7c38e6a7877ffaeeaf6c8ecb7fd4f495dbe8ed012a9a2861004b8f"
                ),
                "storage_bytes": 32_158_175,
                "record_sha256": (
                    "0944138d9b47f7cf720681eb0ea8feda0b612a912992d78434c6bbda0d560fd0"
                ),
            },
            "v2_8_incremental": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "scripted_diagnostic_calls": 48,
            },
            "budget_reset": False,
            "debit_before_new_dispatch": True,
            "manual_reserve_automatic_use": False,
            "projection_reserve_multiplier": 1.25,
            "whole_remaining_matrix_projection_required": True,
            "unknown_price_policy": "stop-before-dispatch",
        },
        "science_design_invariance": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_8,
            "source_science_design_sha256": (PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256),
            "registered_cells": 211,
            "scientific_cells": 209,
            "matrix_profile_id": (
                "local-llama-full-ad-gpt52-fixed-confirmatory-cadb-v1"
            ),
            "seeds_arms_models_stage_order_unchanged": True,
            "provider_profiles_unchanged": True,
            "shock_utility_stop_go_unchanged": True,
            "budget_envelope_unchanged": True,
            "p95_authority_basis_unchanged": True,
        },
        "evidence_lineage": {
            "parent_evidence_namespace": "evidence/current_v2/pilot-v2.8",
            "parent_evidence_status": "complete-with-no-go",
            "parent_evidence_commit": evidence_commit,
            "parent_evidence_merge_commit": evidence_merge_commit,
            "parent_evidence_rewrite": "forbidden",
            "parent_claim_reclassification": "forbidden",
            "v2_9_effect_aggregation_uses_only_v2_9_a_d_cells": True,
            "q_ref_and_stage0_are_prerequisites_not_effect_evidence": True,
        },
    }


def _validate_v2_9_qref_summary_equivalence_amendment(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "qref_summary_equivalence_amendment")
    expected = _v2_9_expected_qref_summary_equivalence_amendment(status=status)
    _strict_keys(
        amendment,
        required=set(expected),
        name="qref_summary_equivalence_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.9 q-ref summary-equivalence amendment drifted")
    return _freeze_json(amendment)


def _v2_10_expected_p95_runner_binding_retry_amendment(
    *,
    status: str,
) -> dict[str, Any]:
    """Return the only accepted V2.10 p95 runner-binding retry amendment."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.10 status must be draft or frozen")
    source_file_sha256 = (
        None if status == "draft" else PILOT_V2_10_SOURCE_MANIFEST_FILE_SHA256
    )
    source_content_sha256 = (
        None if status == "draft" else PILOT_V2_10_SOURCE_MANIFEST_CONTENT_SHA256
    )
    evidence_commit = None if status == "draft" else PILOT_V2_9_EVIDENCE_COMMIT
    evidence_merge_commit = (
        None if status == "draft" else PILOT_V2_9_EVIDENCE_MERGE_COMMIT
    )
    evidence_package_sha256 = (
        None if status == "draft" else PILOT_V2_9_EVIDENCE_PACKAGE_FILE_SHA256
    )
    evidence_checksums_sha256 = (
        None if status == "draft" else PILOT_V2_9_EVIDENCE_CHECKSUMS_FILE_SHA256
    )
    if status == "frozen" and any(
        item is None
        for item in (
            source_file_sha256,
            source_content_sha256,
            evidence_commit,
            evidence_merge_commit,
            evidence_package_sha256,
            evidence_checksums_sha256,
        )
    ):
        raise PilotContractError(
            "V2.10 cannot be frozen before its source and parent-evidence " "bindings"
        )

    return {
        "schema_version": ("finevo-pilot-p95-runner-binding-retry-amendment-v1"),
        "amendment_id": "finevo-pilot-v2.10-p95-runner-binding-retry-1",
        "source_manifest": {
            "path": "experiments/pilot_v2_10_source_manifest.json",
            "schema_version": "finevo-pilot-v2.10-source-manifest-v1",
            "file_sha256": source_file_sha256,
            "content_sha256": source_content_sha256,
        },
        "failure_classification": {
            "parent_contract_id": PILOT_CONTRACT_ID_V2_9,
            "parent_contract_sha256": PILOT_CONTRACT_V2_9_CANONICAL_SHA256,
            "parent_release_tag": PILOT_CONTRACT_TAG_V2_9,
            "parent_release_tag_object": ("ca7871231769ab1d7eb811b71dff79f16de363e9"),
            "parent_release_commit": ("2349ccd41560383965da8880744cf4df366c9ee5"),
            "parent_evidence_commit": evidence_commit,
            "parent_evidence_merge_commit": evidence_merge_commit,
            "terminal_status": "complete-with-no-go",
            "registered_cells": 211,
            "scientific_cells": 209,
            "status_counts": {
                "complete": 26,
                "failed": 185,
            },
            "completed_cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
                "local-experiment-c-offline-candidate-admission": 5,
                "experiment-c-offline-candidate-admission": 5,
            },
            "failed_actor_cell_count": 185,
            "failed_actor_stage_counts": {
                "local-experiment-c": 20,
                "local-experiment-a": 20,
                "local-experiment-d": 35,
                "local-experiment-b": 25,
                "experiment-c": 20,
                "experiment-a": 20,
                "experiment-d": 30,
                "experiment-b": 15,
            },
            "root_cause_code": ("imported-p95-runner-binding-shape-mismatch"),
            "root_cause_message": (
                "The imported authority producer returned nested receipt "
                "identity fields while the runner consumer dereferenced the "
                "legacy flat names."
            ),
            "failure_error_type": "KeyError",
            "failure_message": "'receipt_path'",
            "failure_message_sha256": (
                "d4b516ad7a51dc7a09dcad56e2abe7f7f5236cc49523014fc7b8b8c3fdf2870e"
            ),
            "failure_phase": "before-provider-construction-and-dispatch",
            "partial_actor_streams_persisted": False,
            "actor_action_utility_rule_exposure_outcomes_generated": False,
            "offline_candidate_admission_cells_generated": 10,
            "all_a_d_outcomes_unobserved": False,
            "incremental_cost_usd": 0.0,
            "incremental_hosted_completions": 0,
            "incremental_local_stage_cost_usd": 0.0,
            "run_ledger_internal_sha256": (
                "9cc948d75c37ffeb59a2d7ed569e140668a997fa314d523906a047375011e409"
            ),
            "budget_ledger_internal_sha256": (
                "7e75b9c58bccaa746bdc92b926352fc0d3e56adee8426d3962a80ae5ddd59e10"
            ),
            "evidence_package_manifest_file_sha256": (evidence_package_sha256),
            "evidence_checksums_file_sha256": evidence_checksums_sha256,
        },
        "observation_boundary": {
            "implementation_failure_observed_before_amendment": True,
            "a_d_actor_treatment_effect_outcomes_generated": False,
            "a_d_actor_treatment_effect_outcomes_observed": False,
            "offline_candidate_admission_outcomes_generated": 10,
            "offline_candidate_admission_outcomes_observed": True,
            "all_a_d_outcomes_unobserved_claim_forbidden": True,
            "amendment_is_outcome_blind_for_actor_treatment_effects": True,
            "amendment_is_globally_a_d_outcome_blind": False,
            "q_ref_and_stage0_prerequisites_observed": True,
            "stage0_selected_profile_observed": "nu-0.5",
            "calibration_thresholds_unchanged": True,
            "calibration_tiebreak_order_unchanged": True,
            "calibration_seed_set_unchanged": True,
            "calibration_candidate_profiles_unchanged": True,
            "calibration_model_and_actions_unchanged": True,
            "failed_seed_replacement": "forbidden",
        },
        "source_lineage": {
            "amendment_parent_contract_id": PILOT_CONTRACT_ID_V2_9,
            "amendment_parent_contract_sha256": (PILOT_CONTRACT_V2_9_CANONICAL_SHA256),
            "amendment_parent_raw_namespace": ("experiment_results/pilot-v2.9/raw"),
            "amendment_parent_raw_inventory": {
                "schema_version": "finevo-raw-tree-inventory-v1",
                "canonicalization": "json-sort-keys-compact-utf8-v1",
                "file_count": 623,
                "storage_bytes": 19_288_343,
                "inventory_sha256": (
                    "ae478634a83a98bd206bcafa03f87636fcc392f8dd1e8e234f84696f245ef22f"
                ),
            },
            "child_raw_namespace": "experiment_results/pilot-v2.10/raw",
            "child_snapshot_namespace": (
                "experiment_results/pilot-v2.10/raw/parent-import/" "v2_9_raw_snapshot"
            ),
            "shared_namespace": False,
            "exact_parent_inventory_required": True,
            "source_artifacts_rewritten": False,
            "parent_terminal_no_go_preserved": True,
            "parent_denominator_reclassified": False,
            "parent_evidence_package_required": True,
            "parent_evidence_namespace": ("evidence/current_v2/pilot-v2.9"),
            "parent_evidence_commit": evidence_commit,
            "parent_evidence_merge_commit": evidence_merge_commit,
            "parent_evidence_rewrite": "forbidden",
        },
        "prerequisite_import": {
            "imported_complete_cells": 16,
            "imported_cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            },
            "source_contract_id": PILOT_CONTRACT_ID_V2_9,
            "source_contract_sha256": PILOT_CONTRACT_V2_9_CANONICAL_SHA256,
            "exact_file_and_content_hashes_required": True,
            "source_manifest_reverification_required": True,
            "source_run_manifests_reverification_required": True,
            "q_ref_resolution_reverification_required": True,
            "stage0_selection_reverification_required": True,
            "stage0_selected_profile_id": "nu-0.5",
            "provider_construction_during_import": False,
            "provider_redispatch_for_imported_cells": "forbidden",
            "decoded_completion_reuse_beyond_imported_prerequisites": ("forbidden"),
            "child_artifacts_resealed_to_current_contract_tag_and_head": True,
            "missing_or_malformed_source_policy": "stop-before-dispatch",
            "prerequisites_are_treatment_effect_evidence": False,
        },
        "p95_runner_binding_repair": {
            "policy_id": "finevo-pilot-v2.10-current-release-p95-reseal-v1",
            "profiles": [
                "gpt52_main",
                "llama33_local_controlled",
            ],
            "source_contract_id": PILOT_CONTRACT_ID_V2_8,
            "source_contract_sha256": (PILOT_CONTRACT_V2_8_CANONICAL_SHA256),
            "source_via_exact_v2_9_snapshot": True,
            "source_reservation_values_unchanged": True,
            "current_release_wrapper_required": True,
            "current_release_contract_binding_required": True,
            "current_release_tag_and_head_binding_required": True,
            "current_release_projection_pair_required": True,
            "runner_binding_schema_version": (
                "finevo-pilot-v2.10-runner-p95-binding-v1"
            ),
            "runner_binding_required_fields": [
                "receipt_path",
                "receipt_file_sha256",
                "receipt_content_sha256",
                "git_commit",
                "reservations",
            ],
            "parent_lineage_required_fields": [
                "source_contract_id",
                "source_contract_sha256",
                "source_git_tag",
                "source_git_commit",
                "authority",
                "projection",
            ],
            "receipt_and_projection_rebuilt_from_verified_source": True,
            "receipt_and_projection_exact_pair_validation_required": True,
            "runner_binding_validation_before_provider_construction": True,
            "nested_to_flat_alias_without_current_reseal": "forbidden",
            "v2_9_failed_binding_reuse": "forbidden",
            "missing_or_malformed_binding_policy": "stop-before-dispatch",
        },
        "fresh_science_dispatch": {
            "a_d_cells": 195,
            "stage_counts": {
                "local-experiment-c": 25,
                "local-experiment-a": 20,
                "local-experiment-d": 35,
                "local-experiment-b": 25,
                "experiment-c": 25,
                "experiment-a": 20,
                "experiment-d": 30,
                "experiment-b": 15,
            },
            "a_d_provider_dispatch": "fresh-only",
            "imported_a_d_completions": 0,
            "v2_9_offline_candidate_admission_reuse": "forbidden",
            "decoded_completion_reuse": "forbidden",
            "downstream_dispatch_requires_all_16_prerequisites": True,
            "downstream_dispatch_requires_current_p95_reseal": True,
        },
        "retry_policy": {
            "new_contract_required": True,
            "new_211_cell_denominator_required": True,
            "preserve_parent_denominator": True,
            "v2_9_raw_resume": "forbidden",
            "v2_9_terminal_cell_reclassification": "forbidden",
            "v2_9_status_counts_rewrite": "forbidden",
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "prerequisite_provider_redispatch": "forbidden",
            "a_d_provider_dispatch": "fresh-only",
        },
        "budget_carry_forward": {
            "total_cap_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "cumulative_prior": {
                "stage_bucket": "parent_v23",
                "cost_usd": 3.212770875,
                "hosted_completions": 184,
                "parent_contract_sha256": (PILOT_CONTRACT_V2_9_CANONICAL_SHA256),
                "parent_run_ledger_sha256": (
                    "9cc948d75c37ffeb59a2d7ed569e140668a997fa314d523906a047375011e409"
                ),
                "parent_budget_ledger_sha256": (
                    "7e75b9c58bccaa746bdc92b926352fc0d3e56adee8426d3962a80ae5ddd59e10"
                ),
                "storage_bytes": 50_425_235,
                "record_sha256": (
                    "408b25171d23c172abcc3e5545d736ef0fdb6251524995ab0bb39b34b0b6a5e1"
                ),
            },
            "v2_9_incremental": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "scripted_diagnostic_calls": 48,
                "offline_candidate_admission_cells": 10,
            },
            "budget_reset": False,
            "debit_before_new_dispatch": True,
            "manual_reserve_automatic_use": False,
            "projection_reserve_multiplier": 1.25,
            "whole_remaining_matrix_projection_required": True,
            "unknown_price_policy": "stop-before-dispatch",
        },
        "science_design_invariance": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_9,
            "source_science_design_sha256": (PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256),
            "registered_cells": 211,
            "scientific_cells": 209,
            "prerequisite_cells": 16,
            "fresh_a_d_cells": 195,
            "matrix_profile_id": (
                "local-llama-full-ad-gpt52-fixed-confirmatory-cadb-v1"
            ),
            "seeds_arms_models_stage_order_unchanged": True,
            "provider_profiles_unchanged": True,
            "shock_utility_stop_go_unchanged": True,
            "budget_envelope_unchanged": True,
            "p95_reservation_values_unchanged": True,
        },
        "evidence_lineage": {
            "parent_evidence_namespace": "evidence/current_v2/pilot-v2.9",
            "parent_evidence_status": "complete-with-no-go",
            "parent_evidence_commit": evidence_commit,
            "parent_evidence_merge_commit": evidence_merge_commit,
            "parent_evidence_rewrite": "forbidden",
            "parent_claim_reclassification": "forbidden",
            "v2_10_effect_aggregation_uses_only_fresh_v2_10_a_d_cells": True,
            "all_16_imported_cells_are_prerequisites_not_effect_evidence": True,
            "v2_9_offline_candidate_outcomes_are_not_v2_10_effect_evidence": True,
        },
    }


def _validate_v2_10_p95_runner_binding_retry_amendment(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "p95_runner_binding_retry_amendment")
    expected = _v2_10_expected_p95_runner_binding_retry_amendment(status=status)
    _strict_keys(
        amendment,
        required=set(expected),
        name="p95_runner_binding_retry_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.10 p95 runner-binding retry amendment drifted")
    return _freeze_json(amendment)


def _v2_10_1_expected_qref_receipt_verifier_retry_amendment(
    *,
    status: str,
) -> dict[str, Any]:
    """Return the only accepted V2.10.1 q-ref receipt-verifier amendment."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.10.1 status must be draft or frozen")
    source_file_sha256 = (
        None if status == "draft" else PILOT_V2_10_1_SOURCE_MANIFEST_FILE_SHA256
    )
    source_content_sha256 = (
        None if status == "draft" else PILOT_V2_10_1_SOURCE_MANIFEST_CONTENT_SHA256
    )
    # V2.10 is already terminal and published, so its lineage is non-null in
    # both draft and frozen V2.10.1 contracts.  Only V2.10.1's own source
    # manifest/CI/canonical identities remain draft placeholders.
    evidence_commit = PILOT_V2_10_EVIDENCE_COMMIT
    evidence_merge_commit = PILOT_V2_10_EVIDENCE_MERGE_COMMIT
    evidence_package_sha256 = PILOT_V2_10_EVIDENCE_PACKAGE_FILE_SHA256
    evidence_checksums_sha256 = PILOT_V2_10_EVIDENCE_CHECKSUMS_FILE_SHA256
    raw_file_count = PILOT_V2_10_RAW_FILE_COUNT
    raw_storage_bytes = PILOT_V2_10_RAW_STORAGE_BYTES
    raw_inventory_sha256 = PILOT_V2_10_RAW_INVENTORY_SHA256
    run_ledger_sha256 = PILOT_V2_10_RUN_LEDGER_INTERNAL_SHA256
    budget_ledger_sha256 = PILOT_V2_10_BUDGET_LEDGER_INTERNAL_SHA256
    if status == "frozen" and any(
        item is None
        for item in (
            source_file_sha256,
            source_content_sha256,
            evidence_commit,
            evidence_merge_commit,
            evidence_package_sha256,
            evidence_checksums_sha256,
            raw_file_count,
            raw_storage_bytes,
            raw_inventory_sha256,
            run_ledger_sha256,
            budget_ledger_sha256,
        )
    ):
        raise PilotContractError(
            "V2.10.1 cannot be frozen before its source and immutable "
            "V2.10 terminal lineage bindings"
        )

    return {
        "schema_version": ("finevo-pilot-qref-receipt-verifier-retry-amendment-v1"),
        "amendment_id": ("finevo-pilot-v2.10.1-qref-receipt-verifier-retry-1"),
        "source_manifest": {
            "path": "experiments/pilot_v2_10_1_source_manifest.json",
            "schema_version": "finevo-pilot-v2.10.1-source-manifest-v1",
            "file_sha256": source_file_sha256,
            "content_sha256": source_content_sha256,
        },
        "failure_classification": {
            "parent_contract_id": PILOT_CONTRACT_ID_V2_10,
            "parent_contract_sha256": PILOT_CONTRACT_V2_10_CANONICAL_SHA256,
            "parent_release_tag": PILOT_CONTRACT_TAG_V2_10,
            "parent_release_tag_object": ("7ef3cfa37e1dc2a1c9f49cafe77988624b9f23a9"),
            "parent_release_commit": ("1584629a5f8fd60f42bba878d2a0fcb0eca4bdcf"),
            "parent_evidence_commit": evidence_commit,
            "parent_evidence_merge_commit": evidence_merge_commit,
            "failed_stage_id": "q-ref-resolution",
            "terminal_status": "complete-with-no-go",
            "registered_cells": 211,
            "scientific_cells": 209,
            "status_counts": {
                "complete": 1,
                "integrity-stopped": 210,
            },
            "completed_cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 0,
                "stage0-calibration": 0,
                "a-d": 0,
            },
            "root_cause_code": ("qref-stage-receipt-schema-hash-domain-mismatch"),
            "root_cause_message": (
                "The finevo-pilot-stage-receipt-v2 producer hashes the "
                "artifact after removing the complete integrity object, while "
                "the imported-prerequisite verifier applied the generic "
                "self-hash convention."
            ),
            "failure_error_type": "PilotOrchestrationError",
            "failure_message": (
                "V2.10 q-ref prerequisite failed validation: imported "
                "prerequisite stage_receipt schema or content hash mismatch"
            ),
            "failure_message_sha256": (
                "0d30e7fcfc850e1934d87a3c23b64221bdac6fb1dc96ff7126fa58ed094d5ce4"
            ),
            "failure_phase": "before-provider-construction-and-dispatch",
            "partial_actor_streams_persisted": False,
            "actor_action_utility_rule_exposure_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_generated": False,
            "incremental_cost_usd": 0.0,
            "incremental_hosted_completions": 0,
            "run_ledger_internal_sha256": run_ledger_sha256,
            "run_ledger_file_sha256": (
                "04baa9aef59481d330e7ffe88f174156df32d4ee600eea365b9ceb2bf1b623d9"
            ),
            "budget_ledger_internal_sha256": budget_ledger_sha256,
            "budget_ledger_file_sha256": (
                "60c7244c22d397884784f6a06b540e9291bfc0bb105a2f83c88bad96c3b925cb"
            ),
            "q_ref_failure_stage_receipt_file_sha256": (
                "66dac19579be01cb617fda51c6636a7a27cb3dff5f65d82e66cafb7a3da60823"
            ),
            "q_ref_failure_stage_receipt_content_sha256": (
                "48ae5807da2c3175b3fd427cc023796e7bd81c5b77695789a900474e023da098"
            ),
            "evidence_package_manifest_file_sha256": evidence_package_sha256,
            "evidence_checksums_file_sha256": evidence_checksums_sha256,
        },
        "observation_boundary": {
            "implementation_failure_observed_before_amendment": True,
            "q_ref_receipt_hash_failure_observed": True,
            "q_ref_is_prerequisite_not_treatment_effect_evidence": True,
            "a_d_actor_treatment_effect_outcomes_generated": False,
            "a_d_actor_treatment_effect_outcomes_observed": False,
            "amendment_is_outcome_blind_for_actor_treatment_effects": True,
            "calibration_thresholds_unchanged": True,
            "calibration_tiebreak_order_unchanged": True,
            "calibration_seed_set_unchanged": True,
            "calibration_candidate_profiles_unchanged": True,
            "calibration_model_and_actions_unchanged": True,
            "failed_seed_replacement": "forbidden",
        },
        "source_lineage": {
            "amendment_parent_contract_id": PILOT_CONTRACT_ID_V2_10,
            "amendment_parent_contract_sha256": (PILOT_CONTRACT_V2_10_CANONICAL_SHA256),
            "amendment_parent_raw_namespace": ("experiment_results/pilot-v2.10/raw"),
            "amendment_parent_raw_inventory": {
                "schema_version": "finevo-raw-tree-inventory-v1",
                "canonicalization": "json-sort-keys-compact-utf8-v1",
                "file_count": raw_file_count,
                "storage_bytes": raw_storage_bytes,
                "inventory_sha256": raw_inventory_sha256,
            },
            "child_raw_namespace": "experiment_results/pilot-v2.10.1/raw",
            "shared_namespace": False,
            "exact_parent_inventory_required": True,
            "source_artifacts_rewritten": False,
            "parent_terminal_no_go_preserved": True,
            "parent_denominator_reclassified": False,
            "parent_evidence_package_required": True,
            "parent_evidence_namespace": ("evidence/current_v2/pilot-v2.10"),
            "parent_evidence_commit": evidence_commit,
            "parent_evidence_merge_commit": evidence_merge_commit,
            "parent_evidence_rewrite": "forbidden",
        },
        "qref_receipt_verifier_repair": {
            "policy_id": ("finevo-pilot-v2.10.1-schema-dispatched-receipt-hash-v1"),
            "artifact_schema_version": "finevo-pilot-stage-receipt-v2",
            "canonicalization": PILOT_CONTRACT_CANONICALIZATION,
            "integrity_required_fields": [
                "canonicalization",
                "content_sha256",
            ],
            "content_hash_projection": (
                "canonical-json-of-artifact-after-removing-entire-integrity-object"
            ),
            "generic_self_hash_convention_for_stage_receipt_v2": "forbidden",
            "schema_dispatched_hash_verification_required": True,
            "declared_content_hash_exact_match_required": True,
            "external_file_and_content_binding_exact_match_required": True,
            "q_ref_receipt_content_sha256": (
                "e9865c91ec078043489592813f62e72ca4f1d19239cf935a31699637e9f37d57"
            ),
            "stage0_receipt_content_sha256": (
                "fc45635bbac056f9a72f3d8235286aa743592c793382192c156f7fc0c42c45d5"
            ),
            "source_artifact_reseal_or_rewrite": "forbidden",
            "tamper_policy": "stop-before-dispatch",
            "unknown_schema_policy": "stop-before-dispatch",
            "validation_before_provider_construction": True,
        },
        "prerequisite_retry": {
            "expected_complete_cells_before_a_d_dispatch": 16,
            "expected_cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            },
            "v2_10_complete_parent_import_reverification_required": True,
            "q_ref_and_stage0_source_contract_id": PILOT_CONTRACT_ID_V2_9,
            "q_ref_and_stage0_source_contract_sha256": (
                PILOT_CONTRACT_V2_9_CANONICAL_SHA256
            ),
            "exact_file_and_content_hashes_required": True,
            "source_run_manifests_reverification_required": True,
            "q_ref_resolution_reverification_required": True,
            "stage0_selection_reverification_required": True,
            "stage0_selected_profile_id": "nu-0.5",
            "provider_construction_during_import": False,
            "provider_redispatch_for_imported_cells": "forbidden",
            "decoded_completion_reuse_beyond_prerequisites": "forbidden",
            "child_artifacts_resealed_to_current_contract_tag_and_head": True,
            "missing_or_malformed_source_policy": "stop-before-dispatch",
            "prerequisites_are_treatment_effect_evidence": False,
        },
        "fresh_science_dispatch": {
            "a_d_cells": 195,
            "stage_counts": {
                "local-experiment-c": 25,
                "local-experiment-a": 20,
                "local-experiment-d": 35,
                "local-experiment-b": 25,
                "experiment-c": 25,
                "experiment-a": 20,
                "experiment-d": 30,
                "experiment-b": 15,
            },
            "a_d_provider_dispatch": "fresh-only",
            "imported_a_d_completions": 0,
            "decoded_completion_reuse": "forbidden",
            "downstream_dispatch_requires_all_16_prerequisites": True,
            "whole_remaining_matrix_projection_required": True,
        },
        "retry_policy": {
            "new_contract_required": True,
            "new_211_cell_denominator_required": True,
            "preserve_parent_denominator": True,
            "v2_10_raw_resume": "forbidden",
            "v2_10_terminal_cell_reclassification": "forbidden",
            "v2_10_status_counts_rewrite": "forbidden",
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "prerequisite_provider_redispatch": "forbidden",
            "a_d_provider_dispatch": "fresh-only",
        },
        "budget_carry_forward": {
            "total_cap_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "hosted_confirmatory_cap_usd": (_PILOT_V2_4_HOSTED_STAGE_CAP_USD),
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "cumulative_prior": {
                "stage_bucket": "parent_v23",
                "cost_usd": 3.212770875,
                "hosted_completions": 184,
                "parent_contract_sha256": (PILOT_CONTRACT_V2_10_CANONICAL_SHA256),
                "parent_run_ledger_sha256": run_ledger_sha256,
                "parent_budget_ledger_sha256": budget_ledger_sha256,
                "storage_bytes": 70_035_938,
                "record_sha256": (
                    "4837821a5f059714ef8fa6f8b22522bc693c8adb0edc7603367823a870e94510"
                ),
            },
            "v2_10_incremental": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
            },
            "budget_reset": False,
            "debit_before_new_dispatch": True,
            "manual_reserve_automatic_use": False,
            "projection_reserve_multiplier": 1.25,
            "whole_remaining_matrix_projection_required": True,
            "unknown_price_policy": "stop-before-dispatch",
        },
        "science_design_invariance": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_10,
            "source_contract_sha256": PILOT_CONTRACT_V2_10_CANONICAL_SHA256,
            "source_science_design_sha256": (PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256),
            "registered_cells": 211,
            "scientific_cells": 209,
            "fresh_a_d_cells": 195,
            "matrix_profile_id": (
                "local-llama-full-ad-gpt52-fixed-confirmatory-cadb-v1"
            ),
            "seeds_arms_models_stage_order_unchanged": True,
            "provider_profiles_unchanged": True,
            "shock_utility_stop_go_unchanged": True,
            "budget_envelope_unchanged": True,
            "p95_reservation_values_unchanged": True,
        },
        "evidence_lineage": {
            "parent_evidence_namespace": ("evidence/current_v2/pilot-v2.10"),
            "parent_evidence_status": "complete-with-no-go",
            "parent_evidence_commit": evidence_commit,
            "parent_evidence_merge_commit": evidence_merge_commit,
            "parent_evidence_rewrite": "forbidden",
            "parent_claim_reclassification": "forbidden",
            "v2_10_1_effect_aggregation_uses_only_fresh_v2_10_1_a_d_cells": True,
            "q_ref_and_stage0_are_prerequisites_not_effect_evidence": True,
        },
    }


def _validate_v2_10_1_qref_receipt_verifier_retry_amendment(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "qref_receipt_verifier_retry_amendment")
    expected = _v2_10_1_expected_qref_receipt_verifier_retry_amendment(status=status)
    _strict_keys(
        amendment,
        required=set(expected),
        name="qref_receipt_verifier_retry_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError(
            "V2.10.1 q-ref receipt-verifier retry amendment drifted"
        )
    return _freeze_json(amendment)


def _v2_10_2_expected_p95_consumer_adapter_retry_amendment(
    *,
    status: str,
) -> dict[str, Any]:
    """Return the exact outcome-bounded V2.10.2 retry amendment."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.10.2 amendment status is invalid")
    source_file_sha256 = (
        None if status == "draft" else PILOT_V2_10_2_SOURCE_MANIFEST_FILE_SHA256
    )
    source_content_sha256 = (
        None if status == "draft" else PILOT_V2_10_2_SOURCE_MANIFEST_CONTENT_SHA256
    )
    return {
        "schema_version": ("finevo-pilot-p95-consumer-adapter-retry-amendment-v1"),
        "amendment_id": "finevo-pilot-v2.10.2-p95-consumer-adapter-retry-1",
        "source_manifest": {
            "path": "experiments/pilot_v2_10_2_source_manifest.json",
            "schema_version": "finevo-pilot-v2.10.2-source-manifest-v1",
            "file_sha256": source_file_sha256,
            "content_sha256": source_content_sha256,
        },
        "parent_terminal_failure": {
            "contract_id": PILOT_CONTRACT_ID_V2_10_1,
            "contract_sha256": PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256,
            "science_tag": PILOT_CONTRACT_TAG_V2_10_1,
            "science_tag_object": ("2e6137cb5f4c3c8e5dc174efe8813cf04f2490f5"),
            "science_commit": ("b5bfa9b86d3cdb706cea5be707597bef8ac85aed"),
            "raw_namespace": "experiment_results/pilot-v2.10.1/raw",
            "raw_file_count": PILOT_V2_10_1_RAW_FILE_COUNT,
            "raw_storage_bytes": PILOT_V2_10_1_RAW_STORAGE_BYTES,
            "raw_inventory_sha256": PILOT_V2_10_1_RAW_INVENTORY_SHA256,
            "run_ledger_file_sha256": (
                "33e0df2243f29a9f3e9c9994376703641588f7f5b718805e345fd53fa1f49f10"
            ),
            "run_ledger_internal_sha256": (PILOT_V2_10_1_RUN_LEDGER_INTERNAL_SHA256),
            "run_ledger_event_count": 213,
            "run_ledger_event_head": (
                "217dcefc5d1ad7a9a2f222735613332c530ac27623931b3e6faa298160ead995"
            ),
            "budget_ledger_file_sha256": (
                "692367a14d1d25cc80766442b04c19add0a9c4ef5153cd663c1752d738b0960c"
            ),
            "budget_ledger_internal_sha256": (
                PILOT_V2_10_1_BUDGET_LEDGER_INTERNAL_SHA256
            ),
            "budget_ledger_event_count": 314,
            "budget_ledger_event_head": (
                "ec7e88766e84c1c173e094b1569d8d409b3639a67e886ed3d37c2e17026f0a19"
            ),
            "status_counts": {"complete": 26, "failed": 185},
            "root_cause_code": "observed-p95-consumer-schema-dispatch-gap",
            "failure_error_type": "ValueError",
            "failure_message": (
                "source-backed observed p95 receipt verification failed: "
                "observed-p95 receipt top-level shape or schema drifted"
            ),
            "failure_message_sha256": (
                "39cb7f19f94e435d9eb4873df49beac2507703522f2ad9ffa7f688a5f6b92ef7"
            ),
            "failure_phase": "before-provider-construction-and-dispatch",
            "fresh_provider_calls": 0,
            "incremental_hosted_completions": 0,
            "incremental_hosted_cost_usd": 0.0,
            "offline_candidate_admission_cells_generated": 10,
            "offline_candidate_metrics_observed": True,
            "offline_candidate_metrics_inspected": True,
            "actor_performance_treatment_outcome_blind": True,
            "global_a_d_outcome_blind": False,
        },
        "parent_evidence": {
            "namespace": "evidence/current_v2/pilot-v2.10.1",
            "publication_commit": PILOT_V2_10_1_EVIDENCE_COMMIT,
            "merge_commit": PILOT_V2_10_1_EVIDENCE_MERGE_COMMIT,
            "checksums_file_sha256": (PILOT_V2_10_1_EVIDENCE_CHECKSUMS_FILE_SHA256),
            "package_manifest_file_sha256": (
                PILOT_V2_10_1_EVIDENCE_PACKAGE_FILE_SHA256
            ),
            "aggregate_file_sha256": (
                "d2fe85e42d6f4b7a318a3450b51e38726da7519d6babe674a5dfd78713992232"
            ),
            "failure_ledger_file_sha256": (
                "93c208129a203196628b0bf47f6662f52fd244902de19d72041c7bcbcc2c40e8"
            ),
            "reviewer_report_file_sha256": (
                "53e448bff6e958f14b96e27ec09ef7e17a286ad44ae615f5f570ffbce8cca5cb"
            ),
            "publication_status": "complete-with-no-go",
            "scientific_claim_gates_supported": False,
            "rewrite": "forbidden",
            "reclassification": "forbidden",
        },
        "consumer_adapter_repair": {
            "producer_schema_version": (
                "finevo-pilot-v2.10.2-resealed-observed-p95-authority-v1"
            ),
            "generic_consumer_registry_required": True,
            "dedicated_receipt_and_projection_verifier_required": True,
            "exact_current_release_path_required": True,
            "mapping_only_current_release_input": "reject",
            "unknown_schema_policy": "stop-before-provider-construction",
            "sibling_projection_tamper_policy": ("stop-before-provider-construction"),
            "receipt_replacement_race_policy": ("guarded-read-binding-mismatch-stop"),
            "provider_construction_before_verification": False,
            "validation_before_provider_construction": True,
        },
        "prerequisite_import": {
            "source": (
                "byte-exact-v2.9-prerequisites-nested-inside-immutable-"
                "v2.10.1-release"
            ),
            "expected_complete_cells": 16,
            "cell_breakdown": {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            },
            "provider_construction_during_import": False,
            "provider_redispatch": "forbidden",
            "prerequisites_are_treatment_effect_evidence": False,
            "stage0_selected_profile_id": "nu-0.5",
        },
        "fresh_science_dispatch": {
            "registered_cells": 211,
            "prerequisite_cells": 16,
            "a_d_cells": 195,
            "provider_backed_a_d_cells": 185,
            "offline_candidate_admission_cells": 10,
            "offline_candidate_stage_counts": {
                "experiment-c": 5,
                "local-experiment-c": 5,
            },
            "fresh_provider_dispatch_for_provider_backed_cells": "required",
            "offline_candidate_provider_dispatch": "forbidden",
            "v2_10_1_a_d_cell_reuse": "forbidden",
            "v2_10_1_offline_candidate_cell_reuse": "forbidden",
            "decoded_completion_reuse": "forbidden",
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "stage_counts": {
                "experiment-a": 20,
                "experiment-b": 15,
                "experiment-c": 25,
                "experiment-d": 30,
                "local-experiment-a": 20,
                "local-experiment-b": 25,
                "local-experiment-c": 25,
                "local-experiment-d": 35,
            },
        },
        "budget_carry_forward": {
            "budget_reset": False,
            "cumulative_prior": {
                "cost_usd": 3.212770875,
                "hosted_completions": 184,
                "storage_bytes": 92_541_342,
                "parent_contract_sha256": (PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256),
                "parent_run_ledger_sha256": (PILOT_V2_10_1_RUN_LEDGER_INTERNAL_SHA256),
                "parent_budget_ledger_sha256": (
                    PILOT_V2_10_1_BUDGET_LEDGER_INTERNAL_SHA256
                ),
                "stage_bucket": "parent_v23",
            },
            "debit_before_new_dispatch": True,
            "total_cap_usd": 500.0,
            "hosted_confirmatory_cap_usd": 495.787229125,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "manual_reserve_usd": 1.0,
            "manual_reserve_automatic_use": False,
            "projection_reserve_multiplier": 1.25,
            "whole_remaining_matrix_projection_required": True,
            "unknown_price_policy": "stop-before-dispatch",
        },
        "observation_boundary": {
            "v2_10_1_offline_candidate_metrics_inspected": True,
            "globally_a_d_outcome_blind": False,
            "actor_performance_treatment_outcome_blind": True,
            "scientific_design_selected_before_actor_outcomes": True,
            "offline_candidate_cells_must_be_freshly_rerun": 10,
            "parent_failure_is_model_capability_evidence": False,
            "parent_failure_is_a_d_treatment_effect_evidence": False,
        },
        "science_design_invariance": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_10_1,
            "source_contract_sha256": (PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256),
            "source_science_design_sha256": (PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256),
            "registered_cells": 211,
            "fresh_a_d_cells": 195,
            "seeds_arms_models_stage_order_unchanged": True,
            "shock_utility_stop_go_unchanged": True,
            "provider_profiles_unchanged": True,
            "p95_numeric_values_unchanged": True,
            "budget_envelope_unchanged": True,
        },
    }


def _validate_v2_10_2_p95_consumer_adapter_retry_amendment(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "p95_consumer_adapter_retry_amendment")
    expected = _v2_10_2_expected_p95_consumer_adapter_retry_amendment(
        status=status,
    )
    _strict_keys(
        amendment,
        required=set(expected),
        name="p95_consumer_adapter_retry_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.10.2 p95 consumer-adapter retry amendment drifted")
    return _freeze_json(amendment)


def _v2_11_expected_forward_boundary() -> dict[str, Any]:
    """Exact prospective boundary for the fresh hosted-model denominator."""

    return {
        "schema_version": ("finevo-pilot-v2.11-hosted-model-boundary-v1"),
        "source_manifest": {
            "path": "experiments/pilot_v2_11_source_manifest.json",
            "schema_version": "finevo-pilot-v2.11-source-manifest-v1",
            "file_sha256": (
                "950f115959ace359984d99285aec60ba794162e666d7ea7c36ec56d6f3d76c1d"
            ),
            "content_sha256": (
                "cea1a1b134f89b98ff515b4deead492f738a9df1b9aa8f2c9b891dce5afab48f"
            ),
        },
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_10_2,
            "contract_sha256": (
                "b8de8cfb2560d894dad65d68df8ae9126527d12d3807bef045fa52f5e9d4159e"
            ),
            "science_commit": "2dcc20f8dccc7a6a94a60a00d7f3750a9d61396d",
            "science_tag": PILOT_CONTRACT_TAG_V2_10_2,
            "run_ledger_internal_sha256": (
                "2219a832b9a7dfe235b32db882e126bddc36938f4f201a2ab84ddea6878bb809"
            ),
            "budget_ledger_internal_sha256": (
                "73b1bac2a424147cbfa88bdb4e351d6c924b6e82847050f7cb1d254fe1ea4068"
            ),
            "q_ref_content_sha256": (
                "50d75c846c5e9d2b58fb92faf674da8a06ebb3b0ba7f21a6b1b2ad689034c40c"
            ),
            "stage0_selection_content_sha256": (
                "68c810055fc38683d3a8a7d597c54ffed4fb2c6332c2c02e1964b3ebfb61743c"
            ),
        },
        "calibration_allowlist": {
            "q_ref": 63.50397933257746,
            "utility_profile_id": "nu-0.5",
            "utility_profile": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale_multiplier_of_q_ref": 1.0,
                "discount_factor": 0.99,
            },
            "absolute_flow_utility_threshold": 0.05617208967516696,
        },
        "parent_budget_debit": {
            "schema_version": "finevo-parent-budget-debit-v1",
            "parent_contract_sha256": (
                "b8de8cfb2560d894dad65d68df8ae9126527d12d3807bef045fa52f5e9d4159e"
            ),
            "parent_run_ledger_sha256": (
                "2219a832b9a7dfe235b32db882e126bddc36938f4f201a2ab84ddea6878bb809"
            ),
            "parent_budget_ledger_sha256": (
                "73b1bac2a424147cbfa88bdb4e351d6c924b6e82847050f7cb1d254fe1ea4068"
            ),
            "stage_bucket": "parent_v2102",
            "cost_usd": 16.044922812500005,
            "hosted_completions": 816,
            "storage_bytes": 217010835,
            "record_sha256": (
                "c841dc4cbdfdb548c6917fbb2670c31ba3759f3d4f52ffb0fbb5b9d8bcbbc74d"
            ),
        },
        "import_policy": {
            "imported_effect_cells": 0,
            "effect_metrics_observed": False,
            "effect_artifact_paths": [],
            "imported_p95_authorities": [],
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
        },
        "interface": {
            "max_completion_tokens_by_role": {
                task_id: limits[0]
                for task_id, limits in _SCIENCE_TASK_CAPS_V2_11.items()
            },
            "prompt_token_tier_ceiling": 200000,
            "prompt_token_upper_bound_method": "utf8-bytes-plus-256-v1",
            "prompt_token_ceiling_comparison": (
                "reject-upper-bound-greater-than-or-equal-to-ceiling"
            ),
            "reserve_multiplier": 1.25,
            "minimum_action_samples_per_model": 48,
            "minimum_semantic_samples_per_model": 14,
            "minimum_raw_p95_headroom_fraction": 0.25,
            "adaptive_cap_increase": "forbidden-new-amendment-required",
        },
        "matrix": {
            "ledger_cells": 136,
            "operational_cells": 5,
            "scientific_cells": 131,
            "hosted_provider_calls": 5940,
            "action_calls": 4944,
            "semantic_calls": 996,
            "stage_ledger_cells": {
                "parent-import": 1,
                "capability-gate": 2,
                "long-context-preflight": 2,
                "experiment-c": 25,
                "experiment-a": 20,
                "experiment-d": 55,
                "experiment-b": 25,
                "cross-model": 6,
            },
            "matrix_shrink_after_preflight": False,
        },
        "paid_provenance": {
            "draft_dispatch_allowed": False,
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11,
            "fresh_capability_and_preflight_required": True,
            "historical_effect_reclassification": False,
        },
    }


def _validate_v2_11_forward_boundary(value: Any) -> Mapping[str, Any]:
    boundary = _mapping(value, "v211_forward_boundary")
    expected = _v2_11_expected_forward_boundary()
    _strict_keys(
        boundary,
        required=set(expected),
        name="v211_forward_boundary",
    )
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11 prospective forward boundary drifted")
    return _freeze_json(boundary)


def _v2_11_1_expected_forward_boundary() -> dict[str, Any]:
    """Exact retry boundary after V2.11's zero-dispatch preflight no-go."""

    return {
        "schema_version": "finevo-pilot-v2.11.1-forward-boundary-v1",
        "source_manifest": {
            "path": "experiments/pilot_v2_11_1_source_manifest.json",
            "schema_version": ("finevo-pilot-v2.11.1-parent-source-manifest-v1"),
            "file_sha256": PILOT_V2_11_1_SOURCE_MANIFEST_FILE_SHA256,
            "content_sha256": PILOT_V2_11_1_SOURCE_MANIFEST_CONTENT_SHA256,
        },
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_11,
            "contract_sha256": PILOT_CONTRACT_V2_11_CANONICAL_SHA256,
            "science_commit": "5d6c7920bd4a872b02931fdee8a47b9ac4e7b352",
            "science_tag": PILOT_CONTRACT_TAG_V2_11,
            "science_tag_object": ("c4b457d0cc8e7e48f99c64f0283ab043877cc47f"),
            "run_ledger_file_sha256": (
                "b0a4a0af97ec3fbee3247ceb51b5c7e0241c8d02d7ac0fba55031654bf0b8dbb"
            ),
            "run_ledger_internal_sha256": (
                "d50d89535b0896f46f4ded93d9ca28062558a75a7fb8b9548f989d77233f20a1"
            ),
            "budget_ledger_file_sha256": (
                "842fba225918c472210042597925847566d0639a5d67331cca8ad8bb2c1cb366"
            ),
            "budget_ledger_internal_sha256": (
                "be72be029f0e558153f0a81545ffe13347833031bb9b5611bcd929dc0b0408d8"
            ),
            "terminal_status_counts": {
                "complete": 3,
                "failed": 2,
                "integrity-stopped": 131,
            },
            "failed_preflight_provider_calls": 0,
        },
        "calibration_allowlist": {
            "q_ref": 63.50397933257746,
            "utility_profile_id": "nu-0.5",
            "utility_profile": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale_multiplier_of_q_ref": 1.0,
                "discount_factor": 0.99,
            },
            "absolute_flow_utility_threshold": 0.05617208967516696,
        },
        "parent_budget_debit": {
            "schema_version": "finevo-parent-budget-debit-v1",
            "parent_contract_sha256": PILOT_CONTRACT_V2_11_CANONICAL_SHA256,
            "parent_run_ledger_sha256": (
                "d50d89535b0896f46f4ded93d9ca28062558a75a7fb8b9548f989d77233f20a1"
            ),
            "parent_budget_ledger_sha256": (
                "be72be029f0e558153f0a81545ffe13347833031bb9b5611bcd929dc0b0408d8"
            ),
            "stage_bucket": "parent_v211",
            "cost_usd": 17.166524062500006,
            "hosted_completions": 876,
            "storage_bytes": 217581135,
            "record_sha256": (
                "e5b8406c636d5045040677ca0bd09dd72557afdef2998095f0f5775a0ead8b9c"
            ),
        },
        "import_policy": {
            "imported_operational_capability_cells": 2,
            "historical_capability_calls": 60,
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "imported_effect_cells": 0,
            "effect_metrics_observed": False,
            "effect_artifact_paths": [],
            "imported_p95_authorities": [],
            "failed_preflight_retried_in_parent_namespace": False,
        },
        "interface": {
            "max_completion_tokens_by_role": {
                task_id: limits[0]
                for task_id, limits in _SCIENCE_TASK_CAPS_V2_11.items()
            },
            "prompt_token_tier_ceiling": 200000,
            "prompt_token_upper_bound_method": "utf8-bytes-plus-256-v1",
            "prompt_token_ceiling_comparison": (
                "reject-upper-bound-greater-than-or-equal-to-ceiling"
            ),
            "normal_science_reservation_source": (
                "sealed-long-context-preflight-observed-p95-plus-25pct"
            ),
            "adaptive_cap_increase": "forbidden-new-amendment-required",
        },
        "matrix": {
            "ledger_cells": 136,
            "operational_cells": 5,
            "scientific_cells": 131,
            "fresh_hosted_provider_calls": 5880,
            "fresh_action_calls": 4896,
            "fresh_semantic_calls": 984,
            "historical_imported_capability_calls": 60,
            "stage_ledger_cells": {
                "parent-import": 1,
                "capability-gate": 2,
                "long-context-preflight": 2,
                "experiment-c": 25,
                "experiment-a": 20,
                "experiment-d": 55,
                "experiment-b": 25,
                "cross-model": 6,
            },
            "matrix_shrink_after_preflight": False,
        },
        "paid_provenance": {
            "draft_dispatch_allowed": False,
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11_1,
            "capability_redispatch_allowed": False,
            "fresh_long_context_preflight_required": True,
            "historical_effect_reclassification": False,
        },
    }


def _validate_v2_11_1_forward_boundary(value: Any) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2111_forward_boundary")
    expected = _v2_11_1_expected_forward_boundary()
    _strict_keys(
        boundary,
        required=set(expected),
        name="v2111_forward_boundary",
    )
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.1 forward boundary drifted")
    if (
        PILOT_V2_11_1_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_1_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.1 source manifest hashes must be sealed before rendering"
        )
    return _freeze_json(boundary)


def _v2_11_1_expected_preflight_bootstrap_amendment() -> dict[str, Any]:
    return {
        "schema_version": ("finevo-pilot-v2.11.1-contract-envelope-amendment-v1"),
        "amendment_id": ("finevo-pilot-v2.11.1-long-context-bootstrap-1"),
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_11,
            "contract_sha256": PILOT_CONTRACT_V2_11_CANONICAL_SHA256,
            "tag": PILOT_CONTRACT_TAG_V2_11,
            "commit": "5d6c7920bd4a872b02931fdee8a47b9ac4e7b352",
            "source_manifest_file_sha256": (PILOT_V2_11_1_SOURCE_MANIFEST_FILE_SHA256),
            "source_manifest_content_sha256": (
                PILOT_V2_11_1_SOURCE_MANIFEST_CONTENT_SHA256
            ),
            "run_ledger_file_sha256": (
                "b0a4a0af97ec3fbee3247ceb51b5c7e0241c8d02d7ac0fba55031654bf0b8dbb"
            ),
            "run_ledger_internal_sha256": (
                "d50d89535b0896f46f4ded93d9ca28062558a75a7fb8b9548f989d77233f20a1"
            ),
            "capability_stage_receipt_file_sha256": (
                "6aebcaba39f9f18c35c16dff307f73118c7dac03272e9850d41ebced0ddeb8d9"
            ),
            "capability_stage_receipt_content_sha256": (
                "eaf00790a95bc497e02e44cd2dc2958301ef43c8d835e4d04f09d0fa3d4cf0cc"
            ),
            "capability_taskset_sha256": (
                "633a57690a1a61e8300ca3c1fb506084959d2f47219bc85b087dfd00732091c5"
            ),
        },
        "retry_policy": {
            "allowed_stage": "long-context-preflight",
            "allowed_models": ["gpt52_main", "gpt56_diagnostic"],
            "zero_dispatch_only": True,
            "no_capability_redispatch": True,
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.1/raw",
        },
        "bootstrap_policy": {
            "policy_id": ("finevo-pilot-v2.11.1-long-context-contract-envelope-1"),
            "allowed_execution_mode": "closed_loop_preflight",
            "target_shape": {
                "num_agents": 2,
                "episode_length": 12,
                "action_calls": 24,
                "semantic_calls": 8,
            },
            "source": {
                "contract_id": PILOT_CONTRACT_ID_V2_11,
                "schema_version": "finevo-capability-gate-v5",
                "same_model_required": True,
                "required_sample_counts": {
                    "action": 24,
                    "semantic": 6,
                },
            },
            "capability_audit": {
                "p95_method": ("nearest-rank-with-observed-maximum-floor"),
                "reserve_multiplier": 1.25,
                "dispatch_reservation": False,
            },
            "effective_contract_envelope": {
                "prompt_tokens_per_call": 200000,
                "completion_tokens_per_call": 4096,
                "cached_input_discount_assumed": False,
                "price_basis": ("frozen-provider-profile-dispatch-endpoint"),
            },
            "missing_or_malformed_source_policy": "stop-before-dispatch",
            "scientific_evidence": False,
            "normal_scientific_dispatch_reservation_source": (
                "sealed-long-context-preflight-observed-p95-only"
            ),
        },
    }


def _validate_v2_11_1_preflight_bootstrap_amendment(
    value: Any,
) -> Mapping[str, Any]:
    amendment = _mapping(
        value,
        "v2111_preflight_bootstrap_amendment",
    )
    expected = _v2_11_1_expected_preflight_bootstrap_amendment()
    _strict_keys(
        amendment,
        required=set(expected),
        name="v2111_preflight_bootstrap_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError(
            "V2.11.1 contract-envelope bootstrap amendment drifted"
        )
    return _freeze_json(amendment)


def _v2_11_2_expected_forward_boundary() -> dict[str, Any]:
    """Exact fresh-release boundary after V2.11.1's paid preflight no-go."""

    return {
        "schema_version": "finevo-pilot-v2.11.2-forward-boundary-v1",
        "source_manifest": {
            "path": "experiments/pilot_v2_11_2_source_manifest.json",
            "schema_version": ("finevo-pilot-v2.11.2-parent-source-manifest-v1"),
            "file_sha256": PILOT_V2_11_2_SOURCE_MANIFEST_FILE_SHA256,
            "content_sha256": PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256,
        },
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_1,
            "contract_sha256": PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256,
            "science_commit": "e9871353ad307fdd134f3c74764d201efbc81081",
            "science_tag": PILOT_CONTRACT_TAG_V2_11_1,
            "science_tag_object": ("c12f6bd5b74cb676109b83fcbfdb4376adf7abdf"),
            "run_ledger_file_sha256": (
                "ddd35892843acb4e770c9572d24e95537cb9dee83c044abc321a580f122f97b9"
            ),
            "run_ledger_internal_sha256": (
                "ed9c0210791627128dc0e9942df2cd46269acbbd28a6c52af33454172a4b76c9"
            ),
            "budget_ledger_file_sha256": (
                "d487edda8555e9c66b1d11fdc938daac04c5a62e02afc45b9c3c18fb5947013c"
            ),
            "budget_ledger_internal_sha256": (
                "df9ccffbba39ac6d86375be433c4a34f4e3b51f60a4e0e10d42d31b7c2886330"
            ),
            "preflight_stage_receipt_file_sha256": (
                "88463a813129b93073403b84f1b239869a72b5395290218404e5c75c3167455f"
            ),
            "preflight_stage_receipt_content_sha256": (
                "b729d7b7ba702cd4dd088f7188e26becd3994c5ce589112572f044b3be86a97d"
            ),
            "terminal_status_counts": {
                "complete": 3,
                "failed": 2,
                "integrity-stopped": 131,
            },
            "failed_preflight_provider_calls": 64,
            "post_gate_authority_created": False,
            "checkpoint_artifacts_created": False,
        },
        "calibration_allowlist": {
            "q_ref": 63.50397933257746,
            "utility_profile_id": "nu-0.5",
            "utility_profile": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale_multiplier_of_q_ref": 1.0,
                "discount_factor": 0.99,
            },
            "absolute_flow_utility_threshold": 0.05617208967516696,
        },
        "parent_budget_debit": {
            "schema_version": "finevo-parent-budget-debit-v1",
            "parent_contract_sha256": (PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256),
            "parent_run_ledger_sha256": (
                "ed9c0210791627128dc0e9942df2cd46269acbbd28a6c52af33454172a4b76c9"
            ),
            "parent_budget_ledger_sha256": (
                "df9ccffbba39ac6d86375be433c4a34f4e3b51f60a4e0e10d42d31b7c2886330"
            ),
            "stage_bucket": "parent_v2111",
            "cost_usd": 18.586399812500005,
            "hosted_completions": 940,
            "storage_bytes": 217838625,
            "record_sha256": (
                "678fc5b795e66f1aa358ea7941bebb9167097158f2aed8cee4044567109c5582"
            ),
        },
        "import_policy": {
            "imported_operational_capability_cells": 2,
            "historical_capability_calls": 60,
            "historical_failed_preflight_calls_for_audit": 64,
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "imported_effect_cells": 0,
            "effect_metrics_observed": False,
            "effect_artifact_paths": [],
            "imported_preflight_samples": 0,
            "imported_checkpoint_artifacts": [],
            "imported_p95_authorities": [],
            "failed_preflight_retried_in_parent_namespace": False,
        },
        "interface": {
            "max_completion_tokens_by_role": {
                task_id: limits[0]
                for task_id, limits in _SCIENCE_TASK_CAPS_V2_11.items()
            },
            "prompt_token_tier_ceiling": 200000,
            "prompt_token_upper_bound_method": "utf8-bytes-plus-256-v1",
            "prompt_token_ceiling_comparison": (
                "reject-upper-bound-greater-than-or-equal-to-ceiling"
            ),
            "normal_science_reservation_source": (
                "sealed-v2112-long-context-preflight-observed-p95-plus-25pct"
            ),
            "adaptive_cap_increase": "forbidden-new-amendment-required",
        },
        "matrix": {
            "ledger_cells": 136,
            "operational_cells": 5,
            "scientific_cells": 131,
            "fresh_hosted_provider_calls": 5880,
            "fresh_action_calls": 4896,
            "fresh_semantic_calls": 984,
            "historical_imported_capability_calls": 60,
            "historical_failed_preflight_calls_for_audit": 64,
            "stage_ledger_cells": {
                "parent-import": 1,
                "capability-gate": 2,
                "long-context-preflight": 2,
                "experiment-c": 25,
                "experiment-a": 20,
                "experiment-d": 55,
                "experiment-b": 25,
                "cross-model": 6,
            },
            "matrix_shrink_after_preflight": False,
        },
        "paid_provenance": {
            "draft_dispatch_allowed": False,
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11_2,
            "capability_redispatch_allowed": False,
            "fresh_long_context_preflight_required": True,
            "historical_preflight_reclassification": False,
            "historical_effect_reclassification": False,
        },
    }


def _validate_v2_11_2_forward_boundary(value: Any) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2112_forward_boundary")
    expected = _v2_11_2_expected_forward_boundary()
    _strict_keys(
        boundary,
        required=set(expected),
        name="v2112_forward_boundary",
    )
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.2 forward boundary drifted")
    if (
        PILOT_V2_11_2_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.2 source manifest hashes must be sealed before rendering"
        )
    return _freeze_json(boundary)


def _v2_11_2_expected_recovery_amendment() -> dict[str, Any]:
    """Only the validator defect and fresh preflight release may change."""

    return {
        "schema_version": "finevo-pilot-v2.11.2-recovery-amendment-v1",
        "amendment_id": "finevo-pilot-v2.11.2-lifecycle-preflight-recovery-1",
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_1,
            "contract_sha256": PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256,
            "tag": PILOT_CONTRACT_TAG_V2_11_1,
            "commit": "e9871353ad307fdd134f3c74764d201efbc81081",
            "source_manifest_file_sha256": (PILOT_V2_11_2_SOURCE_MANIFEST_FILE_SHA256),
            "source_manifest_content_sha256": (
                PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256
            ),
            "run_ledger_file_sha256": (
                "ddd35892843acb4e770c9572d24e95537cb9dee83c044abc321a580f122f97b9"
            ),
            "run_ledger_internal_sha256": (
                "ed9c0210791627128dc0e9942df2cd46269acbbd28a6c52af33454172a4b76c9"
            ),
            "preflight_stage_receipt_file_sha256": (
                "88463a813129b93073403b84f1b239869a72b5395290218404e5c75c3167455f"
            ),
            "preflight_stage_receipt_content_sha256": (
                "b729d7b7ba702cd4dd088f7188e26becd3994c5ce589112572f044b3be86a97d"
            ),
        },
        "retry_policy": {
            "allowed_stage": "long-context-preflight",
            "allowed_models": ["gpt52_main", "gpt56_diagnostic"],
            "zero_dispatch_only": True,
            "no_capability_redispatch": True,
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.2/raw",
            "old_failed_journal_use": "audit-only",
        },
        "lifecycle_validator_repair": {
            "policy_id": "finevo-active-rule-hysteresis-validator-v1",
            "changed_condition": (
                "active-current-score-no-longer-reapplies-admission-thresholds"
            ),
            "unchanged_historical_proofs": [
                "activation-threshold-crossing-event-replay",
                "unique-rule-activated-event",
                "post-proposal-activation-episode",
            ],
            "unchanged_retirement_policy": True,
            "prompt_seed_arm_model_environment_metric_changes": False,
        },
        "bootstrap_policy": {
            "policy_id": ("finevo-pilot-v2.11.2-long-context-contract-envelope-1"),
            "allowed_execution_mode": "closed_loop_preflight",
            "target_shape": {
                "num_agents": 2,
                "episode_length": 12,
                "action_calls": 24,
                "semantic_calls": 8,
            },
            "source": {
                "contract_id": PILOT_CONTRACT_ID_V2_11_1,
                "schema_version": (
                    "finevo-pilot-v2.11.1-imported-capability-wrapper-v1"
                ),
                "same_model_required": True,
                "required_sample_counts": {
                    "action": 24,
                    "semantic": 6,
                },
            },
            "capability_audit": {
                "p95_method": "nearest-rank-with-observed-maximum-floor",
                "reserve_multiplier": 1.25,
                "dispatch_reservation": False,
            },
            "effective_contract_envelope": {
                "prompt_tokens_per_call": 200000,
                "completion_tokens_per_call": 4096,
                "cached_input_discount_assumed": False,
                "price_basis": "frozen-provider-profile-dispatch-endpoint",
            },
            "missing_or_malformed_source_policy": "stop-before-dispatch",
            "scientific_evidence": False,
            "normal_scientific_dispatch_reservation_source": (
                "sealed-v2112-long-context-preflight-observed-p95-only"
            ),
        },
    }


def _validate_v2_11_2_recovery_amendment(
    value: Any,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "v2112_recovery_amendment")
    expected = _v2_11_2_expected_recovery_amendment()
    _strict_keys(
        amendment,
        required=set(expected),
        name="v2112_recovery_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.11.2 recovery amendment drifted")
    return _freeze_json(amendment)


def _v2_11_3_expected_forward_boundary() -> dict[str, Any]:
    """Exact fresh denominator after V2.11.2's consumer-adapter no-go."""

    return {
        "schema_version": "finevo-pilot-v2.11.3-forward-boundary-v1",
        "source_manifest": {
            "path": "experiments/pilot_v2_11_3_source_manifest.json",
            "schema_version": "finevo-pilot-v2.11.3-source-manifest-v1",
            "file_sha256": PILOT_V2_11_3_SOURCE_MANIFEST_FILE_SHA256,
            "content_sha256": PILOT_V2_11_3_SOURCE_MANIFEST_CONTENT_SHA256,
        },
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "contract_sha256": PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
            "contract_file_sha256": (
                "130c890f7e6d5d61137b4aa189cfbcca39b8cd7aab455cc2a6f35aeddd8ee3a8"
            ),
            "science_commit": "78870956b528946d415a9be5f5769b0893d16d74",
            "science_tag": PILOT_CONTRACT_TAG_V2_11_2,
            "science_tag_object": ("1b9d9f163934e946255ec19aeebe2f121fba4cc3"),
            "source_manifest_file_sha256": (PILOT_V2_11_2_SOURCE_MANIFEST_FILE_SHA256),
            "source_manifest_content_sha256": (
                PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256
            ),
            "run_ledger_file_sha256": (
                "d8ec96c5aba434368b3d97a1bcc0d6a0116c62325161eae6ce13c9c581198625"
            ),
            "run_ledger_internal_sha256": (
                "686d7f528268e0d9d6ac97ae27d483af9c2eb93be53bd329b4fd621c0ec2ae25"
            ),
            "budget_ledger_file_sha256": (
                "0461fcbb6d4ca01677aa55410c28296e7aba6f2deff8cd044c459981e39c3ecf"
            ),
            "budget_ledger_internal_sha256": (
                "36dd9c62a56c7e87bb647feebeaa7f8d03b0a410d3c7d163834d5029f8da868b"
            ),
            "preflight_stage_receipt_file_sha256": (
                "17c8838edb1c5497311b5c4adba3e7715cd6e601811a33c3cdd59a97d2b359e1"
            ),
            "preflight_stage_receipt_content_sha256": (
                "1d06ad536faa0ad62f2ffec2e100d80f8463acd38c75606ac8b89a24e1568659"
            ),
            "terminal_status_counts": {"complete": 10, "failed": 126},
            "failed_scientific_cells": 126,
            "completed_offline_scientific_cells": 5,
            "scientific_provider_calls": 0,
            "fresh_preflight_provider_calls": 64,
            "failure_signature": (
                "source-backed observed p95 receipt verification failed: "
                "observed-p95 receipt top-level shape or schema drifted"
            ),
        },
        "calibration_allowlist": {
            "q_ref": 63.50397933257746,
            "utility_profile_id": "nu-0.5",
            "utility_profile": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale_multiplier_of_q_ref": 1.0,
                "discount_factor": 0.99,
            },
            "absolute_flow_utility_threshold": 0.05617208967516696,
        },
        "parent_budget_debit": {
            "schema_version": "finevo-parent-budget-debit-v1",
            "parent_contract_sha256": (PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256),
            "parent_run_ledger_sha256": (
                "686d7f528268e0d9d6ac97ae27d483af9c2eb93be53bd329b4fd621c0ec2ae25"
            ),
            "parent_budget_ledger_sha256": (
                "36dd9c62a56c7e87bb647feebeaa7f8d03b0a410d3c7d163834d5029f8da868b"
            ),
            "stage_bucket": "parent_v2112",
            "cost_usd": 19.998220562500006,
            "hosted_completions": 1004,
            "storage_bytes": 221668707,
            "record_sha256": (
                "3ddc22970ff30d1ad9fc3b9efbffe5e4de1f641851bc9e3398aa2fd0977154a1"
            ),
        },
        "import_policy": {
            "imported_parent_budget_cells": 1,
            "imported_capability_authority_cells": 2,
            "imported_preflight_authority_cells": 2,
            "historical_capability_calls": 60,
            "historical_preflight_calls": 64,
            "historical_action_samples_per_model": 24,
            "historical_semantic_samples_per_model": 8,
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "imported_effect_cells": 0,
            "preflight_authority_scientific_evidence": False,
            "effect_artifact_paths": [],
            "v2112_scientific_cells_reused": 0,
        },
        "matrix": {
            "ledger_cells": 136,
            "operational_cells": 5,
            "scientific_cells": 131,
            "provider_backed_scientific_cells": 126,
            "offline_scientific_cells": 5,
            "fresh_scientific_provider_calls": 5816,
            "fresh_action_calls": 4848,
            "fresh_semantic_calls": 968,
            "fresh_calls_by_model": {
                "gpt52_main": {"action": 4560, "semantic": 920},
                "gpt56_diagnostic": {"action": 288, "semantic": 48},
            },
            "stage_ledger_cells": {
                "parent-import": 1,
                "capability-gate": 2,
                "long-context-preflight": 2,
                "experiment-c": 25,
                "experiment-a": 20,
                "experiment-d": 55,
                "experiment-b": 25,
                "cross-model": 6,
            },
            "matrix_shrink_after_release": False,
        },
        "paid_provenance": {
            "draft_dispatch_allowed": False,
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11_3,
            "capability_redispatch_allowed": False,
            "preflight_redispatch_allowed": False,
            "v2112_scientific_cell_reuse_allowed": False,
            "historical_effect_reclassification": False,
        },
    }


def _validate_v2_11_3_forward_boundary(value: Any) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2113_forward_boundary")
    expected = _v2_11_3_expected_forward_boundary()
    _strict_keys(
        boundary,
        required=set(expected),
        name="v2113_forward_boundary",
    )
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.3 forward boundary drifted")
    return _freeze_json(boundary)


def _v2_11_3_expected_consumer_adapter_amendment() -> dict[str, Any]:
    """Outcome-limited repair for the V2.11.2 P95 consumer dispatch gap."""

    return {
        "schema_version": ("finevo-pilot-v2.11.3-p95-consumer-adapter-amendment-v1"),
        "amendment_id": ("finevo-pilot-v2.11.3-p95-consumer-adapter-recovery-1"),
        "parent_terminal_receipt": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "contract_sha256": PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
            "tag": PILOT_CONTRACT_TAG_V2_11_2,
            "tag_object": "1b9d9f163934e946255ec19aeebe2f121fba4cc3",
            "commit": "78870956b528946d415a9be5f5769b0893d16d74",
            "run_ledger_internal_sha256": (
                "686d7f528268e0d9d6ac97ae27d483af9c2eb93be53bd329b4fd621c0ec2ae25"
            ),
            "budget_ledger_internal_sha256": (
                "36dd9c62a56c7e87bb647feebeaa7f8d03b0a410d3c7d163834d5029f8da868b"
            ),
            "terminal_cells": 136,
            "complete_cells": 10,
            "failed_cells": 126,
            "scientific_provider_calls": 0,
            "preflight_provider_calls": 64,
            "failure_signature": (
                "source-backed observed p95 receipt verification failed: "
                "observed-p95 receipt top-level shape or schema drifted"
            ),
        },
        "observation_boundary": {
            "actor_performance_outcomes_observed": False,
            "global_a_to_d_outcome_blind": False,
            "inspected_offline_candidate_cells": 5,
            "inspected_offline_candidate_metrics": [
                "verified_false_activation",
                "unverified_false_activation",
            ],
            "amendment_outcome_blind_for_actor_performance": True,
            "scientific_design_change": False,
        },
        "root_cause": {
            "root_cause_id": "observed-p95-consumer-schema-dispatch-gap",
            "producer_schema_version": ("finevo-pilot-v2.11.2-post-gate-authority-v1"),
            "previous_consumer_path": "generic-observed-p95-v1-shape-validator",
            "failure_point": "before-provider-construction",
            "provider_or_model_failure": False,
            "method_failure": False,
        },
        "consumer_adapter_repair": {
            "registry": "DEDICATED_OBSERVED_P95_BINDING_SCHEMA_REGISTRY",
            "registry_dispatch_key": ("finevo-pilot-v2.11.2-post-gate-authority-v1"),
            "registry_adapter_id": "v2.11.2-post-gate-authority",
            "generic_mapping_only_acceptance_for_current_schema": False,
            "dedicated_verifier": (
                "verified_memory.pilot_v2112_gate."
                "verified_v2112_gate_authority_binding"
            ),
            "adapter_module": "verified_memory/observed_p95_authority.py",
            "source_authority": {
                "path": (
                    "experiment_results/pilot-v2.11.2/raw/"
                    "long-context-preflight/post_gate_authority.json"
                ),
                "schema_version": ("finevo-pilot-v2.11.2-post-gate-authority-v1"),
                "source_commit": "78870956b528946d415a9be5f5769b0893d16d74",
                "file_sha256": (
                    "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
                ),
                "content_sha256": (
                    "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
                ),
            },
            "verification_before_provider_construction": True,
            "unknown_schema_policy": "stop-before-provider-construction",
            "receipt_or_binding_drift_policy": "stop-before-provider-construction",
        },
        "preflight_authority_import": {
            "execution_mode": "preflight_authority_import",
            "source_contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "models": ["gpt52_main", "gpt56_diagnostic"],
            "registered_cells": 2,
            "provider_construction": False,
            "provider_calls": 0,
            "fresh_samples": 0,
            "historical_action_samples_per_model": 24,
            "historical_semantic_samples_per_model": 8,
            "authority_use": "dispatch-reservation-only",
            "scientific_evidence": False,
        },
        "fresh_science_dispatch": {
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.3/raw",
            "registered_scientific_cells": 131,
            "provider_backed_scientific_cells": 126,
            "offline_scientific_cells": 5,
            "registered_provider_calls": 5816,
            "reuse_v2112_scientific_cells": False,
            "reuse_v2112_provider_completions": False,
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "reasoning_or_cap_downgrade": "forbidden",
        },
        "budget_carry_forward": {
            "hard_cap_usd": 500.0,
            "parent_stage_bucket": "parent_v2112",
            "parent_cost_usd": 19.998220562500006,
            "hosted_stage_bucket": "hosted_v2113",
            "hosted_cap_usd": 480.0017794375,
            "manual_reserve_usd": 0.0,
            "parent_hosted_completions": 1004,
            "max_hosted_completions": 7500,
            "parent_storage_bytes": 221668707,
            "max_storage_bytes": 5_000_000_000,
        },
        "immutability": {
            "v2112_denominator_preserved": True,
            "v2112_terminal_statuses_preserved": True,
            "v2112_resume_forbidden": True,
            "v2112_failure_reclassification_forbidden": True,
            "prompt_seed_arm_model_environment_metric_changes": False,
        },
    }


def _validate_v2_11_3_consumer_adapter_amendment(
    value: Any,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "v2113_consumer_adapter_amendment")
    expected = _v2_11_3_expected_consumer_adapter_amendment()
    _strict_keys(
        amendment,
        required=set(expected),
        name="v2113_consumer_adapter_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.11.3 consumer-adapter amendment drifted")
    return _freeze_json(amendment)


def _v2_11_4_expected_forward_boundary(*, status: str) -> dict[str, Any]:
    """Fresh denominator after V2.11.3's zero-call reseal-integrity no-go."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.11.4 forward boundary status is invalid")
    return {
        "schema_version": "finevo-pilot-v2.11.4-forward-boundary-v1",
        "source_manifest": {
            "path": "experiments/pilot_v2_11_4_source_manifest.json",
            "schema_version": "finevo-pilot-v2.11.4-source-manifest-v1",
            "file_sha256": PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256,
            "content_sha256": PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256,
        },
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_3,
            "contract_sha256": PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256,
            "contract_file_sha256": (
                "a0e38d99e52a94c7434bf5e8c1befc9171988d5f722dccc544dc21b400550baf"
            ),
            "science_commit": "65c613cdc9598dfffecbdf3a375cbf6113246782",
            "science_tag": PILOT_CONTRACT_TAG_V2_11_3,
            "science_tag_object": "87a1911284177b627755faf361ad4ea6c8213958",
            "source_manifest_file_sha256": (PILOT_V2_11_3_SOURCE_MANIFEST_FILE_SHA256),
            "source_manifest_content_sha256": (
                PILOT_V2_11_3_SOURCE_MANIFEST_CONTENT_SHA256
            ),
            "run_ledger_file_sha256": (
                "d544145fbcc9028401edd631c8506c0d2413a5e68a79fbf876847e086ad34e31"
            ),
            "run_ledger_internal_sha256": (
                "97216e7b0a23b1b78a1e79d3ae166621147fab5582e5259434e1138c39946f40"
            ),
            "budget_ledger_file_sha256": (
                "d355c48f456d40f0e71b941d5b124cbbaa9959395db05afc2cf30da80cb66189"
            ),
            "budget_ledger_internal_sha256": (
                "366495f3cc4b8075e072c47fcf31c3eed40371996f0057efba64a1709ac5850a"
            ),
            "preflight_stage_receipt_file_sha256": (
                "41a9d8996c010f63b42178610a800c2a59ef6aa13d9666cae639c55ddaebab1a"
            ),
            "preflight_stage_receipt_content_sha256": (
                "1044feb8cf050269c9aafb206bc5fc2c7b6f5b7c0d332d96663d4433ddf967ae"
            ),
            "terminal_status_counts": {
                "complete": 3,
                "integrity-stopped": 133,
            },
            "preflight_stage_status_counts": {"integrity-stopped": 2},
            "completed_operational_cells": 3,
            "integrity_stopped_cells": 133,
            "completed_scientific_cells": 0,
            "fresh_provider_calls": 0,
            "fresh_cost_usd": 0.0,
            "error_type": "V2113PreflightAuthorityImportError",
            "failure_message": (
                "finevo-pilot-v2.11.3--long-context-preflight--gpt52_main--"
                "closed-loop-preflight--none--stage0-selected--s2010922376 "
                "V2.11.3 resealed authority drifted"
            ),
        },
        "reservation_authority_source": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "contract_sha256": PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
            "science_commit": "78870956b528946d415a9be5f5769b0893d16d74",
            "science_tag": PILOT_CONTRACT_TAG_V2_11_2,
            "schema_version": "finevo-pilot-v2.11.2-post-gate-authority-v1",
            "path": (
                "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
                "post_gate_authority.json"
            ),
            "file_sha256": (
                "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
            ),
            "content_sha256": (
                "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
            ),
            "authority_use": "dispatch-reservation-only",
            "scientific_evidence": False,
        },
        "calibration_allowlist": {
            "q_ref": 63.50397933257746,
            "utility_profile_id": "nu-0.5",
            "utility_profile": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale_multiplier_of_q_ref": 1.0,
                "discount_factor": 0.99,
            },
            "absolute_flow_utility_threshold": 0.05617208967516696,
        },
        "parent_budget_debit": {
            "schema_version": "finevo-parent-budget-debit-v1",
            "parent_contract_sha256": PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256,
            "parent_run_ledger_sha256": (
                "97216e7b0a23b1b78a1e79d3ae166621147fab5582e5259434e1138c39946f40"
            ),
            "parent_budget_ledger_sha256": (
                "366495f3cc4b8075e072c47fcf31c3eed40371996f0057efba64a1709ac5850a"
            ),
            "stage_bucket": "parent_v2113",
            "cost_usd": 19.998220562500006,
            "hosted_completions": 1004,
            "storage_bytes": 221838685,
            "record_sha256": (
                "3f75623b4eb5b6c3c1c2e2a7e97687c215da025cbea309f94e861abee47f90ca"
            ),
        },
        "import_policy": {
            "imported_parent_budget_cells": 1,
            "imported_capability_authority_cells": 2,
            "imported_preflight_authority_cells": 2,
            "source_authority_contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "historical_capability_calls": 60,
            "historical_preflight_calls": 64,
            "historical_action_samples_per_model": 24,
            "historical_semantic_samples_per_model": 8,
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "imported_effect_cells": 0,
            "preflight_authority_scientific_evidence": False,
            "effect_artifact_paths": [],
            "v2113_resealed_artifact_use": "failure-audit-only",
            "v2113_scientific_cells_reused": 0,
        },
        "matrix": {
            "ledger_cells": 136,
            "operational_cells": 5,
            "scientific_cells": 131,
            "provider_backed_scientific_cells": 126,
            "offline_scientific_cells": 5,
            "fresh_scientific_provider_calls": 5816,
            "fresh_action_calls": 4848,
            "fresh_semantic_calls": 968,
            "fresh_calls_by_model": {
                "gpt52_main": {"action": 4560, "semantic": 920},
                "gpt56_diagnostic": {"action": 288, "semantic": 48},
            },
            "stage_ledger_cells": {
                "parent-import": 1,
                "capability-gate": 2,
                "long-context-preflight": 2,
                "experiment-c": 25,
                "experiment-a": 20,
                "experiment-d": 55,
                "experiment-b": 25,
                "cross-model": 6,
            },
            "matrix_shrink_after_release": False,
        },
        "paid_provenance": {
            "draft_dispatch_allowed": False,
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11_4,
            "capability_redispatch_allowed": False,
            "preflight_redispatch_allowed": False,
            "v2113_resume_allowed": False,
            "v2113_scientific_cell_reuse_allowed": False,
            "historical_effect_reclassification": False,
        },
    }


def _validate_v2_11_4_forward_boundary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2114_forward_boundary")
    expected = _v2_11_4_expected_forward_boundary(status=status)
    _strict_keys(
        boundary,
        required=set(expected),
        name="v2114_forward_boundary",
    )
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.4 forward boundary drifted")
    if status == "frozen" and (
        PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.4 source manifest hashes must be sealed before freezing"
        )
    return _freeze_json(boundary)


def _v2_11_4_expected_authority_normalization_amendment() -> dict[str, Any]:
    """Exact no-call repair for V2.11.3's representation-level comparison."""

    return {
        "schema_version": ("finevo-pilot-v2.11.4-authority-normalization-amendment-v1"),
        "amendment_id": (
            "finevo-pilot-v2.11.4-resealed-authority-normalization-recovery-1"
        ),
        "parent_terminal_receipt": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_3,
            "contract_sha256": PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256,
            "tag": PILOT_CONTRACT_TAG_V2_11_3,
            "tag_object": "87a1911284177b627755faf361ad4ea6c8213958",
            "commit": "65c613cdc9598dfffecbdf3a375cbf6113246782",
            "run_ledger_internal_sha256": (
                "97216e7b0a23b1b78a1e79d3ae166621147fab5582e5259434e1138c39946f40"
            ),
            "budget_ledger_internal_sha256": (
                "366495f3cc4b8075e072c47fcf31c3eed40371996f0057efba64a1709ac5850a"
            ),
            "preflight_stage_receipt_content_sha256": (
                "1044feb8cf050269c9aafb206bc5fc2c7b6f5b7c0d332d96663d4433ddf967ae"
            ),
            "terminal_cells": 136,
            "complete_cells": 3,
            "integrity_stopped_cells": 133,
            "scientific_provider_calls": 0,
            "fresh_provider_calls": 0,
            "fresh_cost_usd": 0.0,
            "error_type": "V2113PreflightAuthorityImportError",
            "failure_message": (
                "finevo-pilot-v2.11.3--long-context-preflight--gpt52_main--"
                "closed-loop-preflight--none--stage0-selected--s2010922376 "
                "V2.11.3 resealed authority drifted"
            ),
        },
        "observation_boundary": {
            "actor_performance_outcomes_observed": False,
            "a_to_d_scientific_outcomes_observed": False,
            "inspected_artifacts": [
                "source-preflight-wrapper reservation structure",
                "resealed-authority reservation structure",
                "failed equality predicate",
            ],
            "amendment_outcome_blind_for_scientific_performance": True,
            "scientific_design_change": False,
        },
        "root_cause": {
            "root_cause_id": "resealed-authority-representation-comparison-mismatch",
            "failed_predicate": (
                "source-wrapper reservations equal enriched resealed runtime "
                "reservations"
            ),
            "source_wrapper_reservations_sha256": (
                "06788d94b9259c24b753d467683ae88c8015f34ce4989fed47ba71e7eeb823da"
            ),
            "enriched_runtime_reservations_sha256": (
                "8e3e514360cfaff6c40103838d7106078984606d92c779a1f0fb00f93dcb5770"
            ),
            "reservation_payloads_drifted": False,
            "non_allowlisted_authority_fields_drifted": False,
            "provider_or_model_failure": False,
            "method_failure": False,
            "failure_point": "before-provider-construction",
        },
        "authority_normalization_repair": {
            "source_authority_contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "source_authority_schema_version": (
                "finevo-pilot-v2.11.2-post-gate-authority-v1"
            ),
            "call_kinds": ["action", "semantic"],
            "normalization_scope": "each-call-kind-authority-object-only",
            "reseal_only_authority_fields": [
                "source_authority_receipt_content_sha256",
                "source_authority_receipt_file_sha256",
                "source_authority_receipt_path",
                "source_release_commit",
            ],
            "comparison_direction": (
                "strip-exact-reseal-only-allowlist-from-enriched-runtime-then-"
                "compare-to-immutable-source-wrapper"
            ),
            "source_wrapper_mutation": "forbidden",
            "reservation_payload_exact_equality_required": True,
            "non_allowlisted_authority_exact_equality_required": True,
            "removed_fields_verified_against_source_gate_receipt": True,
            "missing_allowlisted_field_policy": "stop-before-provider-construction",
            "unexpected_extra_field_policy": "stop-before-provider-construction",
            "provenance_value_mismatch_policy": "stop-before-provider-construction",
            "verification_before_provider_construction": True,
            "provider_construction": False,
            "provider_calls": 0,
        },
        "preflight_authority_import": {
            "execution_mode": "preflight_authority_import",
            "source_contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "models": ["gpt52_main", "gpt56_diagnostic"],
            "registered_cells": 2,
            "provider_construction": False,
            "provider_calls": 0,
            "fresh_samples": 0,
            "historical_action_samples_per_model": 24,
            "historical_semantic_samples_per_model": 8,
            "authority_use": "dispatch-reservation-only",
            "scientific_evidence": False,
        },
        "fresh_science_dispatch": {
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.4/raw",
            "registered_scientific_cells": 131,
            "provider_backed_scientific_cells": 126,
            "offline_scientific_cells": 5,
            "registered_provider_calls": 5816,
            "reuse_v2113_scientific_cells": False,
            "reuse_v2113_provider_completions": False,
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "reasoning_or_cap_downgrade": "forbidden",
        },
        "budget_carry_forward": {
            "hard_cap_usd": 500.0,
            "parent_stage_bucket": "parent_v2113",
            "parent_cost_usd": 19.998220562500006,
            "hosted_stage_bucket": "hosted_v2114",
            "hosted_cap_usd": 480.0017794375,
            "manual_reserve_usd": 0.0,
            "parent_hosted_completions": 1004,
            "max_hosted_completions": 7500,
            "parent_storage_bytes": 221838685,
            "max_storage_bytes": 5_000_000_000,
        },
        "immutability": {
            "v2113_denominator_preserved": True,
            "v2113_terminal_statuses_preserved": True,
            "v2113_resume_forbidden": True,
            "v2113_failure_reclassification_forbidden": True,
            "v2113_partial_reseal_use": "failure-audit-only",
            "prompt_seed_arm_model_environment_metric_changes": False,
        },
    }


def _validate_v2_11_4_authority_normalization_amendment(
    value: Any,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "v2114_authority_normalization_amendment")
    expected = _v2_11_4_expected_authority_normalization_amendment()
    _strict_keys(
        amendment,
        required=set(expected),
        name="v2114_authority_normalization_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError("V2.11.4 authority-normalization amendment drifted")
    return _freeze_json(amendment)


def _v2_11_5_expected_forward_boundary(*, status: str) -> dict[str, Any]:
    """Fresh denominator after V2.11.4's pre-dispatch acceptance no-go."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.11.5 forward boundary status is invalid")
    return {
        "schema_version": "finevo-pilot-v2.11.5-forward-boundary-v1",
        "source_manifest": {
            "path": "experiments/pilot_v2_11_5_source_manifest.json",
            "schema_version": "finevo-pilot-v2.11.5-source-manifest-v1",
            "file_sha256": PILOT_V2_11_5_SOURCE_MANIFEST_FILE_SHA256,
            "content_sha256": PILOT_V2_11_5_SOURCE_MANIFEST_CONTENT_SHA256,
        },
        "parent": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_4,
            "contract_sha256": PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256,
            "contract_file_sha256": (
                "300abdbb0b2eb27fc1807a1a6af401ef9ad5b0ae9b43ecfceb98dee7fd30dd27"
            ),
            "science_commit": "74f6c05dafc58fadf8d1b658ef3764d244676f76",
            "science_tag": PILOT_CONTRACT_TAG_V2_11_4,
            "science_tag_object": "d774465ad006e9ae974f927ff7b4de94fd5f5147",
            "release_attestation_file_sha256": (
                "9737c690799394481d5c9e4632887d20e0364801d37e7287f9f3418ba0b5f3db"
            ),
            "release_attestation_content_sha256": (
                "50a371cbe79f47529e48b4dfd0e365762aa4b9c56ad2e593209c2942ea67b334"
            ),
            "scientific_launch_input_file_sha256": (
                "59f4f3b8b0142e2753f701595e97266a21f7a72a9a108b14e1174b772b25d094"
            ),
            "scientific_launch_input_content_sha256": (
                "cfc72441b959724e719622a9cbeaa871747635bf993aecdcafcaa7d2908ffedf"
            ),
            "run_ledger_file_sha256": (
                "49c06b1e23c72d280bcab6734fdb40040a2e65e31670970b3804ebcde34c7f13"
            ),
            "run_ledger_internal_sha256": (
                "f0064120e279137fbd7dd5f5cec474aa384745b7c405d9371c90ac5c4f448656"
            ),
            "run_ledger_event_count": 7,
            "run_ledger_event_head_sha256": (
                "6c5c297032aa5ea01b35556b34b55ec4f091a783fd0f90383304006794235861"
            ),
            "budget_ledger_file_sha256": (
                "bdc771f1590fa1111978de7c12152d63b4524078324ceaa008dc717e7a7f2924"
            ),
            "budget_ledger_internal_sha256": (
                "d4ce8beebe1e462003039db2d39e6616c76cd33897c5b3e77e45989ced9d8789"
            ),
            "budget_ledger_event_count": 12,
            "budget_ledger_event_head_sha256": (
                "737162cf298a9c5ec1d116c0238da531fd89ee9b91b231d9b135deb6bebb6fc6"
            ),
            "raw_inventory": {
                "schema_version": "finevo-raw-tree-inventory-v1",
                "file_count": 27,
                "storage_bytes": 427703,
                "inventory_sha256": (
                    "b3e6228fc1873e64f792b4ab51af18104c38b402dbd13b2a7f14ef2ddcb0c89c"
                ),
            },
            "status_counts": {"complete": 5, "scheduled": 131},
            "completed_operational_cells": 5,
            "scheduled_scientific_cells": 131,
            "completed_scientific_cells": 0,
            "fresh_provider_calls": 0,
            "fresh_cost_usd": 0.0,
            "current_attempt_storage_bytes": 210017,
            "acceptance_receipt_present": False,
            "publication_status": "immutable-pre-dispatch-acceptance-no-go",
            "failure_point": "before-provider-construction",
            "failed_predicate": (
                "per-model imported authority equals current global-gate authority"
            ),
        },
        "reservation_authority_source": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "contract_sha256": PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
            "science_commit": "78870956b528946d415a9be5f5769b0893d16d74",
            "science_tag": PILOT_CONTRACT_TAG_V2_11_2,
            "schema_version": "finevo-pilot-v2.11.2-post-gate-authority-v1",
            "path": (
                "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
                "post_gate_authority.json"
            ),
            "file_sha256": (
                "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
            ),
            "content_sha256": (
                "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
            ),
            "authority_use": "dispatch-reservation-only",
            "scientific_evidence": False,
        },
        "calibration_allowlist": {
            "q_ref": 63.50397933257746,
            "utility_profile_id": "nu-0.5",
            "utility_profile": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale_multiplier_of_q_ref": 1.0,
                "discount_factor": 0.99,
            },
            "absolute_flow_utility_threshold": 0.05617208967516696,
        },
        "parent_budget_debit": {
            "schema_version": "finevo-parent-budget-debit-v1",
            "parent_contract_sha256": PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256,
            "parent_run_ledger_sha256": (
                "f0064120e279137fbd7dd5f5cec474aa384745b7c405d9371c90ac5c4f448656"
            ),
            "parent_budget_ledger_sha256": (
                "d4ce8beebe1e462003039db2d39e6616c76cd33897c5b3e77e45989ced9d8789"
            ),
            "stage_bucket": "parent_v2114",
            "cost_usd": 19.998220562500006,
            "hosted_completions": 1004,
            "storage_bytes": 222048702,
            "record_sha256": (
                "9595d037a21f429a59fd37febd4abd8283287e080ee9eb506ebf999e3d1e81a5"
            ),
        },
        "import_policy": {
            "imported_parent_budget_cells": 1,
            "imported_capability_authority_cells": 2,
            "imported_preflight_authority_cells": 2,
            "source_authority_contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "historical_capability_calls": 60,
            "historical_preflight_calls": 64,
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "imported_effect_cells": 0,
            "preflight_authority_scientific_evidence": False,
            "effect_artifact_paths": [],
            "v2114_artifact_use": "failure-audit-only",
            "v2114_scientific_cells_reused": 0,
        },
        "matrix": {
            "ledger_cells": 136,
            "operational_cells": 5,
            "scientific_cells": 131,
            "provider_backed_scientific_cells": 126,
            "offline_scientific_cells": 5,
            "fresh_scientific_provider_calls": 5816,
            "fresh_action_calls": 4848,
            "fresh_semantic_calls": 968,
            "fresh_calls_by_model": {
                "gpt52_main": {"action": 4560, "semantic": 920},
                "gpt56_diagnostic": {"action": 288, "semantic": 48},
            },
            "stage_ledger_cells": {
                "parent-import": 1,
                "capability-gate": 2,
                "long-context-preflight": 2,
                "experiment-c": 25,
                "experiment-a": 20,
                "experiment-d": 55,
                "experiment-b": 25,
                "cross-model": 6,
            },
            "matrix_shrink_after_release": False,
        },
        "paid_provenance": {
            "draft_dispatch_allowed": False,
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11_5,
            "capability_redispatch_allowed": False,
            "preflight_redispatch_allowed": False,
            "v2114_resume_allowed": False,
            "v2114_scientific_cell_reuse_allowed": False,
            "historical_effect_reclassification": False,
        },
    }


def _validate_v2_11_5_forward_boundary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2115_forward_boundary")
    expected = _v2_11_5_expected_forward_boundary(status=status)
    _strict_keys(boundary, required=set(expected), name="v2115_forward_boundary")
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.5 forward boundary drifted")
    if status == "frozen" and (
        PILOT_V2_11_5_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_5_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.5 source manifest hashes must be sealed before freezing"
        )
    return _freeze_json(boundary)


def _v2_11_5_expected_consumer_authority_normalization_amendment() -> dict[str, Any]:
    """Normalize authority provenance at the final reservation consumer."""

    stable_fields = [
        "authority_id",
        "source_kind",
        "source_projection_schema_version",
        "source_preflight_run_id",
        "source_preflight_run_spec_sha256",
        "source_model_id",
        "source_served_model",
        "source_execution_artifact_sha256",
        "source_provider_call_journal_sha256",
    ]
    generation_fields = [
        "pilot_contract_hash",
        "pilot_tag",
        "source_authority_receipt_content_sha256",
        "source_authority_receipt_file_sha256",
        "source_authority_receipt_path",
        "source_projection_content_sha256",
        "source_projection_file_sha256",
        "source_release_commit",
    ]
    return {
        "schema_version": (
            "finevo-pilot-v2.11.5-consumer-authority-normalization-amendment-v1"
        ),
        "amendment_id": (
            "finevo-pilot-v2.11.5-final-consumer-generation-binding-recovery-1"
        ),
        "parent_acceptance_no_go": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_4,
            "contract_sha256": PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256,
            "tag": PILOT_CONTRACT_TAG_V2_11_4,
            "tag_object": "d774465ad006e9ae974f927ff7b4de94fd5f5147",
            "commit": "74f6c05dafc58fadf8d1b658ef3764d244676f76",
            "run_ledger_internal_sha256": (
                "f0064120e279137fbd7dd5f5cec474aa384745b7c405d9371c90ac5c4f448656"
            ),
            "budget_ledger_internal_sha256": (
                "d4ce8beebe1e462003039db2d39e6616c76cd33897c5b3e77e45989ced9d8789"
            ),
            "registered_cells": 136,
            "complete_cells": 5,
            "scheduled_cells": 131,
            "scientific_complete_cells": 0,
            "fresh_provider_calls": 0,
            "fresh_cost_usd": 0.0,
            "acceptance_receipt_present": False,
            "classification": "pre-dispatch-acceptance-no-go",
        },
        "observation_boundary": {
            "actor_performance_outcomes_observed": False,
            "a_to_d_scientific_outcomes_observed": False,
            "inspected_artifacts": [
                "V2.11.2 per-model reservation authorities",
                "V2.11.4 global-gate reservation authorities",
                "V2.11.4 failed final-consumer equality predicate",
            ],
            "amendment_outcome_blind_for_scientific_performance": True,
            "scientific_design_change": False,
        },
        "root_cause": {
            "root_cause_id": (
                "final-consumer-generation-provenance-comparison-mismatch"
            ),
            "failed_predicate": (
                "per-model imported authority object exactly equals current "
                "global-gate authority object"
            ),
            "reservation_payloads_drifted": False,
            "stable_authority_fields_drifted": False,
            "generation_authority_fields_drifted": True,
            "provider_or_model_failure": False,
            "method_failure": False,
            "failure_point": "before-provider-construction",
        },
        "consumer_authority_normalization_repair": {
            "normalization_scope": (
                "per-model-per-call-kind-final-reservation-consumer"
            ),
            "source_authority_contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "source_authority_schema_version": (
                "finevo-pilot-v2.11.2-post-gate-authority-v1"
            ),
            "stable_authority_fields": stable_fields,
            "generation_authority_fields": generation_fields,
            "stable_field_count": 9,
            "generation_field_count": 8,
            "source_generation_values": {
                "pilot_contract_hash": PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
                "pilot_tag": PILOT_CONTRACT_TAG_V2_11_2,
                "source_authority_receipt_content_sha256": (
                    "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
                ),
                "source_authority_receipt_file_sha256": (
                    "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
                ),
                "source_authority_receipt_path": (
                    "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
                    "post_gate_authority.json"
                ),
                "source_projection_content_sha256": (
                    "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
                ),
                "source_projection_file_sha256": (
                    "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
                ),
                "source_release_commit": ("78870956b528946d415a9be5f5769b0893d16d74"),
            },
            "parent_generation_values": {
                "pilot_contract_hash": PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256,
                "pilot_tag": PILOT_CONTRACT_TAG_V2_11_4,
                "source_authority_receipt_content_sha256": (
                    "f556add3a9c4d362a915e77231efcd58ffad4783fa363d31b8f271eba93a2e71"
                ),
                "source_authority_receipt_file_sha256": (
                    "3ddc7c322991c3363059710173614d5dd00bc12b7122417ff571dbbbf0fd6cfc"
                ),
                "source_authority_receipt_path": (
                    "experiment_results/pilot-v2.11.4/raw/long-context-preflight/"
                    "post_gate_authority.json"
                ),
                "source_projection_content_sha256": (
                    "f556add3a9c4d362a915e77231efcd58ffad4783fa363d31b8f271eba93a2e71"
                ),
                "source_projection_file_sha256": (
                    "3ddc7c322991c3363059710173614d5dd00bc12b7122417ff571dbbbf0fd6cfc"
                ),
                "source_release_commit": ("74f6c05dafc58fadf8d1b658ef3764d244676f76"),
            },
            "current_generation_rules": {
                "pilot_contract_hash": "current-contract-canonical-sha256",
                "pilot_tag": "current-contract-required-annotated-tag",
                "source_authority_receipt_content_sha256": (
                    "current-global-gate-receipt-content-sha256"
                ),
                "source_authority_receipt_file_sha256": (
                    "current-global-gate-receipt-file-sha256"
                ),
                "source_authority_receipt_path": (
                    "current-raw-namespace/long-context-preflight/"
                    "post_gate_authority.json"
                ),
                "source_projection_content_sha256": (
                    "current-global-gate-receipt-content-sha256"
                ),
                "source_projection_file_sha256": (
                    "current-global-gate-receipt-file-sha256"
                ),
                "source_release_commit": "current-annotated-tag-peeled-commit",
            },
            "reservation_payload_sha256_by_model_call_kind": {
                "gpt52_main": {
                    "action": (
                        "fdbcf2d6696969452ea3e0aa1af7475618dcd058ad0f741f92886f23aa05a9f3"
                    ),
                    "semantic": (
                        "61c8bb94d82ec715eb7fc93d73b51fda9caf194c1e58627e737c5c13e69996de"
                    ),
                },
                "gpt56_diagnostic": {
                    "action": (
                        "4da97d868764ff87ac45cf517f8de3e08db1cb4debf4462cb3279dc75f4d9b3f"
                    ),
                    "semantic": (
                        "715708d6a4ad18a46d02c628c6688bf398d5f4ed5e88c519ade2bee642df9c7a"
                    ),
                },
            },
            "reservation_payload_exact_equality_required": True,
            "stable_field_exact_equality_required": True,
            "generation_fields_must_match_current_generation_rules": True,
            "unknown_authority_field_policy": "stop-before-provider-construction",
            "missing_authority_field_policy": "stop-before-provider-construction",
            "validation_before_provider_construction": True,
            "provider_construction": False,
            "provider_calls": 0,
        },
        "fresh_science_dispatch": {
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.5/raw",
            "registered_scientific_cells": 131,
            "provider_backed_scientific_cells": 126,
            "offline_scientific_cells": 5,
            "registered_provider_calls": 5816,
            "reuse_v2114_scientific_cells": False,
            "reuse_v2114_provider_completions": False,
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "reasoning_or_cap_downgrade": "forbidden",
        },
        "budget_carry_forward": {
            "hard_cap_usd": 500.0,
            "parent_stage_bucket": "parent_v2114",
            "parent_cost_usd": 19.998220562500006,
            "hosted_stage_bucket": "hosted_v2115",
            "hosted_cap_usd": 480.0017794375,
            "manual_reserve_usd": 0.0,
            "parent_hosted_completions": 1004,
            "max_hosted_completions": 7500,
            "parent_storage_bytes": 222048702,
            "max_storage_bytes": 5_000_000_000,
        },
        "immutability": {
            "v2114_denominator_preserved": True,
            "v2114_statuses_preserved": True,
            "v2114_resume_forbidden": True,
            "v2114_failure_reclassification_forbidden": True,
            "v2114_artifact_use": "failure-audit-only",
            "prompt_seed_arm_model_environment_metric_changes": False,
        },
    }


def _validate_v2_11_5_consumer_authority_normalization_amendment(
    value: Any,
) -> Mapping[str, Any]:
    amendment = _mapping(value, "v2115_consumer_authority_normalization_amendment")
    expected = _v2_11_5_expected_consumer_authority_normalization_amendment()
    _strict_keys(
        amendment,
        required=set(expected),
        name="v2115_consumer_authority_normalization_amendment",
    )
    if _json_copy(amendment) != expected:
        raise PilotContractError(
            "V2.11.5 consumer-authority normalization amendment drifted"
        )
    return _freeze_json(amendment)


def _v2_11_6_expected_continuation_boundary(*, status: str) -> dict[str, Any]:
    """Bind the immutable V2.11.5 prefix and its 86 unstarted science rows.

    V2.11.6 is intentionally not a fresh 136-cell retry.  One zero-provider
    import cell authenticates the 50 terminal parent rows and maps only the 86
    still-scheduled D/B/cross-model rows into a fresh execution namespace.
    """

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.11.6 continuation boundary status is invalid")
    return {
        "schema_version": "finevo-pilot-v2.11.6-continuation-boundary-v1",
        "source_manifest": {
            "path": "experiments/pilot_v2_11_6_source_manifest.json",
            "schema_version": "finevo-pilot-v2.11.6-source-manifest-v1",
            "file_sha256": PILOT_V2_11_6_SOURCE_MANIFEST_FILE_SHA256,
            "content_sha256": PILOT_V2_11_6_SOURCE_MANIFEST_CONTENT_SHA256,
        },
        "parent_release": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_5,
            "contract_path": "experiments/pilot_v2_11_5.yaml",
            "contract_file_sha256": (
                "b96438430231f0c46fd6c5f15ba749713534feb15f964c496aa02606cf11103b"
            ),
            "contract_sha256": PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256,
            "science_tag": PILOT_CONTRACT_TAG_V2_11_5,
            "science_tag_object": "bccfb13cee7d592470d1873cfacc3b12bed38be4",
            "science_commit": "2351ac2283f9fedb9dce70067174020be56ed9cc",
            "source_manifest_path": "experiments/pilot_v2_11_5_source_manifest.json",
            "source_manifest_file_sha256": (
                "fea5a276fb64fdd5bf0539014687ea39a891e9d305205b1d2046a2c15a892d16"
            ),
            "source_manifest_content_sha256": (
                "be84d33f561a5ab8927f13e0753f5109b5f018dc790ae180d5e0e6e0228af559"
            ),
        },
        "parent_terminal_prefix": {
            "run_ledger": {
                "path": "experiment_results/pilot-v2.11.5/raw/run_ledger.json",
                "schema_version": "finevo-pilot-run-ledger-v2",
                "file_sha256": (
                    "16c368ab7ad5c3da2d53ba9b15a17ef83ce7a72639af0adc317eaa3605e27c28"
                ),
                "ledger_sha256": (
                    "8a86231f0906ea117626190cc7a2699933c968ce555612cb1bc6378473601fa7"
                ),
                "prefix_event_count": 53,
                "prefix_event_head_sha256": (
                    "61489ef64e71400e603e2fb1110e5e8af3ba772ac083361338a4ccff9641022f"
                ),
                "registered_rows": 136,
                "terminal_rows": 50,
                "scheduled_rows": 86,
                "status_counts": {
                    "complete": 47,
                    "failed": 3,
                    "scheduled": 86,
                },
            },
            "budget_ledger": {
                "path": "experiment_results/pilot-v2.11.5/raw/budget_ledger.json",
                "schema_version": "finevo-pilot-budget-ledger-v2",
                "file_sha256": (
                    "49c21f5b2b21b773a8a0814ffe115ccd69cc67d947b67b5b4e3ecdaf2b5a2afb"
                ),
                "ledger_sha256": (
                    "53e70f6c0b9053674408de385e1a5b5bf42ace7e82dc8e0c6f227ea124b7a38f"
                ),
                "prefix_event_count": 103,
                "prefix_event_head_sha256": (
                    "f745a8c4087b310d5d5cfb74645df8ddb0f2f80ef3d269b9642d3715f7de5834"
                ),
                "terminal_run_rows": 50,
            },
            "source_raw_inventory": {
                "root": "experiment_results/pilot-v2.11.5/raw",
                "schema_version": "finevo-raw-tree-inventory-v1",
                "canonicalization": "json-sort-keys-compact-utf8-v1",
                "excluded_operational_paths": [".real-stage-execution.lock"],
                "file_count": 691,
                "storage_bytes": 48820556,
                "inventory_sha256": (
                    "f2fdb1ccedcb70e6793d3b8f3c87425f0d602552f0a3e0e7f35db9c5777c6746"
                ),
            },
        },
        "parent_stage_receipts": {
            "experiment-a": {
                "path": (
                    "experiment_results/pilot-v2.11.5/raw/experiment-a/"
                    "stage_receipt.json"
                ),
                "file_sha256": (
                    "8193f3449663f63c9cf0c881ee5e7759d2682f320f214c4941040489c81734f9"
                ),
                "content_sha256": (
                    "177dc8ce4d1957eac0734bb1716279676f77931e30b3a1d10dd2c138a43a5457"
                ),
                "status": "complete-with-no-go",
                "go": False,
                "execution_progression_go": True,
                "scientific_matrix_complete": False,
                "registered_run_count": 20,
                "status_counts": {"complete": 17, "failed": 3},
            },
            "experiment-c": {
                "path": (
                    "experiment_results/pilot-v2.11.5/raw/experiment-c/"
                    "stage_receipt.json"
                ),
                "file_sha256": (
                    "958cb161785c144c89861da3e9536e53069e8f1070a64c03f54647cbfe05b322"
                ),
                "content_sha256": (
                    "39a9d35f4961fee4b0bc59ac67f7a9a2da0c3f95fddf77a418b92e518b6e2eba"
                ),
                "status": "complete-with-no-go",
                "go": False,
                "execution_progression_go": True,
                "scientific_matrix_complete": True,
                "registered_run_count": 25,
                "status_counts": {"complete": 25},
            },
        },
        "imported_authority": {
            "parent_import_receipt_path": (
                "experiment_results/pilot-v2.11.5/raw/parent-import/"
                "parent_import_receipt.json"
            ),
            "parent_import_receipt_file_sha256": (
                "104966506803b93e009730fb3e2f742a1eded17e6f5c210faacff1f5ffc5ace8"
            ),
            "parent_import_receipt_content_sha256": (
                "3eb37262ca7a3df78436e964bc202768e7b5ca4417d947c7a61f419eec24a658"
            ),
            "preflight_authority_path": (
                "experiment_results/pilot-v2.11.5/raw/long-context-preflight/"
                "post_gate_authority.json"
            ),
            "preflight_authority_file_sha256": (
                "08b33162c91a07b392bacefc40d6abee9d89633600608a3de798f171e427a35a"
            ),
            "preflight_authority_receipt_sha256": (
                "19e18c1641ecaf55e48340694e126416ff30f0b39307c66f5335c9c9e9a46abc"
            ),
            "preflight_authority_source_contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "authority_use": "dispatch-reservation-only",
            "provider_construction": False,
            "provider_calls": 0,
            "scientific_evidence": False,
        },
        "parent_budget_debit": {
            "schema_version": "finevo-parent-budget-debit-v1",
            "parent_contract_sha256": PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256,
            "parent_run_ledger_sha256": (
                "8a86231f0906ea117626190cc7a2699933c968ce555612cb1bc6378473601fa7"
            ),
            "parent_budget_ledger_sha256": (
                "53e70f6c0b9053674408de385e1a5b5bf42ace7e82dc8e0c6f227ea124b7a38f"
            ),
            "stage_bucket": "parent_v2115",
            "cost_usd": 63.1196450625,
            "hosted_completions": 3440,
            "storage_bytes": 270188235,
            "record_sha256": (
                "bada157f174d33344370c621f0bd480d57cf8ff5adcde498d7e02426a4363270"
            ),
        },
        "continuation_budget": {
            "hard_cap_usd": 500.0,
            "parent_stage_bucket": "parent_v2115",
            "hosted_stage_bucket": "hosted_v2116",
            "hosted_cap_usd": 436.8803549375,
            "fresh_projected_cost_usd": 149.3301875,
            "fresh_registered_provider_calls": 3256,
            "fresh_storage_reservation_bytes": 1020000000,
            "projected_cumulative_cost_usd": 212.4498325625,
            "projected_cumulative_hosted_completions": 6696,
            "projected_cumulative_storage_bytes": 1290188235,
            "max_hosted_completions": 7500,
            "max_storage_bytes": 5000000000,
            "within_all_hard_caps": True,
        },
        "continuation_matrix": {
            "ledger_cells": 87,
            "operational_import_cells": 1,
            "fresh_scientific_cells": 86,
            "fresh_provider_backed_scientific_cells": 86,
            "fresh_provider_calls": 3256,
            "fresh_calls_by_stage": {
                "experiment-d": 1480,
                "experiment-b": 1440,
                "cross-model": 336,
            },
            "stage_ledger_cells": {
                "parent-import": 1,
                "experiment-d": 55,
                "experiment-b": 25,
                "cross-model": 6,
            },
            "stage_order": [
                "parent-import",
                "experiment-d",
                "experiment-b",
                "cross-model",
            ],
            "source_scheduled_rows": 86,
            "normalized_source_spec_sha256": (
                "9968bb55b9c56ced90f56826bc8e186f72299e0a8bb40dfdb4fbb1e637af1632"
            ),
            "normalized_source_spec_sha256_by_stage": {
                "experiment-d": (
                    "38a3d4ccfc15d2d9fac2ef6fc4d071a5392f742c11452e921687588d54ba6159"
                ),
                "experiment-b": (
                    "b1e6be0a3076c5ac2f7b867c132f5ef15af098e2711f4314018be03edac7f866"
                ),
                "cross-model": (
                    "2d89f7b6be1ce63a11f4ba5b80da49890567d749fafc5b1a436e1aba7f166351"
                ),
            },
            "source_to_continuation_mapping": "normalized-one-to-one",
            "combined_registered_denominator": 136,
            "combined_parent_terminal_rows": 50,
            "combined_continuation_rows": 86,
            "imported_a_c_remain_parent_evidence": True,
            "a_c_reclassified_as_v2116": False,
        },
        "immutability": {
            "parent_raw_read_only": True,
            "parent_terminal_rows_redispatch_forbidden": True,
            "parent_a_c_resume_forbidden": True,
            "parent_failure_reclassification_forbidden": True,
            "failed_seed_replacement": "forbidden",
            "matrix_shrink": "forbidden",
            "reasoning_or_cap_downgrade": "forbidden",
            "remaining_spec_changes": False,
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.6/raw",
        },
        "paid_provenance": {
            "draft_dispatch_allowed": False,
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11_6,
            "parent_prefix_validation_before_provider_construction": True,
            "source_manifest_validation_before_provider_construction": True,
            "parent_import_provider_calls": 0,
        },
    }


def _validate_v2_11_6_continuation_boundary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2116_continuation_boundary")
    expected = _v2_11_6_expected_continuation_boundary(status=status)
    _strict_keys(
        boundary,
        required=set(expected),
        name="v2116_continuation_boundary",
    )
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.6 continuation boundary drifted")
    if status == "frozen" and (
        PILOT_V2_11_6_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_6_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.6 source manifest hashes must be sealed before freezing"
        )
    return _freeze_json(boundary)


def _v2_11_7_expected_recovery_boundary(*, status: str) -> dict[str, Any]:
    """Bind V2.11.6's zero-call no-go and the untouched V2.11.5 rows."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.11.7 recovery boundary status is invalid")
    boundary = _json_copy(_v2_11_6_expected_continuation_boundary(status="frozen"))
    boundary["schema_version"] = (
        "finevo-pilot-v2.11.7-accounting-scope-recovery-boundary-v1"
    )
    boundary["source_manifest"] = {
        "path": "experiments/pilot_v2_11_7_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.7-source-manifest-v1",
        "file_sha256": PILOT_V2_11_7_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": PILOT_V2_11_7_SOURCE_MANIFEST_CONTENT_SHA256,
    }
    boundary["failed_release_no_go"] = {
        "contract_id": PILOT_CONTRACT_ID_V2_11_6,
        "contract_path": "experiments/pilot_v2_11_6.yaml",
        "contract_file_sha256": (
            "8670a2c464214f8c63b5c4712baf946f26cb85fc727dec7f6c1c6a933979792a"
        ),
        "contract_sha256": PILOT_CONTRACT_V2_11_6_CANONICAL_SHA256,
        "science_tag": PILOT_CONTRACT_TAG_V2_11_6,
        "science_tag_object": "6355d2329d800c95595c89f5e41e032ba6129fb7",
        "science_commit": "0a7eb29a76c5f9c90486052a4c335ad1d2000bf0",
        "source_manifest_path": "experiments/pilot_v2_11_6_source_manifest.json",
        "source_manifest_file_sha256": (
            "710db4414471005d088cd64fb1e1a7c4a46fd99f8852b05f3f17f2acaead240d"
        ),
        "source_manifest_content_sha256": (
            "c510941c565d1120604199139d193990948d6b65be15a823ba1d4850968f2ce0"
        ),
        "raw_inventory": {
            "root": "experiment_results/pilot-v2.11.6/raw",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "excluded_operational_paths": [".real-stage-execution.lock"],
            "file_count": 5,
            "storage_bytes": 215033,
            "inventory_sha256": (
                "0dbf0a293b9b2c00c642aa8d7724eb7b585e0b23ad327504e15edaf63e5e234d"
            ),
        },
        "run_ledger": {
            "path": "experiment_results/pilot-v2.11.6/raw/run_ledger.json",
            "file_sha256": (
                "e43ee536232369cabc27215e8c066a2d1834c27505da138652c76a578ea8af40"
            ),
            "ledger_sha256": (
                "0fcd05fea8cf93574e69fc2b7a0c94171a952e85e7932d035ea85784fbb8594d"
            ),
            "event_count": 89,
            "event_head_sha256": (
                "d2749697e1139ea403e43a164901c0ff36b1f83814a8035a5d292ff47b0f2100"
            ),
            "registered_rows": 87,
            "status_counts": {"integrity-stopped": 87},
        },
        "budget_ledger": {
            "path": "experiment_results/pilot-v2.11.6/raw/budget_ledger.json",
            "file_sha256": (
                "d36b2cded3ea7ddd8e7354173aaa5ad26e72ad8430d5d65071ef59d121ff1b0a"
            ),
            "ledger_sha256": (
                "a35780543ba195257da51a66dbfc7f9f662c1b6546b728e97e9041920795afea"
            ),
            "event_count": 4,
            "event_head_sha256": (
                "16a1a891adbee9b44e92ebce969485ec1ee8bb785aa8c70a5df4542e162bcbf2"
            ),
            "current_actual": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "storage_bytes": 1696,
            },
        },
        "stage_receipt": {
            "path": (
                "experiment_results/pilot-v2.11.6/raw/parent-import/"
                "stage_receipt.json"
            ),
            "file_sha256": (
                "f91d6631b718265cc7ff8682089b184780a4fa390237484b335c05d580fa1d0f"
            ),
            "content_sha256": (
                "52905fa9f9fcf3cb9e65579e49f5b8bca77dd0d252b314cbfbf7fd34ab0e6e69"
            ),
            "status": "integrity-stopped",
            "go": False,
            "execution_progression_go": False,
            "failure_error_type": "V2116ParentImportIntegrityError",
            "failure_cause_type": "PilotV2116ContinuationError",
            "failure_message": "V2.11.5 current-release actual debit drifted",
        },
        "release_attestation": {
            "path": "experiment_results/pilot-v2.11.6/raw/release_attestation.json",
            "file_sha256": (
                "456ac803bd6120b5675417251041ebf8aa3b4ba1c928d195b0fb831c2c676cef"
            ),
            "attestation_sha256": (
                "7f6d245ebb010237248d51db44fc466a1f4061190d10d8fba5046ba103b7671d"
            ),
            "status": "pass",
        },
        "scientific_launch_input": {
            "path": (
                "experiment_results/pilot-v2.11.6/raw/" "scientific_launch_input.json"
            ),
            "file_sha256": (
                "cb4df8813f13c297a78da2e9d7c4c280deb4fbb991c3e80f89d10fe3cade4c62"
            ),
            "launch_input_sha256": (
                "660b549503d4209b7b1ed4e859504a9ecb2f01f6fc6a0abc01a86c058a8f6d00"
            ),
        },
        "acceptance_receipt_present": False,
        "science_reservations": 0,
        "provider_construction": False,
        "provider_calls": 0,
        "scientific_evidence": False,
        "resume_forbidden": True,
        "failure_reclassification_forbidden": True,
    }
    boundary["authority_current_actual_decomposition"] = {
        "aggregation_scope": "all-current-budget-run-rows",
        "hosted_v2115": {
            "row_count": 47,
            "cost_usd": 43.1214245,
            "hosted_completions": 2436,
            "storage_bytes": 47975380,
        },
        "operational_parent_v2114": {
            "row_count": 3,
            "cost_usd": 0.0,
            "hosted_completions": 0,
            "storage_bytes": 164153,
        },
        "all_current": {
            "row_count": 50,
            "cost_usd": 43.1214245,
            "hosted_completions": 2436,
            "storage_bytes": 48139533,
        },
        "inherited_parent": {
            "cost_usd": 19.998220562500006,
            "hosted_completions": 1004,
            "storage_bytes": 222048702,
        },
        "cumulative_v2115": {
            "cost_usd": 63.1196450625,
            "hosted_completions": 3440,
            "storage_bytes": 270188235,
        },
        "observed_storage_difference_bytes": 164153,
        "repair_changes_scientific_design": False,
        "scientific_outcomes_inspected_for_repair": False,
    }
    boundary["parent_budget_debit"] = {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": PILOT_CONTRACT_V2_11_6_CANONICAL_SHA256,
        "parent_run_ledger_sha256": (
            "0fcd05fea8cf93574e69fc2b7a0c94171a952e85e7932d035ea85784fbb8594d"
        ),
        "parent_budget_ledger_sha256": (
            "a35780543ba195257da51a66dbfc7f9f662c1b6546b728e97e9041920795afea"
        ),
        "stage_bucket": "parent_v2116",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3440,
        "storage_bytes": 270189931,
        "record_sha256": (
            "1118a572ce7fe713f0428bbddd155808e20db6ac7bd845f8a180145c50f7b46a"
        ),
    }
    boundary["continuation_budget"].update(
        {
            "parent_stage_bucket": "parent_v2116",
            "hosted_stage_bucket": "hosted_v2117",
            "projected_cumulative_storage_bytes": 1290189931,
        }
    )
    matrix = boundary["continuation_matrix"]
    matrix["a_c_reclassified_as_v2117"] = matrix.pop("a_c_reclassified_as_v2116")
    matrix["failed_v2116_rows_are_aborted_release_audit_only"] = True
    matrix["per_row_source_mapping_required"] = True
    matrix["per_row_source_contract_id"] = PILOT_CONTRACT_ID_V2_11_5
    matrix["per_row_mapping_key_fields"] = [
        "source_run_id",
        "logical_cell_sha256",
    ]
    matrix["per_row_mapping_duplicate_policy"] = "reject"
    matrix["canonical_86_row_mapping_sha256"] = (
        "88aef768d311653c8335f7ad769400c84e0c0430c9c82183611f87d0f6906fcd"
    )
    matrix["logical_registered_denominator_after_cross_release_dedup"] = 136
    matrix["logical_scientific_denominator_after_cross_release_dedup"] = 131
    boundary["immutability"].update(
        {
            "v2116_no_go_raw_read_only": True,
            "v2116_statuses_preserved": True,
            "v2116_resume_forbidden": True,
            "v2116_failure_reclassification_forbidden": True,
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.7/raw",
        }
    )
    boundary["paid_provenance"].update(
        {
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11_7,
            "failed_release_validation_before_provider_construction": True,
            "authority_release_validation_before_provider_construction": True,
        }
    )
    return boundary


def _validate_v2_11_7_recovery_boundary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2117_recovery_boundary")
    expected = _v2_11_7_expected_recovery_boundary(status=status)
    _strict_keys(boundary, required=set(expected), name="v2117_recovery_boundary")
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.7 recovery boundary drifted")
    if status == "frozen" and (
        PILOT_V2_11_7_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_7_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.7 source manifest hashes must be sealed before freezing"
        )
    return _freeze_json(boundary)


def _v2_11_8_expected_recovery_boundary(*, status: str) -> dict[str, Any]:
    """Bind V2.11.7's zero-call no-go and recover its repository context."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.11.8 recovery boundary status is invalid")
    boundary = _json_copy(_v2_11_7_expected_recovery_boundary(status="frozen"))
    boundary["schema_version"] = (
        "finevo-pilot-v2.11.8-observed-p95-context-recovery-boundary-v1"
    )
    boundary["source_manifest"] = {
        "path": "experiments/pilot_v2_11_8_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.8-source-manifest-v1",
        "file_sha256": PILOT_V2_11_8_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": PILOT_V2_11_8_SOURCE_MANIFEST_CONTENT_SHA256,
    }

    failed = boundary["failed_release_no_go"]
    failed.update(
        {
            "contract_id": PILOT_CONTRACT_ID_V2_11_7,
            "contract_path": "experiments/pilot_v2_11_7.yaml",
            "contract_file_sha256": (
                "4b570b212f391c1887b4a7cb3554ab65e6fac77b6d35e0a3aa8b0509e84c8d85"
            ),
            "contract_sha256": PILOT_CONTRACT_V2_11_7_CANONICAL_SHA256,
            "science_tag": PILOT_CONTRACT_TAG_V2_11_7,
            "science_tag_object": "6ce166fecfb126c07788bc87c31fcdc6ecb42078",
            "science_commit": "57c53588440dc2647f6b6ffae519049db4cd4844",
            "source_manifest_path": ("experiments/pilot_v2_11_7_source_manifest.json"),
            "source_manifest_file_sha256": (
                "dd124c09359d0bd08411add3486cc43887cbee207fdbb6f9bc929e5c1eb81ef9"
            ),
            "source_manifest_content_sha256": (
                "64be1bf836d131d8ec0542e68388dbc328314af7e891549600f5871f8f61f2b0"
            ),
        }
    )
    failed["raw_inventory"].update(
        {
            "root": "experiment_results/pilot-v2.11.7/raw",
            "storage_bytes": 224071,
            "inventory_sha256": (
                "af4053b3e7fc2b706707f47d552d56ac25dfff4fbf5df5d58a6739e375f160ec"
            ),
        }
    )
    failed["run_ledger"].update(
        {
            "path": "experiment_results/pilot-v2.11.7/raw/run_ledger.json",
            "file_sha256": (
                "8dc580d7030f7aab182429bc7dd7bc72c6a0d61e7944477bce8a52836ac324cc"
            ),
            "ledger_sha256": (
                "bb6d497308097cf6f348c282339f2f6d4cb6721950604744c1e6b0751e913681"
            ),
            "event_head_sha256": (
                "5446a8981e9f8579893c1faea39a0722972ee709967967e99086be614414dab6"
            ),
        }
    )
    failed["budget_ledger"].update(
        {
            "path": "experiment_results/pilot-v2.11.7/raw/budget_ledger.json",
            "file_sha256": (
                "f9e0216cbc8e5d3ea6ceb7728ca1a4df0fd71ecfb49124a1ccacd3b0758d9272"
            ),
            "ledger_sha256": (
                "bc6cc622beaff05e2480e866408929f3edd7f02a7555bdb26202fe94ae3e9c77"
            ),
            "event_head_sha256": (
                "66333a703087c3aae041171eb2d2a96ff2f7e3ff60aa454d8b55a08bda2f5fdd"
            ),
        }
    )
    failed["budget_ledger"]["current_actual"]["storage_bytes"] = 1797
    failed["stage_receipt"].update(
        {
            "path": (
                "experiment_results/pilot-v2.11.7/raw/parent-import/"
                "stage_receipt.json"
            ),
            "file_sha256": (
                "d8e276f8eaa725b1b32666f09508c984d536c9e5eb05e54cf6bb6faaff2b0ddc"
            ),
            "content_sha256": (
                "914ee31e0b2f1102d77b18b26e2ae133247df13ae403c9df97545be80637abba"
            ),
            "failure_error_type": "V2117ParentImportIntegrityError",
            "failure_cause_type": "PilotV2115AcceptanceError",
            "failure_message": (
                "Experiment D group gpt52_main/617806385 failed validation: "
                "source-backed observed p95 release commit differs from the "
                "annotated tag or current HEAD"
            ),
        }
    )
    failed["release_attestation"].update(
        {
            "path": ("experiment_results/pilot-v2.11.7/raw/release_attestation.json"),
            "file_sha256": (
                "07805c14c5a6673805de1f0d1d3a423a98270d9b420227c5ac1865e2ffc64a7e"
            ),
            "attestation_sha256": (
                "df033a800e85d3a0a918e10b4627f01bba3bb8aa046c04a3a3440370e6ea226c"
            ),
        }
    )
    failed["scientific_launch_input"].update(
        {
            "path": (
                "experiment_results/pilot-v2.11.7/raw/scientific_launch_input.json"
            ),
            "file_sha256": (
                "0a29ceb2d7d21f3d99032d40543c074cc0095f5ac5acebf206e3c37cb8215e27"
            ),
            "launch_input_sha256": (
                "2ae474f0958a31e070b36d85a4cbe28a6b2301c8f9bf2b8a47da2e30303b8f2b"
            ),
        }
    )

    boundary["parent_budget_debit"].update(
        {
            "parent_contract_sha256": PILOT_CONTRACT_V2_11_7_CANONICAL_SHA256,
            "parent_run_ledger_sha256": (
                "bb6d497308097cf6f348c282339f2f6d4cb6721950604744c1e6b0751e913681"
            ),
            "parent_budget_ledger_sha256": (
                "bc6cc622beaff05e2480e866408929f3edd7f02a7555bdb26202fe94ae3e9c77"
            ),
            "stage_bucket": "parent_v2117",
            "storage_bytes": 270191728,
            "record_sha256": (
                "a8281fea88c404d504792b08d8bef75ee5d33d890ee5a44ed91962012ba87f1e"
            ),
        }
    )
    boundary["continuation_budget"].update(
        {
            "parent_stage_bucket": "parent_v2117",
            "hosted_stage_bucket": "hosted_v2118",
            "projected_cumulative_storage_bytes": 1290191728,
        }
    )
    matrix = boundary["continuation_matrix"]
    matrix["a_c_reclassified_as_v2118"] = matrix.pop("a_c_reclassified_as_v2117")
    matrix["failed_v2117_rows_are_aborted_release_audit_only"] = matrix.pop(
        "failed_v2116_rows_are_aborted_release_audit_only"
    )
    matrix["canonical_86_row_mapping_sha256"] = (
        "812781508a1cbfb8a827de0a981c3b6e189cff497decb7e9343ce6f8aa4d4ca5"
    )
    immutability = boundary["immutability"]
    for key in (
        "v2116_no_go_raw_read_only",
        "v2116_statuses_preserved",
        "v2116_resume_forbidden",
        "v2116_failure_reclassification_forbidden",
    ):
        immutability.pop(key)
    immutability.update(
        {
            "v2117_no_go_raw_read_only": True,
            "v2117_statuses_preserved": True,
            "v2117_resume_forbidden": True,
            "v2117_failure_reclassification_forbidden": True,
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.8/raw",
        }
    )
    boundary["paid_provenance"]["required_annotated_tag"] = PILOT_CONTRACT_TAG_V2_11_8
    return boundary


def _validate_v2_11_8_recovery_boundary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2118_recovery_boundary")
    expected = _v2_11_8_expected_recovery_boundary(status=status)
    _strict_keys(boundary, required=set(expected), name="v2118_recovery_boundary")
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.8 recovery boundary drifted")
    if status == "frozen" and (
        PILOT_V2_11_8_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_8_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.8 source manifest hashes must be sealed before freezing"
        )
    return _freeze_json(boundary)


def _v2_11_9_expected_recovery_boundary(*, status: str) -> dict[str, Any]:
    """Bind V2.11.8's zero-call no-go and recover release recomputation."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.11.9 recovery boundary status is invalid")
    boundary = _json_copy(_v2_11_8_expected_recovery_boundary(status="frozen"))
    boundary["schema_version"] = (
        "finevo-pilot-v2.11.9-release-binding-recovery-boundary-v1"
    )
    boundary["source_manifest"] = {
        "path": "experiments/pilot_v2_11_9_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.9-source-manifest-v1",
        "file_sha256": PILOT_V2_11_9_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": PILOT_V2_11_9_SOURCE_MANIFEST_CONTENT_SHA256,
    }
    boundary["runtime_input_binding"] = {
        "cwd_must_equal_release_root": True,
        "profile_path": "data/profiles.json",
        "profile_file_sha256": (
            "1bc90a92ef8e32f3da6e474f787207b79b1c82cc0b7b13c5ea3bd6cd1439b223"
        ),
        "profile_regular_non_symlink_required": True,
        "verification_points": [
            "scientific-dispatch-acceptance",
            "each-scientific-stage-before-provider-construction",
        ],
    }
    boundary["source_coverage"] = {
        "complete_verified_memory_python_tree": True,
        "complete_foundation_python_tree": True,
        "llm_provider_and_unique_cli": True,
        "release_renderers": True,
        "contract_module_normalization": "three-literal-v2119-cycle-pins-only",
        "ci_module_normalization": "two-literal-v2119-source-anchor-pins-only",
    }

    failed = boundary["failed_release_no_go"]
    failed.update(
        {
            "contract_id": PILOT_CONTRACT_ID_V2_11_8,
            "contract_path": "experiments/pilot_v2_11_8.yaml",
            "contract_file_sha256": (
                "c355c1f1fe7eaa3571f4101f2770bd3c9ef8a5fc41553c337439b7aa1148390a"
            ),
            "contract_sha256": PILOT_CONTRACT_V2_11_8_CANONICAL_SHA256,
            "science_tag": PILOT_CONTRACT_TAG_V2_11_8,
            "science_tag_object": "a5564d374762aed5ea2493706888e2950b6e97fa",
            "science_commit": "67aa0fcce68fa5ac43b48dd3b81b849112137093",
            "source_manifest_path": ("experiments/pilot_v2_11_8_source_manifest.json"),
            "source_manifest_file_sha256": (
                "acfc9dc6c751e8ab9f314133de856bae7a0a4021c067f693ed8ebff938b230a6"
            ),
            "source_manifest_content_sha256": (
                "104b63db289234820aebf14f42808c26cd01d9f8a19029fef793887bfff47cd3"
            ),
        }
    )
    failed["raw_inventory"].update(
        {
            "root": "experiment_results/pilot-v2.11.8/raw",
            "file_count": 5,
            "storage_bytes": 221847,
            "inventory_sha256": (
                "07919624f2bfaeef1c9c54883f089b543f454de4d3775bb73cdf2f7230427596"
            ),
        }
    )
    failed["run_ledger"].update(
        {
            "path": "experiment_results/pilot-v2.11.8/raw/run_ledger.json",
            "file_sha256": (
                "c9f4b2991428e00c367cde3dd770938ba9e88f87d4b68ae2a6d9f3d69a289628"
            ),
            "ledger_sha256": (
                "ab419bf9db32a9948b3ebac6d1ccd055d6e622e3a28a03ba1aae33f0564b7237"
            ),
            "event_count": 89,
            "event_head_sha256": (
                "99d7a1eafba692773d5eba830489770badaef484780480f813508287f127b97e"
            ),
            "registered_rows": 87,
            "status_counts": {"integrity-stopped": 87},
        }
    )
    failed["budget_ledger"].update(
        {
            "path": "experiment_results/pilot-v2.11.8/raw/budget_ledger.json",
            "file_sha256": (
                "893c486213f0f348a230e5b5a0996887d359e17d6b50dc3134cbce662f396df1"
            ),
            "ledger_sha256": (
                "341f2e448e2162895fc7a58870b629dda3ebaaad9add453d26fe031d430dc339"
            ),
            "event_count": 4,
            "event_head_sha256": (
                "0262f83823ebbcc13a813cc6415b2377b76630e3f99bb8eb444e642094ed0336"
            ),
            "current_actual": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "storage_bytes": 1772,
            },
        }
    )
    failed["stage_receipt"].update(
        {
            "path": (
                "experiment_results/pilot-v2.11.8/raw/parent-import/"
                "stage_receipt.json"
            ),
            "file_sha256": (
                "66394ac6547c4485a12a96187a4af5a1a5db1185eda07d10ac673c3ba920643f"
            ),
            "content_sha256": (
                "8bc3e8fb226273878429d944fe3db40c3002d8c204a51b16c52c8ed29c846e68"
            ),
            "failure_error_type": "V2118ParentImportIntegrityError",
            "failure_cause_type": "PilotV2118ContinuationError",
            "failure_message": (
                "V2.11.5 acceptance revalidation failed: scientific-dispatch "
                "acceptance field 'release' differs from source recomputation"
            ),
        }
    )
    failed["release_attestation"].update(
        {
            "path": ("experiment_results/pilot-v2.11.8/raw/release_attestation.json"),
            "file_sha256": (
                "839b7168ac365e5ad8e23005697facb312ed077b3be3f938107fe02f709ba0f0"
            ),
            "attestation_sha256": (
                "55fd978e7c2692ac80f138ed3837b0768167c9aced0e5f8ab4a5083d4c53b252"
            ),
        }
    )
    failed["scientific_launch_input"].update(
        {
            "path": (
                "experiment_results/pilot-v2.11.8/raw/scientific_launch_input.json"
            ),
            "file_sha256": (
                "08d1910ffec80269db0671f84596ca3d84017fdf40c9b967bd835a20b54ed278"
            ),
            "launch_input_sha256": (
                "21b31c327999bf3205d6aadf63707cbd088f2c745e6f65d0368b77c6a52d0af8"
            ),
        }
    )

    boundary["parent_budget_debit"].update(
        {
            "parent_contract_sha256": PILOT_CONTRACT_V2_11_8_CANONICAL_SHA256,
            "parent_run_ledger_sha256": (
                "ab419bf9db32a9948b3ebac6d1ccd055d6e622e3a28a03ba1aae33f0564b7237"
            ),
            "parent_budget_ledger_sha256": (
                "341f2e448e2162895fc7a58870b629dda3ebaaad9add453d26fe031d430dc339"
            ),
            "stage_bucket": "parent_v2118",
            "cost_usd": 63.1196450625,
            "hosted_completions": 3440,
            "storage_bytes": 270193500,
            "record_sha256": (
                "e5d18a013b0f2cd2faa4bf0d95c62c191a76ce8a0dcdff4a4d684e27956e42cd"
            ),
        }
    )
    boundary["continuation_budget"].update(
        {
            "parent_stage_bucket": "parent_v2118",
            "hosted_stage_bucket": "hosted_v2119",
            "fresh_registered_provider_calls": 3256,
            "fresh_projected_cost_usd": 149.3301875,
            "projected_cumulative_cost_usd": 212.4498325625,
            "projected_cumulative_hosted_completions": 6696,
            "projected_cumulative_storage_bytes": 1290193500,
        }
    )
    matrix = boundary["continuation_matrix"]
    matrix["a_c_reclassified_as_v2119"] = matrix.pop("a_c_reclassified_as_v2118")
    matrix["failed_v2118_rows_are_aborted_release_audit_only"] = matrix.pop(
        "failed_v2117_rows_are_aborted_release_audit_only"
    )
    matrix["canonical_86_row_mapping_sha256"] = (
        "7d958b7e1c9caf1ac7a2b019c534b6ff7b599b8ec420c2b9ab386ea678a70346"
    )
    immutability = boundary["immutability"]
    for key in (
        "v2117_no_go_raw_read_only",
        "v2117_statuses_preserved",
        "v2117_resume_forbidden",
        "v2117_failure_reclassification_forbidden",
    ):
        immutability.pop(key)
    immutability.update(
        {
            "v2118_no_go_raw_read_only": True,
            "v2118_statuses_preserved": True,
            "v2118_resume_forbidden": True,
            "v2118_failure_reclassification_forbidden": True,
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.9/raw",
        }
    )
    boundary["paid_provenance"]["required_annotated_tag"] = PILOT_CONTRACT_TAG_V2_11_9
    return boundary


def _validate_v2_11_9_recovery_boundary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    boundary = _mapping(value, "v2119_recovery_boundary")
    expected = _v2_11_9_expected_recovery_boundary(status=status)
    _strict_keys(boundary, required=set(expected), name="v2119_recovery_boundary")
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.9 recovery boundary drifted")
    if status == "frozen" and (
        PILOT_V2_11_9_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_9_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.9 source manifest hashes must be sealed before freezing"
        )
    return _freeze_json(boundary)


def _v2_11_10_expected_recovery_boundary(*, status: str) -> dict[str, Any]:
    """Bind V2.11.9's zero-completion no-go and repair P95 layering."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.11.10 recovery boundary status is invalid")
    boundary = _json_copy(_v2_11_9_expected_recovery_boundary(status="frozen"))
    boundary["schema_version"] = (
        "finevo-pilot-v2.11.10-p95-authority-layer-recovery-boundary-v1"
    )
    boundary["source_manifest"] = {
        "path": "experiments/pilot_v2_11_10_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.10-source-manifest-v1",
        "file_sha256": PILOT_V2_11_10_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": PILOT_V2_11_10_SOURCE_MANIFEST_CONTENT_SHA256,
    }
    boundary["runtime_input_binding"] = {
        "cwd_must_equal_release_root": True,
        "profile_path": "data/profiles.json",
        "profile_file_sha256": (
            "1bc90a92ef8e32f3da6e474f787207b79b1c82cc0b7b13c5ea3bd6cd1439b223"
        ),
        "profile_regular_non_symlink_required": True,
        "verification_points": [
            "scientific-dispatch-acceptance",
            "each-scientific-stage-before-provider-construction",
        ],
    }
    boundary["source_coverage"] = {
        "complete_verified_memory_python_tree": True,
        "complete_foundation_python_tree": True,
        "llm_provider_and_unique_cli": True,
        "release_renderers": True,
        "contract_module_normalization": "three-literal-v21110-cycle-pins-only",
        "ci_module_normalization": "two-literal-v21110-source-anchor-pins-only",
    }

    boundary["failed_release_no_go"] = {
        "contract_id": PILOT_CONTRACT_ID_V2_11_9,
        "contract_path": "experiments/pilot_v2_11_9.yaml",
        "contract_file_sha256": (
            "160f82b4bd57ba3ccc8ae711542ffc1d41f26503b48b80031b277b1f199c5d47"
        ),
        "contract_sha256": PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256,
        "science_tag": PILOT_CONTRACT_TAG_V2_11_9,
        "science_tag_object": "f0af244b64a69b3ee4571452df6d3611fd8c6220",
        "science_commit": "d850902af6218c72a6b0e71275c62c81c9143fb9",
        "source_manifest_path": "experiments/pilot_v2_11_9_source_manifest.json",
        "source_manifest_file_sha256": (
            "609adf9d12543b4caa7adb0cbddb8c8a9073a10f689adf52a8670608d16e9cb1"
        ),
        "source_manifest_content_sha256": (
            "36a790fe5edd6269218d6010046ec9293c3c418d8bc58a4dd5d89a6a70a547d6"
        ),
        "raw_inventory": {
            "root": "experiment_results/pilot-v2.11.9/raw",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "excluded_operational_paths": [".real-stage-execution.lock"],
            "file_count": 89,
            "storage_bytes": 1_219_091,
            "inventory_sha256": (
                "1f98a77a4262542e050003305600e07f0fef0e2e4e8eef8cc51a56fe9a8111bc"
            ),
        },
        "complete_raw_inventory": {
            "root": "experiment_results/pilot-v2.11.9/raw",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "excluded_operational_paths": [],
            "file_count": 90,
            "storage_bytes": 1_219_229,
            "inventory_sha256": (
                "268654caa5404b5e49729e418f4ebe1896f5b06bd52bba4de5a533021dd9e206"
            ),
        },
        "run_ledger": {
            "path": "experiment_results/pilot-v2.11.9/raw/run_ledger.json",
            "file_sha256": (
                "ea1dbe11bdb229a89eae21e37c58bbd47283599de17d8402edec952e39482cc1"
            ),
            "ledger_sha256": (
                "b2891fb152825cac846955b9c2fe4a041e80eab8cbebef9bc4d861d2313fc923"
            ),
            "event_count": 90,
            "event_head_sha256": (
                "3b255d0cfd972b5491f01588d3987006e5babc27ffafcff0dddd6c7c2f880bd6"
            ),
            "registered_rows": 87,
            "status_counts": {"complete": 1, "failed": 86},
        },
        "budget_ledger": {
            "path": "experiment_results/pilot-v2.11.9/raw/budget_ledger.json",
            "file_sha256": (
                "f4164290d4a66f56843bb3bb47bee2662de77fec98e911d601c51f9f06374267"
            ),
            "ledger_sha256": (
                "02adeb470b823664c67d09cd34df8787a68760e6270f46b59cca204701e3465d"
            ),
            "event_count": 77,
            "event_head_sha256": (
                "ab37a7c425d7b5e16d96dd6fcb75fa1af95cd93f96a7bb0bb2182857a53eda51"
            ),
            "owner_rows": 37,
            "status_counts": {"complete": 1, "failed": 36},
            "current_actual": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "storage_bytes": 800_162,
            },
        },
        "stage_receipts": {
            "parent-import": {
                "path": (
                    "experiment_results/pilot-v2.11.9/raw/parent-import/"
                    "stage_receipt.json"
                ),
                "file_sha256": (
                    "49caccf869b8515baea84f5e016a53bd8bacdbb22aebc85a61324cd9799a3da4"
                ),
                "content_sha256": (
                    "2d5e4cbb2e08ef231537646e2ad31a2f432a0349c585b3645d544abdf9126754"
                ),
                "status": "complete",
                "go": True,
                "execution_progression_go": True,
                "registered_run_count": 1,
                "status_counts": {"complete": 1},
                "scientific_matrix_complete": True,
            },
            "experiment-d": {
                "path": (
                    "experiment_results/pilot-v2.11.9/raw/experiment-d/"
                    "stage_receipt.json"
                ),
                "file_sha256": (
                    "5a66db97dbea5abb380a6ba43bdb632bf43c65226bb57b86b6fd4632185c2ff0"
                ),
                "content_sha256": (
                    "111d77c6525b5d17cf3ad65da45649221f7dd5a17207fcbdf68c979f01cc01b6"
                ),
                "status": "complete-with-no-go",
                "go": False,
                "execution_progression_go": True,
                "registered_run_count": 55,
                "status_counts": {"failed": 55},
                "scientific_matrix_complete": False,
            },
            "experiment-b": {
                "path": (
                    "experiment_results/pilot-v2.11.9/raw/experiment-b/"
                    "stage_receipt.json"
                ),
                "file_sha256": (
                    "1278621c8a8d10f7e4d03bcdd46eb4647bb24eb61f615a978650b711e6cc93c9"
                ),
                "content_sha256": (
                    "cc710beb91422372cec53e77996ad53572006c294b7b24efbcd5fdd47fca56f1"
                ),
                "status": "complete-with-no-go",
                "go": False,
                "execution_progression_go": True,
                "registered_run_count": 25,
                "status_counts": {"failed": 25},
                "scientific_matrix_complete": False,
            },
            "cross-model": {
                "path": (
                    "experiment_results/pilot-v2.11.9/raw/cross-model/"
                    "stage_receipt.json"
                ),
                "file_sha256": (
                    "fb25de2c70b619370b2e7a455e14bd17fe07f6a53b8480e716e29988000f928a"
                ),
                "content_sha256": (
                    "98d07a4db01d1e1011544e8b3e814471ec3e65f8121fa574520f07c02eed7e18"
                ),
                "status": "complete-with-no-go",
                "go": False,
                "execution_progression_go": False,
                "registered_run_count": 6,
                "status_counts": {"failed": 6},
                "scientific_matrix_complete": False,
            },
        },
        "scientific_dispatch_acceptance": {
            "path": (
                "experiment_results/pilot-v2.11.9/raw/"
                "scientific_dispatch_acceptance.json"
            ),
            "file_sha256": (
                "f9d49c118b319e7612c0e84c428e5fdfa1571c33915d160afa6f93036864be94"
            ),
            "content_sha256": (
                "2cac2d2b5f9f9e97054473c982e94df8ae0ed823852a03c39f1cc59d9f5a4e06"
            ),
            "status": "go",
            "go": True,
            "scientific_evidence": False,
        },
        "release_attestation": {
            "path": "experiment_results/pilot-v2.11.9/raw/release_attestation.json",
            "file_sha256": (
                "15a0a4b519466fc84fd08340e4643bd5a4b5fe7c7abdf38baee6601058397b38"
            ),
            "attestation_sha256": (
                "1b02503d7a7488b3d4b166413a5b4fd7ee376fe175e58cbff0b080d52ef765e9"
            ),
            "status": "pass",
        },
        "scientific_launch_input": {
            "path": (
                "experiment_results/pilot-v2.11.9/raw/scientific_launch_input.json"
            ),
            "file_sha256": (
                "c8d92522fc02ce7f2fb4905cdd28f6e6ce759eeb29cce7f9221b4adbd5369dfa"
            ),
            "launch_input_sha256": (
                "ad4bf87f65582f63ba3a675891b7cb9e2ef29a88ca2efa5e889a40461512c16d"
            ),
        },
        "acceptance_receipt_present": True,
        "science_reservations": 36,
        "provider_construction": False,
        "provider_calls": 0,
        "hosted_completions": 0,
        "scientific_evidence": False,
        "resume_forbidden": True,
        "failure_reclassification_forbidden": True,
        "failure_profile": {
            "failure_manifest_count": 36,
            "failed_scientific_cells": 86,
            "gpt52_failure_count": 80,
            "gpt56_failure_count": 6,
            "gpt52_message": (
                "source-backed observed p95 source authority differs for "
                "openai/gpt-5.2-2025-12-11::action"
            ),
            "gpt56_message": (
                "source-backed observed p95 source authority differs for "
                "openai/gpt-5.6-sol::action"
            ),
        },
    }

    boundary["parent_budget_debit"].update(
        {
            "parent_contract_sha256": PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256,
            "parent_run_ledger_sha256": (
                "b2891fb152825cac846955b9c2fe4a041e80eab8cbebef9bc4d861d2313fc923"
            ),
            "parent_budget_ledger_sha256": (
                "02adeb470b823664c67d09cd34df8787a68760e6270f46b59cca204701e3465d"
            ),
            "stage_bucket": "parent_v2119",
            "cost_usd": 63.1196450625,
            "hosted_completions": 3_440,
            "storage_bytes": 270_993_662,
            "record_sha256": (
                "5e0c39817c32c845c2f771a02320c55e85e9a6bfb5f3e705046b822593b4c592"
            ),
        }
    )
    boundary["continuation_budget"].update(
        {
            "parent_stage_bucket": "parent_v2119",
            "hosted_stage_bucket": "hosted_v21110",
            "fresh_registered_provider_calls": 3_256,
            "fresh_projected_cost_usd": 149.3301875,
            "projected_cumulative_cost_usd": 212.4498325625,
            "projected_cumulative_hosted_completions": 6_696,
            "projected_cumulative_storage_bytes": 1_290_993_662,
        }
    )
    matrix = boundary["continuation_matrix"]
    matrix["a_c_reclassified_as_v21110"] = matrix.pop("a_c_reclassified_as_v2119")
    matrix["failed_v2119_rows_are_aborted_release_audit_only"] = matrix.pop(
        "failed_v2118_rows_are_aborted_release_audit_only"
    )
    matrix["canonical_86_row_mapping_sha256"] = (
        "d876dbf6ae604a80e9cc6d29f857b944fbdcc58f0a6a279c85e88f5127468d15"
    )
    immutability = boundary["immutability"]
    for key in (
        "v2118_no_go_raw_read_only",
        "v2118_statuses_preserved",
        "v2118_resume_forbidden",
        "v2118_failure_reclassification_forbidden",
    ):
        immutability.pop(key)
    immutability.update(
        {
            "v2119_no_go_raw_read_only": True,
            "v2119_statuses_preserved": True,
            "v2119_resume_forbidden": True,
            "v2119_failure_reclassification_forbidden": True,
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.10/raw",
        }
    )
    boundary["paid_provenance"]["required_annotated_tag"] = PILOT_CONTRACT_TAG_V2_11_10
    return boundary


def _validate_v2_11_10_recovery_boundary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    boundary = _mapping(value, "v21110_recovery_boundary")
    expected = _v2_11_10_expected_recovery_boundary(status=status)
    _strict_keys(boundary, required=set(expected), name="v21110_recovery_boundary")
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.10 recovery boundary drifted")
    if status == "frozen" and (
        PILOT_V2_11_10_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_10_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.10 source manifest hashes must be sealed before freezing"
        )
    return _freeze_json(boundary)


def _v2_11_11_expected_dispatch_refresh() -> dict[str, Any]:
    """Return the exact, non-scientific 20-call dispatch refresh plan."""

    system = "Return exactly one JSON object and no surrounding text."
    definitions: list[dict[str, Any]] = []
    for call_kind in ("action", "semantic"):
        for index in range(5):
            probe_id = f"{call_kind}-{index}"
            expected = {
                "kind": call_kind,
                "probe_id": probe_id,
                "status": "ok",
            }
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"FinEvo V2.11.11 dispatch refresh probe {probe_id}. "
                        "Reply with exactly this JSON object: "
                        + json.dumps(expected, sort_keys=True, separators=(",", ":"))
                    ),
                },
            ]
            encoded = json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            definitions.append(
                {
                    "probe_id": probe_id,
                    "call_kind": call_kind,
                    "messages": messages,
                    "messages_sha256": hashlib.sha256(encoded).hexdigest(),
                    # Bytes conservatively upper-bound BPE tokens.
                    "prompt_token_upper_bound": len(encoded),
                    "expected_json": expected,
                    "expected_json_sha256": hashlib.sha256(
                        json.dumps(
                            expected,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )

    prices = {
        "gpt52_main": {"input": 1.75, "output": 14.0},
        "gpt56_diagnostic": {"input": 5.0, "output": 30.0},
    }
    caps = {"action": 8_192, "semantic": 4_096}
    rows: list[dict[str, Any]] = []
    for profile_id in ("gpt52_main", "gpt56_diagnostic"):
        for definition in definitions:
            call_kind = str(definition["call_kind"])
            prompt_bound = int(definition["prompt_token_upper_bound"])
            full_cap_cost = (
                prompt_bound * prices[profile_id]["input"]
                + caps[call_kind] * prices[profile_id]["output"]
            ) / 1_000_000.0
            rows.append(
                {
                    "run_id": (
                        f"{PILOT_CONTRACT_ID_V2_11_11}--dispatch-refresh--"
                        f"{profile_id}--{definition['probe_id']}"
                    ),
                    "profile_id": profile_id,
                    "probe_id": definition["probe_id"],
                    "call_kind": call_kind,
                    "max_completion_tokens": caps[call_kind],
                    "prompt_token_upper_bound": prompt_bound,
                    "reserved_cost_usd": round(full_cap_cost * 1.25, 12),
                }
            )
    reserve = round(sum(float(row["reserved_cost_usd"]) for row in rows), 12)
    return {
        "schema_version": "finevo-pilot-v2.11.11-dispatch-refresh-plan-v1",
        "evidence_class": "operational",
        "scientific_evidence": False,
        "claim_boundary": "identity-interface-length-cap-refresh-not-p95",
        "denominator_scope": "independent-non-scientific-authority-ledger",
        "model_profiles": ["gpt52_main", "gpt56_diagnostic"],
        "call_kinds": ["action", "semantic"],
        "probes_per_model_call_kind": 5,
        "provider_calls": 20,
        "max_attempts_per_call": 1,
        "hosted_max_in_flight": 1,
        "service_tier": "default",
        "short_context_prompt_token_ceiling": 272_000,
        "reserve_multiplier": 1.25,
        "cached_input_discount_assumed": False,
        "failure_policy": "global-science-no-go-no-retry-no-replacement",
        "required_finish_reason": "stop",
        "required_response_completed": True,
        "required_exact_json": True,
        "definitions": definitions,
        "rows": rows,
        "reserved_cost_usd": reserve,
    }


def _v2_11_11_expected_fresh_cohort_boundary(*, status: str) -> dict[str, Any]:
    """Bind the immutable evidence roots and the exact fresh-cohort envelope."""

    if status not in {"draft", "frozen"}:
        raise PilotContractError("V2.11.11 fresh-cohort status is invalid")
    dispatch_refresh = _v2_11_11_expected_dispatch_refresh()
    refresh_cost = float(dispatch_refresh["reserved_cost_usd"])
    parent_cost = 78.3237413125
    science_cost = 404.46984
    parent_completions = 4_192
    refresh_completions = 20
    science_completions = 3_256
    return {
        "schema_version": "finevo-pilot-v2.11.11-fresh-cohort-boundary-v1",
        "source_manifest": {
            "path": "experiments/pilot_v2_11_11_source_manifest.json",
            "schema_version": "finevo-pilot-v2.11.11-source-manifest-v1",
            "file_sha256": PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256,
            "content_sha256": PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256,
        },
        "v2115_scientific_authority": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_5,
            "science_tag": PILOT_CONTRACT_TAG_V2_11_5,
            "science_tag_object": "bccfb13cee7d592470d1873cfacc3b12bed38be4",
            "science_commit": "2351ac2283f9fedb9dce70067174020be56ed9cc",
            "contract_sha256": (
                "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
            ),
            "contract_file_sha256": (
                "b96438430231f0c46fd6c5f15ba749713534feb15f964c496aa02606cf11103b"
            ),
            "raw_inventory": {
                "file_count": 691,
                "storage_bytes": 48_820_556,
                "inventory_sha256": (
                    "f2fdb1ccedcb70e6793d3b8f3c87425f0d602552f0a3e0e7f35db9c5777c6746"
                ),
            },
            "publication_checksums_file_sha256": (
                "1c57592cc14689eee3ed9832996cacb3a4edde764647e1d704f5765fe2920576"
            ),
            "publication_package_manifest_file_sha256": (
                "99d4db05cc4dbfd2b9339f9034748396d36f49d6db46f91c2b73970388d3b333"
            ),
            "read_only": True,
        },
        "v21110_terminal_release": {
            "contract_id": PILOT_CONTRACT_ID_V2_11_10,
            "science_tag": PILOT_CONTRACT_TAG_V2_11_10,
            "science_tag_object": "7c3de14ddb604436a1dfee1dabfb781849ea68a7",
            "science_commit": "aa05e94ee6097916db53c40c0276b044c05a44cc",
            "contract_sha256": (
                "76a03b7781b1bea317855010d9a3b34b49fcfba3f27cc344954daf19abcd2b1f"
            ),
            "contract_file_sha256": (
                "13ef71917db5d73b6793e359d519c1e066de81c5bd3f3699650d426c2068f6fd"
            ),
            "raw_inventory": {
                "excluded_operational_paths": [".real-stage-execution.lock"],
                "file_count": 77,
                "storage_bytes": 10_294_111,
                "inventory_sha256": (
                    "bf033f4e37e5ed721d1ff832d97a0af3ba13b9d90e9e7c27dec8c0d09288ecad"
                ),
            },
            "run_ledger": {
                "file_sha256": (
                    "3a1a4c431e52e7dda20b4fdd5064130ba2b1cc8dc10ff42596e32ab6aa281a44"
                ),
                "ledger_sha256": (
                    "a483012b15644fc831459c1f7b64ffea753c72d697b5ee25e2aad3483cc25282"
                ),
                "registered_rows": 87,
                "status_counts": {
                    "complete": 2,
                    "failed": 55,
                    "integrity-stopped": 30,
                },
            },
            "budget_ledger": {
                "file_sha256": (
                    "331e5f880a76fb7bc12e64dd7e47eb813e670da450cc5bd48099ea7d11f394d3"
                ),
                "ledger_sha256": (
                    "e8819ea65b94befedbb4c96a301b8a82eb0b5c597e70748d6267c58bf963216f"
                ),
                "current_actual": {
                    "cost_usd": 15.20409625,
                    "hosted_completions": 752,
                    "storage_bytes": 9_951_755,
                },
            },
            "publication_checksums_file_sha256": (
                "5ef502757af0a0c07c3ba13ecd93bd0cf6d1fd24dc937cd04e7105ae18879dfb"
            ),
            "publication_package_manifest_file_sha256": (
                "db8a54dd08b09fc8edc7866dfc7c1495fff2853feb1bae6903fc874608e28907"
            ),
            "resume_forbidden": True,
            "failure_reclassification_forbidden": True,
            "effect_import_forbidden": True,
            "read_only": True,
        },
        "parent_budget_debit": {
            "schema_version": "finevo-parent-budget-debit-v1",
            "parent_contract_sha256": (
                "76a03b7781b1bea317855010d9a3b34b49fcfba3f27cc344954daf19abcd2b1f"
            ),
            "parent_run_ledger_sha256": (
                "a483012b15644fc831459c1f7b64ffea753c72d697b5ee25e2aad3483cc25282"
            ),
            "parent_budget_ledger_sha256": (
                "e8819ea65b94befedbb4c96a301b8a82eb0b5c597e70748d6267c58bf963216f"
            ),
            "stage_bucket": "parent_v21110",
            "cost_usd": 78.3237413125,
            "hosted_completions": 4_192,
            "storage_bytes": 280_945_417,
            "record_sha256": (
                "72e9a0440ae2df3595a6c2bd8897b000298ebf3dc29b3889c104657257e8da70"
            ),
        },
        "fresh_cohort": {
            "fresh_seed_set": [
                877_361,
                1_410_637_959,
                416_755_402,
                357_136_200,
                1_541_219_789,
            ],
            "seed_overlap_with_v2115_or_v21110": 0,
            "scientific_cells": 86,
            "operational_cells": 1,
            "calls_by_stage": {
                "experiment-b": 1_440,
                "experiment-d": 1_480,
                "cross-model": 336,
            },
            "simulated_provider_calls": 3_256,
            "acceptance_provider_calls": 0,
            "lineage_registered_denominator": 222,
            "lineage_scientific_denominator": 217,
            "old_cells_retried_or_replaced": False,
        },
        "budget_envelope": {
            "hard_cap_usd": 500.0,
            "hard_completion_cap": 7_500,
            "parent_cost_usd": 78.3237413125,
            "parent_hosted_completions": 4_192,
            "fresh_full_cap_reserve_usd": {
                "experiment-b": 155.02914,
                "experiment-d": 170.47086,
                "cross-model": 78.96984,
                "total": 404.46984,
            },
            "dispatch_refresh_full_cap_reserve_usd": refresh_cost,
            "projected_cumulative_cost_usd": round(
                parent_cost + refresh_cost + science_cost, 12
            ),
            "projected_cumulative_hosted_completions": (
                parent_completions + refresh_completions + science_completions
            ),
            "remaining_cost_usd": round(
                500.0 - parent_cost - refresh_cost - science_cost, 12
            ),
            "remaining_hosted_completions": (
                7_500 - parent_completions - refresh_completions - science_completions
            ),
            "price_and_served_identity_refresh_after_acceptance_before_science": True,
            "refresh_must_fit_without_matrix_shrink": True,
        },
        "dispatch_refresh": dispatch_refresh,
        "execution_policy": {
            "actor_max_completion_tokens": 8_192,
            "semantic_max_completion_tokens": 4_096,
            "hosted_max_in_flight": 1,
            "request_timeout_seconds": 300,
            "provider_attempts_per_request": 1,
            "experiment_d_prefix_failure_scope": "all-eleven-seed-cells",
            "experiment_d_branch_failure_scope": "single-branch-cell",
            "experiment_d_branch_resume": "untouched-branches-only",
            "cross_stage_failure_propagation": False,
        },
        "evidence_partition": {
            "source": "contract-stage-evidence_class",
            "classes": {
                "operational": ["parent-import"],
                "scientific": ["experiment-b", "experiment-d", "cross-model"],
            },
            "hard_coded_stage_partition_forbidden": True,
        },
        "immutability": {
            "v2115_raw_and_evidence_read_only": True,
            "v21110_raw_and_evidence_read_only": True,
            "new_raw_namespace_required": "experiment_results/pilot-v2.11.11/raw",
            "decoded_completion_reuse": False,
            "partial_branch_reuse": False,
            "failed_seed_replacement": False,
        },
        "paid_provenance": {
            "required_annotated_tag": PILOT_CONTRACT_TAG_V2_11_11,
            "clean_main_required": True,
            "provider_credentials_loaded_after_acceptance_only": True,
        },
    }


def _v2_11_11_expected_seed_generation() -> dict[str, Any]:
    """Replay the locally recorded pre-dispatch SHA-256 seed derivation."""

    salt = "finevo-pilot-v2.11.11|fresh-seed-v1"
    historical = [
        617806385,
        760687867,
        959809858,
        1099057501,
        1421875452,
        1769977770,
        1942013315,
        2010922376,
    ]
    exclusion = set(historical)
    accepted: list[dict[str, Any]] = []
    seen: set[int] = set()
    counter = 0
    while len(accepted) < 5:
        preimage = f"{salt}|main|{counter}"
        digest = hashlib.sha256(preimage.encode("utf-8")).digest()
        candidate = int.from_bytes(digest[:8], "big") % 2_147_483_647
        rejected_reason: Optional[str] = None
        if candidate == 0:
            rejected_reason = "zero"
        elif candidate in seen:
            rejected_reason = "duplicate"
        elif candidate in exclusion:
            rejected_reason = "historical-exclusion"
        if rejected_reason is None:
            seen.add(candidate)
        accepted.append(
            {
                "counter": counter,
                "preimage": preimage,
                "digest_sha256": digest.hex(),
                "candidate": candidate,
                "status": "accepted" if rejected_reason is None else "rejected",
                "rejected_reason": rejected_reason,
            }
        )
        counter += 1
        if rejected_reason is not None:
            accepted.pop()

    values = [int(item["candidate"]) for item in accepted]
    preflight_preimage = f"{salt}|preflight|0"
    preflight_digest = hashlib.sha256(preflight_preimage.encode("utf-8")).digest()
    preflight_candidate = int.from_bytes(preflight_digest[:8], "big") % 2_147_483_647
    return {
        "method": "sha256-counter-v1",
        "provenance_class": (
            "pre-dispatch locally recorded, agent-proposed SHA-256 derivation"
        ),
        "recorded_at": "2026-08-01T17:09:40.387Z",
        "timing_boundary": (
            "after V2.11.5/V2.11.10 results and before any V2.11.11 raw/run"
        ),
        "salt": salt,
        "stream": "main",
        "preimage_format": "{salt}|{stream}|{counter}",
        "encoding": "UTF-8",
        "newline": False,
        "digest": "SHA-256",
        "digest_slice": "digest[0:8]",
        "integer_encoding": "unsigned-big-endian",
        "modulus": 2_147_483_647,
        "valid_range": [1, 2_147_483_646],
        "counter_start": 0,
        "counter_increment_after_every_candidate": True,
        "rejection_rules": ["zero", "duplicate", "historical-exclusion"],
        "generated_before_results": True,
        "values": values,
        "derivation_trace": accepted,
        "historical_seed_registry": historical,
        "historical_seed_registry_sha256": canonical_sha256(historical),
        "fresh_values_overlap_historical_registry": [],
        "unused_preflight_candidate": {
            "denominator_role": "unused-non-denominator",
            "preimage": preflight_preimage,
            "digest_sha256": preflight_digest.hex(),
            "candidate": preflight_candidate,
        },
        "random_sampling_claimed": False,
        "public_preregistration_claimed": False,
        "user_selected_claimed": False,
        "claim_boundary": (
            "This exact replay proves a locally recorded seed derivation before "
            "any V2.11.11 dispatch or result. It is not a public preregistration, "
            "was not user-selected, was recorded after earlier V2.11.5/V2.11.10 "
            "results, and does not prove random sampling."
        ),
    }


def _validate_v2_11_11_fresh_cohort_boundary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    boundary = _mapping(value, "v21111_fresh_cohort_boundary")
    expected = _v2_11_11_expected_fresh_cohort_boundary(status=status)
    _strict_keys(
        boundary,
        required=set(expected),
        name="v21111_fresh_cohort_boundary",
    )
    if _json_copy(boundary) != expected:
        raise PilotContractError("V2.11.11 fresh-cohort boundary drifted")
    if status == "frozen" and (
        PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256 is None
        or PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256 is None
    ):
        raise PilotContractError(
            "V2.11.11 source manifest hashes must be sealed before freezing"
        )
    return _freeze_json(boundary)


def _v2_11_expected_model_roles() -> dict[str, Any]:
    hosted_roles = [
        "capability-choice",
        "capability-proposal",
        "actor-action",
        "semantic-proposal",
    ]
    return {
        "gpt52_main": {
            "profile_id": "gpt52_main",
            "role": "primary",
            "dispatch_eligible": True,
            "ineligibility_reason": None,
            "allowed_stages": [
                "capability-gate",
                "long-context-preflight",
                "experiment-c",
                "experiment-a",
                "experiment-d",
                "experiment-b",
            ],
            "allowed_call_roles": [
                *hosted_roles,
                "offline-verifier",
                "checkpoint-branch",
            ],
        },
        "gpt56_diagnostic": {
            "profile_id": "gpt56_diagnostic",
            "role": "secondary_diagnostic",
            "dispatch_eligible": True,
            "ineligibility_reason": None,
            "allowed_stages": [
                "capability-gate",
                "long-context-preflight",
                "cross-model",
            ],
            "allowed_call_roles": hosted_roles,
        },
        "qref_scripted": {
            "profile_id": "qref_scripted",
            "role": "calibration_only",
            "dispatch_eligible": True,
            "ineligibility_reason": None,
            "allowed_stages": ["parent-import"],
            "allowed_call_roles": ["parent-authority-import"],
        },
    }


def _v2_11_stage(
    stage_id: str,
    *,
    model_id: str,
    arms: Sequence[str],
    seed_set: str,
    prerequisites: Sequence[str],
    call_roles: Sequence[str],
    execution_mode: str = "actor_run",
    narratives: Sequence[str] = ("none",),
    budget_bucket: str = "hosted_v211",
    num_agents: int = 4,
    episode_length: int = 12,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "enabled": True,
        "budget_bucket": budget_bucket,
        "num_agents": num_agents,
        "episode_length": episode_length,
        "seed_set": seed_set,
        "utility_profiles": [
            (
                "provider-preflight-default"
                if stage_id in {"parent-import", "capability-gate"}
                else "stage0-selected"
            )
        ],
        "shock_id": (
            "baseline-3pct"
            if stage_id in {"parent-import", "capability-gate"}
            else "registered-rate-shock"
        ),
        "cells": [
            {
                "models": [model_id],
                "arms": list(arms),
                "narratives": list(narratives),
                "execution_mode": execution_mode,
            }
        ],
        "prerequisites": list(prerequisites),
        "reuse": [],
        "call_roles": list(call_roles),
    }


def _v2_11_expected_stages() -> list[dict[str, Any]]:
    parent = _v2_11_stage(
        "parent-import",
        budget_bucket="parent_v2102",
        model_id="qref_scripted",
        arms=["parent-import"],
        seed_set="preflight",
        prerequisites=[],
        call_roles=["parent-authority-import"],
        execution_mode="parent_authority_import",
        num_agents=2,
        episode_length=1,
    )
    capability = _v2_11_stage(
        "capability-gate",
        model_id="gpt52_main",
        arms=["capability-probe"],
        seed_set="preflight",
        prerequisites=["parent-import"],
        call_roles=["capability-choice", "capability-proposal"],
        execution_mode="capability_probe",
        num_agents=2,
        episode_length=1,
    )
    capability["cells"][0]["models"] = [
        "gpt52_main",
        "gpt56_diagnostic",
    ]
    preflight = _v2_11_stage(
        "long-context-preflight",
        model_id="gpt52_main",
        arms=["closed-loop-preflight"],
        seed_set="preflight",
        prerequisites=["capability-gate"],
        call_roles=["actor-action", "semantic-proposal"],
        execution_mode="closed_loop_preflight",
        num_agents=2,
    )
    preflight["cells"][0]["models"] = [
        "gpt52_main",
        "gpt56_diagnostic",
    ]
    experiment_c = _v2_11_stage(
        "experiment-c",
        model_id="gpt52_main",
        arms=[
            "full",
            "unverified-dual",
            "verified-error-candidate",
            "verified-error-forced",
            "unverified-error-forced",
        ],
        seed_set="main",
        prerequisites=["long-context-preflight"],
        call_roles=[
            "actor-action",
            "semantic-proposal",
            "offline-verifier",
        ],
    )
    experiment_a = _v2_11_stage(
        "experiment-a",
        model_id="gpt52_main",
        arms=["no-context", "prompt-only", "retrieval-only", "full"],
        seed_set="main",
        prerequisites=["experiment-c"],
        call_roles=["actor-action", "semantic-proposal"],
    )
    experiment_d = _v2_11_stage(
        "experiment-d",
        model_id="gpt52_main",
        arms=[
            "matched-a",
            "matched-b",
            "no-memory",
            "shuffled-episodic",
            "wrong-context",
            "error-verified",
            "error-unverified",
        ],
        seed_set="main",
        prerequisites=["experiment-c", "experiment-a"],
        call_roles=[
            "actor-action",
            "semantic-proposal",
            "checkpoint-branch",
        ],
        execution_mode="checkpoint_continuation",
    )
    experiment_d["cells"].append(
        {
            "models": ["gpt52_main"],
            "arms": ["narrative-content"],
            "narratives": ["none", "aligned", "paraphrase", "opposite"],
            "execution_mode": "checkpoint_continuation",
        }
    )
    experiment_b = _v2_11_stage(
        "experiment-b",
        model_id="gpt52_main",
        arms=[
            "no-memory",
            "episodic-only",
            "semantic-only",
            "unverified-dual",
            "full",
        ],
        seed_set="main",
        prerequisites=["experiment-d"],
        call_roles=["actor-action", "semantic-proposal"],
    )
    cross_model = _v2_11_stage(
        "cross-model",
        model_id="gpt56_diagnostic",
        arms=["full", "no-memory"],
        seed_set="cross-model",
        prerequisites=["long-context-preflight", "experiment-b"],
        call_roles=["actor-action", "semantic-proposal"],
    )
    return [
        parent,
        capability,
        preflight,
        experiment_c,
        experiment_a,
        experiment_d,
        experiment_b,
        cross_model,
    ]


def _v2_11_1_expected_stages() -> list[dict[str, Any]]:
    """V2.11 matrix with imported capability cells and a fresh budget namespace."""

    stages = _json_copy(_v2_11_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v211"
            if stage["stage_id"] in {"parent-import", "capability-gate"}
            else "hosted_v2111"
        )
        if stage["stage_id"] == "capability-gate":
            stage["cells"][0]["execution_mode"] = "capability_authority_import"
            stage["call_roles"] = ["parent-authority-import"]
    return stages


def _v2_11_2_expected_stages() -> list[dict[str, Any]]:
    """Unchanged V2.11.1 matrix under a fresh release and budget namespace."""

    stages = _json_copy(_v2_11_1_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v2111"
            if stage["stage_id"] in {"parent-import", "capability-gate"}
            else "hosted_v2112"
        )
    return stages


def _v2_11_3_expected_stages() -> list[dict[str, Any]]:
    """V2.11.2 science matrix with zero-provider operational imports."""

    stages = _json_copy(_v2_11_2_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v2112"
            if stage["stage_id"] in {"parent-import", "capability-gate"}
            else "hosted_v2113"
        )
        if stage["stage_id"] == "long-context-preflight":
            stage["cells"][0]["execution_mode"] = "preflight_authority_import"
            stage["call_roles"] = ["parent-authority-import"]
    return stages


def _v2_11_4_expected_stages() -> list[dict[str, Any]]:
    """Unchanged V2.11.3 denominator under a fresh no-go-preserving namespace."""

    stages = _json_copy(_v2_11_3_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v2113"
            if stage["stage_id"] in {"parent-import", "capability-gate"}
            else "hosted_v2114"
        )
    return stages


def _v2_11_5_expected_stages() -> list[dict[str, Any]]:
    """Unchanged V2.11.4 denominator under the V2.11.5 budget namespace."""

    stages = _json_copy(_v2_11_4_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v2114"
            if stage["stage_id"] in {"parent-import", "capability-gate"}
            else "hosted_v2115"
        )
    return stages


def _v2_11_6_expected_stages() -> list[dict[str, Any]]:
    """The exact unstarted V2.11.5 D/B/cross rows plus one prefix import."""

    parent_stages = {
        stage["stage_id"]: stage for stage in _json_copy(_v2_11_5_expected_stages())
    }
    result: list[dict[str, Any]] = []
    for stage_id in (
        "parent-import",
        "experiment-d",
        "experiment-b",
        "cross-model",
    ):
        stage = parent_stages[stage_id]
        stage["budget_bucket"] = (
            "parent_v2115" if stage_id == "parent-import" else "hosted_v2116"
        )
        stage["prerequisites"] = {
            "parent-import": [],
            "experiment-d": ["parent-import"],
            "experiment-b": ["experiment-d"],
            "cross-model": ["experiment-b"],
        }[stage_id]
        result.append(stage)
    return result


def _v2_11_7_expected_stages() -> list[dict[str, Any]]:
    """The same 86 logical science cells in the prospective recovery namespace."""

    stages = _json_copy(_v2_11_6_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v2116" if stage["stage_id"] == "parent-import" else "hosted_v2117"
        )
    return stages


def _v2_11_8_expected_stages() -> list[dict[str, Any]]:
    """Retain all 86 science rows under the fresh recovery budget buckets."""

    stages = _json_copy(_v2_11_7_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v2117" if stage["stage_id"] == "parent-import" else "hosted_v2118"
        )
    return stages


def _v2_11_9_expected_stages() -> list[dict[str, Any]]:
    """Retain all 86 science rows under the release-binding recovery buckets."""

    stages = _json_copy(_v2_11_8_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v2118" if stage["stage_id"] == "parent-import" else "hosted_v2119"
        )
    return stages


def _v2_11_10_expected_stages() -> list[dict[str, Any]]:
    """Retain the 86 science rows under fresh V2.11.10 budget buckets."""

    stages = _json_copy(_v2_11_9_expected_stages())
    for stage in stages:
        stage["budget_bucket"] = (
            "parent_v2119" if stage["stage_id"] == "parent-import" else "hosted_v21110"
        )
    return stages


def _v2_11_11_expected_stages() -> list[dict[str, Any]]:
    """Fresh B/D/cross cohort with contract-declared evidence partitions.

    Every scientific stage depends only on the zero-provider lineage import.
    This prevents an integrity or provider failure in one experiment family
    from terminalizing untouched cells in another family.
    """

    parent = {
        stage["stage_id"]: stage for stage in _json_copy(_v2_11_10_expected_stages())
    }
    result: list[dict[str, Any]] = []
    for stage_id in ("parent-import", "experiment-b", "experiment-d", "cross-model"):
        stage = parent[stage_id]
        stage["budget_bucket"] = (
            "parent_v21110" if stage_id == "parent-import" else "hosted_v21111"
        )
        stage["prerequisites"] = (
            [] if stage_id == "parent-import" else ["parent-import"]
        )
        stage["evidence_class"] = (
            "operational" if stage_id == "parent-import" else "scientific"
        )
        result.append(stage)
    return result


def _v2_11_3_expected_preflight_arm() -> dict[str, Any]:
    return {
        "arm_id": "closed-loop-preflight",
        "execution_mode": "preflight_authority_import",
        "family": "preflight-authority-import",
        "parameters": {
            "source_contract_id": PILOT_CONTRACT_ID_V2_11_2,
            "source_authority_schema_version": (
                "finevo-pilot-v2.11.2-post-gate-authority-v1"
            ),
            "provider_construction": False,
            "provider_calls": 0,
            "fresh_samples": 0,
            "historical_action_samples_per_model": 24,
            "historical_semantic_samples_per_model": 8,
            "authority_use": "dispatch-reservation-only",
            "scientific_evidence": False,
        },
    }


def _v2_11_4_expected_preflight_arm() -> dict[str, Any]:
    """Reuse the immutable V2.11.2 reservation authority with zero calls."""

    return _json_copy(_v2_11_3_expected_preflight_arm())


def _v2_11_5_expected_preflight_arm() -> dict[str, Any]:
    """Reuse the immutable V2.11.2 reservation payloads with zero calls."""

    return _json_copy(_v2_11_4_expected_preflight_arm())


def _v2_11_6_expected_parent_import_arm() -> dict[str, Any]:
    """Declare continuation without copying terminal parent rows as child rows."""

    return {
        "arm_id": "parent-import",
        "execution_mode": "parent_authority_import",
        "family": "parent-authority-import",
        "parameters": {
            "parent_artifacts_read_only": True,
            "parent_denominator_continued": True,
            "terminal_parent_rows_imported_as_child_rows": False,
            "mapped_scheduled_parent_rows": 86,
            "provider_calls": 0,
            "scientific_evidence": False,
        },
    }


def _v2_11_7_expected_parent_import_arm() -> dict[str, Any]:
    """Authenticate both releases without turning either audit row into evidence."""

    return {
        "arm_id": "parent-import",
        "execution_mode": "parent_authority_import",
        "family": "parent-authority-import",
        "parameters": {
            "failed_release_artifacts_read_only": True,
            "authority_artifacts_read_only": True,
            "authority_denominator_continued": True,
            "failed_release_rows_imported_as_science": False,
            "terminal_authority_rows_imported_as_child_rows": False,
            "mapped_never_dispatched_authority_rows": 86,
            "mapping_key_fields": ["source_run_id", "logical_cell_sha256"],
            "cross_release_logical_deduplication_required": True,
            "provider_calls": 0,
            "scientific_evidence": False,
        },
    }


def _v2_11_8_expected_parent_import_arm() -> dict[str, Any]:
    """Keep the V2.11.7 dual-lineage import semantics unchanged."""

    return _json_copy(_v2_11_7_expected_parent_import_arm())


def _v2_11_9_expected_parent_import_arm() -> dict[str, Any]:
    """Keep the V2.11.8 dual-lineage import semantics unchanged."""

    return _json_copy(_v2_11_8_expected_parent_import_arm())


def _v2_11_10_expected_parent_import_arm() -> dict[str, Any]:
    """Keep the dual-lineage import semantics unchanged."""

    return _json_copy(_v2_11_9_expected_parent_import_arm())


def _v2_11_11_expected_parent_import_arm() -> dict[str, Any]:
    """Bind both immutable releases without importing an old effect cell."""

    return {
        "arm_id": "parent-import",
        "execution_mode": "parent_authority_import",
        "family": "parent-authority-import",
        "parameters": {
            "v2115_authority_artifacts_read_only": True,
            "v21110_terminal_artifacts_read_only": True,
            "v21110_rows_imported_as_science": False,
            "v21110_outcomes_reused": False,
            "fresh_scientific_cells": 86,
            "old_failed_cells_retried_or_replaced": False,
            "provider_calls": 0,
            "scientific_evidence": False,
        },
    }


def _v2_11_1_expected_model_roles() -> dict[str, Any]:
    roles = _json_copy(_v2_11_expected_model_roles())
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        allowed = roles[model_id]["allowed_call_roles"]
        if "parent-authority-import" not in allowed:
            allowed.append("parent-authority-import")
    return roles


def _v2_11_2_expected_model_roles() -> dict[str, Any]:
    return _json_copy(_v2_11_1_expected_model_roles())


def _v2_11_3_expected_model_roles() -> dict[str, Any]:
    return _json_copy(_v2_11_2_expected_model_roles())


def _v2_11_4_expected_model_roles() -> dict[str, Any]:
    return _json_copy(_v2_11_3_expected_model_roles())


def _v2_11_5_expected_model_roles() -> dict[str, Any]:
    return _json_copy(_v2_11_4_expected_model_roles())


def _v2_11_6_expected_model_roles() -> dict[str, Any]:
    """Retain model/call capabilities while limiting them to continuation stages."""

    roles = _json_copy(_v2_11_5_expected_model_roles())
    roles["gpt52_main"]["allowed_stages"] = ["experiment-d", "experiment-b"]
    roles["gpt56_diagnostic"]["allowed_stages"] = ["cross-model"]
    roles["qref_scripted"]["allowed_stages"] = ["parent-import"]
    return roles


def _v2_11_7_expected_model_roles() -> dict[str, Any]:
    """Keep the V2.11.6 role and stage surface unchanged."""

    return _json_copy(_v2_11_6_expected_model_roles())


def _v2_11_8_expected_model_roles() -> dict[str, Any]:
    """Keep the V2.11.7 role and stage surface unchanged."""

    return _json_copy(_v2_11_7_expected_model_roles())


def _v2_11_9_expected_model_roles() -> dict[str, Any]:
    """Keep the V2.11.8 role and stage surface unchanged."""

    return _json_copy(_v2_11_8_expected_model_roles())


def _v2_11_10_expected_model_roles() -> dict[str, Any]:
    """Keep the V2.11.9 role and stage surface unchanged."""

    return _json_copy(_v2_11_9_expected_model_roles())


def _v2_11_11_expected_model_roles() -> dict[str, Any]:
    """Keep the two hosted roles while admitting only fresh-cohort stages."""

    return _json_copy(_v2_11_10_expected_model_roles())


def _v2_11_expected_non_claims() -> list[str]:
    return [
        (
            "V2.10.2 remains an immutable complete-with-no-go denominator; "
            "no V2.10.2 treatment effect or P95 authority is imported."
        ),
        (
            "The parent-import cell performs zero provider calls and imports "
            "only hash-bound calibration inputs and cumulative budget debit."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated "
            "weight snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The 4-agent by 12-month mechanism micro-pilot is not the "
            "10x24x5 confirmatory pilot and does not support 100x240 claims."
        ),
        "No paid V2.11 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_1_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11 remains an immutable terminal preflight no-go denominator; "
            "its failed preflight cells are not retried or deleted."
        ),
        (
            "V2.11.1 imports only hash-bound calibration, capability, failure, "
            "and cumulative-budget authorities with zero provider calls and no "
            "treatment-effect cells."
        ),
        (
            "The contract-envelope reservation is a conservative operational "
            "bootstrap for long-context preflight, not an observed P95 authority "
            "and not scientific evidence."
        ),
        (
            "All post-preflight scientific dispatch still requires a newly sealed "
            "per-model, per-call-role observed P95 plus 25 percent headroom."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated "
            "weight snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The 4-agent by 12-month mechanism micro-pilot is not the "
            "10x24x5 confirmatory pilot and does not support 100x240 claims."
        ),
        "No paid V2.11.1 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_2_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.1 remains an immutable terminal preflight no-go denominator; "
            "its two failed cells and all 64 provider calls are retained and are "
            "never retried in the V2.11.1 namespace."
        ),
        (
            "V2.11.2 imports only hash-bound calibration and capability wrappers "
            "with zero provider calls; V2.11.1 failed-preflight journals are "
            "budget and failure-audit evidence only."
        ),
        (
            "No V2.11.1 preflight sample, checkpoint, exactness receipt, provider "
            "journal, bootstrap projection, or P95 authority is reclassified as "
            "V2.11.2 scientific or dispatch evidence."
        ),
        (
            "The lifecycle repair corrects validation of an already-active rule "
            "inside the frozen admission-retirement hysteresis band; it does not "
            "change prompts, thresholds, seeds, arms, models, environment, or metrics."
        ),
        (
            "The contract-envelope reservation is a conservative operational "
            "bootstrap for fresh V2.11.2 long-context preflight, not an observed "
            "P95 authority and not scientific evidence."
        ),
        (
            "All post-preflight scientific dispatch still requires a newly sealed "
            "V2.11.2 per-model, per-call-role observed P95 plus 25 percent headroom."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated "
            "weight snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The 4-agent by 12-month mechanism micro-pilot is not the "
            "10x24x5 confirmatory pilot and does not support 100x240 claims."
        ),
        "No paid V2.11.2 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_3_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.2 remains an immutable 136-cell terminal denominator with 10 "
            "complete and 126 failed cells; no failed cell is resumed, deleted, or "
            "reclassified in the V2.11.2 namespace."
        ),
        (
            "V2.11.3 imports five hash-bound operational authority cells with zero "
            "provider calls; the V2.11.2 preflight authority is dispatch-reservation "
            "evidence only, not treatment-effect or scientific evidence."
        ),
        (
            "No V2.11.2 scientific cell, provider completion, action outcome, "
            "checkpoint continuation, or treatment-effect metric is reused as a "
            "V2.11.3 scientific result."
        ),
        (
            "The consumer-adapter amendment is actor-performance outcome-blind but "
            "not globally A-D outcome-blind because five offline candidate-admission "
            "cells were inspected; their metrics do not select or alter the matrix."
        ),
        (
            "The adapter repair changes schema dispatch and dedicated verification "
            "before provider construction; it does not change prompts, seeds, arms, "
            "models, environment dynamics, utility, thresholds, or metrics."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated weight "
            "snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The 4-agent by 12-month mechanism micro-pilot is not the 10x24x5 "
            "confirmatory pilot and does not support 100x240 claims."
        ),
        "No paid V2.11.3 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_4_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.3 remains an immutable 136-cell terminal denominator with 3 "
            "complete and 133 integrity-stopped cells; no stopped cell is resumed, "
            "deleted, or reclassified in the V2.11.3 namespace."
        ),
        (
            "V2.11.4 imports five hash-bound operational authority cells with zero "
            "provider calls; dispatch reservations still originate from the exact "
            "V2.11.2 post-gate authority and are not scientific evidence."
        ),
        (
            "No V2.11.3 scientific cell, provider completion, action outcome, "
            "checkpoint continuation, or treatment-effect metric is reused as a "
            "V2.11.4 scientific result."
        ),
        (
            "The authority-normalization amendment is scientific-performance "
            "outcome-blind; only the failed equality predicate and provenance "
            "structures were inspected."
        ),
        (
            "Normalization removes exactly four reseal-only provenance fields for "
            "source-core comparison and verifies them separately; it does not change "
            "prompts, seeds, arms, models, environment dynamics, utility, thresholds, "
            "or metrics."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated weight "
            "snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The 4-agent by 12-month mechanism micro-pilot is not the 10x24x5 "
            "confirmatory pilot and does not support 100x240 claims."
        ),
        "No paid V2.11.4 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_5_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.4 remains an immutable pre-dispatch acceptance no-go with 5 "
            "complete operational cells and 131 scheduled scientific cells; it "
            "is not a terminal 136-cell scientific denominator."
        ),
        (
            "V2.11.5 imports five hash-bound operational authority cells with zero "
            "provider calls; reservation payloads still originate from the exact "
            "V2.11.2 post-gate authority and are not scientific evidence."
        ),
        (
            "No V2.11.4 scientific cell, provider completion, action outcome, "
            "checkpoint continuation, or treatment-effect metric is reused as a "
            "V2.11.5 scientific result."
        ),
        (
            "The final-consumer authority normalization is scientific-performance "
            "outcome-blind; only the failed equality predicate and provenance "
            "structures were inspected."
        ),
        (
            "Normalization requires exact equality for nine stable authority fields "
            "and reservation payloads while validating eight release-generation "
            "fields against the current release; it does not change prompts, seeds, "
            "arms, models, environment dynamics, utility, thresholds, or metrics."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated weight "
            "snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The 4-agent by 12-month mechanism micro-pilot is not the 10x24x5 "
            "confirmatory pilot and does not support 100x240 claims."
        ),
        "No paid V2.11.5 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_6_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.5 remains the immutable parent denominator: 47 complete, 3 "
            "failed, and 86 scheduled rows. Its terminal A/C and operational rows "
            "are not retried, deleted, or reclassified as V2.11.6 evidence."
        ),
        (
            "V2.11.6 is an 87-cell continuation overlay containing one zero-provider "
            "parent import and exactly the 86 previously scheduled D/B/cross-model "
            "rows; it is not a fresh 136-cell rerun."
        ),
        (
            "The parent import authenticates lineage, cumulative budget, calibration, "
            "capability, and preflight reservation authority only. It creates no new "
            "scientific observation and makes no provider call."
        ),
        (
            "The V2.11.5 Experiment A retrieval-effect gate remains a negative result, "
            "and its three ITT failures remain failures; continuation cannot revise "
            "that conclusion or replace those seeds."
        ),
        (
            "The V2.11.5 Experiment C formal publishability result remains a no-go "
            "because its preregistered offline sensitivity artifact is absent; its "
            "narrow forced-active diagnostic is not upgraded to a full reliability claim."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated weight "
            "snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The combined V2.11.5 parent plus V2.11.6 continuation is a 4-agent by "
            "12-month mechanism micro-pilot, not the 10x24x5 confirmatory pilot and "
            "not evidence for 100x240 claims."
        ),
        "No paid V2.11.6 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_7_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.5 remains the immutable scientific authority: 47 complete, 3 "
            "failed, and 86 never-dispatched scheduled rows. Its terminal A/C and "
            "operational rows are not retried, deleted, or reclassified."
        ),
        (
            "V2.11.6 remains an immutable pre-dispatch accounting no-go with 87 "
            "integrity-stopped rows and zero provider calls. Those rows are retained "
            "only as aborted-release audit records and are not scientific evidence."
        ),
        (
            "V2.11.7 contains one zero-provider dual-lineage import and maps exactly "
            "the 86 V2.11.5 never-dispatched D/B/cross-model cells one-to-one; "
            "cross-release deduplication preserves the original 136-cell registered "
            "and 131-cell scientific denominator."
        ),
        (
            "The accounting repair sums all 50 V2.11.5 current-release budget rows; "
            "it is outcome-blind and changes no prompt, seed, arm, model, environment, "
            "utility, threshold, metric, or scientific interpretation."
        ),
        (
            "The V2.11.5 Experiment A retrieval-effect gate remains a negative result, "
            "and its three ITT failures remain failures; continuation cannot revise "
            "that conclusion or replace those seeds."
        ),
        (
            "The V2.11.5 Experiment C formal publishability result remains a no-go "
            "because its preregistered offline sensitivity artifact is absent; its "
            "narrow forced-active diagnostic is not upgraded to a full reliability claim."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated weight "
            "snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The combined lineage is a 4-agent by 12-month mechanism micro-pilot, "
            "not the 10x24x5 confirmatory pilot and not evidence for 100x240 claims."
        ),
        "No paid V2.11.7 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_8_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.5 remains the immutable scientific authority: 47 complete, 3 "
            "failed, and 86 never-dispatched scheduled rows. Its terminal A/C and "
            "operational rows are not retried, deleted, or reclassified."
        ),
        (
            "V2.11.7 remains an immutable 87-row observed-p95 context no-go with "
            "zero provider calls. Its integrity-stopped rows remain audit-only and "
            "are neither resumed nor reclassified as V2.11.8 evidence."
        ),
        (
            "V2.11.8 contains one zero-provider dual-lineage import and maps exactly "
            "the 86 V2.11.5 never-dispatched D/B/cross-model cells one-to-one; "
            "cross-release deduplication preserves the original 136-cell registered "
            "and 131-cell scientific denominator."
        ),
        (
            "The recovery changes only the repository context used while reverifying "
            "source-backed observed-p95 authority. It is performance-outcome-blind "
            "and changes no prompt, seed, arm, model, environment, utility, threshold, "
            "metric, decoding output, or scientific interpretation."
        ),
        (
            "The V2.11.5 Experiment A retrieval-effect gate remains a negative result, "
            "and its three ITT failures remain failures; continuation cannot revise "
            "that conclusion or replace those seeds."
        ),
        (
            "The V2.11.5 Experiment C formal publishability result remains a no-go "
            "because its preregistered offline sensitivity artifact is absent; its "
            "narrow forced-active diagnostic is not upgraded to a full reliability claim."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated weight "
            "snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The combined lineage is a 4-agent by 12-month mechanism micro-pilot, "
            "not the 10x24x5 confirmatory pilot and not evidence for 100x240 claims."
        ),
        "No paid V2.11.8 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_9_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.5 remains the immutable scientific authority: 47 complete, 3 "
            "failed, and 86 never-dispatched scheduled rows. Its terminal A/C and "
            "operational rows are not retried, deleted, or reclassified."
        ),
        (
            "V2.11.8 remains an immutable 87-row release-binding no-go with zero "
            "provider calls. Its integrity-stopped rows remain audit-only and are "
            "neither resumed nor reclassified as V2.11.9 evidence."
        ),
        (
            "V2.11.9 contains one zero-provider dual-lineage import and maps exactly "
            "the 86 V2.11.5 never-dispatched D/B/cross-model cells one-to-one; "
            "cross-release deduplication preserves the original 136-cell registered "
            "and 131-cell scientific denominator."
        ),
        (
            "The recovery changes only reconstruction of the historical V2.11.5 "
            "contract provenance binding during acceptance recomputation. It is "
            "performance-outcome-blind and changes no prompt, seed, arm, model, "
            "environment, utility, threshold, metric, decoding output, or scientific "
            "interpretation."
        ),
        (
            "The V2.11.5 Experiment A retrieval-effect gate remains a negative result, "
            "and its three ITT failures remain failures; continuation cannot revise "
            "that conclusion or replace those seeds."
        ),
        (
            "The V2.11.5 Experiment C formal publishability result remains a no-go "
            "because its preregistered offline sensitivity artifact is absent; its "
            "narrow forced-active diagnostic is not upgraded to a full reliability claim."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated weight "
            "snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The combined lineage is a 4-agent by 12-month mechanism micro-pilot, "
            "not the 10x24x5 confirmatory pilot and not evidence for 100x240 claims."
        ),
        "No paid V2.11.9 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_10_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.5 remains the immutable scientific authority: 47 complete, 3 "
            "failed, and 86 originally scheduled D/B/cross-model rows. Its terminal "
            "A/C and operational rows are not retried, deleted, or reclassified."
        ),
        (
            "V2.11.9 remains an immutable 87-row implementation no-go: its one "
            "operational import completed, all 86 scientific cells failed before "
            "the first hosted completion, and its zero-cost failures remain audit-only."
        ),
        (
            "V2.11.10 contains one zero-provider dual-lineage import and maps exactly "
            "the same 86 V2.11.5 D/B/cross-model scientific specifications one-to-one; "
            "cross-release deduplication preserves the original 136-cell registered "
            "and 131-cell scientific denominator."
        ),
        (
            "The recovery changes only observed-P95 producer/runner authority layering "
            "and requires its real producer-to-consumer round trip during zero-provider "
            "acceptance. It is outcome-blind and changes no prompt, seed, arm, model, "
            "environment, utility, threshold, metric, or decoding configuration."
        ),
        (
            "The V2.11.5 Experiment A retrieval-effect gate remains a negative result, "
            "and its three ITT failures remain failures; continuation cannot revise "
            "that conclusion or replace those seeds."
        ),
        (
            "The V2.11.5 Experiment C formal publishability result remains a no-go "
            "because its preregistered offline sensitivity artifact is absent; its "
            "narrow forced-active diagnostic is not upgraded to a full reliability claim."
        ),
        (
            "GPT-5.6 is a route/model-ID boundary without an immutable dated weight "
            "snapshot or a model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The combined lineage is a 4-agent by 12-month mechanism micro-pilot, "
            "not the 10x24x5 confirmatory pilot and not evidence for 100x240 claims."
        ),
        "No paid V2.11.10 dispatch is authorized while the contract status is draft.",
    ]


def _v2_11_11_expected_non_claims() -> list[str]:
    return [
        (
            "V2.11.5 remains immutable scientific evidence. Its Experiment A no-go, "
            "Experiment C publication no-go, three failed cells, and all original "
            "denominators are not revised by this cohort."
        ),
        (
            "V2.11.10 remains an immutable terminal 87-row audit record: two cells "
            "completed, 55 failed, and 30 were integrity-stopped. No old outcome, "
            "failure, seed, or partial branch is retried, replaced, or reclassified."
        ),
        (
            "V2.11.11 registers 86 new scientific cells using five fresh seeds and "
            "one zero-provider operational lineage cell. The combined lineage has "
            "222 registered and 217 scientific cells."
        ),
        (
            "Increasing the actor output ceiling from 4096 to 8192 tokens addresses "
            "prospective truncation only; it cannot turn a provider or integrity "
            "failure into a scientific effect."
        ),
        (
            "GPT-5.6 remains a route/model-ID boundary without an immutable dated "
            "weight snapshot or model-specific matched A/A null."
        ),
        (
            "The three-seed GPT-5.6 lane can support only a capability-qualified "
            "directional small-pilot statement, never backbone independence."
        ),
        (
            "The combined evidence remains a 4-agent by 12-month mechanism "
            "micro-pilot, not the 10x24x5 confirmatory pilot and not evidence for "
            "100x240 claims."
        ),
        "No paid V2.11.11 dispatch is authorized while the contract status is draft.",
    ]


@dataclass(frozen=True, slots=True)
class PilotContract:
    schema_version: str
    contract_id: str
    status: str
    implementation: Mapping[str, Any]
    seeds: Mapping[str, Any]
    provider_profiles: Mapping[str, ProviderRequestProfile]
    arms: Mapping[str, Any]
    narratives: Mapping[str, Any]
    shocks: Mapping[str, Any]
    utility: Mapping[str, Any]
    budgets: Mapping[str, Any]
    stop_go: Mapping[str, Any]
    stages: tuple[PilotStage, ...]
    parameter_dispatch_policy: Optional[ParameterDispatchPolicy]
    task_output_contracts: Mapping[str, TaskOutputContract]
    model_roles: Mapping[str, ModelRolePolicy]
    denominator_policy: Optional[DenominatorPolicy]
    release_requirements: Optional[ReleaseRequirements]
    operational_amendment: Optional[Mapping[str, Any]]
    evaluator_amendment: Optional[Mapping[str, Any]]
    preflight_bootstrap_amendment: Optional[Mapping[str, Any]]
    matrix_amendment: Optional[Mapping[str, Any]]
    parent_import_retry_amendment: Optional[Mapping[str, Any]]
    p95_authority_retry_amendment: Optional[Mapping[str, Any]]
    stage0_evaluator_retry_amendment: Optional[Mapping[str, Any]]
    qref_identity_retry_amendment: Optional[Mapping[str, Any]]
    qref_summary_equivalence_amendment: Optional[Mapping[str, Any]]
    p95_runner_binding_retry_amendment: Optional[Mapping[str, Any]]
    qref_receipt_verifier_retry_amendment: Optional[Mapping[str, Any]]
    p95_consumer_adapter_retry_amendment: Optional[Mapping[str, Any]]
    non_claims: tuple[str, ...]
    canonicalization: str
    declared_sha256: str
    v211_forward_boundary: Optional[Mapping[str, Any]] = None
    v2111_forward_boundary: Optional[Mapping[str, Any]] = None
    v2111_preflight_bootstrap_amendment: Optional[Mapping[str, Any]] = None
    v2112_forward_boundary: Optional[Mapping[str, Any]] = None
    v2112_recovery_amendment: Optional[Mapping[str, Any]] = None
    v2113_forward_boundary: Optional[Mapping[str, Any]] = None
    v2113_consumer_adapter_amendment: Optional[Mapping[str, Any]] = None
    v2114_forward_boundary: Optional[Mapping[str, Any]] = None
    v2114_authority_normalization_amendment: Optional[Mapping[str, Any]] = None
    v2115_forward_boundary: Optional[Mapping[str, Any]] = None
    v2115_consumer_authority_normalization_amendment: Optional[Mapping[str, Any]] = None
    v2116_continuation_boundary: Optional[Mapping[str, Any]] = None
    v2117_recovery_boundary: Optional[Mapping[str, Any]] = None
    v2118_recovery_boundary: Optional[Mapping[str, Any]] = None
    v2119_recovery_boundary: Optional[Mapping[str, Any]] = None
    v21110_recovery_boundary: Optional[Mapping[str, Any]] = None
    v21111_fresh_cohort_boundary: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotContract":
        value = _mapping(value, "pilot contract")
        base_fields = {
            "schema_version",
            "contract_id",
            "status",
            "implementation",
            "seeds",
            "provider_profiles",
            "arms",
            "narratives",
            "shocks",
            "utility",
            "budgets",
            "stop_go",
            "stages",
            "non_claims",
            "integrity",
        }
        v2_fields = {
            "parameter_dispatch_policy",
            "task_output_contracts",
            "model_roles",
            "denominator_policy",
            "release_requirements",
        }
        schema_version = value.get("schema_version")
        if schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V1:
            fields = base_fields
            is_v2 = False
            is_v2_1 = False
            is_v2_2 = False
            is_v2_3 = False
            is_v2_4 = False
            is_v2_5 = False
            is_v2_6 = False
            is_v2_7 = False
            is_v2_8 = False
            is_v2_9 = False
            is_v2_10 = False
            is_v2_10_1 = False
            is_v2_10_2 = False
            is_v2_11 = False
            is_v2_11_1 = False
            is_v2_11_2 = False
            is_v2_11_3 = False
            is_v2_11_4 = False
            is_v2_11_5 = False
            is_v2_11_6 = False
            is_v2_11_7 = False
            is_v2_11_8 = False
            is_v2_11_9 = False
            is_v2_11_10 = False
            is_v2_11_11 = False
        elif schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V2:
            fields = base_fields | v2_fields
            is_v2 = True
            contract_id = value.get("contract_id")
            is_v2_1 = contract_id == PILOT_CONTRACT_ID_V2_1
            is_v2_2 = contract_id == PILOT_CONTRACT_ID_V2_2
            is_v2_3 = contract_id == PILOT_CONTRACT_ID_V2_3
            is_v2_4 = contract_id == PILOT_CONTRACT_ID_V2_4
            is_v2_5 = contract_id == PILOT_CONTRACT_ID_V2_5
            is_v2_6 = contract_id == PILOT_CONTRACT_ID_V2_6
            is_v2_7 = contract_id == PILOT_CONTRACT_ID_V2_7
            is_v2_8 = contract_id == PILOT_CONTRACT_ID_V2_8
            is_v2_9 = contract_id == PILOT_CONTRACT_ID_V2_9
            is_v2_10 = contract_id == PILOT_CONTRACT_ID_V2_10
            is_v2_10_1 = contract_id == PILOT_CONTRACT_ID_V2_10_1
            is_v2_10_2 = contract_id == PILOT_CONTRACT_ID_V2_10_2
            is_v2_11 = contract_id == PILOT_CONTRACT_ID_V2_11
            is_v2_11_1 = contract_id == PILOT_CONTRACT_ID_V2_11_1
            is_v2_11_2 = contract_id == PILOT_CONTRACT_ID_V2_11_2
            is_v2_11_3 = contract_id == PILOT_CONTRACT_ID_V2_11_3
            is_v2_11_4 = contract_id == PILOT_CONTRACT_ID_V2_11_4
            is_v2_11_5 = contract_id == PILOT_CONTRACT_ID_V2_11_5
            is_v2_11_6 = contract_id == PILOT_CONTRACT_ID_V2_11_6
            is_v2_11_7 = contract_id == PILOT_CONTRACT_ID_V2_11_7
            is_v2_11_8 = contract_id == PILOT_CONTRACT_ID_V2_11_8
            is_v2_11_9 = contract_id == PILOT_CONTRACT_ID_V2_11_9
            is_v2_11_10 = contract_id == PILOT_CONTRACT_ID_V2_11_10
            is_v2_11_11 = contract_id == PILOT_CONTRACT_ID_V2_11_11
            if is_v2_1:
                fields = fields | {"operational_amendment"}
            elif is_v2_2:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                }
            elif is_v2_3:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                    "preflight_bootstrap_amendment",
                }
            elif is_v2_4:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                    "preflight_bootstrap_amendment",
                    "matrix_amendment",
                }
            elif is_v2_5:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                    "preflight_bootstrap_amendment",
                    "matrix_amendment",
                    "parent_import_retry_amendment",
                }
            elif is_v2_6:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                    "preflight_bootstrap_amendment",
                    "matrix_amendment",
                    "parent_import_retry_amendment",
                    "p95_authority_retry_amendment",
                }
            elif is_v2_7:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                    "preflight_bootstrap_amendment",
                    "matrix_amendment",
                    "parent_import_retry_amendment",
                    "p95_authority_retry_amendment",
                    "stage0_evaluator_retry_amendment",
                }
            elif is_v2_8:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                    "preflight_bootstrap_amendment",
                    "matrix_amendment",
                    "parent_import_retry_amendment",
                    "p95_authority_retry_amendment",
                    "stage0_evaluator_retry_amendment",
                    "qref_identity_retry_amendment",
                }
            elif is_v2_9:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                    "preflight_bootstrap_amendment",
                    "matrix_amendment",
                    "parent_import_retry_amendment",
                    "p95_authority_retry_amendment",
                    "stage0_evaluator_retry_amendment",
                    "qref_identity_retry_amendment",
                    "qref_summary_equivalence_amendment",
                }
            elif is_v2_10 or is_v2_10_1 or is_v2_10_2:
                fields = fields | {
                    "operational_amendment",
                    "evaluator_amendment",
                    "preflight_bootstrap_amendment",
                    "matrix_amendment",
                    "parent_import_retry_amendment",
                    "p95_authority_retry_amendment",
                    "stage0_evaluator_retry_amendment",
                    "qref_identity_retry_amendment",
                    "qref_summary_equivalence_amendment",
                    "p95_runner_binding_retry_amendment",
                }
                if is_v2_10_1 or is_v2_10_2:
                    fields = fields | {
                        "qref_receipt_verifier_retry_amendment",
                    }
                if is_v2_10_2:
                    fields = fields | {
                        "p95_consumer_adapter_retry_amendment",
                    }
            elif is_v2_11:
                fields = fields | {"v211_forward_boundary"}
            elif is_v2_11_1:
                fields = fields | {
                    "v2111_forward_boundary",
                    "v2111_preflight_bootstrap_amendment",
                }
            elif is_v2_11_2:
                fields = fields | {
                    "v2112_forward_boundary",
                    "v2112_recovery_amendment",
                }
            elif is_v2_11_3:
                fields = fields | {
                    "v2113_forward_boundary",
                    "v2113_consumer_adapter_amendment",
                }
            elif is_v2_11_4:
                fields = fields | {
                    "v2114_forward_boundary",
                    "v2114_authority_normalization_amendment",
                }
            elif is_v2_11_5:
                fields = fields | {
                    "v2115_forward_boundary",
                    "v2115_consumer_authority_normalization_amendment",
                }
            elif is_v2_11_6:
                fields = fields | {"v2116_continuation_boundary"}
            elif is_v2_11_7:
                fields = fields | {"v2117_recovery_boundary"}
            elif is_v2_11_8:
                fields = fields | {"v2118_recovery_boundary"}
            elif is_v2_11_9:
                fields = fields | {"v2119_recovery_boundary"}
            elif is_v2_11_10:
                fields = fields | {"v21110_recovery_boundary"}
            elif is_v2_11_11:
                fields = fields | {"v21111_fresh_cohort_boundary"}
        else:
            raise PilotContractError("unsupported pilot contract schema")
        is_v211_family = (
            is_v2_11
            or is_v2_11_1
            or is_v2_11_2
            or is_v2_11_3
            or is_v2_11_4
            or is_v2_11_5
            or is_v2_11_6
            or is_v2_11_7
            or is_v2_11_8
            or is_v2_11_9
            or is_v2_11_10
            or is_v2_11_11
        )
        _strict_keys(value, required=fields, name="pilot contract")
        if value["status"] != "frozen" and not (
            (
                is_v2_1
                or is_v2_2
                or is_v2_3
                or is_v2_4
                or is_v2_5
                or is_v2_6
                or is_v2_7
                or is_v2_8
                or is_v2_9
                or is_v2_10
                or is_v2_10_1
                or is_v2_10_2
                or is_v2_11
                or is_v2_11_1
                or is_v2_11_2
                or is_v2_11_3
                or is_v2_11_4
                or is_v2_11_5
                or is_v2_11_6
                or is_v2_11_7
                or is_v2_11_8
                or is_v2_11_9
                or is_v2_11_10
                or is_v2_11_11
            )
            and value["status"] == "draft"
        ):
            raise PilotContractError(
                "pilot contract status must be frozen, except an amendment draft"
            )
        if (
            is_v2_4
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_4_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.4 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_5
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_5_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.5 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_6
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_6_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.6 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_7
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_7_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.7 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_8
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_8_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.8 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_9
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_9_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.9 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_10
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_10_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.10 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_10_1
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.10.1 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_10_2
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_10_2_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.10.2 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_1
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.1 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_2
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.2 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_3
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.3 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_4
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.4 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_5
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.5 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_6
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_6_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.6 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_7
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_7_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.7 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_8
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_8_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.8 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_9
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.9 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_10
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.10 cannot be frozen before its canonical hash and CI inventory"
            )
        if (
            is_v2_11_11
            and value["status"] == "frozen"
            and PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256 is None
        ):
            raise PilotContractError(
                "V2.11.11 cannot be frozen before its canonical hash and CI inventory"
            )
        if is_v2 and value["contract_id"] not in {
            PILOT_CONTRACT_ID_V2,
            PILOT_CONTRACT_ID_V2_1,
            PILOT_CONTRACT_ID_V2_2,
            PILOT_CONTRACT_ID_V2_3,
            PILOT_CONTRACT_ID_V2_4,
            PILOT_CONTRACT_ID_V2_5,
            PILOT_CONTRACT_ID_V2_6,
            PILOT_CONTRACT_ID_V2_7,
            PILOT_CONTRACT_ID_V2_8,
            PILOT_CONTRACT_ID_V2_9,
            PILOT_CONTRACT_ID_V2_10,
            PILOT_CONTRACT_ID_V2_10_1,
            PILOT_CONTRACT_ID_V2_10_2,
            PILOT_CONTRACT_ID_V2_11,
            PILOT_CONTRACT_ID_V2_11_1,
            PILOT_CONTRACT_ID_V2_11_2,
            PILOT_CONTRACT_ID_V2_11_3,
            PILOT_CONTRACT_ID_V2_11_4,
            PILOT_CONTRACT_ID_V2_11_5,
            PILOT_CONTRACT_ID_V2_11_6,
            PILOT_CONTRACT_ID_V2_11_7,
            PILOT_CONTRACT_ID_V2_11_8,
            PILOT_CONTRACT_ID_V2_11_9,
            PILOT_CONTRACT_ID_V2_11_10,
            PILOT_CONTRACT_ID_V2_11_11,
        }:
            raise PilotContractError("unsupported V2 contract_id")
        if (is_v2_1 or is_v2_2 or is_v2_3) and science_design_sha256(
            value
        ) != PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256:
            raise PilotContractError(
                f"{value['contract_id']} science-design fieldset differs from frozen V2"
            )
        if (
            is_v2_5
            or is_v2_6
            or is_v2_7
            or is_v2_8
            or is_v2_9
            or is_v2_10
            or is_v2_10_1
            or is_v2_10_2
        ) and science_design_sha256(value) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
            raise PilotContractError(
                f"{value['contract_id']} science-design fieldset differs from frozen V2.4"
            )
        if (
            is_v2_11_3
            and science_design_sha256(value)
            != PILOT_CONTRACT_V2_11_3_SCIENCE_DESIGN_SHA256
        ):
            raise PilotContractError(
                "V2.11.3 science-design fieldset differs from its frozen draft"
            )
        if is_v2_11_4:
            if PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256 is None:
                raise PilotContractError(
                    "V2.11.4 science-design hash has not been bootstrapped"
                )
            if (
                science_design_sha256(value)
                != PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256
            ):
                raise PilotContractError(
                    "V2.11.4 science-design fieldset differs from its frozen draft"
                )
        if is_v2_11_5:
            if PILOT_CONTRACT_V2_11_5_SCIENCE_DESIGN_SHA256 is None:
                raise PilotContractError(
                    "V2.11.5 science-design hash has not been bootstrapped"
                )
            if (
                science_design_sha256(value)
                != PILOT_CONTRACT_V2_11_5_SCIENCE_DESIGN_SHA256
            ):
                raise PilotContractError(
                    "V2.11.5 science-design fieldset differs from its frozen draft"
                )
        if is_v2_11_6:
            if PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256 is None:
                raise PilotContractError(
                    "V2.11.6 science-design hash has not been bootstrapped"
                )
            if (
                science_design_sha256(value)
                != PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256
            ):
                raise PilotContractError(
                    "V2.11.6 science-design fieldset differs from its frozen draft"
                )
        if is_v2_11_7:
            if PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256 is None:
                raise PilotContractError(
                    "V2.11.7 science-design hash has not been bootstrapped"
                )
            if (
                science_design_sha256(value)
                != PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256
            ):
                raise PilotContractError(
                    "V2.11.7 science-design fieldset differs from its frozen draft"
                )
        if is_v2_11_8:
            if PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256 is None:
                raise PilotContractError(
                    "V2.11.8 science-design hash has not been bootstrapped"
                )
            if (
                science_design_sha256(value)
                != PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256
            ):
                raise PilotContractError(
                    "V2.11.8 science-design fieldset differs from its frozen draft"
                )
        if is_v2_11_9:
            if PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256 is None:
                raise PilotContractError(
                    "V2.11.9 science-design hash has not been bootstrapped"
                )
            if (
                science_design_sha256(value)
                != PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256
            ):
                raise PilotContractError(
                    "V2.11.9 science-design fieldset differs from its frozen draft"
                )
        if is_v2_11_10:
            if PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256 is None:
                raise PilotContractError(
                    "V2.11.10 science-design hash has not been bootstrapped"
                )
            if (
                science_design_sha256(value)
                != PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256
            ):
                raise PilotContractError(
                    "V2.11.10 science-design fieldset differs from its frozen draft"
                )
        if is_v2_11_11:
            if PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256 is None:
                raise PilotContractError(
                    "V2.11.11 science-design hash has not been bootstrapped"
                )
            if (
                science_design_sha256(value)
                != PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256
            ):
                raise PilotContractError(
                    "V2.11.11 science-design fieldset differs from its frozen draft"
                )

        implementation = _mapping(value["implementation"], "implementation")
        implementation_fields = {
            "required_git_tag",
            "commit_resolution",
            "required_git_commit",
            "p0_base_commit",
            "require_clean_worktree",
        }
        if is_v2:
            implementation_fields.add("required_git_branch")
        _strict_keys(
            implementation,
            required=implementation_fields,
            name="implementation",
        )
        if implementation["commit_resolution"] != "annotated_tag_peel":
            raise PilotContractError(
                "implementation commit_resolution must be annotated_tag_peel"
            )
        _text(implementation["required_git_tag"], "required_git_tag")
        _git_commit(implementation["p0_base_commit"], "p0_base_commit")
        _boolean(implementation["require_clean_worktree"], "require_clean_worktree")
        if is_v2:
            expected_tag = (
                PILOT_CONTRACT_TAG_V2_11_7
                if is_v2_11_7
                else (
                    PILOT_CONTRACT_TAG_V2_11_6
                    if is_v2_11_6
                    else (
                        PILOT_CONTRACT_TAG_V2_11_5
                        if is_v2_11_5
                        else (
                            PILOT_CONTRACT_TAG_V2_11_4
                            if is_v2_11_4
                            else (
                                PILOT_CONTRACT_TAG_V2_11_3
                                if is_v2_11_3
                                else (
                                    PILOT_CONTRACT_TAG_V2_11_2
                                    if is_v2_11_2
                                    else (
                                        PILOT_CONTRACT_TAG_V2_11_1
                                        if is_v2_11_1
                                        else (
                                            PILOT_CONTRACT_TAG_V2_11
                                            if is_v2_11
                                            else (
                                                PILOT_CONTRACT_TAG_V2_10_2
                                                if is_v2_10_2
                                                else (
                                                    PILOT_CONTRACT_TAG_V2_10_1
                                                    if is_v2_10_1
                                                    else (
                                                        PILOT_CONTRACT_TAG_V2_10
                                                        if is_v2_10
                                                        else (
                                                            PILOT_CONTRACT_TAG_V2_9
                                                            if is_v2_9
                                                            else (
                                                                PILOT_CONTRACT_TAG_V2_8
                                                                if is_v2_8
                                                                else (
                                                                    PILOT_CONTRACT_TAG_V2_7
                                                                    if is_v2_7
                                                                    else (
                                                                        PILOT_CONTRACT_TAG_V2_6
                                                                        if is_v2_6
                                                                        else (
                                                                            PILOT_CONTRACT_TAG_V2_5
                                                                            if is_v2_5
                                                                            else (
                                                                                PILOT_CONTRACT_TAG_V2_4
                                                                                if is_v2_4
                                                                                else (
                                                                                    PILOT_CONTRACT_TAG_V2_3
                                                                                    if is_v2_3
                                                                                    else (
                                                                                        PILOT_CONTRACT_TAG_V2_2
                                                                                        if is_v2_2
                                                                                        else (
                                                                                            PILOT_CONTRACT_TAG_V2_1
                                                                                            if is_v2_1
                                                                                            else PILOT_CONTRACT_TAG_V2
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if is_v2_11_8:
                expected_tag = PILOT_CONTRACT_TAG_V2_11_8
            if is_v2_11_9:
                expected_tag = PILOT_CONTRACT_TAG_V2_11_9
            if is_v2_11_10:
                expected_tag = PILOT_CONTRACT_TAG_V2_11_10
            if is_v2_11_11:
                expected_tag = PILOT_CONTRACT_TAG_V2_11_11
            if implementation["required_git_tag"] != expected_tag:
                raise PilotContractError(
                    f"{value['contract_id']} must require {expected_tag}"
                )
            if implementation["required_git_branch"] != "main":
                raise PilotContractError("V2 implementation branch must be main")
        required_commit = implementation["required_git_commit"]
        if required_commit is not None:
            _git_commit(required_commit, "required_git_commit")
        elif implementation["commit_resolution"] != "annotated_tag_peel":
            raise PilotContractError(
                "null required_git_commit requires annotated_tag_peel"
            )

        parameter_dispatch_policy: Optional[ParameterDispatchPolicy] = None
        task_output_contracts: dict[str, TaskOutputContract] = {}
        model_roles: dict[str, ModelRolePolicy] = {}
        denominator_policy: Optional[DenominatorPolicy] = None
        release_requirements: Optional[ReleaseRequirements] = None
        operational_amendment: Optional[Mapping[str, Any]] = None
        evaluator_amendment: Optional[Mapping[str, Any]] = None
        preflight_bootstrap_amendment: Optional[Mapping[str, Any]] = None
        matrix_amendment: Optional[Mapping[str, Any]] = None
        parent_import_retry_amendment: Optional[Mapping[str, Any]] = None
        p95_authority_retry_amendment: Optional[Mapping[str, Any]] = None
        stage0_evaluator_retry_amendment: Optional[Mapping[str, Any]] = None
        qref_identity_retry_amendment: Optional[Mapping[str, Any]] = None
        qref_summary_equivalence_amendment: Optional[Mapping[str, Any]] = None
        p95_runner_binding_retry_amendment: Optional[Mapping[str, Any]] = None
        qref_receipt_verifier_retry_amendment: Optional[Mapping[str, Any]] = None
        p95_consumer_adapter_retry_amendment: Optional[Mapping[str, Any]] = None
        v211_forward_boundary: Optional[Mapping[str, Any]] = None
        v2111_forward_boundary: Optional[Mapping[str, Any]] = None
        v2111_preflight_bootstrap_amendment: Optional[Mapping[str, Any]] = None
        v2112_forward_boundary: Optional[Mapping[str, Any]] = None
        v2112_recovery_amendment: Optional[Mapping[str, Any]] = None
        v2113_forward_boundary: Optional[Mapping[str, Any]] = None
        v2113_consumer_adapter_amendment: Optional[Mapping[str, Any]] = None
        v2114_forward_boundary: Optional[Mapping[str, Any]] = None
        v2114_authority_normalization_amendment: Optional[Mapping[str, Any]] = None
        v2115_forward_boundary: Optional[Mapping[str, Any]] = None
        v2115_consumer_authority_normalization_amendment: Optional[
            Mapping[str, Any]
        ] = None
        v2116_continuation_boundary: Optional[Mapping[str, Any]] = None
        v2117_recovery_boundary: Optional[Mapping[str, Any]] = None
        v2118_recovery_boundary: Optional[Mapping[str, Any]] = None
        v2119_recovery_boundary: Optional[Mapping[str, Any]] = None
        v21110_recovery_boundary: Optional[Mapping[str, Any]] = None
        v21111_fresh_cohort_boundary: Optional[Mapping[str, Any]] = None
        if is_v2:
            parameter_dispatch_policy = ParameterDispatchPolicy.from_dict(
                _mapping(
                    value["parameter_dispatch_policy"],
                    "parameter_dispatch_policy",
                )
            )
            task_rows = _mapping(
                value["task_output_contracts"], "task_output_contracts"
            )
            for task_id, row in task_rows.items():
                task = TaskOutputContract.from_dict(
                    _mapping(row, f"task_output_contracts.{task_id}")
                )
                if task.task_id != task_id:
                    raise PilotContractError(
                        f"task output key {task_id!r} does not match task_id"
                    )
                task_output_contracts[task_id] = task
            if set(task_output_contracts) != set(_SCIENCE_TASK_CAPS):
                raise PilotContractError(
                    "V2 task_output_contracts must define exactly four call roles"
                )
            expected_task_caps = (
                _SCIENCE_TASK_CAPS_V2_11_11
                if is_v2_11_11
                else _SCIENCE_TASK_CAPS_V2_11 if is_v211_family else _SCIENCE_TASK_CAPS
            )
            actual_task_caps = {
                task_id: (
                    task.max_completion_tokens,
                    task.max_visible_json_bytes,
                )
                for task_id, task in task_output_contracts.items()
            }
            if actual_task_caps != expected_task_caps:
                raise PilotContractError(
                    f"{value['contract_id']} task output limits differ from "
                    "the contract-specific frozen caps"
                )
            role_rows = _mapping(value["model_roles"], "model_roles")
            for profile_id, row in role_rows.items():
                role = ModelRolePolicy.from_dict(
                    _mapping(row, f"model_roles.{profile_id}")
                )
                if role.profile_id != profile_id:
                    raise PilotContractError(
                        f"model role key {profile_id!r} does not match profile_id"
                    )
                model_roles[profile_id] = role
            denominator_policy = DenominatorPolicy.from_dict(
                _mapping(value["denominator_policy"], "denominator_policy")
            )
            if (
                is_v211_family
                and denominator_policy.policy_id != f"{value['contract_id']}-itt"
            ):
                raise PilotContractError(
                    f"{value['contract_id']} denominator policy identifier drifted"
                )
            release_requirements = ReleaseRequirements.from_dict(
                _mapping(value["release_requirements"], "release_requirements")
            )
            expected_tag = (
                PILOT_CONTRACT_TAG_V2_11_7
                if is_v2_11_7
                else (
                    PILOT_CONTRACT_TAG_V2_11_6
                    if is_v2_11_6
                    else (
                        PILOT_CONTRACT_TAG_V2_11_5
                        if is_v2_11_5
                        else (
                            PILOT_CONTRACT_TAG_V2_11_4
                            if is_v2_11_4
                            else (
                                PILOT_CONTRACT_TAG_V2_11_3
                                if is_v2_11_3
                                else (
                                    PILOT_CONTRACT_TAG_V2_11_2
                                    if is_v2_11_2
                                    else (
                                        PILOT_CONTRACT_TAG_V2_11_1
                                        if is_v2_11_1
                                        else (
                                            PILOT_CONTRACT_TAG_V2_11
                                            if is_v2_11
                                            else (
                                                PILOT_CONTRACT_TAG_V2_10_2
                                                if is_v2_10_2
                                                else (
                                                    PILOT_CONTRACT_TAG_V2_10_1
                                                    if is_v2_10_1
                                                    else (
                                                        PILOT_CONTRACT_TAG_V2_10
                                                        if is_v2_10
                                                        else (
                                                            PILOT_CONTRACT_TAG_V2_9
                                                            if is_v2_9
                                                            else (
                                                                PILOT_CONTRACT_TAG_V2_8
                                                                if is_v2_8
                                                                else (
                                                                    PILOT_CONTRACT_TAG_V2_7
                                                                    if is_v2_7
                                                                    else (
                                                                        PILOT_CONTRACT_TAG_V2_6
                                                                        if is_v2_6
                                                                        else (
                                                                            PILOT_CONTRACT_TAG_V2_5
                                                                            if is_v2_5
                                                                            else (
                                                                                PILOT_CONTRACT_TAG_V2_4
                                                                                if is_v2_4
                                                                                else (
                                                                                    PILOT_CONTRACT_TAG_V2_3
                                                                                    if is_v2_3
                                                                                    else (
                                                                                        PILOT_CONTRACT_TAG_V2_2
                                                                                        if is_v2_2
                                                                                        else (
                                                                                            PILOT_CONTRACT_TAG_V2_1
                                                                                            if is_v2_1
                                                                                            else PILOT_CONTRACT_TAG_V2
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if is_v2_11_8:
                expected_tag = PILOT_CONTRACT_TAG_V2_11_8
            if is_v2_11_9:
                expected_tag = PILOT_CONTRACT_TAG_V2_11_9
            if is_v2_11_10:
                expected_tag = PILOT_CONTRACT_TAG_V2_11_10
            if is_v2_11_11:
                expected_tag = PILOT_CONTRACT_TAG_V2_11_11
            if release_requirements.tag != expected_tag:
                raise PilotContractError(
                    "release tag differs from implementation contract version"
                )
            if is_v2_11:
                v211_forward_boundary = _validate_v2_11_forward_boundary(
                    value["v211_forward_boundary"]
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_6:
                v2116_continuation_boundary = _validate_v2_11_6_continuation_boundary(
                    value["v2116_continuation_boundary"],
                    status=str(value["status"]),
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_7:
                v2117_recovery_boundary = _validate_v2_11_7_recovery_boundary(
                    value["v2117_recovery_boundary"],
                    status=str(value["status"]),
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_8:
                v2118_recovery_boundary = _validate_v2_11_8_recovery_boundary(
                    value["v2118_recovery_boundary"],
                    status=str(value["status"]),
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_9:
                v2119_recovery_boundary = _validate_v2_11_9_recovery_boundary(
                    value["v2119_recovery_boundary"],
                    status=str(value["status"]),
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_10:
                v21110_recovery_boundary = _validate_v2_11_10_recovery_boundary(
                    value["v21110_recovery_boundary"],
                    status=str(value["status"]),
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_11:
                v21111_fresh_cohort_boundary = _validate_v2_11_11_fresh_cohort_boundary(
                    value["v21111_fresh_cohort_boundary"],
                    status=str(value["status"]),
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_1:
                v2111_forward_boundary = _validate_v2_11_1_forward_boundary(
                    value["v2111_forward_boundary"]
                )
                v2111_preflight_bootstrap_amendment = (
                    _validate_v2_11_1_preflight_bootstrap_amendment(
                        value["v2111_preflight_bootstrap_amendment"]
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_2:
                v2112_forward_boundary = _validate_v2_11_2_forward_boundary(
                    value["v2112_forward_boundary"]
                )
                v2112_recovery_amendment = _validate_v2_11_2_recovery_amendment(
                    value["v2112_recovery_amendment"]
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_3:
                v2113_forward_boundary = _validate_v2_11_3_forward_boundary(
                    value["v2113_forward_boundary"]
                )
                v2113_consumer_adapter_amendment = (
                    _validate_v2_11_3_consumer_adapter_amendment(
                        value["v2113_consumer_adapter_amendment"]
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_4:
                v2114_forward_boundary = _validate_v2_11_4_forward_boundary(
                    value["v2114_forward_boundary"],
                    status=str(value["status"]),
                )
                v2114_authority_normalization_amendment = (
                    _validate_v2_11_4_authority_normalization_amendment(
                        value["v2114_authority_normalization_amendment"]
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_11_5:
                v2115_forward_boundary = _validate_v2_11_5_forward_boundary(
                    value["v2115_forward_boundary"],
                    status=str(value["status"]),
                )
                v2115_consumer_authority_normalization_amendment = (
                    _validate_v2_11_5_consumer_authority_normalization_amendment(
                        value["v2115_consumer_authority_normalization_amendment"]
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_1:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_10_1 or is_v2_10_2:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                matrix_amendment = _validate_v2_4_matrix_amendment(
                    value["matrix_amendment"]
                )
                parent_import_retry_amendment = (
                    _validate_v2_5_parent_import_retry_amendment(
                        value["parent_import_retry_amendment"]
                    )
                )
                p95_authority_retry_amendment = (
                    _validate_v2_6_p95_authority_retry_amendment(
                        value["p95_authority_retry_amendment"],
                        status="frozen",
                    )
                )
                stage0_evaluator_retry_amendment = (
                    _validate_v2_7_stage0_evaluator_retry_amendment(
                        value["stage0_evaluator_retry_amendment"],
                        status="frozen",
                    )
                )
                qref_identity_retry_amendment = (
                    _validate_v2_8_qref_identity_retry_amendment(
                        value["qref_identity_retry_amendment"],
                        status="frozen",
                    )
                )
                qref_summary_equivalence_amendment = (
                    _validate_v2_9_qref_summary_equivalence_amendment(
                        value["qref_summary_equivalence_amendment"],
                        status="frozen",
                    )
                )
                p95_runner_binding_retry_amendment = (
                    _validate_v2_10_p95_runner_binding_retry_amendment(
                        value["p95_runner_binding_retry_amendment"],
                        status="frozen",
                    )
                )
                qref_receipt_verifier_retry_amendment = (
                    _validate_v2_10_1_qref_receipt_verifier_retry_amendment(
                        value["qref_receipt_verifier_retry_amendment"],
                        status=("frozen" if is_v2_10_2 else str(value["status"])),
                    )
                )
                if is_v2_10_2:
                    p95_consumer_adapter_retry_amendment = (
                        _validate_v2_10_2_p95_consumer_adapter_retry_amendment(
                            value["p95_consumer_adapter_retry_amendment"],
                            status=str(value["status"]),
                        )
                    )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_10:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                matrix_amendment = _validate_v2_4_matrix_amendment(
                    value["matrix_amendment"]
                )
                parent_import_retry_amendment = (
                    _validate_v2_5_parent_import_retry_amendment(
                        value["parent_import_retry_amendment"]
                    )
                )
                p95_authority_retry_amendment = (
                    _validate_v2_6_p95_authority_retry_amendment(
                        value["p95_authority_retry_amendment"],
                        status="frozen",
                    )
                )
                stage0_evaluator_retry_amendment = (
                    _validate_v2_7_stage0_evaluator_retry_amendment(
                        value["stage0_evaluator_retry_amendment"],
                        status="frozen",
                    )
                )
                qref_identity_retry_amendment = (
                    _validate_v2_8_qref_identity_retry_amendment(
                        value["qref_identity_retry_amendment"],
                        status="frozen",
                    )
                )
                qref_summary_equivalence_amendment = (
                    _validate_v2_9_qref_summary_equivalence_amendment(
                        value["qref_summary_equivalence_amendment"],
                        status="frozen",
                    )
                )
                p95_runner_binding_retry_amendment = (
                    _validate_v2_10_p95_runner_binding_retry_amendment(
                        value["p95_runner_binding_retry_amendment"],
                        status=str(value["status"]),
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_9:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                matrix_amendment = _validate_v2_4_matrix_amendment(
                    value["matrix_amendment"]
                )
                parent_import_retry_amendment = (
                    _validate_v2_5_parent_import_retry_amendment(
                        value["parent_import_retry_amendment"]
                    )
                )
                p95_authority_retry_amendment = (
                    _validate_v2_6_p95_authority_retry_amendment(
                        value["p95_authority_retry_amendment"],
                        status="frozen",
                    )
                )
                stage0_evaluator_retry_amendment = (
                    _validate_v2_7_stage0_evaluator_retry_amendment(
                        value["stage0_evaluator_retry_amendment"],
                        status="frozen",
                    )
                )
                qref_identity_retry_amendment = (
                    _validate_v2_8_qref_identity_retry_amendment(
                        value["qref_identity_retry_amendment"],
                        status="frozen",
                    )
                )
                qref_summary_equivalence_amendment = (
                    _validate_v2_9_qref_summary_equivalence_amendment(
                        value["qref_summary_equivalence_amendment"],
                        status=str(value["status"]),
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_8:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                matrix_amendment = _validate_v2_4_matrix_amendment(
                    value["matrix_amendment"]
                )
                parent_import_retry_amendment = (
                    _validate_v2_5_parent_import_retry_amendment(
                        value["parent_import_retry_amendment"]
                    )
                )
                p95_authority_retry_amendment = (
                    _validate_v2_6_p95_authority_retry_amendment(
                        value["p95_authority_retry_amendment"],
                        status="frozen",
                    )
                )
                stage0_evaluator_retry_amendment = (
                    _validate_v2_7_stage0_evaluator_retry_amendment(
                        value["stage0_evaluator_retry_amendment"],
                        status="frozen",
                    )
                )
                qref_identity_retry_amendment = (
                    _validate_v2_8_qref_identity_retry_amendment(
                        value["qref_identity_retry_amendment"],
                        status=str(value["status"]),
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_2:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_3:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_4:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                matrix_amendment = _validate_v2_4_matrix_amendment(
                    value["matrix_amendment"]
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_5:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                matrix_amendment = _validate_v2_4_matrix_amendment(
                    value["matrix_amendment"]
                )
                parent_import_retry_amendment = (
                    _validate_v2_5_parent_import_retry_amendment(
                        value["parent_import_retry_amendment"]
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_6:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                matrix_amendment = _validate_v2_4_matrix_amendment(
                    value["matrix_amendment"]
                )
                parent_import_retry_amendment = (
                    _validate_v2_5_parent_import_retry_amendment(
                        value["parent_import_retry_amendment"]
                    )
                )
                p95_authority_retry_amendment = (
                    _validate_v2_6_p95_authority_retry_amendment(
                        value["p95_authority_retry_amendment"],
                        status=str(value["status"]),
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )
            elif is_v2_7:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                evaluator_amendment = _validate_v2_2_evaluator_amendment(
                    value["evaluator_amendment"]
                )
                preflight_bootstrap_amendment = (
                    _validate_v2_3_preflight_bootstrap_amendment(
                        value["preflight_bootstrap_amendment"]
                    )
                )
                matrix_amendment = _validate_v2_4_matrix_amendment(
                    value["matrix_amendment"]
                )
                parent_import_retry_amendment = (
                    _validate_v2_5_parent_import_retry_amendment(
                        value["parent_import_retry_amendment"]
                    )
                )
                p95_authority_retry_amendment = (
                    _validate_v2_6_p95_authority_retry_amendment(
                        value["p95_authority_retry_amendment"],
                        status="frozen",
                    )
                )
                stage0_evaluator_retry_amendment = (
                    _validate_v2_7_stage0_evaluator_retry_amendment(
                        value["stage0_evaluator_retry_amendment"],
                        status=str(value["status"]),
                    )
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
                )

        profiles_value = _mapping(value["provider_profiles"], "provider_profiles")
        profiles: dict[str, ProviderRequestProfile] = {}
        for profile_id, row in profiles_value.items():
            profile = ProviderRequestProfile.from_dict(
                _mapping(row, f"provider_profiles.{profile_id}")
            )
            if profile.profile_id != profile_id:
                raise PilotContractError(
                    f"provider profile key {profile_id!r} does not match profile_id"
                )
            if profile.transport in {"openai", "openrouter"}:
                profile.price_snapshot.assert_positive_for_hosted_dispatch()
            else:
                profile.price_snapshot.assert_known_for_dispatch()
            if is_v2 and not profile.decoding_fields:
                raise PilotContractError(
                    f"V2 profile {profile_id} lacks decoding_fields"
                )
            profiles[profile_id] = profile
        if not profiles:
            raise PilotContractError("provider_profiles must not be empty")
        if is_v2_11_11:
            expected_openai = {
                "gpt52_main": {
                    "input": 1.75,
                    "cached": 0.175,
                    "output": 14.0,
                    "model_reference": (
                        "https://developers.openai.com/api/docs/models/gpt-5.2"
                    ),
                },
                "gpt56_diagnostic": {
                    "input": 5.0,
                    "cached": 0.5,
                    "output": 30.0,
                    "model_reference": (
                        "https://developers.openai.com/api/docs/models/gpt-5.6-sol"
                    ),
                },
            }
            for profile_id, expected in expected_openai.items():
                profile = profiles.get(profile_id)
                if profile is None:
                    raise PilotContractError(
                        f"V2.11.11 lacks required profile {profile_id}"
                    )
                price = profile.price_snapshot
                if (
                    profile.transport != "openai"
                    or profile.service_tier != "default"
                    or profile.short_context_prompt_token_ceiling != 272_000
                    or price.captured_at != "2026-08-02"
                    or price.source != "https://developers.openai.com/api/docs/pricing"
                    or price.model_reference != expected["model_reference"]
                    or price.dispatch_input != expected["input"]
                    or price.dispatch_cached_input != expected["cached"]
                    or price.dispatch_output != expected["output"]
                ):
                    raise PilotContractError(
                        f"V2.11.11 {profile_id} service-tier or price snapshot drifted"
                    )
        if is_v2:
            if set(model_roles) != set(profiles):
                raise PilotContractError(
                    "V2 model_roles must cover every provider profile exactly"
                )
            for profile_id, profile in profiles.items():
                role = model_roles[profile_id]
                if profile.dispatch_eligible != role.dispatch_eligible:
                    raise PilotContractError(
                        f"profile/model-role dispatch eligibility differs for {profile_id}"
                    )
                if profile.ineligibility_reason != role.ineligibility_reason:
                    raise PilotContractError(
                        f"profile/model-role ineligibility reason differs for {profile_id}"
                    )
            if not (
                is_v2_4
                or is_v2_5
                or is_v2_6
                or is_v2_7
                or is_v2_8
                or is_v2_9
                or is_v2_10
                or is_v2_10_1
                or is_v2_10_2
                or is_v2_11
                or is_v2_11_1
                or is_v2_11_2
                or is_v2_11_3
                or is_v2_11_4
                or is_v2_11_5
                or is_v2_11_6
                or is_v2_11_7
                or is_v2_11_8
                or is_v2_11_9
                or is_v2_11_10
                or is_v2_11_11
            ):
                opus = profiles.get("opus48_no_go")
                opus_role = model_roles.get("opus48_no_go")
                if (
                    opus is None
                    or opus_role is None
                    or opus.dispatch_eligible
                    or opus.ineligibility_reason
                    != "cross_model_budget_no_go_under_nonshrink_policy"
                    or opus_role.role != "capability_no_go"
                ):
                    raise PilotContractError(
                        "Opus must remain zero-dispatch under the frozen "
                        "cross-model non-shrink budget gate"
                    )
            for profile_id, profile in profiles.items():
                decoding = dict(profile.decoding_fields)
                if set(decoding) != set(parameter_dispatch_policy.fields):
                    raise PilotContractError(
                        f"profile {profile_id} does not implement uniform dispatch fields"
                    )
                if profile.transport in {"openai", "openrouter"} and any(
                    not field.catalog_evidence_required for field in decoding.values()
                ):
                    raise PilotContractError(
                        f"hosted profile {profile_id} requires catalog evidence "
                        "for every dispatch disposition"
                    )
                seed_dispatch = decoding["seed"]
                if (
                    profile.seed_capability == "unsupported"
                    and seed_dispatch.dispatch_mode != "documented_unsupported_omitted"
                ):
                    raise PilotContractError(
                        f"seed-unsupported profile {profile_id} must omit seed"
                    )
                response_dispatch = decoding["response_format"]
                if (
                    profile.json_mode == "json_object"
                    and response_dispatch.dispatch_mode != "explicit_supported"
                ):
                    raise PilotContractError(
                        f"JSON profile {profile_id} must explicitly dispatch "
                        "response_format"
                    )
            if not is_v211_family:
                local = profiles.get("llama33_local_controlled")
                if (
                    local is None
                    or local.transport != "ollama"
                    or local.json_mode != "json_object"
                    or set(dict(local.artifact_identity))
                    != {
                        "manifest_sha256",
                        "model_layer_digest",
                        "model_layer_size_bytes",
                        "ollama_version",
                        "adapter",
                        "base_url",
                    }
                ):
                    raise PilotContractError(
                        "controlled local Llama must freeze JSON mode and runtime identity"
                    )

        seeds = _mapping(value["seeds"], "seeds")
        _strict_keys(
            seeds,
            required={
                "generation",
                "preflight_seed",
                "sets",
                *(("failed_seed_replacement",) if is_v2 else ()),
            },
            name="seeds",
        )
        if is_v2 and seeds["failed_seed_replacement"] != "forbidden":
            raise PilotContractError("V2 failed seeds cannot be replaced")
        generation = _mapping(seeds["generation"], "seeds.generation")
        generation_fields = {
            "method",
            "salt",
            "generated_before_results",
            "values",
        }
        if is_v2_11_11:
            generation_fields |= {
                "provenance_class",
                "recorded_at",
                "timing_boundary",
                "stream",
                "preimage_format",
                "encoding",
                "newline",
                "digest",
                "digest_slice",
                "integer_encoding",
                "modulus",
                "valid_range",
                "counter_start",
                "counter_increment_after_every_candidate",
                "rejection_rules",
                "derivation_trace",
                "historical_seed_registry",
                "historical_seed_registry_sha256",
                "fresh_values_overlap_historical_registry",
                "unused_preflight_candidate",
                "random_sampling_claimed",
                "public_preregistration_claimed",
                "user_selected_claimed",
                "claim_boundary",
            }
        _strict_keys(
            generation,
            required=generation_fields,
            name="seeds.generation",
        )
        seed_method = _text(generation["method"], "seed generation method")
        if seed_method not in {
            "sha256-counter-v1",
            "user-preregistered-v1",
        }:
            raise PilotContractError(
                "seed generation method must be sha256-counter-v1 or "
                "an explicitly preregistered vector method"
            )
        salt = _text(generation["salt"], "seed generation salt")
        if not _boolean(
            generation["generated_before_results"], "generated_before_results"
        ):
            raise PilotContractError("seeds must be frozen before results")
        raw_values = generation["values"]
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            raise PilotContractError("seed generation values must be an array")
        main_values = tuple(
            _integer(item, "seed", minimum=0, maximum=2**31 - 2) for item in raw_values
        )
        if len(main_values) != 5 or len(set(main_values)) != 5:
            raise PilotContractError("the main pilot requires five unique frozen seeds")
        if is_v2_11_11 and _json_copy(generation) != (
            _v2_11_11_expected_seed_generation()
        ):
            raise PilotContractError(
                "V2.11.11 seed vector commitment or historical exclusion drifted"
            )
        if seed_method == "sha256-counter-v1" and not is_v2_11_11:
            derived = tuple(
                int.from_bytes(
                    hashlib.sha256(f"{salt}|{index}".encode("utf-8")).digest()[:8],
                    "big",
                )
                % (2**31 - 1)
                for index in range(5)
            )
            if main_values != derived:
                raise PilotContractError(
                    "frozen seed values do not match their derivation"
                )
        preflight_seed = _integer(
            seeds["preflight_seed"],
            "preflight_seed",
            minimum=0,
            maximum=2**31 - 2,
        )
        if preflight_seed in main_values:
            raise PilotContractError("preflight seed must be distinct from main seeds")
        seed_sets = _mapping(seeds["sets"], "seed sets")
        normalized_seed_sets: dict[str, tuple[int, ...]] = {}
        for set_id, items in seed_sets.items():
            if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
                raise PilotContractError(f"seed set {set_id} must be an array")
            normalized = tuple(
                _integer(item, f"seed_sets.{set_id}", minimum=0, maximum=2**31 - 2)
                for item in items
            )
            if not normalized or len(normalized) != len(set(normalized)):
                raise PilotContractError(f"seed set {set_id} is empty or duplicated")
            normalized_seed_sets[str(set_id)] = normalized
        if set(normalized_seed_sets) != {
            "preflight",
            "q-ref",
            "calibration",
            "main",
            "cross-model",
        }:
            raise PilotContractError("seed sets must match the frozen pilot registry")
        if normalized_seed_sets["preflight"] != (preflight_seed,):
            raise PilotContractError("preflight seed set does not match preflight_seed")
        if normalized_seed_sets["q-ref"] != (preflight_seed,):
            raise PilotContractError("q-ref seed set must reuse the preflight seed")
        if normalized_seed_sets["main"] != main_values:
            raise PilotContractError("main seed set does not match frozen main values")
        if normalized_seed_sets["cross-model"] != main_values[:3]:
            raise PilotContractError(
                "cross-model seed set must use the first three main seeds"
            )
        calibration_values = normalized_seed_sets["calibration"]
        if len(calibration_values) != 2:
            raise PilotContractError("calibration seed set requires exactly two seeds")
        if set(calibration_values) & {*main_values, preflight_seed}:
            raise PilotContractError(
                "calibration seeds must be distinct from preflight and main seeds"
            )
        if is_v2:
            expected_main_values = (
                (
                    877361,
                    1410637959,
                    416755402,
                    357136200,
                    1541219789,
                )
                if is_v2_11_11
                else (
                    1099057501,
                    1421875452,
                    1769977770,
                    959809858,
                    617806385,
                )
            )
            if main_values != expected_main_values:
                raise PilotContractError("V2 main seed registry drifted")
            if preflight_seed != 2010922376:
                raise PilotContractError("V2 preflight seed drifted")
            if calibration_values != (1942013315, 760687867):
                raise PilotContractError("V2 calibration seed registry drifted")

        arms = _mapping(value["arms"], "arms")
        narratives = _mapping(value["narratives"], "narratives")
        shocks = _mapping(value["shocks"], "shocks")
        utility = _mapping(value["utility"], "utility")
        budgets = _mapping(value["budgets"], "budgets")
        stop_go = _mapping(value["stop_go"], "stop_go")
        if not all((arms, narratives, shocks, utility, budgets, stop_go)):
            raise PilotContractError("contract sections must not be empty")
        if budgets.get("completion_scope") != "hosted-api-only":
            raise PilotContractError(
                "pilot provider-completion cap must use hosted-api-only scope"
            )
        for arm_id, arm in arms.items():
            row = _mapping(arm, f"arms.{arm_id}")
            if row.get("arm_id") != arm_id:
                raise PilotContractError(f"arm key {arm_id!r} does not match arm_id")
        for narrative_id, narrative in narratives.items():
            row = _mapping(narrative, f"narratives.{narrative_id}")
            _strict_keys(
                row,
                required={"narrative_id", "relation_to_shock", "text"},
                name=f"narratives.{narrative_id}",
            )
            if row.get("narrative_id") != narrative_id:
                raise PilotContractError(
                    f"narrative key {narrative_id!r} does not match narrative_id"
                )
        narrative_texts = {
            str(narrative_id): row.get("text")
            for narrative_id, narrative in narratives.items()
            for row in (_mapping(narrative, f"narratives.{narrative_id}"),)
        }
        if narrative_texts != PILOT_V1_NARRATIVE_FIXTURES:
            raise PilotContractError(
                "pilot-v1 narrative fixture text drifted from the frozen "
                "continuation intervention"
            )
        for shock_id, shock in shocks.items():
            row = _mapping(shock, f"shocks.{shock_id}")
            if row.get("shock_id") != shock_id:
                raise PilotContractError(
                    f"shock key {shock_id!r} does not match shock_id"
                )
        utility_profiles = _mapping(utility.get("profiles"), "utility.profiles")
        if is_v2:
            shock = _mapping(
                shocks.get("registered-rate-shock"),
                "shocks.registered-rate-shock",
            )
            expected_schedule = (
                {
                    "start": 0,
                    "end": 4,
                    "interest_rate": 0.03,
                    "phase": "pre-shock",
                },
                {
                    "start": 5,
                    "end": 7,
                    "interest_rate": 0.08,
                    "phase": "shock",
                },
                {
                    "start": 8,
                    "end": 11,
                    "interest_rate": 0.03,
                    "phase": "recovery",
                },
            )
            schedule = shock.get("schedule")
            if (
                isinstance(schedule, (str, bytes))
                or not isinstance(schedule, Sequence)
                or tuple(dict(_mapping(row, "shock schedule row")) for row in schedule)
                != expected_schedule
            ):
                raise PilotContractError("V2 registered shock schedule drifted")
            hook = _mapping(shock.get("hook_semantics"), "shock hook_semantics")
            if dict(hook) != {
                "prompt_effective_before_decision": True,
                "environment_effective_before_step": True,
                "write_independent_event_stream": True,
                "future_values_hidden": True,
            }:
                raise PilotContractError("V2 shock hook semantics drifted")

            expected_budget = {
                "total_usd": (
                    _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD
                    if (
                        is_v2_4
                        or is_v2_5
                        or is_v2_6
                        or is_v2_7
                        or is_v2_8
                        or is_v2_9
                        or is_v2_10
                        or is_v2_10_1
                        or is_v2_10_2
                        or is_v2_11
                        or is_v2_11_1
                        or is_v2_11_2
                        or is_v2_11_3
                        or is_v2_11_4
                        or is_v2_11_5
                        or is_v2_11_6
                        or is_v2_11_7
                        or is_v2_11_8
                        or is_v2_11_9
                        or is_v2_11_10
                        or is_v2_11_11
                    )
                    else 25.0
                ),
                "max_provider_completions": 7500,
                "completion_scope": "hosted-api-only",
                "max_storage_bytes": 5_000_000_000,
                "automatic_reserve_usd": 0.0 if is_v211_family else 1.0,
            }
            if any(
                budgets.get(key) != expected
                for key, expected in expected_budget.items()
            ):
                raise PilotContractError("V2 global budget limits drifted")
            caps = _mapping(budgets.get("stage_usd_caps"), "budgets.stage_usd_caps")
            expected_caps = (
                {
                    "parent_v21110": 78.3237413125,
                    "dispatch_refresh": float(
                        _v2_11_11_expected_dispatch_refresh()["reserved_cost_usd"]
                    ),
                    "hosted_v21111": 404.46984,
                    "unallocated_headroom": 13.8044374375,
                    "manual_reserve": 0.0,
                }
                if is_v2_11_11
                else (
                    {
                        "parent_v2119": 63.1196450625,
                        "hosted_v21110": 436.8803549375,
                        "manual_reserve": 0.0,
                    }
                    if is_v2_11_10
                    else (
                        {
                            "parent_v2118": 63.1196450625,
                            "hosted_v2119": 436.8803549375,
                            "manual_reserve": 0.0,
                        }
                        if is_v2_11_9
                        else (
                            {
                                "parent_v2117": 63.1196450625,
                                "hosted_v2118": 436.8803549375,
                                "manual_reserve": 0.0,
                            }
                            if is_v2_11_8
                            else (
                                {
                                    "parent_v2116": 63.1196450625,
                                    "hosted_v2117": 436.8803549375,
                                    "manual_reserve": 0.0,
                                }
                                if is_v2_11_7
                                else (
                                    {
                                        "parent_v2115": 63.1196450625,
                                        "hosted_v2116": 436.8803549375,
                                        "manual_reserve": 0.0,
                                    }
                                    if is_v2_11_6
                                    else (
                                        {
                                            "parent_v2114": 19.998220562500006,
                                            "hosted_v2115": 480.0017794375,
                                            "manual_reserve": 0.0,
                                        }
                                        if is_v2_11_5
                                        else (
                                            {
                                                "parent_v2113": 19.998220562500006,
                                                "hosted_v2114": 480.0017794375,
                                                "manual_reserve": 0.0,
                                            }
                                            if is_v2_11_4
                                            else (
                                                {
                                                    "parent_v2112": 19.998220562500006,
                                                    "hosted_v2113": 480.0017794375,
                                                    "manual_reserve": 0.0,
                                                }
                                                if is_v2_11_3
                                                else (
                                                    {
                                                        "parent_v2111": 18.586399812500005,
                                                        "hosted_v2112": 481.4136001875,
                                                        "manual_reserve": 0.0,
                                                    }
                                                    if is_v2_11_2
                                                    else (
                                                        {
                                                            "parent_v211": 17.166524062500006,
                                                            "hosted_v2111": 482.8334759375,
                                                            "manual_reserve": 0.0,
                                                        }
                                                        if is_v2_11_1
                                                        else (
                                                            {
                                                                "parent_v2102": 16.044922812500005,
                                                                "hosted_v211": 483.9550771875,
                                                                "manual_reserve": 0.0,
                                                            }
                                                            if is_v2_11
                                                            else (
                                                                {
                                                                    "parent_v23": 3.212770875,
                                                                    "local": 0.0,
                                                                    "hosted_confirmatory": _PILOT_V2_4_HOSTED_STAGE_CAP_USD,
                                                                    "manual_reserve": 1.0,
                                                                }
                                                                if (
                                                                    is_v2_4
                                                                    or is_v2_5
                                                                    or is_v2_6
                                                                    or is_v2_7
                                                                    or is_v2_8
                                                                    or is_v2_9
                                                                    or is_v2_10
                                                                    or is_v2_10_1
                                                                    or is_v2_10_2
                                                                )
                                                                else (
                                                                    {
                                                                        "capability": 3.0701145,
                                                                        "calibration": 3.0,
                                                                        "core": 13.0,
                                                                        "cross_model": 4.9298855,
                                                                        "manual_reserve": 1.0,
                                                                    }
                                                                    if (
                                                                        is_v2_1
                                                                        or is_v2_2
                                                                        or is_v2_3
                                                                    )
                                                                    else {
                                                                        "capability": 2.0,
                                                                        "calibration": 3.0,
                                                                        "core": 13.0,
                                                                        "cross_model": 6.0,
                                                                        "manual_reserve": 1.0,
                                                                    }
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if dict(caps) != expected_caps:
                raise PilotContractError("V2 stage budget caps drifted")
            projection = _mapping(
                budgets.get("pre_dispatch_projection"),
                "budgets.pre_dispatch_projection",
            )
            if dict(projection) != {
                "required": True,
                "basis": "model-by-call-role preflight p95",
                "reserve_multiplier": 1.25,
                "unknown_price_policy": "stop-before-dispatch",
                "over_budget_policy": "no-go-no-matrix-shrink",
            }:
                raise PilotContractError("V2 budget projection policy drifted")

            q_ref = _mapping(
                utility.get("q_ref_resolution"), "utility.q_ref_resolution"
            )
            if (
                q_ref.get("seed") != 2010922376
                or q_ref.get("num_agents") != 4
                or q_ref.get("episode_length") != 12
                or q_ref.get("aggregation") != "median"
                or q_ref.get("gate") != "finite_and_strictly_positive"
                or tuple(q_ref.get("work_fraction_cycle", ())) != (0.25, 0.5, 0.75, 0.5)
                or tuple(q_ref.get("consumption_fraction_cycle", ()))
                != (0.3, 0.35, 0.3, 0.25)
                or q_ref.get("expected_rows") != 48
            ):
                raise PilotContractError("V2 q_ref calibration contract drifted")
            selection = _mapping(
                utility.get("selection_rule"), "utility.selection_rule"
            )
            if (
                selection.get("method") != "guardrail-then-registered-tiebreak-v1"
                or selection.get("outcome_blind") is not True
                or tuple(selection.get("tiebreak_order", ()))
                != (
                    "maximize mean interior action coverage",
                    "minimize component-balance log distance from one",
                    "minimize normalized center distance",
                    "declaration order only for an exact remaining tie",
                )
            ):
                raise PilotContractError(
                    "V2 utility selection must remain outcome-blind"
                )
            profile_fields = (
                "rho",
                "labor_weight",
                "inverse_frisch",
                "consumption_scale",
                "consumption_scale_multiplier_of_q_ref",
                "discount_factor",
                "evidence_use",
            )
            expected_profile_signatures = {
                "provider-preflight-default": (
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    None,
                    0.99,
                    "capability-only",
                ),
                "center": (
                    1.0,
                    2.0,
                    1.0,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "psi-1": (
                    1.0,
                    1.0,
                    1.0,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "psi-4": (
                    1.0,
                    4.0,
                    1.0,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "nu-0.5": (
                    1.0,
                    2.0,
                    0.5,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "nu-2": (
                    1.0,
                    2.0,
                    2.0,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "q0-0.5x": (
                    1.0,
                    2.0,
                    1.0,
                    None,
                    0.5,
                    0.99,
                    "stage0-candidate",
                ),
                "q0-2x": (
                    1.0,
                    2.0,
                    1.0,
                    None,
                    2.0,
                    0.99,
                    "stage0-candidate",
                ),
                "stage0-selected": (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "resolved-from-stage0-selection-artifact",
                ),
            }
            profile_signatures = {
                profile_id: tuple(
                    _mapping(row, f"utility.profiles.{profile_id}").get(field)
                    for field in profile_fields
                )
                for profile_id, row in utility_profiles.items()
            }
            if profile_signatures != expected_profile_signatures:
                raise PilotContractError("V2 utility OFAT profile grid drifted")
            calibration_gate = _mapping(
                stop_go.get("calibration"), "stop_go.calibration"
            )
            expected_calibration = {
                "max_abs_budget_residual": 1e-8,
                "clipping_count": 0,
                "ceiling_labor_rate_max": 0.5,
                "zero_labor_rate_max": 0.25,
                "interior_labor_rate_min": 0.5,
                "interior_consumption_rate_min": 0.75,
                "median_labor_disutility_to_consumption_utility": [0.5, 2.0],
                "no_candidate_action": "stop",
            }
            if _json_copy(calibration_gate) != expected_calibration:
                raise PilotContractError("V2 calibration stop/go contract drifted")
            expected_capability = {
                "required_task_families": [
                    "action-generation",
                    "m3-proposal",
                    "evidence-citation",
                    "context-scope",
                    "strict-json",
                    "long-memory-context",
                ],
                "interface_valid_required": True,
                "strict_parse_required": True,
                "semantic_candidate_acceptance_required": True,
                "all_provider_and_parse_outcomes_in_denominator": True,
                "recovery_is_report_only": True,
                "truncation_is_failure": True,
            }
            if (
                _json_copy(_mapping(stop_go.get("capability"), "stop_go.capability"))
                != expected_capability
            ):
                raise PilotContractError("V2 capability gate contract drifted")
            expected_preflight = (
                {
                    "action_parse_success": "24/24",
                    "semantic_proposal_outcomes_accounted": "8/8",
                    "provider_rows": 32,
                    "clipping_count": 0,
                    "provider_failure_count": 0,
                    "route_metadata_complete": True,
                    "usage_metadata_complete": True,
                    "cost_metadata_complete": True,
                    "finish_reason_stop_required": True,
                    "served_model_exact": True,
                    "provider_pin_exact": True,
                    "fallback_observed": False,
                    "attempts_per_request": 1,
                    "prompt_token_tier_ceiling": 200000,
                }
                if is_v211_family
                else {
                    "action_parse_success": "12/12",
                    "semantic_proposals_all_accounted": True,
                    "clipping_count": 0,
                    "provider_failure_count": 0,
                    "route_metadata_complete": True,
                    "usage_metadata_complete": True,
                    "cost_metadata_complete": True,
                    "served_model_exact": True,
                    "provider_pin_exact": True,
                    "fallback_observed": False,
                    "attempts_per_request": 1,
                }
            )
            if (
                _json_copy(
                    _mapping(
                        stop_go.get("closed_loop_preflight"),
                        "stop_go.closed_loop_preflight",
                    )
                )
                != expected_preflight
            ):
                raise PilotContractError("V2 closed-loop preflight gate drifted")
            expected_a = {
                "complete_pairs_min": 4,
                "same_direction_min": 4,
                "total_registered_pairs": 5,
                "median_relative_effect_min": 0.05,
                "route_manipulation_checks_required": True,
            }
            if (
                _json_copy(
                    _mapping(stop_go.get("experiment_a"), "stop_go.experiment_a")
                )
                != expected_a
            ):
                raise PilotContractError("V2 Experiment A stop/go drifted")
            if _json_copy(
                _mapping(
                    stop_go.get("core_completeness"),
                    "stop_go.core_completeness",
                )
            ) != {
                "complete_pairs_min": 4,
                "total_registered_pairs": 5,
                "failed_and_missing_runs_remain_in_itt_denominator": True,
            }:
                raise PilotContractError("V2 core completeness policy drifted")
            expected_cross_model = {
                "reportable_complete_pairs_min": 2,
                "total_registered_pairs": 3,
                "direction_replication_complete_pairs": 3,
                "direction_replication_requires_capability_pass": True,
                "seed_unsupported_directional_replication_requires_registered_matched_a_a_null": (
                    not is_v211_family
                ),
                "missing_matched_a_a_null_action": (
                    "directional-only-no-model-specific-repeatability-null"
                    if is_v211_family
                    else "uncalibrated-diagnostic-no-registered-matched-a-a-null"
                ),
            }
            if (
                _json_copy(_mapping(stop_go.get("cross_model"), "stop_go.cross_model"))
                != expected_cross_model
            ):
                raise PilotContractError("V2 cross-model stop/go drifted")
            if _json_copy(_mapping(stop_go.get("global"), "stop_go.global")) != {
                "contract_hash_match": True,
                "annotated_tag_peel_match": True,
                "clean_worktree_required": True,
                "all_registered_runs_have_terminal_ledger_rows": True,
                "provider_and_parse_failures_remain_in_denominator": True,
                "budget_or_storage_projection_failure": "stop-before-dispatch",
            }:
                raise PilotContractError("V2 global stop/go drifted")

        experiment_c = _mapping(stop_go.get("experiment_c"), "stop_go.experiment_c")
        if is_v2:
            if tuple(experiment_c.get("required_directions", ())) != (
                "verifier lowers false activation",
                "verifier lowers harmful exposure",
                "verifier lowers cumulative utility loss",
            ) or experiment_c.get("failure_action") != (
                "withdraw-or-narrow-rule-reliability-claim"
            ):
                raise PilotContractError("V2 Experiment C stop/go drifted")
        sensitivity = _mapping(
            experiment_c.get("zero_api_sensitivity"),
            "stop_go.experiment_c.zero_api_sensitivity",
        )
        _strict_keys(
            sensitivity,
            required={
                "alternative_success_weights",
                "outcome_definitions",
                "absolute_flow_threshold",
                "effectiveness_gate",
                "descriptive_only",
            },
            name="stop_go.experiment_c.zero_api_sensitivity",
        )
        sensitivity_weights = sensitivity["alternative_success_weights"]
        if (
            isinstance(sensitivity_weights, (str, bytes))
            or not isinstance(sensitivity_weights, Sequence)
            or tuple(sensitivity_weights) != PILOT_V1_SENSITIVITY_WEIGHTS
        ):
            raise PilotContractError(
                "pilot-v1 sensitivity weights differ from the frozen 3-cell grid"
            )
        if (
            _string_tuple(
                sensitivity["outcome_definitions"],
                "sensitivity outcome definitions",
            )
            != PILOT_V1_SENSITIVITY_OUTCOMES
        ):
            raise PilotContractError(
                "pilot-v1 sensitivity outcomes differ from the frozen 3-cell grid"
            )
        threshold = _mapping(
            sensitivity["absolute_flow_threshold"],
            "stop_go.experiment_c.zero_api_sensitivity.absolute_flow_threshold",
        )
        expected_threshold = {
            "source_stage": "stage0-calibration",
            "source_profile": "selected-profile-only",
            "source_seeds": "all-two-calibration-seeds",
            "field": "flow_utility",
            "aggregation": "median",
            "derived_after_profile_selection": True,
            "treatment_outcomes_inspected": False,
        }
        if dict(threshold) != expected_threshold:
            raise PilotContractError(
                "absolute flow-utility threshold derivation is not the frozen "
                "Stage-0 selected-profile median"
            )
        if (
            _boolean(
                sensitivity["effectiveness_gate"],
                "sensitivity.effectiveness_gate",
            )
            is not False
            or _boolean(
                sensitivity["descriptive_only"],
                "sensitivity.descriptive_only",
            )
            is not True
        ):
            raise PilotContractError(
                "zero-API sensitivity must remain descriptive and outside the "
                "effectiveness gate"
            )

        experiment_d = _mapping(stop_go.get("experiment_d"), "stop_go.experiment_d")
        if is_v2:
            expected_d_scalars = {
                "complete_pairs_min": 4,
                "same_direction_min": 4,
                "total_registered_pairs": 5,
                "effect_exceeds_matched_a_b_max_null": True,
                "effect_exceeds_one_action_bin": True,
            }
            if any(
                experiment_d.get(key) != expected
                for key, expected in expected_d_scalars.items()
            ):
                raise PilotContractError("V2 Experiment D stop/go drifted")
            expected_memory_pulse = {
                "schema_version": "finevo-pilot-d-memory-pulse-v1",
                "treatment_arms": [
                    "no-memory",
                    "shuffled-episodic",
                    "wrong-context",
                ],
                "focal_agent_id": 0,
                "wrong_context_source_agent_id": 1,
                "decision_t": 6,
                "duration_decisions": 1,
                "continuation_horizon_steps": 6,
                "pulse_at_first_continuation_step": True,
                "direct_treatment_only_at_pulse": True,
                "claim_label": (
                    "focal-agent decision-6 memory pulse with six-step "
                    "downstream continuation"
                ),
            }
            expected_shuffle = {
                "algorithm": ("checkpoint-bound-sha256-rank-permutation-v1"),
                "non_identity_required": True,
                "fixed_reversal_prohibited_for_three_or_more_items": True,
                "checkpoint_hash_bound": True,
            }
            expected_journal = {
                "required": True,
                "calls_per_branch": 24,
                "completion_events_per_branch": 24,
                "terminal_parse_dispositions_per_branch": 24,
                "raw_output_storage": "sha256-and-byte-count-only",
            }
            expected_narrative_pulse = {
                "schema_version": "finevo-pilot-d-narrative-pulse-v1",
                "treatment_narratives": [
                    "aligned",
                    "paraphrase",
                    "opposite",
                ],
                "focal_agent_id": 0,
                "decision_t": 6,
                "duration_decisions": 1,
                "continuation_horizon_steps": 6,
                "pulse_at_first_continuation_step": True,
                "direct_treatment_only_at_pulse": True,
            }
            expected_narrative_gate = {
                "primary_contrast": "aligned-minus-opposite",
                "directional_action_metric": ("focal_first_consumption_rate"),
                "expected_sign": "negative",
                "same_direction_min": 4,
                "must_exceed_matched_a_b_max_null": True,
                "must_exceed_one_consumption_action_bin": True,
                "labor_action_metric": "diagnostic-only",
                "paraphrase_equivalence": (
                    "aligned-within-one-labor-and-consumption-action-bin"
                ),
            }
            expected_nested = {
                "memory_pulse_contract": expected_memory_pulse,
                "shuffle_policy": expected_shuffle,
                "branch_provider_call_journal": expected_journal,
                "narrative_pulse_contract": expected_narrative_pulse,
                "narrative_semantic_gate": expected_narrative_gate,
                "source_schema_versions": {
                    "continuation": "finevo-pilot-continuation-v2",
                    "narrative": "finevo-pilot-narrative-v2",
                },
            }
            if any(
                _json_copy(
                    _mapping(
                        experiment_d.get(key),
                        f"stop_go.experiment_d.{key}",
                    )
                )
                != expected
                for key, expected in expected_nested.items()
            ):
                raise PilotContractError(
                    "V2 Experiment D pulse/journal/narrative contract drifted"
                )
        action_grid = _mapping(
            experiment_d.get("action_grid"),
            "stop_go.experiment_d.action_grid",
        )
        if dict(action_grid) != PILOT_V1_ACTION_GRID:
            raise PilotContractError(
                "pilot-v1 Experiment D action grid drifted from the frozen bins"
            )
        fixture_hash = _sha256(
            experiment_d.get("narrative_fixture_hash"),
            "stop_go.experiment_d.narrative_fixture_hash",
        )
        if fixture_hash != canonical_sha256(PILOT_V1_NARRATIVE_FIXTURES):
            raise PilotContractError(
                "pilot-v1 narrative fixture hash does not match the exact texts"
            )

        stages_value = value["stages"]
        if isinstance(stages_value, (str, bytes)) or not isinstance(
            stages_value, Sequence
        ):
            raise PilotContractError("stages must be an array")
        stages = tuple(PilotStage.from_dict(stage) for stage in stages_value)
        stage_ids = tuple(stage.stage_id for stage in stages)
        if len(stage_ids) != len(set(stage_ids)):
            raise PilotContractError("stage IDs must be unique")
        if (
            is_v2_4
            or is_v2_5
            or is_v2_6
            or is_v2_7
            or is_v2_8
            or is_v2_9
            or is_v2_10
            or is_v2_10_1
            or is_v2_10_2
        ):
            if [stage.to_dict() for stage in stages] != _v2_4_expected_stages():
                raise PilotContractError(
                    "V2.4-V2.10 stages differ from the 211-cell matrix"
                )
            if {
                key: role.to_dict() for key, role in model_roles.items()
            } != _v2_4_expected_model_roles():
                raise PilotContractError("V2.4-V2.10 active model roles drifted")
            for role in model_roles.values():
                if not set(role.allowed_stages) <= set(stage_ids):
                    raise PilotContractError(
                        f"model role {role.profile_id} references an unknown stage"
                    )
                if not set(role.allowed_call_roles) <= {
                    *task_output_contracts,
                    "qref-scripted",
                    "offline-verifier",
                    "checkpoint-branch",
                    "parent-authority-import",
                }:
                    raise PilotContractError(
                        f"model role {role.profile_id} references an unknown call role"
                    )
        elif is_v211_family:
            expected_v211_stages = (
                _v2_11_11_expected_stages()
                if is_v2_11_11
                else (
                    _v2_11_10_expected_stages()
                    if is_v2_11_10
                    else (
                        _v2_11_9_expected_stages()
                        if is_v2_11_9
                        else (
                            _v2_11_8_expected_stages()
                            if is_v2_11_8
                            else (
                                _v2_11_7_expected_stages()
                                if is_v2_11_7
                                else (
                                    _v2_11_6_expected_stages()
                                    if is_v2_11_6
                                    else (
                                        _v2_11_5_expected_stages()
                                        if is_v2_11_5
                                        else (
                                            _v2_11_4_expected_stages()
                                            if is_v2_11_4
                                            else (
                                                _v2_11_3_expected_stages()
                                                if is_v2_11_3
                                                else (
                                                    _v2_11_2_expected_stages()
                                                    if is_v2_11_2
                                                    else (
                                                        _v2_11_1_expected_stages()
                                                        if is_v2_11_1
                                                        else _v2_11_expected_stages()
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if [stage.to_dict() for stage in stages] != expected_v211_stages:
                if (
                    is_v2_11_6
                    or is_v2_11_7
                    or is_v2_11_8
                    or is_v2_11_9
                    or is_v2_11_10
                    or is_v2_11_11
                ):
                    raise PilotContractError(
                        f"{value['contract_id']} stages differ from its exact "
                        "registered matrix"
                    )
                raise PilotContractError(
                    f"{value['contract_id']} stages differ from the prospective "
                    "136-cell matrix"
                )
            expected_v211_roles = (
                _v2_11_11_expected_model_roles()
                if is_v2_11_11
                else (
                    _v2_11_10_expected_model_roles()
                    if is_v2_11_10
                    else (
                        _v2_11_9_expected_model_roles()
                        if is_v2_11_9
                        else (
                            _v2_11_8_expected_model_roles()
                            if is_v2_11_8
                            else (
                                _v2_11_7_expected_model_roles()
                                if is_v2_11_7
                                else (
                                    _v2_11_6_expected_model_roles()
                                    if is_v2_11_6
                                    else (
                                        _v2_11_5_expected_model_roles()
                                        if is_v2_11_5
                                        else (
                                            _v2_11_4_expected_model_roles()
                                            if is_v2_11_4
                                            else (
                                                _v2_11_3_expected_model_roles()
                                                if is_v2_11_3
                                                else (
                                                    _v2_11_2_expected_model_roles()
                                                    if is_v2_11_2
                                                    else (
                                                        _v2_11_1_expected_model_roles()
                                                        if is_v2_11_1
                                                        else _v2_11_expected_model_roles()
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if {
                key: role.to_dict() for key, role in model_roles.items()
            } != expected_v211_roles:
                raise PilotContractError(
                    f"{value['contract_id']} active model roles drifted"
                )
            for role in model_roles.values():
                if not set(role.allowed_stages) <= set(stage_ids):
                    raise PilotContractError(
                        f"model role {role.profile_id} references an unknown stage"
                    )
                if not set(role.allowed_call_roles) <= {
                    *task_output_contracts,
                    "offline-verifier",
                    "checkpoint-branch",
                    "parent-authority-import",
                }:
                    raise PilotContractError(
                        f"model role {role.profile_id} references an unknown call role"
                    )
        if is_v2 and not (
            is_v2_4
            or is_v2_5
            or is_v2_6
            or is_v2_7
            or is_v2_8
            or is_v2_9
            or is_v2_10
            or is_v2_10_1
            or is_v2_10_2
            or is_v2_11
            or is_v2_11_1
            or is_v2_11_2
            or is_v2_11_3
            or is_v2_11_4
            or is_v2_11_5
            or is_v2_11_6
            or is_v2_11_7
            or is_v2_11_8
            or is_v2_11_9
            or is_v2_11_10
            or is_v2_11_11
        ):
            expected_stage_order = (
                "capability-gate",
                "closed-loop-preflight",
                "secondary-capability-gate",
                "secondary-closed-loop-preflight",
                "q-ref-resolution",
                "stage0-calibration",
                "experiment-a",
                "experiment-c",
                "experiment-d",
                "experiment-b",
                "controlled-second",
                "cross-model-diagnostics",
            )
            if stage_ids != expected_stage_order:
                raise PilotContractError(
                    "V2 stages must keep capability/preflight split and A-C-D-B order"
                )
            stage_map = {stage.stage_id: stage for stage in stages}
            models_by_stage = {
                stage.stage_id: {model for cell in stage.cells for model in cell.models}
                for stage in stages
            }
            primary_gate_models = {"gpt52_main", "llama33_local_controlled"}
            secondary_gate_models = {
                "gpt56_diagnostic",
                "gemini35_flash_diagnostic",
                "llama4_maverick_diagnostic",
            }
            if (
                models_by_stage["capability-gate"] != primary_gate_models
                or models_by_stage["closed-loop-preflight"] != primary_gate_models
                or models_by_stage["secondary-capability-gate"] != secondary_gate_models
                or models_by_stage["secondary-closed-loop-preflight"]
                != secondary_gate_models
            ):
                raise PilotContractError(
                    "V2 primary and secondary capability tiers drifted"
                )
            if (
                stage_map["capability-gate"].budget_bucket != "capability"
                or stage_map["closed-loop-preflight"].budget_bucket != "capability"
                or stage_map["secondary-capability-gate"].budget_bucket != "cross_model"
                or stage_map["secondary-closed-loop-preflight"].budget_bucket
                != "cross_model"
            ):
                raise PilotContractError(
                    "V2 capability tiers must use their frozen budget buckets"
                )
            if stage_map["secondary-capability-gate"].prerequisites != (
                "closed-loop-preflight",
            ) or stage_map["secondary-closed-loop-preflight"].prerequisites != (
                "secondary-capability-gate",
            ):
                raise PilotContractError(
                    "V2 secondary gates must follow primary closed-loop preflight"
                )
            if (
                "secondary-closed-loop-preflight"
                not in stage_map["cross-model-diagnostics"].prerequisites
            ):
                raise PilotContractError(
                    "cross-model diagnostics require secondary preflight"
                )
            if "experiment-b" in stage_map["experiment-c"].prerequisites:
                raise PilotContractError("Experiment C cannot depend on Experiment B")
            if stage_map["experiment-d"].prerequisites != (
                "experiment-a",
                "experiment-c",
            ):
                raise PilotContractError("Experiment D must depend on A and C only")
            if "experiment-d" not in stage_map["experiment-b"].prerequisites:
                raise PilotContractError("Experiment B must run after Experiment D")

            expected_roles = {
                "gpt52_main": "primary",
                "llama33_local_controlled": "controlled_second",
                "gpt56_diagnostic": "secondary_diagnostic",
                "gemini35_flash_diagnostic": "secondary_diagnostic",
                "llama4_maverick_diagnostic": "secondary_diagnostic",
                "opus48_no_go": "capability_no_go",
                "qref_scripted": "calibration_only",
            }
            if {key: role.role for key, role in model_roles.items()} != expected_roles:
                raise PilotContractError("V2 model scientific roles drifted")
            allowed_special_roles = {
                "qref-scripted",
                "offline-verifier",
                "checkpoint-branch",
            }
            for role in model_roles.values():
                if not set(role.allowed_stages) <= set(stage_ids):
                    raise PilotContractError(
                        f"model role {role.profile_id} references an unknown stage"
                    )
                if not set(role.allowed_call_roles) <= {
                    *task_output_contracts,
                    *allowed_special_roles,
                }:
                    raise PilotContractError(
                        f"model role {role.profile_id} references an unknown call role"
                    )
        for stage in stages:
            if stage.seed_set not in normalized_seed_sets:
                raise PilotContractError(
                    f"stage {stage.stage_id} references unknown seed set"
                )
            if stage.shock_id not in shocks:
                raise PilotContractError(
                    f"stage {stage.stage_id} references unknown shock"
                )
            if stage.budget_bucket not in _mapping(
                budgets.get("stage_usd_caps"), "budgets.stage_usd_caps"
            ):
                raise PilotContractError(
                    f"stage {stage.stage_id} references unknown budget bucket"
                )
            if not set(stage.utility_profiles) <= set(utility_profiles):
                raise PilotContractError(
                    f"stage {stage.stage_id} references unknown utility profile"
                )
            for prerequisite in stage.prerequisites:
                if prerequisite not in stage_ids:
                    raise PilotContractError(
                        f"stage {stage.stage_id} has unknown prerequisite"
                    )
            if is_v2:
                if not stage.call_roles:
                    raise PilotContractError(
                        f"V2 stage {stage.stage_id} must declare call_roles"
                    )
                if not set(stage.call_roles) <= {
                    *task_output_contracts,
                    "qref-scripted",
                    "offline-verifier",
                    "checkpoint-branch",
                    "parent-authority-import",
                }:
                    raise PilotContractError(
                        f"stage {stage.stage_id} has an unknown V2 call role"
                    )
            for cell in stage.cells:
                if not set(cell.models) <= set(profiles):
                    raise PilotContractError(
                        f"stage {stage.stage_id} references unknown model profile"
                    )
                if not set(cell.arms) <= set(arms):
                    raise PilotContractError(
                        f"stage {stage.stage_id} references unknown arm"
                    )
                if not set(cell.narratives) <= set(narratives):
                    raise PilotContractError(
                        f"stage {stage.stage_id} references unknown narrative"
                    )
                if is_v2:
                    for model_id in cell.models:
                        role = model_roles[model_id]
                        if not role.dispatch_eligible:
                            raise PilotContractError(
                                f"dispatch-ineligible profile {model_id} appears in "
                                f"stage {stage.stage_id}"
                            )
                        if stage.stage_id not in role.allowed_stages:
                            raise PilotContractError(
                                f"profile {model_id} is not eligible for stage "
                                f"{stage.stage_id}"
                            )
                        if not set(stage.call_roles) <= set(role.allowed_call_roles):
                            raise PilotContractError(
                                f"profile {model_id} is not eligible for all call "
                                f"roles in stage {stage.stage_id}"
                            )

        if (
            is_v2_4
            or is_v2_5
            or is_v2_6
            or is_v2_7
            or is_v2_8
            or is_v2_9
            or is_v2_10
            or is_v2_10_1
            or is_v2_10_2
        ):
            if set(profiles) != {
                "gpt52_main",
                "llama33_local_controlled",
                "qref_scripted",
            }:
                raise PilotContractError("V2.4-V2.10 active provider profiles drifted")
            if _json_copy(arms.get("parent-import")) != (
                _v2_4_expected_parent_import_arm()
            ):
                raise PilotContractError("V2.4-V2.10 parent-import arm drifted")
            if list(value["non_claims"]) != _v2_4_expected_non_claims():
                raise PilotContractError("V2.4-V2.10 non-claim boundary drifted")
            registered_cells = sum(
                len(normalized_seed_sets[stage.seed_set])
                * len(stage.utility_profiles)
                * sum(
                    len(cell.models) * len(cell.arms) * len(cell.narratives)
                    for cell in stage.cells
                )
                for stage in stages
                if stage.enabled
            )
            if registered_cells != 211:
                raise PilotContractError(
                    "V2.4/V2.5/V2.6 denominator must contain exactly 211 registered cells"
                )
        elif is_v211_family:
            if set(profiles) != {
                "gpt52_main",
                "gpt56_diagnostic",
                "qref_scripted",
            }:
                raise PilotContractError(
                    f"{value['contract_id']} provider profiles must contain exactly "
                    "the two "
                    "fresh hosted routes and the zero-provider import route"
                )
            expected_parent_import_arm = (
                _v2_11_11_expected_parent_import_arm()
                if is_v2_11_11
                else (
                    _v2_11_10_expected_parent_import_arm()
                    if is_v2_11_10
                    else (
                        _v2_11_9_expected_parent_import_arm()
                        if is_v2_11_9
                        else (
                            _v2_11_8_expected_parent_import_arm()
                            if is_v2_11_8
                            else (
                                _v2_11_7_expected_parent_import_arm()
                                if is_v2_11_7
                                else (
                                    _v2_11_6_expected_parent_import_arm()
                                    if is_v2_11_6
                                    else _v2_4_expected_parent_import_arm()
                                )
                            )
                        )
                    )
                )
            )
            if _json_copy(arms.get("parent-import")) != expected_parent_import_arm:
                raise PilotContractError(
                    f"{value['contract_id']} parent-import arm drifted"
                )
            if (
                is_v2_11_1
                or is_v2_11_2
                or is_v2_11_3
                or is_v2_11_4
                or is_v2_11_5
                or is_v2_11_6
                or is_v2_11_7
                or is_v2_11_8
                or is_v2_11_9
                or is_v2_11_10
                or is_v2_11_11
            ) and _mapping(
                arms.get("capability-probe"),
                "arms.capability-probe",
            ).get(
                "execution_mode"
            ) != "capability_authority_import":
                raise PilotContractError(
                    f"{value['contract_id']} capability cells must be zero-call "
                    "authority imports"
                )
            if (
                is_v2_11_3
                or is_v2_11_4
                or is_v2_11_5
                or is_v2_11_6
                or is_v2_11_7
                or is_v2_11_8
                or is_v2_11_9
                or is_v2_11_10
                or is_v2_11_11
            ) and _json_copy(
                _mapping(
                    arms.get("closed-loop-preflight"),
                    "arms.closed-loop-preflight",
                )
            ) != (
                _v2_11_5_expected_preflight_arm()
                if (
                    is_v2_11_5
                    or is_v2_11_6
                    or is_v2_11_7
                    or is_v2_11_8
                    or is_v2_11_9
                    or is_v2_11_10
                    or is_v2_11_11
                )
                else (
                    _v2_11_4_expected_preflight_arm()
                    if is_v2_11_4
                    else _v2_11_3_expected_preflight_arm()
                )
            ):
                raise PilotContractError(
                    f"{value['contract_id']} preflight cells must be exact zero-call "
                    "authority imports"
                )
            expected_non_claims = (
                _v2_11_11_expected_non_claims()
                if is_v2_11_11
                else (
                    _v2_11_10_expected_non_claims()
                    if is_v2_11_10
                    else (
                        _v2_11_9_expected_non_claims()
                        if is_v2_11_9
                        else (
                            _v2_11_8_expected_non_claims()
                            if is_v2_11_8
                            else (
                                _v2_11_7_expected_non_claims()
                                if is_v2_11_7
                                else (
                                    _v2_11_6_expected_non_claims()
                                    if is_v2_11_6
                                    else (
                                        _v2_11_5_expected_non_claims()
                                        if is_v2_11_5
                                        else (
                                            _v2_11_4_expected_non_claims()
                                            if is_v2_11_4
                                            else (
                                                _v2_11_3_expected_non_claims()
                                                if is_v2_11_3
                                                else (
                                                    _v2_11_2_expected_non_claims()
                                                    if is_v2_11_2
                                                    else (
                                                        _v2_11_1_expected_non_claims()
                                                        if is_v2_11_1
                                                        else _v2_11_expected_non_claims()
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if list(value["non_claims"]) != expected_non_claims:
                raise PilotContractError(
                    f"{value['contract_id']} non-claim boundary drifted"
                )
            registered_cells = sum(
                len(normalized_seed_sets[stage.seed_set])
                * len(stage.utility_profiles)
                * sum(
                    len(cell.models) * len(cell.arms) * len(cell.narratives)
                    for cell in stage.cells
                )
                for stage in stages
                if stage.enabled
            )
            expected_registered_cells = (
                87
                if (
                    is_v2_11_6
                    or is_v2_11_7
                    or is_v2_11_8
                    or is_v2_11_9
                    or is_v2_11_10
                    or is_v2_11_11
                )
                else 136
            )
            if registered_cells != expected_registered_cells:
                raise PilotContractError(
                    f"{value['contract_id']} denominator must contain exactly "
                    f"{expected_registered_cells} registered cells"
                )
            operational_stage_ids = (
                {
                    stage.stage_id
                    for stage in stages
                    if stage.evidence_class == "operational"
                }
                if is_v2_11_11
                else {
                    "parent-import",
                    "capability-gate",
                    "long-context-preflight",
                }
            )
            if is_v2_11_11 and (
                any(stage.evidence_class is None for stage in stages)
                or operational_stage_ids != {"parent-import"}
                or {
                    stage.stage_id
                    for stage in stages
                    if stage.evidence_class == "scientific"
                }
                != {"experiment-b", "experiment-d", "cross-model"}
            ):
                raise PilotContractError(
                    "V2.11.11 stage evidence_class partition drifted"
                )
            scientific_cells = sum(
                len(normalized_seed_sets[stage.seed_set])
                * len(stage.utility_profiles)
                * sum(
                    len(cell.models) * len(cell.arms) * len(cell.narratives)
                    for cell in stage.cells
                )
                for stage in stages
                if stage.enabled and stage.stage_id not in operational_stage_ids
            )
            expected_scientific_cells = (
                86
                if (
                    is_v2_11_6
                    or is_v2_11_7
                    or is_v2_11_8
                    or is_v2_11_9
                    or is_v2_11_10
                    or is_v2_11_11
                )
                else 131
            )
            if scientific_cells != expected_scientific_cells:
                raise PilotContractError(
                    f"{value['contract_id']} denominator must contain exactly "
                    f"{expected_scientific_cells} scientific cells"
                )

        integrity = _mapping(value["integrity"], "integrity")
        _strict_keys(
            integrity,
            required={"canonicalization", "declared_sha256"},
            name="integrity",
        )
        if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
            raise PilotContractError("unsupported contract canonicalization")
        declared = _sha256(integrity["declared_sha256"], "declared_sha256")
        actual = canonical_contract_sha256(value)
        if declared != actual:
            raise PilotContractError(
                f"pilot contract hash mismatch: declared {declared}, actual {actual}"
            )
        if (
            is_v2_4
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_4_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.4 frozen canonical hash drifted")
        if (
            is_v2_5
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_5_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.5 frozen canonical hash drifted")
        if (
            is_v2_6
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_6_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.6 frozen canonical hash drifted")
        if (
            is_v2_7
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_7_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.7 frozen canonical hash drifted")
        if (
            is_v2_8
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_8_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.8 frozen canonical hash drifted")
        if (
            is_v2_9
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_9_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.9 frozen canonical hash drifted")
        if (
            is_v2_10
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_10_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.10 frozen canonical hash drifted")
        if (
            is_v2_10_1
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.10.1 frozen canonical hash drifted")
        if (
            is_v2_10_2
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_10_2_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.10.2 frozen canonical hash drifted")
        if (
            is_v2_11
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11 frozen canonical hash drifted")
        if (
            is_v2_11_1
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.1 frozen canonical hash drifted")
        if (
            is_v2_11_2
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.2 frozen canonical hash drifted")
        if (
            is_v2_11_3
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.3 frozen canonical hash drifted")
        if (
            is_v2_11_4
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.4 frozen canonical hash drifted")
        if (
            is_v2_11_5
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.5 frozen canonical hash drifted")
        if (
            is_v2_11_6
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_6_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.6 frozen canonical hash drifted")
        if (
            is_v2_11_7
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_7_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.7 frozen canonical hash drifted")
        if (
            is_v2_11_8
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_8_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.8 frozen canonical hash drifted")
        if (
            is_v2_11_9
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.9 frozen canonical hash drifted")
        if (
            is_v2_11_10
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.10 frozen canonical hash drifted")
        if (
            is_v2_11_11
            and value["status"] == "frozen"
            and actual != PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256
        ):
            raise PilotContractError("V2.11.11 frozen canonical hash drifted")

        non_claims = _string_tuple(value["non_claims"], "non_claims")
        return cls(
            schema_version=value["schema_version"],
            contract_id=_text(value["contract_id"], "contract_id"),
            status=value["status"],
            implementation=_freeze_json(implementation),
            seeds=_freeze_json(
                {
                    **dict(seeds),
                    "sets": {
                        key: list(items) for key, items in normalized_seed_sets.items()
                    },
                }
            ),
            provider_profiles=MappingProxyType(dict(profiles)),
            arms=_freeze_json(arms),
            narratives=_freeze_json(narratives),
            shocks=_freeze_json(shocks),
            utility=_freeze_json(utility),
            budgets=_freeze_json(budgets),
            stop_go=_freeze_json(stop_go),
            stages=stages,
            parameter_dispatch_policy=parameter_dispatch_policy,
            task_output_contracts=MappingProxyType(dict(task_output_contracts)),
            model_roles=MappingProxyType(dict(model_roles)),
            denominator_policy=denominator_policy,
            release_requirements=release_requirements,
            operational_amendment=operational_amendment,
            evaluator_amendment=evaluator_amendment,
            preflight_bootstrap_amendment=preflight_bootstrap_amendment,
            matrix_amendment=matrix_amendment,
            parent_import_retry_amendment=parent_import_retry_amendment,
            p95_authority_retry_amendment=p95_authority_retry_amendment,
            stage0_evaluator_retry_amendment=(stage0_evaluator_retry_amendment),
            qref_identity_retry_amendment=(qref_identity_retry_amendment),
            qref_summary_equivalence_amendment=(qref_summary_equivalence_amendment),
            p95_runner_binding_retry_amendment=(p95_runner_binding_retry_amendment),
            qref_receipt_verifier_retry_amendment=(
                qref_receipt_verifier_retry_amendment
            ),
            p95_consumer_adapter_retry_amendment=(p95_consumer_adapter_retry_amendment),
            v211_forward_boundary=v211_forward_boundary,
            v2111_forward_boundary=v2111_forward_boundary,
            v2111_preflight_bootstrap_amendment=(v2111_preflight_bootstrap_amendment),
            v2112_forward_boundary=v2112_forward_boundary,
            v2112_recovery_amendment=v2112_recovery_amendment,
            v2113_forward_boundary=v2113_forward_boundary,
            v2113_consumer_adapter_amendment=(v2113_consumer_adapter_amendment),
            v2114_forward_boundary=v2114_forward_boundary,
            v2114_authority_normalization_amendment=(
                v2114_authority_normalization_amendment
            ),
            v2115_forward_boundary=v2115_forward_boundary,
            v2115_consumer_authority_normalization_amendment=(
                v2115_consumer_authority_normalization_amendment
            ),
            v2116_continuation_boundary=v2116_continuation_boundary,
            v2117_recovery_boundary=v2117_recovery_boundary,
            v2118_recovery_boundary=v2118_recovery_boundary,
            v2119_recovery_boundary=v2119_recovery_boundary,
            v21110_recovery_boundary=v21110_recovery_boundary,
            v21111_fresh_cohort_boundary=v21111_fresh_cohort_boundary,
            non_claims=non_claims,
            canonicalization=integrity["canonicalization"],
            declared_sha256=declared,
        )

    @property
    def canonical_hash(self) -> str:
        return canonical_contract_sha256(self.to_dict())

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self.provider_profiles)

    @property
    def arm_ids(self) -> tuple[str, ...]:
        return tuple(self.arms)

    def stage(self, stage_id: str) -> PilotStage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"unknown pilot stage: {stage_id}")

    def models_for_stage(self, stage_id: str) -> tuple[str, ...]:
        stage = self.stage(stage_id)
        return tuple(
            dict.fromkeys(model for cell in stage.cells for model in cell.models)
        )

    def arms_for_stage(self, stage_id: str) -> tuple[str, ...]:
        stage = self.stage(stage_id)
        return tuple(dict.fromkeys(arm for cell in stage.cells for arm in cell.arms))

    def expand(
        self,
        *,
        stage: Optional[str] = None,
        model: Optional[str] = None,
        arm: Optional[str] = None,
        include_disabled: bool = False,
    ) -> tuple[PilotRunSpec, ...]:
        """Expand the frozen stage/model/arm/seed/utility/narrative matrix."""

        if stage is not None and stage not in self.stage_ids:
            raise KeyError(f"unknown pilot stage: {stage}")
        if model is not None and model not in self.provider_profiles:
            raise KeyError(f"unknown pilot model: {model}")
        if arm is not None and arm not in self.arms:
            raise KeyError(f"unknown pilot arm: {arm}")
        seed_sets = _mapping(self.seeds["sets"], "seed sets")
        result: list[PilotRunSpec] = []
        for stage_spec in self.stages:
            if stage is not None and stage_spec.stage_id != stage:
                continue
            if not stage_spec.enabled and not include_disabled:
                continue
            seeds = tuple(int(item) for item in seed_sets[stage_spec.seed_set])
            for cell in stage_spec.cells:
                for model_id in cell.models:
                    if model is not None and model_id != model:
                        continue
                    profile = self.provider_profiles[model_id]
                    for arm_id in cell.arms:
                        if arm is not None and arm_id != arm:
                            continue
                        arm_row = _mapping(self.arms[arm_id], f"arms.{arm_id}")
                        execution_mode = str(
                            arm_row.get("execution_mode", cell.execution_mode)
                        )
                        for narrative_id in cell.narratives:
                            for utility_id in stage_spec.utility_profiles:
                                for seed_value in seeds:
                                    result.append(
                                        PilotRunSpec(
                                            contract_id=self.contract_id,
                                            stage_id=stage_spec.stage_id,
                                            model_id=model_id,
                                            requested_model=profile.requested_model,
                                            arm_id=arm_id,
                                            narrative_id=narrative_id,
                                            environment_seed=seed_value,
                                            decoding_seed=(
                                                (
                                                    seed_value
                                                    if dict(profile.decoding_fields)[
                                                        "seed"
                                                    ].dispatch_mode
                                                    == "explicit_supported"
                                                    else None
                                                )
                                                if profile.decoding_fields
                                                else (
                                                    None
                                                    if profile.seed_capability
                                                    == "unsupported"
                                                    else seed_value
                                                )
                                            ),
                                            utility_profile_id=utility_id,
                                            shock_id=stage_spec.shock_id,
                                            budget_bucket=stage_spec.budget_bucket,
                                            num_agents=stage_spec.num_agents,
                                            episode_length=stage_spec.episode_length,
                                            execution_mode=execution_mode,
                                        )
                                    )
        run_ids = [item.run_id for item in result]
        if len(run_ids) != len(set(run_ids)):
            raise PilotContractError("expanded pilot matrix contains duplicate run IDs")
        if any(selector is not None for selector in (stage, model, arm)) and not result:
            raise KeyError("pilot selector combination matches no run")
        return tuple(result)

    def validate_provenance(self, git_commit: str, git_tag: str) -> dict[str, Any]:
        """Validate a caller-resolved annotated-tag binding for a run manifest.

        The caller must verify that ``git_tag`` is annotated and peel it to
        ``git_commit``.  This pure method verifies the frozen identity and
        returns the exact manifest fields that bind the peeled commit to the
        contract hash.
        """

        if self.status != "frozen":
            raise PilotContractError(
                "paid provenance cannot be validated from a draft contract"
            )
        if self.contract_id in {
            PILOT_CONTRACT_ID_V2_1,
            PILOT_CONTRACT_ID_V2_2,
            PILOT_CONTRACT_ID_V2_3,
            PILOT_CONTRACT_ID_V2_4,
            PILOT_CONTRACT_ID_V2_5,
            PILOT_CONTRACT_ID_V2_6,
            PILOT_CONTRACT_ID_V2_7,
            PILOT_CONTRACT_ID_V2_8,
            PILOT_CONTRACT_ID_V2_9,
            PILOT_CONTRACT_ID_V2_10,
            PILOT_CONTRACT_ID_V2_10_1,
            PILOT_CONTRACT_ID_V2_10_2,
            PILOT_CONTRACT_ID_V2_11,
            PILOT_CONTRACT_ID_V2_11_1,
            PILOT_CONTRACT_ID_V2_11_2,
            PILOT_CONTRACT_ID_V2_11_3,
            PILOT_CONTRACT_ID_V2_11_4,
            PILOT_CONTRACT_ID_V2_11_5,
            PILOT_CONTRACT_ID_V2_11_6,
            PILOT_CONTRACT_ID_V2_11_7,
            PILOT_CONTRACT_ID_V2_11_8,
            PILOT_CONTRACT_ID_V2_11_9,
            PILOT_CONTRACT_ID_V2_11_10,
        }:
            if self.release_requirements is None:  # pragma: no cover - parser
                raise PilotContractError(
                    f"{self.contract_id} lacks release requirements"
                )
            _validate_v2_1_expected_ci_state(
                self.release_requirements.expected_ci,
                status=self.status,
                name="release expected_ci",
            )

        resolved = _git_commit(git_commit, "git_commit")
        actual_tag = _text(git_tag, "git_tag")
        required_tag = str(self.implementation["required_git_tag"])
        if actual_tag != required_tag:
            raise PilotContractError(
                f"pilot requires annotated tag {required_tag!r}, not {actual_tag!r}"
            )
        required_commit = self.implementation["required_git_commit"]
        if required_commit is not None and resolved != required_commit:
            raise PilotContractError("git commit does not match frozen contract")
        return {
            "git_tag": actual_tag,
            "resolved_git_commit": resolved,
            "commit_resolution": self.implementation["commit_resolution"],
            "p0_base_commit": self.implementation["p0_base_commit"],
            "contract_id": self.contract_id,
            "contract_sha256": self.canonical_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "status": self.status,
            "implementation": _thaw_json(self.implementation),
            "seeds": _thaw_json(self.seeds),
            "provider_profiles": {
                key: profile.to_dict()
                for key, profile in self.provider_profiles.items()
            },
            "arms": _thaw_json(self.arms),
            "narratives": _thaw_json(self.narratives),
            "shocks": _thaw_json(self.shocks),
            "utility": _thaw_json(self.utility),
            "budgets": _thaw_json(self.budgets),
            "stop_go": _thaw_json(self.stop_go),
            "stages": [stage.to_dict() for stage in self.stages],
            "non_claims": list(self.non_claims),
            "integrity": {
                "canonicalization": self.canonicalization,
                "declared_sha256": self.declared_sha256,
            },
        }
        if self.schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V2:
            if (
                self.parameter_dispatch_policy is None
                or self.denominator_policy is None
                or self.release_requirements is None
            ):
                raise PilotContractError("incomplete typed V2 contract")
            result.update(
                {
                    "parameter_dispatch_policy": (
                        self.parameter_dispatch_policy.to_dict()
                    ),
                    "task_output_contracts": {
                        key: item.to_dict()
                        for key, item in self.task_output_contracts.items()
                    },
                    "model_roles": {
                        key: item.to_dict() for key, item in self.model_roles.items()
                    },
                    "denominator_policy": self.denominator_policy.to_dict(),
                    "release_requirements": self.release_requirements.to_dict(),
                }
            )
            if self.contract_id == PILOT_CONTRACT_ID_V2_11:
                if self.v211_forward_boundary is None:
                    raise PilotContractError(
                        "V2.11 contract lacks its prospective forward boundary"
                    )
                if any(
                    amendment is not None
                    for amendment in (
                        self.operational_amendment,
                        self.evaluator_amendment,
                        self.preflight_bootstrap_amendment,
                        self.matrix_amendment,
                        self.parent_import_retry_amendment,
                        self.p95_authority_retry_amendment,
                        self.stage0_evaluator_retry_amendment,
                        self.qref_identity_retry_amendment,
                        self.qref_summary_equivalence_amendment,
                        self.p95_runner_binding_retry_amendment,
                        self.qref_receipt_verifier_retry_amendment,
                        self.p95_consumer_adapter_retry_amendment,
                    )
                ):
                    raise PilotContractError(
                        "V2.11 is independently canonicalized and cannot carry "
                        "the V2.1-V2.10.2 amendment chain"
                    )
                result["v211_forward_boundary"] = _thaw_json(self.v211_forward_boundary)
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_1:
                if (
                    self.v2111_forward_boundary is None
                    or self.v2111_preflight_bootstrap_amendment is None
                ):
                    raise PilotContractError(
                        "V2.11.1 contract lacks its forward boundary or "
                        "contract-envelope bootstrap amendment"
                    )
                if self.v211_forward_boundary is not None or any(
                    amendment is not None
                    for amendment in (
                        self.operational_amendment,
                        self.evaluator_amendment,
                        self.preflight_bootstrap_amendment,
                        self.matrix_amendment,
                        self.parent_import_retry_amendment,
                        self.p95_authority_retry_amendment,
                        self.stage0_evaluator_retry_amendment,
                        self.qref_identity_retry_amendment,
                        self.qref_summary_equivalence_amendment,
                        self.p95_runner_binding_retry_amendment,
                        self.qref_receipt_verifier_retry_amendment,
                        self.p95_consumer_adapter_retry_amendment,
                    )
                ):
                    raise PilotContractError(
                        "V2.11.1 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2111_forward_boundary"] = _thaw_json(
                    self.v2111_forward_boundary
                )
                result["v2111_preflight_bootstrap_amendment"] = _thaw_json(
                    self.v2111_preflight_bootstrap_amendment
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_2:
                if (
                    self.v2112_forward_boundary is None
                    or self.v2112_recovery_amendment is None
                ):
                    raise PilotContractError(
                        "V2.11.2 contract lacks its forward boundary or "
                        "recovery amendment"
                    )
                if (
                    self.v211_forward_boundary is not None
                    or self.v2111_forward_boundary is not None
                    or self.v2111_preflight_bootstrap_amendment is not None
                    or any(
                        amendment is not None
                        for amendment in (
                            self.operational_amendment,
                            self.evaluator_amendment,
                            self.preflight_bootstrap_amendment,
                            self.matrix_amendment,
                            self.parent_import_retry_amendment,
                            self.p95_authority_retry_amendment,
                            self.stage0_evaluator_retry_amendment,
                            self.qref_identity_retry_amendment,
                            self.qref_summary_equivalence_amendment,
                            self.p95_runner_binding_retry_amendment,
                            self.qref_receipt_verifier_retry_amendment,
                            self.p95_consumer_adapter_retry_amendment,
                        )
                    )
                ):
                    raise PilotContractError(
                        "V2.11.2 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2112_forward_boundary"] = _thaw_json(
                    self.v2112_forward_boundary
                )
                result["v2112_recovery_amendment"] = _thaw_json(
                    self.v2112_recovery_amendment
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_3:
                if (
                    self.v2113_forward_boundary is None
                    or self.v2113_consumer_adapter_amendment is None
                ):
                    raise PilotContractError(
                        "V2.11.3 contract lacks its forward boundary or "
                        "consumer-adapter amendment"
                    )
                if (
                    self.v211_forward_boundary is not None
                    or self.v2111_forward_boundary is not None
                    or self.v2111_preflight_bootstrap_amendment is not None
                    or self.v2112_forward_boundary is not None
                    or self.v2112_recovery_amendment is not None
                    or any(
                        amendment is not None
                        for amendment in (
                            self.operational_amendment,
                            self.evaluator_amendment,
                            self.preflight_bootstrap_amendment,
                            self.matrix_amendment,
                            self.parent_import_retry_amendment,
                            self.p95_authority_retry_amendment,
                            self.stage0_evaluator_retry_amendment,
                            self.qref_identity_retry_amendment,
                            self.qref_summary_equivalence_amendment,
                            self.p95_runner_binding_retry_amendment,
                            self.qref_receipt_verifier_retry_amendment,
                            self.p95_consumer_adapter_retry_amendment,
                        )
                    )
                ):
                    raise PilotContractError(
                        "V2.11.3 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2113_forward_boundary"] = _thaw_json(
                    self.v2113_forward_boundary
                )
                result["v2113_consumer_adapter_amendment"] = _thaw_json(
                    self.v2113_consumer_adapter_amendment
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_4:
                if (
                    self.v2114_forward_boundary is None
                    or self.v2114_authority_normalization_amendment is None
                ):
                    raise PilotContractError(
                        "V2.11.4 contract lacks its forward boundary or "
                        "authority-normalization amendment"
                    )
                if (
                    self.v211_forward_boundary is not None
                    or self.v2111_forward_boundary is not None
                    or self.v2111_preflight_bootstrap_amendment is not None
                    or self.v2112_forward_boundary is not None
                    or self.v2112_recovery_amendment is not None
                    or self.v2113_forward_boundary is not None
                    or self.v2113_consumer_adapter_amendment is not None
                    or any(
                        amendment is not None
                        for amendment in (
                            self.operational_amendment,
                            self.evaluator_amendment,
                            self.preflight_bootstrap_amendment,
                            self.matrix_amendment,
                            self.parent_import_retry_amendment,
                            self.p95_authority_retry_amendment,
                            self.stage0_evaluator_retry_amendment,
                            self.qref_identity_retry_amendment,
                            self.qref_summary_equivalence_amendment,
                            self.p95_runner_binding_retry_amendment,
                            self.qref_receipt_verifier_retry_amendment,
                            self.p95_consumer_adapter_retry_amendment,
                        )
                    )
                ):
                    raise PilotContractError(
                        "V2.11.4 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2114_forward_boundary"] = _thaw_json(
                    self.v2114_forward_boundary
                )
                result["v2114_authority_normalization_amendment"] = _thaw_json(
                    self.v2114_authority_normalization_amendment
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_5:
                if (
                    self.v2115_forward_boundary is None
                    or self.v2115_consumer_authority_normalization_amendment is None
                ):
                    raise PilotContractError(
                        "V2.11.5 contract lacks its forward boundary or "
                        "consumer-authority normalization amendment"
                    )
                if (
                    self.v211_forward_boundary is not None
                    or self.v2111_forward_boundary is not None
                    or self.v2111_preflight_bootstrap_amendment is not None
                    or self.v2112_forward_boundary is not None
                    or self.v2112_recovery_amendment is not None
                    or self.v2113_forward_boundary is not None
                    or self.v2113_consumer_adapter_amendment is not None
                    or self.v2114_forward_boundary is not None
                    or self.v2114_authority_normalization_amendment is not None
                    or any(
                        amendment is not None
                        for amendment in (
                            self.operational_amendment,
                            self.evaluator_amendment,
                            self.preflight_bootstrap_amendment,
                            self.matrix_amendment,
                            self.parent_import_retry_amendment,
                            self.p95_authority_retry_amendment,
                            self.stage0_evaluator_retry_amendment,
                            self.qref_identity_retry_amendment,
                            self.qref_summary_equivalence_amendment,
                            self.p95_runner_binding_retry_amendment,
                            self.qref_receipt_verifier_retry_amendment,
                            self.p95_consumer_adapter_retry_amendment,
                        )
                    )
                ):
                    raise PilotContractError(
                        "V2.11.5 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2115_forward_boundary"] = _thaw_json(
                    self.v2115_forward_boundary
                )
                result["v2115_consumer_authority_normalization_amendment"] = _thaw_json(
                    self.v2115_consumer_authority_normalization_amendment
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_6:
                if self.v2116_continuation_boundary is None:
                    raise PilotContractError(
                        "V2.11.6 contract lacks its continuation boundary"
                    )
                if any(
                    item is not None
                    for item in (
                        self.v211_forward_boundary,
                        self.v2111_forward_boundary,
                        self.v2111_preflight_bootstrap_amendment,
                        self.v2112_forward_boundary,
                        self.v2112_recovery_amendment,
                        self.v2113_forward_boundary,
                        self.v2113_consumer_adapter_amendment,
                        self.v2114_forward_boundary,
                        self.v2114_authority_normalization_amendment,
                        self.v2115_forward_boundary,
                        self.v2115_consumer_authority_normalization_amendment,
                        self.v2117_recovery_boundary,
                        self.v2118_recovery_boundary,
                        self.v2119_recovery_boundary,
                        self.v21110_recovery_boundary,
                        self.operational_amendment,
                        self.evaluator_amendment,
                        self.preflight_bootstrap_amendment,
                        self.matrix_amendment,
                        self.parent_import_retry_amendment,
                        self.p95_authority_retry_amendment,
                        self.stage0_evaluator_retry_amendment,
                        self.qref_identity_retry_amendment,
                        self.qref_summary_equivalence_amendment,
                        self.p95_runner_binding_retry_amendment,
                        self.qref_receipt_verifier_retry_amendment,
                        self.p95_consumer_adapter_retry_amendment,
                    )
                ):
                    raise PilotContractError(
                        "V2.11.6 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2116_continuation_boundary"] = _thaw_json(
                    self.v2116_continuation_boundary
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_7:
                if self.v2117_recovery_boundary is None:
                    raise PilotContractError(
                        "V2.11.7 contract lacks its recovery boundary"
                    )
                if any(
                    item is not None
                    for item in (
                        self.v211_forward_boundary,
                        self.v2111_forward_boundary,
                        self.v2111_preflight_bootstrap_amendment,
                        self.v2112_forward_boundary,
                        self.v2112_recovery_amendment,
                        self.v2113_forward_boundary,
                        self.v2113_consumer_adapter_amendment,
                        self.v2114_forward_boundary,
                        self.v2114_authority_normalization_amendment,
                        self.v2115_forward_boundary,
                        self.v2115_consumer_authority_normalization_amendment,
                        self.v2116_continuation_boundary,
                        self.v2118_recovery_boundary,
                        self.v2119_recovery_boundary,
                        self.v21110_recovery_boundary,
                        self.operational_amendment,
                        self.evaluator_amendment,
                        self.preflight_bootstrap_amendment,
                        self.matrix_amendment,
                        self.parent_import_retry_amendment,
                        self.p95_authority_retry_amendment,
                        self.stage0_evaluator_retry_amendment,
                        self.qref_identity_retry_amendment,
                        self.qref_summary_equivalence_amendment,
                        self.p95_runner_binding_retry_amendment,
                        self.qref_receipt_verifier_retry_amendment,
                        self.p95_consumer_adapter_retry_amendment,
                    )
                ):
                    raise PilotContractError(
                        "V2.11.7 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2117_recovery_boundary"] = _thaw_json(
                    self.v2117_recovery_boundary
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_8:
                if self.v2118_recovery_boundary is None:
                    raise PilotContractError(
                        "V2.11.8 contract lacks its recovery boundary"
                    )
                if any(
                    item is not None
                    for item in (
                        self.v211_forward_boundary,
                        self.v2111_forward_boundary,
                        self.v2111_preflight_bootstrap_amendment,
                        self.v2112_forward_boundary,
                        self.v2112_recovery_amendment,
                        self.v2113_forward_boundary,
                        self.v2113_consumer_adapter_amendment,
                        self.v2114_forward_boundary,
                        self.v2114_authority_normalization_amendment,
                        self.v2115_forward_boundary,
                        self.v2115_consumer_authority_normalization_amendment,
                        self.v2116_continuation_boundary,
                        self.v2117_recovery_boundary,
                        self.v2119_recovery_boundary,
                        self.v21110_recovery_boundary,
                        self.operational_amendment,
                        self.evaluator_amendment,
                        self.preflight_bootstrap_amendment,
                        self.matrix_amendment,
                        self.parent_import_retry_amendment,
                        self.p95_authority_retry_amendment,
                        self.stage0_evaluator_retry_amendment,
                        self.qref_identity_retry_amendment,
                        self.qref_summary_equivalence_amendment,
                        self.p95_runner_binding_retry_amendment,
                        self.qref_receipt_verifier_retry_amendment,
                        self.p95_consumer_adapter_retry_amendment,
                    )
                ):
                    raise PilotContractError(
                        "V2.11.8 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2118_recovery_boundary"] = _thaw_json(
                    self.v2118_recovery_boundary
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_9:
                if self.v2119_recovery_boundary is None:
                    raise PilotContractError(
                        "V2.11.9 contract lacks its recovery boundary"
                    )
                if any(
                    item is not None
                    for item in (
                        self.v211_forward_boundary,
                        self.v2111_forward_boundary,
                        self.v2111_preflight_bootstrap_amendment,
                        self.v2112_forward_boundary,
                        self.v2112_recovery_amendment,
                        self.v2113_forward_boundary,
                        self.v2113_consumer_adapter_amendment,
                        self.v2114_forward_boundary,
                        self.v2114_authority_normalization_amendment,
                        self.v2115_forward_boundary,
                        self.v2115_consumer_authority_normalization_amendment,
                        self.v2116_continuation_boundary,
                        self.v2117_recovery_boundary,
                        self.v2118_recovery_boundary,
                        self.v21110_recovery_boundary,
                        self.operational_amendment,
                        self.evaluator_amendment,
                        self.preflight_bootstrap_amendment,
                        self.matrix_amendment,
                        self.parent_import_retry_amendment,
                        self.p95_authority_retry_amendment,
                        self.stage0_evaluator_retry_amendment,
                        self.qref_identity_retry_amendment,
                        self.qref_summary_equivalence_amendment,
                        self.p95_runner_binding_retry_amendment,
                        self.qref_receipt_verifier_retry_amendment,
                        self.p95_consumer_adapter_retry_amendment,
                    )
                ):
                    raise PilotContractError(
                        "V2.11.9 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v2119_recovery_boundary"] = _thaw_json(
                    self.v2119_recovery_boundary
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_10:
                if self.v21110_recovery_boundary is None:
                    raise PilotContractError(
                        "V2.11.10 contract lacks its recovery boundary"
                    )
                if any(
                    item is not None
                    for item in (
                        self.v211_forward_boundary,
                        self.v2111_forward_boundary,
                        self.v2111_preflight_bootstrap_amendment,
                        self.v2112_forward_boundary,
                        self.v2112_recovery_amendment,
                        self.v2113_forward_boundary,
                        self.v2113_consumer_adapter_amendment,
                        self.v2114_forward_boundary,
                        self.v2114_authority_normalization_amendment,
                        self.v2115_forward_boundary,
                        self.v2115_consumer_authority_normalization_amendment,
                        self.v2116_continuation_boundary,
                        self.v2117_recovery_boundary,
                        self.v2118_recovery_boundary,
                        self.v2119_recovery_boundary,
                        self.operational_amendment,
                        self.evaluator_amendment,
                        self.preflight_bootstrap_amendment,
                        self.matrix_amendment,
                        self.parent_import_retry_amendment,
                        self.p95_authority_retry_amendment,
                        self.stage0_evaluator_retry_amendment,
                        self.qref_identity_retry_amendment,
                        self.qref_summary_equivalence_amendment,
                        self.p95_runner_binding_retry_amendment,
                        self.qref_receipt_verifier_retry_amendment,
                        self.p95_consumer_adapter_retry_amendment,
                    )
                ):
                    raise PilotContractError(
                        "V2.11.10 is independently canonicalized and cannot "
                        "carry an earlier amendment or forward boundary"
                    )
                result["v21110_recovery_boundary"] = _thaw_json(
                    self.v21110_recovery_boundary
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_11_11:
                if self.v21111_fresh_cohort_boundary is None:
                    raise PilotContractError(
                        "V2.11.11 contract lacks its fresh-cohort boundary"
                    )
                if any(
                    item is not None
                    for item in (
                        self.v211_forward_boundary,
                        self.v2111_forward_boundary,
                        self.v2111_preflight_bootstrap_amendment,
                        self.v2112_forward_boundary,
                        self.v2112_recovery_amendment,
                        self.v2113_forward_boundary,
                        self.v2113_consumer_adapter_amendment,
                        self.v2114_forward_boundary,
                        self.v2114_authority_normalization_amendment,
                        self.v2115_forward_boundary,
                        self.v2115_consumer_authority_normalization_amendment,
                        self.v2116_continuation_boundary,
                        self.v2117_recovery_boundary,
                        self.v2118_recovery_boundary,
                        self.v2119_recovery_boundary,
                        self.v21110_recovery_boundary,
                        self.operational_amendment,
                        self.evaluator_amendment,
                        self.preflight_bootstrap_amendment,
                        self.matrix_amendment,
                        self.parent_import_retry_amendment,
                        self.p95_authority_retry_amendment,
                        self.stage0_evaluator_retry_amendment,
                        self.qref_identity_retry_amendment,
                        self.qref_summary_equivalence_amendment,
                        self.p95_runner_binding_retry_amendment,
                        self.qref_receipt_verifier_retry_amendment,
                        self.p95_consumer_adapter_retry_amendment,
                    )
                ):
                    raise PilotContractError(
                        "V2.11.11 is independently canonicalized and cannot "
                        "carry an earlier amendment or boundary"
                    )
                result["v21111_fresh_cohort_boundary"] = _thaw_json(
                    self.v21111_fresh_cohort_boundary
                )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_1:
                if self.operational_amendment is None:
                    raise PilotContractError(
                        "V2.1 contract lacks its operational amendment"
                    )
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                if self.evaluator_amendment is not None:
                    raise PilotContractError(
                        "V2.1 contract cannot carry an evaluator amendment"
                    )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_2:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                ):
                    raise PilotContractError("V2.2 contract lacks its amendment chain")
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                if self.preflight_bootstrap_amendment is not None:
                    raise PilotContractError(
                        "V2.2 contract cannot carry a preflight amendment"
                    )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_3:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                    or self.preflight_bootstrap_amendment is None
                ):
                    raise PilotContractError("V2.3 contract lacks its amendment chain")
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                result["preflight_bootstrap_amendment"] = _thaw_json(
                    self.preflight_bootstrap_amendment
                )
                if self.matrix_amendment is not None:
                    raise PilotContractError(
                        "V2.3 contract cannot carry a matrix amendment"
                    )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_4:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                    or self.preflight_bootstrap_amendment is None
                    or self.matrix_amendment is None
                ):
                    raise PilotContractError(
                        "V2.4 contract lacks its parent amendment chain"
                    )
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                result["preflight_bootstrap_amendment"] = _thaw_json(
                    self.preflight_bootstrap_amendment
                )
                result["matrix_amendment"] = _thaw_json(self.matrix_amendment)
                if self.parent_import_retry_amendment is not None:
                    raise PilotContractError(
                        "V2.4 contract cannot carry a parent-import retry amendment"
                    )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_5:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                    or self.preflight_bootstrap_amendment is None
                    or self.matrix_amendment is None
                    or self.parent_import_retry_amendment is None
                ):
                    raise PilotContractError(
                        "V2.5 contract lacks its parent amendment chain"
                    )
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                result["preflight_bootstrap_amendment"] = _thaw_json(
                    self.preflight_bootstrap_amendment
                )
                result["matrix_amendment"] = _thaw_json(self.matrix_amendment)
                result["parent_import_retry_amendment"] = _thaw_json(
                    self.parent_import_retry_amendment
                )
                if self.p95_authority_retry_amendment is not None:
                    raise PilotContractError(
                        "V2.5 contract cannot carry a p95-authority retry amendment"
                    )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_6:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                    or self.preflight_bootstrap_amendment is None
                    or self.matrix_amendment is None
                    or self.parent_import_retry_amendment is None
                    or self.p95_authority_retry_amendment is None
                ):
                    raise PilotContractError(
                        "V2.6 contract lacks its parent amendment chain"
                    )
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                result["preflight_bootstrap_amendment"] = _thaw_json(
                    self.preflight_bootstrap_amendment
                )
                result["matrix_amendment"] = _thaw_json(self.matrix_amendment)
                result["parent_import_retry_amendment"] = _thaw_json(
                    self.parent_import_retry_amendment
                )
                result["p95_authority_retry_amendment"] = _thaw_json(
                    self.p95_authority_retry_amendment
                )
                if self.stage0_evaluator_retry_amendment is not None:
                    raise PilotContractError(
                        "V2.6 contract cannot carry a Stage-0 evaluator retry amendment"
                    )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_7:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                    or self.preflight_bootstrap_amendment is None
                    or self.matrix_amendment is None
                    or self.parent_import_retry_amendment is None
                    or self.p95_authority_retry_amendment is None
                    or self.stage0_evaluator_retry_amendment is None
                ):
                    raise PilotContractError(
                        "V2.7 contract lacks its parent amendment chain"
                    )
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                result["preflight_bootstrap_amendment"] = _thaw_json(
                    self.preflight_bootstrap_amendment
                )
                result["matrix_amendment"] = _thaw_json(self.matrix_amendment)
                result["parent_import_retry_amendment"] = _thaw_json(
                    self.parent_import_retry_amendment
                )
                result["p95_authority_retry_amendment"] = _thaw_json(
                    self.p95_authority_retry_amendment
                )
                result["stage0_evaluator_retry_amendment"] = _thaw_json(
                    self.stage0_evaluator_retry_amendment
                )
                if self.qref_identity_retry_amendment is not None:
                    raise PilotContractError(
                        "V2.7 contract cannot carry a q-ref identity retry " "amendment"
                    )
            elif self.contract_id in {
                PILOT_CONTRACT_ID_V2_10,
                PILOT_CONTRACT_ID_V2_10_1,
                PILOT_CONTRACT_ID_V2_10_2,
            }:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                    or self.preflight_bootstrap_amendment is None
                    or self.matrix_amendment is None
                    or self.parent_import_retry_amendment is None
                    or self.p95_authority_retry_amendment is None
                    or self.stage0_evaluator_retry_amendment is None
                    or self.qref_identity_retry_amendment is None
                    or self.qref_summary_equivalence_amendment is None
                    or self.p95_runner_binding_retry_amendment is None
                ):
                    raise PilotContractError(
                        f"{self.contract_id} contract lacks its parent amendment chain"
                    )
                if (
                    self.contract_id
                    in {
                        PILOT_CONTRACT_ID_V2_10_1,
                        PILOT_CONTRACT_ID_V2_10_2,
                    }
                    and self.qref_receipt_verifier_retry_amendment is None
                ):
                    raise PilotContractError(
                        f"{self.contract_id} contract lacks its q-ref receipt-verifier "
                        "retry amendment"
                    )
                if (
                    self.contract_id == PILOT_CONTRACT_ID_V2_10
                    and self.qref_receipt_verifier_retry_amendment is not None
                ):
                    raise PilotContractError(
                        "V2.10 contract cannot carry a V2.10.1 q-ref "
                        "receipt-verifier retry amendment"
                    )
                if (
                    self.contract_id == PILOT_CONTRACT_ID_V2_10_2
                    and self.p95_consumer_adapter_retry_amendment is None
                ):
                    raise PilotContractError(
                        "V2.10.2 contract lacks its p95 consumer-adapter "
                        "retry amendment"
                    )
                if (
                    self.contract_id != PILOT_CONTRACT_ID_V2_10_2
                    and self.p95_consumer_adapter_retry_amendment is not None
                ):
                    raise PilotContractError(
                        "pre-V2.10.2 contract cannot carry a p95 "
                        "consumer-adapter retry amendment"
                    )
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                result["preflight_bootstrap_amendment"] = _thaw_json(
                    self.preflight_bootstrap_amendment
                )
                result["matrix_amendment"] = _thaw_json(self.matrix_amendment)
                result["parent_import_retry_amendment"] = _thaw_json(
                    self.parent_import_retry_amendment
                )
                result["p95_authority_retry_amendment"] = _thaw_json(
                    self.p95_authority_retry_amendment
                )
                result["stage0_evaluator_retry_amendment"] = _thaw_json(
                    self.stage0_evaluator_retry_amendment
                )
                result["qref_identity_retry_amendment"] = _thaw_json(
                    self.qref_identity_retry_amendment
                )
                result["qref_summary_equivalence_amendment"] = _thaw_json(
                    self.qref_summary_equivalence_amendment
                )
                result["p95_runner_binding_retry_amendment"] = _thaw_json(
                    self.p95_runner_binding_retry_amendment
                )
                if self.qref_receipt_verifier_retry_amendment is not None:
                    result["qref_receipt_verifier_retry_amendment"] = _thaw_json(
                        self.qref_receipt_verifier_retry_amendment
                    )
                if self.p95_consumer_adapter_retry_amendment is not None:
                    result["p95_consumer_adapter_retry_amendment"] = _thaw_json(
                        self.p95_consumer_adapter_retry_amendment
                    )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_9:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                    or self.preflight_bootstrap_amendment is None
                    or self.matrix_amendment is None
                    or self.parent_import_retry_amendment is None
                    or self.p95_authority_retry_amendment is None
                    or self.stage0_evaluator_retry_amendment is None
                    or self.qref_identity_retry_amendment is None
                    or self.qref_summary_equivalence_amendment is None
                ):
                    raise PilotContractError(
                        "V2.9 contract lacks its parent amendment chain"
                    )
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                result["preflight_bootstrap_amendment"] = _thaw_json(
                    self.preflight_bootstrap_amendment
                )
                result["matrix_amendment"] = _thaw_json(self.matrix_amendment)
                result["parent_import_retry_amendment"] = _thaw_json(
                    self.parent_import_retry_amendment
                )
                result["p95_authority_retry_amendment"] = _thaw_json(
                    self.p95_authority_retry_amendment
                )
                result["stage0_evaluator_retry_amendment"] = _thaw_json(
                    self.stage0_evaluator_retry_amendment
                )
                result["qref_identity_retry_amendment"] = _thaw_json(
                    self.qref_identity_retry_amendment
                )
                result["qref_summary_equivalence_amendment"] = _thaw_json(
                    self.qref_summary_equivalence_amendment
                )
                if self.p95_runner_binding_retry_amendment is not None:
                    raise PilotContractError(
                        "V2.9 contract cannot carry a p95 runner-binding "
                        "retry amendment"
                    )
            elif self.contract_id == PILOT_CONTRACT_ID_V2_8:
                if (
                    self.operational_amendment is None
                    or self.evaluator_amendment is None
                    or self.preflight_bootstrap_amendment is None
                    or self.matrix_amendment is None
                    or self.parent_import_retry_amendment is None
                    or self.p95_authority_retry_amendment is None
                    or self.stage0_evaluator_retry_amendment is None
                    or self.qref_identity_retry_amendment is None
                ):
                    raise PilotContractError(
                        "V2.8 contract lacks its parent amendment chain"
                    )
                result["operational_amendment"] = _thaw_json(self.operational_amendment)
                result["evaluator_amendment"] = _thaw_json(self.evaluator_amendment)
                result["preflight_bootstrap_amendment"] = _thaw_json(
                    self.preflight_bootstrap_amendment
                )
                result["matrix_amendment"] = _thaw_json(self.matrix_amendment)
                result["parent_import_retry_amendment"] = _thaw_json(
                    self.parent_import_retry_amendment
                )
                result["p95_authority_retry_amendment"] = _thaw_json(
                    self.p95_authority_retry_amendment
                )
                result["stage0_evaluator_retry_amendment"] = _thaw_json(
                    self.stage0_evaluator_retry_amendment
                )
                result["qref_identity_retry_amendment"] = _thaw_json(
                    self.qref_identity_retry_amendment
                )
            elif self.operational_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry an operational amendment"
                )
            elif self.evaluator_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry an evaluator amendment"
                )
            elif self.preflight_bootstrap_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a preflight amendment"
                )
            elif self.matrix_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a matrix amendment"
                )
            elif self.parent_import_retry_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a parent-import retry amendment"
                )
            elif self.p95_authority_retry_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a p95-authority retry amendment"
                )
            elif self.stage0_evaluator_retry_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a Stage-0 evaluator retry amendment"
                )
            elif self.qref_identity_retry_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a q-ref identity retry "
                    "amendment"
                )
            elif self.qref_summary_equivalence_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a q-ref summary "
                    "equivalence amendment"
                )
            elif self.p95_runner_binding_retry_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a p95 runner-binding "
                    "retry amendment"
                )
            elif self.qref_receipt_verifier_retry_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a q-ref "
                    "receipt-verifier retry amendment"
                )
            elif self.p95_consumer_adapter_retry_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry a p95 "
                    "consumer-adapter retry amendment"
                )
        return result


def _assert_v2_1_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Fail closed if the amendment changes any scientific design field."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.1 science-critical field {field!r} differs from V2"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.1 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.1 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.1 release requirements differ beyond tag/CI placeholders"
        )

    expected_budgets = _json_copy(base["budgets"])
    expected_budgets["total_usd"] = 25.0
    expected_budgets["automatic_reserve_usd"] = 1.0
    expected_budgets["stage_usd_caps"] = {
        "capability": 3.0701145,
        "calibration": 3.0,
        "core": 13.0,
        "cross_model": 4.9298855,
        "manual_reserve": 1.0,
    }
    if _json_copy(expanded["budgets"]) != expected_budgets:
        raise PilotContractError("V2.1 budget reallocation drifted")

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.1 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_1
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_1
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_1
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.1-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.1 allowed identifier/CI amendment drifted")


def _expand_v2_1_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the compact V2.1 operational amendment over the frozen V2."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "operational_amendment",
            "integrity",
        },
        name="V2.1 amendment overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_1
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.1 amendment overlay identity drifted")

    integrity = _mapping(value["integrity"], "V2.1 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.1 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.1 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.1 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.1 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.1 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.1 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2,
        "canonical_sha256": PILOT_CONTRACT_V2_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.1 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError("V2.1 base contract must be the sibling pilot_v2.yaml")
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.1 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.1 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "budgets",
            "denominator_policy",
        },
        name="V2.1 changes",
    )
    implementation_change = _mapping(
        changes["implementation"],
        "V2.1 changes.implementation",
    )
    _strict_keys(
        implementation_change,
        required={"required_git_tag"},
        name="V2.1 changes.implementation",
    )
    if implementation_change["required_git_tag"] != PILOT_CONTRACT_TAG_V2_1:
        raise PilotContractError("V2.1 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.1 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.1 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.1 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.1 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_1:
        raise PilotContractError("V2.1 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.1 changes.release_requirements.expected_ci",
    )

    budget_change = _mapping(changes["budgets"], "V2.1 changes.budgets")
    _strict_keys(
        budget_change,
        required={
            "total_usd",
            "automatic_reserve_usd",
            "stage_usd_caps",
        },
        name="V2.1 changes.budgets",
    )
    expected_budget_change = {
        "total_usd": 25.0,
        "automatic_reserve_usd": 1.0,
        "stage_usd_caps": {
            "capability": 3.0701145,
            "calibration": 3.0,
            "core": 13.0,
            "cross_model": 4.9298855,
            "manual_reserve": 1.0,
        },
    }
    if _json_copy(budget_change) != expected_budget_change:
        raise PilotContractError("V2.1 overlay budget caps drifted")

    denominator_change = _mapping(
        changes["denominator_policy"],
        "V2.1 changes.denominator_policy",
    )
    _strict_keys(
        denominator_change,
        required={"policy_id"},
        name="V2.1 changes.denominator_policy",
    )
    if denominator_change["policy_id"] != "finevo-pilot-v2.1-itt":
        raise PilotContractError("V2.1 denominator policy identifier drifted")

    amendment = _validate_v2_1_operational_amendment(value["operational_amendment"])
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_1
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_1
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_1
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["budgets"]["total_usd"] = 25.0
    expanded["budgets"]["automatic_reserve_usd"] = 1.0
    expanded["budgets"]["stage_usd_caps"] = _json_copy(budget_change["stage_usd_caps"])
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.1-itt"
    expanded["operational_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_1_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    return expanded


def _assert_v2_2_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only release identity plus the evaluator-amendment chain to change."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.2 science-critical field {field!r} differs from V2.1"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.2 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.2 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.2 release requirements differ beyond tag/CI placeholders"
        )

    if _json_copy(expanded["budgets"]) != _json_copy(base["budgets"]):
        raise PilotContractError("V2.2 budgets differ from V2.1")
    if _json_copy(expanded["operational_amendment"]) != _json_copy(
        base["operational_amendment"]
    ):
        raise PilotContractError(
            "V2.2 altered the immutable V2.1 operational amendment"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.2 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_2
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_2
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_2
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.2-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.2 allowed identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.2 science-design hash drifted")


def _expand_v2_2_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the compact evaluator-only V2.2 amendment over frozen V2.1."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "evaluator_amendment",
            "integrity",
        },
        name="V2.2 evaluator amendment overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_2
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_2
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.2 evaluator overlay identity drifted")

    integrity = _mapping(value["integrity"], "V2.2 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.2 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.2 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.2 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.2 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.2 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.2 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_1.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_1,
        "canonical_sha256": PILOT_CONTRACT_V2_1_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.2 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_1.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.2 base contract must be the sibling pilot_v2_1.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_1
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_1_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.2 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.2 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.2 changes",
    )
    implementation_change = _mapping(
        changes["implementation"],
        "V2.2 changes.implementation",
    )
    _strict_keys(
        implementation_change,
        required={"required_git_tag"},
        name="V2.2 changes.implementation",
    )
    if implementation_change["required_git_tag"] != PILOT_CONTRACT_TAG_V2_2:
        raise PilotContractError("V2.2 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.2 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.2 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.2 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.2 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_2:
        raise PilotContractError("V2.2 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.2 changes.release_requirements.expected_ci",
    )

    denominator_change = _mapping(
        changes["denominator_policy"],
        "V2.2 changes.denominator_policy",
    )
    _strict_keys(
        denominator_change,
        required={"policy_id"},
        name="V2.2 changes.denominator_policy",
    )
    if denominator_change["policy_id"] != "finevo-pilot-v2.2-itt":
        raise PilotContractError("V2.2 denominator policy identifier drifted")

    evaluator_amendment = _validate_v2_2_evaluator_amendment(
        value["evaluator_amendment"]
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_2
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_2
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_2
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.2-itt"
    expanded["evaluator_amendment"] = _thaw_json(evaluator_amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_2_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    return expanded


def _assert_v2_3_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Permit only release identity plus the audited bootstrap amendment."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.3 science-critical field {field!r} differs from V2.2"
            )
    if _json_copy(expanded["budgets"]) != _json_copy(base["budgets"]):
        raise PilotContractError("V2.3 budget caps differ from V2.2")
    if _json_copy(expanded["operational_amendment"]) != _json_copy(
        base["operational_amendment"]
    ) or _json_copy(expanded["evaluator_amendment"]) != _json_copy(
        base["evaluator_amendment"]
    ):
        raise PilotContractError("V2.3 inherited amendment chain drifted")

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.3 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.3 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.3 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.3 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_3
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_3
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_3
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.3-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.3 allowed identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.3 science-design hash drifted")


def _expand_v2_3_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the V2.3 preflight-bootstrap amendment over frozen V2.2."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "preflight_bootstrap_amendment",
            "integrity",
        },
        name="V2.3 preflight bootstrap overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_3
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_3
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.3 preflight overlay identity drifted")

    integrity = _mapping(value["integrity"], "V2.3 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.3 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.3 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.3 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.3 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.3 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.3 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_2.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_2,
        "canonical_sha256": PILOT_CONTRACT_V2_2_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.3 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_2.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.3 base contract must be the sibling pilot_v2_2.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_2
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_2_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.3 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.3 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.3 changes",
    )
    implementation_change = _mapping(
        changes["implementation"],
        "V2.3 changes.implementation",
    )
    _strict_keys(
        implementation_change,
        required={"required_git_tag"},
        name="V2.3 changes.implementation",
    )
    if implementation_change["required_git_tag"] != PILOT_CONTRACT_TAG_V2_3:
        raise PilotContractError("V2.3 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.3 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.3 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.3 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.3 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_3:
        raise PilotContractError("V2.3 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.3 changes.release_requirements.expected_ci",
    )

    denominator_change = _mapping(
        changes["denominator_policy"],
        "V2.3 changes.denominator_policy",
    )
    _strict_keys(
        denominator_change,
        required={"policy_id"},
        name="V2.3 changes.denominator_policy",
    )
    if denominator_change["policy_id"] != "finevo-pilot-v2.3-itt":
        raise PilotContractError("V2.3 denominator policy identifier drifted")

    amendment = _validate_v2_3_preflight_bootstrap_amendment(
        value["preflight_bootstrap_amendment"]
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_3
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_3
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_3
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.3-itt"
    expanded["preflight_bootstrap_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_3_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    return expanded


def _assert_v2_4_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Permit only the exact prospective local-first matrix amendment."""

    for field in (
        "seeds",
        "narratives",
        "shocks",
        "stop_go",
        "parameter_dispatch_policy",
        "task_output_contracts",
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.4 inherited field {field!r} differs from V2.3"
            )

    expected_profiles = {
        profile_id: _json_copy(base["provider_profiles"][profile_id])
        for profile_id in (
            "gpt52_main",
            "llama33_local_controlled",
            "qref_scripted",
        )
    }
    if _json_copy(expanded["provider_profiles"]) != expected_profiles:
        raise PilotContractError("V2.4 active provider profile selection drifted")

    expected_arms = _json_copy(base["arms"])
    expected_arms["parent-import"] = _v2_4_expected_parent_import_arm()
    if _json_copy(expanded["arms"]) != expected_arms:
        raise PilotContractError("V2.4 arm registry drifted")

    expected_utility = _json_copy(base["utility"])
    expected_utility["q_ref_resolution"]["required_before"] = [
        "stage0-calibration",
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    ]
    expected_utility["selection_rule"]["required_before"] = [
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    ]
    if _json_copy(expanded["utility"]) != expected_utility:
        raise PilotContractError("V2.4 outcome-blind utility dependencies drifted")

    if _json_copy(expanded["model_roles"]) != _v2_4_expected_model_roles():
        raise PilotContractError("V2.4 model roles drifted")
    if _json_copy(expanded["stages"]) != _v2_4_expected_stages():
        raise PilotContractError("V2.4 registered stage matrix drifted")
    if _json_copy(expanded["non_claims"]) != _v2_4_expected_non_claims():
        raise PilotContractError("V2.4 non-claim boundary drifted")
    if _json_copy(expanded["matrix_amendment"]) != (_v2_4_expected_matrix_amendment()):
        raise PilotContractError("V2.4 parent/matrix amendment drifted")

    expected_budgets = _json_copy(base["budgets"])
    expected_budgets.update(
        {
            "total_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
            "automatic_reserve_usd": 1.0,
            "max_provider_completions": 7500,
            "max_storage_bytes": 5_000_000_000,
            "stage_usd_caps": {
                "parent_v23": 3.212770875,
                "local": 0.0,
                "hosted_confirmatory": _PILOT_V2_4_HOSTED_STAGE_CAP_USD,
                "manual_reserve": 1.0,
            },
        }
    )
    if _json_copy(expanded["budgets"]) != expected_budgets:
        raise PilotContractError("V2.4 authorized budget envelope drifted")

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.4 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.4 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.4 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.4 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_4
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_4
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_4
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.4-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.4 identifier/CI amendment drifted")


def _expand_v2_4_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the prospective V2.4 matrix amendment over immutable V2.3."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "matrix_amendment",
            "integrity",
        },
        name="V2.4 matrix amendment overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_4
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_4
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.4 matrix overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_4_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.4 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.4 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.4 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.4 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.4 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.4 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.4 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.4 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_3.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_3,
        "canonical_sha256": PILOT_CONTRACT_V2_3_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.4 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_3.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.4 base contract must be the sibling pilot_v2_3.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_3
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_3_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.4 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.4 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
            "active_provider_profile_ids",
            "budgets",
            "matrix_profile_id",
        },
        name="V2.4 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_4
    }:
        raise PilotContractError("V2.4 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.4 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.4 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.4 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.4 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_4:
        raise PilotContractError("V2.4 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.4 changes.release_requirements.expected_ci",
    )

    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.4-itt"
    }:
        raise PilotContractError("V2.4 denominator identifier drifted")
    if list(changes["active_provider_profile_ids"]) != [
        "gpt52_main",
        "llama33_local_controlled",
        "qref_scripted",
    ]:
        raise PilotContractError("V2.4 active provider profile list drifted")
    expected_budget_change = {
        "total_usd": _PILOT_V2_4_AUTHORIZED_HARD_CAP_USD,
        "automatic_reserve_usd": 1.0,
        "max_provider_completions": 7500,
        "max_storage_bytes": 5_000_000_000,
        "stage_usd_caps": {
            "parent_v23": 3.212770875,
            "local": 0.0,
            "hosted_confirmatory": _PILOT_V2_4_HOSTED_STAGE_CAP_USD,
            "manual_reserve": 1.0,
        },
    }
    if _json_copy(changes["budgets"]) != expected_budget_change:
        raise PilotContractError("V2.4 authorized budget change drifted")
    if changes["matrix_profile_id"] != (
        "local-llama-full-ad-gpt52-fixed-confirmatory-cadb-v1"
    ):
        raise PilotContractError("V2.4 matrix profile identifier drifted")

    amendment = _validate_v2_4_matrix_amendment(value["matrix_amendment"])
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_4
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_4
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_4
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.4-itt"
    expanded["provider_profiles"] = {
        profile_id: expanded["provider_profiles"][profile_id]
        for profile_id in changes["active_provider_profile_ids"]
    }
    expanded["arms"]["parent-import"] = _v2_4_expected_parent_import_arm()
    expanded["model_roles"] = _v2_4_expected_model_roles()
    expanded["stages"] = _v2_4_expected_stages()
    expanded["utility"]["q_ref_resolution"]["required_before"] = [
        "stage0-calibration",
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    ]
    expanded["utility"]["selection_rule"]["required_before"] = [
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    ]
    expanded["budgets"].update(_json_copy(changes["budgets"]))
    expanded["non_claims"] = _v2_4_expected_non_claims()
    expanded["matrix_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_4_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_4_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.4 frozen canonical hash drifted")
    return expanded


def _assert_v2_5_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only a new release identity and the zero-call retry metadata."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.5 science-critical field {field!r} differs from V2.4"
            )
    for field in (
        "budgets",
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
        "matrix_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.5 inherited field {field!r} differs from V2.4"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.5 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.5 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.5 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.5 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_5
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_5
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_5
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.5-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.5 identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.5 science-design hash differs from V2.4")
    if _json_copy(expanded["parent_import_retry_amendment"]) != (
        _v2_5_expected_parent_import_retry_amendment()
    ):
        raise PilotContractError("V2.5 retry amendment drifted")


def _expand_v2_5_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the zero-call V2.5 retry over immutable terminal V2.4."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "parent_import_retry_amendment",
            "integrity",
        },
        name="V2.5 parent-import retry overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_5
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_5
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.5 retry overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_5_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.5 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.5 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.5 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.5 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.5 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.5 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.5 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.5 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_4.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_4,
        "canonical_sha256": PILOT_CONTRACT_V2_4_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.5 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_4.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.5 base contract must be the sibling pilot_v2_4.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_4
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_4_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.5 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.5 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.5 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_5
    }:
        raise PilotContractError("V2.5 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.5 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.5 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.5 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.5 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_5:
        raise PilotContractError("V2.5 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.5 changes.release_requirements.expected_ci",
    )
    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.5-itt"
    }:
        raise PilotContractError("V2.5 denominator identifier drifted")

    amendment = _validate_v2_5_parent_import_retry_amendment(
        value["parent_import_retry_amendment"]
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_5
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_5
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_5
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.5-itt"
    expanded["parent_import_retry_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_5_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_5_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.5 frozen canonical hash drifted")
    return expanded


def _assert_v2_6_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only V2.6 release, parent-debit, and authority retry metadata."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.6 science-critical field {field!r} differs from V2.5"
            )
    for field in (
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
        "matrix_amendment",
        "parent_import_retry_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.6 inherited field {field!r} differs from V2.5"
            )

    if _json_copy(expanded["budgets"]) != _json_copy(base["budgets"]):
        raise PilotContractError("V2.6 budgets differ from V2.5")

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.6 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.6 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.6 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.6 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_6
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_6
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_6
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.6-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.6 identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.6 science-design hash differs from V2.5")
    if _json_copy(expanded["p95_authority_retry_amendment"]) != (
        _v2_6_expected_p95_authority_retry_amendment(
            status=overlay_status,
        )
    ):
        raise PilotContractError("V2.6 p95-authority retry amendment drifted")


def _expand_v2_6_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the V2.6 authority-interface retry over terminal V2.5."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "p95_authority_retry_amendment",
            "integrity",
        },
        name="V2.6 p95-authority retry overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_6
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_6
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.6 retry overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_6_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.6 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.6 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.6 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.6 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.6 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.6 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.6 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.6 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_5.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_5,
        "canonical_sha256": PILOT_CONTRACT_V2_5_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.6 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_5.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.6 base contract must be the sibling pilot_v2_5.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_5
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_5_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.6 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.6 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.6 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_6
    }:
        raise PilotContractError("V2.6 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.6 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.6 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.6 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.6 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_6:
        raise PilotContractError("V2.6 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.6 changes.release_requirements.expected_ci",
    )
    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.6-itt"
    }:
        raise PilotContractError("V2.6 denominator identifier drifted")
    amendment = _validate_v2_6_p95_authority_retry_amendment(
        value["p95_authority_retry_amendment"],
        status=str(value["status"]),
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_6
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_6
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_6
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.6-itt"
    expanded["p95_authority_retry_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_6_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_6_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.6 frozen canonical hash drifted")
    return expanded


def _assert_v2_7_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only V2.7 release identity and the Stage-0 retry metadata."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.7 science-critical field {field!r} differs from V2.6"
            )
    for field in (
        "budgets",
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
        "matrix_amendment",
        "parent_import_retry_amendment",
        "p95_authority_retry_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.7 inherited field {field!r} differs from V2.6"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.7 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.7 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.7 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.7 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_7
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_7
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_7
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.7-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.7 identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.7 science-design hash differs from V2.6")
    if _json_copy(expanded["stage0_evaluator_retry_amendment"]) != (
        _v2_7_expected_stage0_evaluator_retry_amendment(
            status=overlay_status,
        )
    ):
        raise PilotContractError("V2.7 Stage-0 evaluator retry amendment drifted")


def _expand_v2_7_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the V2.7 Stage-0 evaluator/import retry over terminal V2.6."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "stage0_evaluator_retry_amendment",
            "integrity",
        },
        name="V2.7 Stage-0 evaluator retry overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_7
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_7
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.7 retry overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_7_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.7 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.7 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.7 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.7 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.7 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.7 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.7 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.7 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_6.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_6,
        "canonical_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.7 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_6.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.7 base contract must be the sibling pilot_v2_6.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_6
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_6_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.7 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.7 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.7 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_7
    }:
        raise PilotContractError("V2.7 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.7 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.7 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.7 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.7 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_7:
        raise PilotContractError("V2.7 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.7 changes.release_requirements.expected_ci",
    )
    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.7-itt"
    }:
        raise PilotContractError("V2.7 denominator identifier drifted")

    amendment = _validate_v2_7_stage0_evaluator_retry_amendment(
        value["stage0_evaluator_retry_amendment"],
        status=str(value["status"]),
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_7
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_7
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_7
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.7-itt"
    expanded["stage0_evaluator_retry_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_7_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_7_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.7 frozen canonical hash drifted")
    return expanded


def _assert_v2_8_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only V2.8 release identity and q-ref retry metadata."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.8 science-critical field {field!r} differs from V2.7"
            )
    for field in (
        "budgets",
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
        "matrix_amendment",
        "parent_import_retry_amendment",
        "p95_authority_retry_amendment",
        "stage0_evaluator_retry_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.8 inherited field {field!r} differs from V2.7"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.8 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.8 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.8 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.8 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_8
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_8
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_8
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.8-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.8 identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.8 science-design hash differs from V2.7")
    if _json_copy(expanded["qref_identity_retry_amendment"]) != (
        _v2_8_expected_qref_identity_retry_amendment(
            status=overlay_status,
        )
    ):
        raise PilotContractError("V2.8 q-ref identity retry amendment drifted")


def _expand_v2_8_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the V2.8 q-ref identity retry over terminal V2.7."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "qref_identity_retry_amendment",
            "integrity",
        },
        name="V2.8 q-ref identity retry overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_8
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_8
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.8 retry overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_8_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.8 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.8 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.8 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.8 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.8 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.8 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.8 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.8 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_7.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_7,
        "canonical_sha256": PILOT_CONTRACT_V2_7_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.8 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_7.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.8 base contract must be the sibling pilot_v2_7.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_7
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_7_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.8 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.8 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.8 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_8
    }:
        raise PilotContractError("V2.8 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.8 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.8 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.8 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.8 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_8:
        raise PilotContractError("V2.8 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.8 changes.release_requirements.expected_ci",
    )
    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.8-itt"
    }:
        raise PilotContractError("V2.8 denominator identifier drifted")

    amendment = _validate_v2_8_qref_identity_retry_amendment(
        value["qref_identity_retry_amendment"],
        status=str(value["status"]),
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_8
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_8
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_8
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.8-itt"
    expanded["qref_identity_retry_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_8_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_8_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.8 frozen canonical hash drifted")
    return expanded


def _assert_v2_9_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only V2.9 release identity and summary-equivalence metadata."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.9 science-critical field {field!r} differs from V2.8"
            )
    for field in (
        "budgets",
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
        "matrix_amendment",
        "parent_import_retry_amendment",
        "p95_authority_retry_amendment",
        "stage0_evaluator_retry_amendment",
        "qref_identity_retry_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.9 inherited field {field!r} differs from V2.8"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.9 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.9 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.9 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.9 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_9
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_9
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_9
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.9-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.9 identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.9 science-design hash differs from V2.8")
    if _json_copy(expanded["qref_summary_equivalence_amendment"]) != (
        _v2_9_expected_qref_summary_equivalence_amendment(
            status=overlay_status,
        )
    ):
        raise PilotContractError("V2.9 q-ref summary-equivalence amendment drifted")


def _expand_v2_9_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the V2.9 deterministic-summary retry over terminal V2.8."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "qref_summary_equivalence_amendment",
            "integrity",
        },
        name="V2.9 q-ref summary-equivalence overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_9
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_9
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.9 retry overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_9_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.9 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.9 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.9 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.9 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.9 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.9 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.9 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.9 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_8.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_8,
        "canonical_sha256": PILOT_CONTRACT_V2_8_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.9 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_8.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.9 base contract must be the sibling pilot_v2_8.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_8
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_8_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.9 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.9 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.9 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_9
    }:
        raise PilotContractError("V2.9 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.9 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.9 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.9 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.9 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_9:
        raise PilotContractError("V2.9 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.9 changes.release_requirements.expected_ci",
    )
    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.9-itt"
    }:
        raise PilotContractError("V2.9 denominator identifier drifted")

    amendment = _validate_v2_9_qref_summary_equivalence_amendment(
        value["qref_summary_equivalence_amendment"],
        status=str(value["status"]),
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_9
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_9
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_9
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.9-itt"
    expanded["qref_summary_equivalence_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_9_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_9_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.9 frozen canonical hash drifted")
    return expanded


def _assert_v2_10_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only V2.10 release identity and p95-binding retry metadata."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.10 science-critical field {field!r} differs from V2.9"
            )
    for field in (
        "budgets",
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
        "matrix_amendment",
        "parent_import_retry_amendment",
        "p95_authority_retry_amendment",
        "stage0_evaluator_retry_amendment",
        "qref_identity_retry_amendment",
        "qref_summary_equivalence_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.10 inherited field {field!r} differs from V2.9"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.10 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError("V2.10 implementation differs beyond its release tag")

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.10 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.10 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_10
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_10
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_10
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.10-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.10 identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.10 science-design hash differs from V2.9")
    if _json_copy(expanded["p95_runner_binding_retry_amendment"]) != (
        _v2_10_expected_p95_runner_binding_retry_amendment(
            status=overlay_status,
        )
    ):
        raise PilotContractError("V2.10 p95 runner-binding retry amendment drifted")


def _expand_v2_10_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the V2.10 p95 runner-binding retry over terminal V2.9."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "p95_runner_binding_retry_amendment",
            "integrity",
        },
        name="V2.10 p95 runner-binding retry overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_10
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.10 retry overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_10_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.10 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.10 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.10 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.10 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.10 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.10 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.10 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.10 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_9.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_9,
        "canonical_sha256": PILOT_CONTRACT_V2_9_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.10 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_9.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.10 base contract must be the sibling pilot_v2_9.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_9
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_9_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.10 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.10 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.10 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_10
    }:
        raise PilotContractError("V2.10 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.10 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.10 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.10 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.10 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_10:
        raise PilotContractError("V2.10 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.10 changes.release_requirements.expected_ci",
    )
    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.10-itt"
    }:
        raise PilotContractError("V2.10 denominator identifier drifted")

    amendment = _validate_v2_10_p95_runner_binding_retry_amendment(
        value["p95_runner_binding_retry_amendment"],
        status=str(value["status"]),
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_10
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_10
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_10
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.10-itt"
    expanded["p95_runner_binding_retry_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_10_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_10_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.10 frozen canonical hash drifted")
    return expanded


def _assert_v2_10_1_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only V2.10.1 release identity and receipt-verifier metadata."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.10.1 science-critical field {field!r} differs from V2.10"
            )
    for field in (
        "budgets",
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
        "matrix_amendment",
        "parent_import_retry_amendment",
        "p95_authority_retry_amendment",
        "stage0_evaluator_retry_amendment",
        "qref_identity_retry_amendment",
        "qref_summary_equivalence_amendment",
        "p95_runner_binding_retry_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.10.1 inherited field {field!r} differs from V2.10"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.10.1 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError(
            "V2.10.1 implementation differs beyond its release tag"
        )

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.10.1 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.10.1 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_10_1
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_10_1
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_10_1
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.10.1-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.10.1 identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.10.1 science-design hash differs from V2.10")
    if _json_copy(
        expanded["qref_receipt_verifier_retry_amendment"]
    ) != _v2_10_1_expected_qref_receipt_verifier_retry_amendment(
        status=overlay_status,
    ):
        raise PilotContractError(
            "V2.10.1 q-ref receipt-verifier retry amendment drifted"
        )


def _expand_v2_10_1_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand V2.10.1's receipt-verifier retry over terminal V2.10."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "qref_receipt_verifier_retry_amendment",
            "integrity",
        },
        name="V2.10.1 q-ref receipt-verifier retry overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_10_1
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.10.1 retry overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.10.1 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.10.1 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.10.1 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.10.1 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.10.1 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.10.1 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.10.1 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.10.1 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_10.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_10,
        "canonical_sha256": PILOT_CONTRACT_V2_10_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.10.1 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_10.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.10.1 base contract must be the sibling pilot_v2_10.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_10
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_10_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.10.1 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.10.1 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.10.1 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_10_1
    }:
        raise PilotContractError("V2.10.1 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.10.1 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.10.1 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.10.1 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.10.1 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_10_1:
        raise PilotContractError("V2.10.1 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.10.1 changes.release_requirements.expected_ci",
    )
    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.10.1-itt"
    }:
        raise PilotContractError("V2.10.1 denominator identifier drifted")

    amendment = _validate_v2_10_1_qref_receipt_verifier_retry_amendment(
        value["qref_receipt_verifier_retry_amendment"],
        status=str(value["status"]),
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_10_1
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_10_1
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_10_1
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.10.1-itt"
    expanded["qref_receipt_verifier_retry_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_10_1_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.10.1 frozen canonical hash drifted")
    return expanded


def _assert_v2_10_2_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Allow only V2.10.2 identity and consumer-adapter retry metadata."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.10.2 science-critical field {field!r} differs from V2.10.1"
            )
    for field in (
        "budgets",
        "operational_amendment",
        "evaluator_amendment",
        "preflight_bootstrap_amendment",
        "matrix_amendment",
        "parent_import_retry_amendment",
        "p95_authority_retry_amendment",
        "stage0_evaluator_retry_amendment",
        "qref_identity_retry_amendment",
        "qref_summary_equivalence_amendment",
        "p95_runner_binding_retry_amendment",
        "qref_receipt_verifier_retry_amendment",
    ):
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.10.2 inherited field {field!r} differs from V2.10.1"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.10.2 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError(
            "V2.10.2 implementation differs beyond its release tag"
        )

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.10.2 release requirements differ beyond tag/CI identity"
        )

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.10.2 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_10_2
        or expanded["implementation"]["required_git_tag"] != PILOT_CONTRACT_TAG_V2_10_2
        or expanded["release_requirements"]["tag"] != PILOT_CONTRACT_TAG_V2_10_2
        or expanded["denominator_policy"]["policy_id"] != "finevo-pilot-v2.10.2-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.10.2 identifier/CI amendment drifted")
    if science_design_sha256(expanded) != PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256:
        raise PilotContractError("V2.10.2 science-design hash differs from V2.10.1")
    if _json_copy(
        expanded["p95_consumer_adapter_retry_amendment"]
    ) != _v2_10_2_expected_p95_consumer_adapter_retry_amendment(
        status=overlay_status,
    ):
        raise PilotContractError("V2.10.2 p95 consumer-adapter retry amendment drifted")


def _expand_v2_10_2_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand V2.10.2's consumer-adapter retry over terminal V2.10.1."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "p95_consumer_adapter_retry_amendment",
            "integrity",
        },
        name="V2.10.2 p95 consumer-adapter retry overlay",
    )
    if (
        value["schema_version"] != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_2
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_10_2
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.10.2 retry overlay identity drifted")
    if value["status"] == "frozen" and PILOT_CONTRACT_V2_10_2_CANONICAL_SHA256 is None:
        raise PilotContractError(
            "V2.10.2 cannot be frozen before its canonical hash and CI inventory"
        )

    integrity = _mapping(value["integrity"], "V2.10.2 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.10.2 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.10.2 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.10.2 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.10.2 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.10.2 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.10.2 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2_10_1.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2_10_1,
        "canonical_sha256": PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.10.2 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2_10_1.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.10.2 base contract must be the sibling pilot_v2_10_1.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2_10_1
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.10.2 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.10.2 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "denominator_policy",
        },
        name="V2.10.2 changes",
    )
    if _json_copy(changes["implementation"]) != {
        "required_git_tag": PILOT_CONTRACT_TAG_V2_10_2
    }:
        raise PilotContractError("V2.10.2 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.10.2 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.10.2 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.10.2 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.10.2 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_10_2:
        raise PilotContractError("V2.10.2 release tag drifted")
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.10.2 changes.release_requirements.expected_ci",
    )
    if _json_copy(changes["denominator_policy"]) != {
        "policy_id": "finevo-pilot-v2.10.2-itt"
    }:
        raise PilotContractError("V2.10.2 denominator identifier drifted")

    amendment = _validate_v2_10_2_p95_consumer_adapter_retry_amendment(
        value["p95_consumer_adapter_retry_amendment"],
        status=str(value["status"]),
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_10_2
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_10_2
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_10_2
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.10.2-itt"
    expanded["p95_consumer_adapter_retry_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    _assert_v2_10_2_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    if (
        value["status"] == "frozen"
        and expanded["integrity"]["declared_sha256"]
        != PILOT_CONTRACT_V2_10_2_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.10.2 frozen canonical hash drifted")
    return expanded


def _validate_v2_4_parent_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
) -> None:
    binding = _mapping(
        amendment.get("parent_source_manifest"),
        "matrix_amendment.parent_source_manifest",
    )
    expected_path = "experiments/pilot_v2_4_parent_source_manifest.json"
    if binding.get("path") != expected_path:
        raise PilotContractError("V2.4 parent source manifest path drifted")
    manifest_path = source.parent / "pilot_v2_4_parent_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_4_parent_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.4 parent source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding.get("file_sha256"):
        raise PilotContractError("V2.4 parent source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.4 parent source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError(
            "V2.4 parent source manifest is not canonical JSON"
        ) from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.4 parent source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding.get("schema_version")
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding.get("content_sha256")
    ):
        raise PilotContractError("V2.4 parent source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding.get("content_sha256"):
        raise PilotContractError("V2.4 parent source manifest content hash drifted")


def _validate_v2_5_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
) -> None:
    binding = _mapping(
        amendment.get("source_manifest"),
        "parent_import_retry_amendment.source_manifest",
    )
    expected_binding = _v2_5_expected_parent_import_retry_amendment()["source_manifest"]
    if _thaw_json(binding) != expected_binding:
        raise PilotContractError("V2.5 source manifest binding drifted")
    manifest_path = source.parent / "pilot_v2_5_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_5_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.5 source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
        raise PilotContractError("V2.5 source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.5 source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError("V2.5 source manifest is not canonical JSON") from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.5 source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding["schema_version"]
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding["content_sha256"]
    ):
        raise PilotContractError("V2.5 source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding["content_sha256"]:
        raise PilotContractError("V2.5 source manifest content hash drifted")

    v24 = _mapping(
        manifest.get("v2_4_terminal_parent"),
        "V2.5 source manifest V2.4 parent",
    )
    v24_contract = _mapping(v24.get("contract"), "V2.4 parent contract")
    denominator = _mapping(
        v24.get("terminal_denominator"),
        "V2.4 parent terminal denominator",
    )
    v24_run_ledger = _mapping(
        denominator.get("run_ledger"),
        "V2.4 parent run-ledger binding",
    )
    failure = _mapping(
        v24.get("parent_import_failure"),
        "V2.4 parent import failure",
    )
    incremental = _mapping(
        v24.get("incremental_budget_debit"),
        "V2.4 incremental budget debit",
    )
    if (
        v24_contract.get("contract_id") != PILOT_CONTRACT_ID_V2_4
        or v24_contract.get("canonical_sha256") != PILOT_CONTRACT_V2_4_CANONICAL_SHA256
        or denominator.get("registered_cells") != 211
        or denominator.get("terminal_cells") != 211
        or denominator.get("status_counts") != {"integrity-stopped": 211}
        or denominator.get("scientific_complete") is not False
        or denominator.get("scientific_matrix_complete") is not False
        or v24_run_ledger.get("file_sha256")
        != "35da084e4a2ef5e5eb97603bb9b004561f87de6ce04ba0e6d1182a4ff04a85e4"
        or v24_run_ledger.get("internal_sha256")
        != "6ef976205f37fe675169b05fcec8806c16085aceffdafeaa4a471a002f194fd1"
        or v24_run_ledger.get("event_count") != 213
        or v24_run_ledger.get("event_chain_head")
        != "0eb71399880c1491917b5158b8ddfe5a823a5fad0613c840a78f9f3cfc92e059"
        or failure.get("provider_calls") != 0
        or failure.get("scientific_evidence") is not False
        or failure.get("scientific_effect_outcomes_available") is not False
        or failure.get("scientific_effect_outcomes_inspected") is not False
        or incremental
        != {
            "cost_usd": 0.0,
            "hosted_completions": 0,
            "storage_bytes": 518_235,
        }
    ):
        raise PilotContractError("V2.5 V2.4 terminal provenance drifted")

    v23 = _mapping(
        manifest.get("v2_3_authority_parent"),
        "V2.5 source manifest V2.3 parent",
    )
    v23_manifest = _mapping(
        v23.get("source_manifest"),
        "V2.5 V2.3 source manifest binding",
    )
    v23_debit = _mapping(
        v23.get("cumulative_budget_debit"),
        "V2.5 V2.3 budget debit",
    )
    if (
        v23.get("contract_id") != PILOT_CONTRACT_ID_V2_3
        or v23.get("contract_sha256") != PILOT_CONTRACT_V2_3_CANONICAL_SHA256
        or v23.get("science_tag") != PILOT_CONTRACT_TAG_V2_3
        or v23.get("science_tag_object") != "e985abd6749471363db6b27bda66485c0b578bb3"
        or v23.get("science_commit") != PILOT_V2_4_PARENT_RELEASE_COMMIT
        or v23_debit.get("cost_usd") != 3.212770875
        or v23_debit.get("hosted_completions") != 184
        or v23_debit.get("storage_bytes") != 4_196_087
        or v23_manifest.get("file_sha256")
        != "d6a867cd7add43818127af7778a447d579ac1ab31ed6d053bcd29d69b3cf0f33"
        or v23_manifest.get("content_sha256")
        != "7ae427fe6eac5aa6e04eddd3efa9e63405e128c782013ed3f67c35808be3cec5"
    ):
        raise PilotContractError("V2.5 V2.3 authority provenance drifted")
    v23_manifest_path = source.parent / "pilot_v2_4_parent_source_manifest.json"
    if not v23_manifest_path.is_file() or hashlib.sha256(
        v23_manifest_path.read_bytes()
    ).hexdigest() != v23_manifest.get("file_sha256"):
        raise PilotContractError("V2.5 referenced V2.3 authority manifest file drifted")

    published = _mapping(
        manifest.get("v2_4_published_evidence"),
        "V2.5 V2.4 published evidence",
    )
    if (
        published.get("status") != "complete-with-no-go"
        or published.get("scientific_complete") is not False
        or published.get("scientific_matrix_complete") is not False
        or published.get("registered_cells") != 211
        or published.get("terminal_cells") != 211
    ):
        raise PilotContractError("V2.5 V2.4 evidence boundary drifted")


def _validate_v2_6_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
    *,
    status: str,
) -> None:
    binding = _mapping(
        amendment.get("source_manifest"),
        "p95_authority_retry_amendment.source_manifest",
    )
    expected_binding = _v2_6_expected_p95_authority_retry_amendment(
        status=status,
    )["source_manifest"]
    if _thaw_json(binding) != expected_binding:
        raise PilotContractError("V2.6 source manifest binding drifted")
    if status == "draft":
        if (
            binding.get("file_sha256") is not None
            or binding.get("content_sha256") is not None
        ):
            raise PilotContractError("V2.6 draft source-manifest hashes must be null")
        return

    manifest_path = source.parent / "pilot_v2_6_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_6_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.6 source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
        raise PilotContractError("V2.6 source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.6 source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError("V2.6 source manifest is not canonical JSON") from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.6 source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding["schema_version"]
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding["content_sha256"]
    ):
        raise PilotContractError("V2.6 source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding["content_sha256"]:
        raise PilotContractError("V2.6 source manifest content hash drifted")


def _validate_v2_7_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
    *,
    status: str,
) -> None:
    binding = _mapping(
        amendment.get("source_manifest"),
        "stage0_evaluator_retry_amendment.source_manifest",
    )
    expected_binding = _v2_7_expected_stage0_evaluator_retry_amendment(
        status=status,
    )["source_manifest"]
    if _thaw_json(binding) != expected_binding:
        raise PilotContractError("V2.7 source manifest binding drifted")
    if status == "draft":
        if (
            binding.get("file_sha256") is not None
            or binding.get("content_sha256") is not None
        ):
            raise PilotContractError("V2.7 draft source-manifest hashes must be null")
        return

    manifest_path = source.parent / "pilot_v2_7_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_7_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.7 source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
        raise PilotContractError("V2.7 source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.7 source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError("V2.7 source manifest is not canonical JSON") from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.7 source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding["schema_version"]
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding["content_sha256"]
    ):
        raise PilotContractError("V2.7 source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding["content_sha256"]:
        raise PilotContractError("V2.7 source manifest content hash drifted")


def _validate_v2_8_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
    *,
    status: str,
) -> None:
    binding = _mapping(
        amendment.get("source_manifest"),
        "qref_identity_retry_amendment.source_manifest",
    )
    expected_binding = _v2_8_expected_qref_identity_retry_amendment(
        status=status,
    )["source_manifest"]
    if _thaw_json(binding) != expected_binding:
        raise PilotContractError("V2.8 source manifest binding drifted")
    if status == "draft":
        if (
            binding.get("file_sha256") is not None
            or binding.get("content_sha256") is not None
        ):
            raise PilotContractError("V2.8 draft source-manifest hashes must be null")
        return

    manifest_path = source.parent / "pilot_v2_8_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_8_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.8 source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
        raise PilotContractError("V2.8 source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.8 source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError("V2.8 source manifest is not canonical JSON") from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.8 source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding["schema_version"]
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding["content_sha256"]
    ):
        raise PilotContractError("V2.8 source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding["content_sha256"]:
        raise PilotContractError("V2.8 source manifest content hash drifted")


def _validate_v2_9_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
    *,
    status: str,
) -> None:
    binding = _mapping(
        amendment.get("source_manifest"),
        "qref_summary_equivalence_amendment.source_manifest",
    )
    expected_binding = _v2_9_expected_qref_summary_equivalence_amendment(
        status=status,
    )["source_manifest"]
    if _thaw_json(binding) != expected_binding:
        raise PilotContractError("V2.9 source manifest binding drifted")
    if status == "draft":
        if (
            binding.get("file_sha256") is not None
            or binding.get("content_sha256") is not None
        ):
            raise PilotContractError("V2.9 draft source-manifest hashes must be null")
        return

    manifest_path = source.parent / "pilot_v2_9_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_9_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.9 source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
        raise PilotContractError("V2.9 source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.9 source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError("V2.9 source manifest is not canonical JSON") from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.9 source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding["schema_version"]
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding["content_sha256"]
    ):
        raise PilotContractError("V2.9 source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding["content_sha256"]:
        raise PilotContractError("V2.9 source manifest content hash drifted")


def _validate_v2_10_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
    *,
    status: str,
) -> None:
    binding = _mapping(
        amendment.get("source_manifest"),
        "p95_runner_binding_retry_amendment.source_manifest",
    )
    expected_binding = _v2_10_expected_p95_runner_binding_retry_amendment(
        status=status,
    )["source_manifest"]
    if _thaw_json(binding) != expected_binding:
        raise PilotContractError("V2.10 source manifest binding drifted")
    if status == "draft":
        if (
            binding.get("file_sha256") is not None
            or binding.get("content_sha256") is not None
        ):
            raise PilotContractError("V2.10 draft source-manifest hashes must be null")
        return

    manifest_path = source.parent / "pilot_v2_10_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_10_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.10 source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
        raise PilotContractError("V2.10 source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.10 source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError("V2.10 source manifest is not canonical JSON") from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.10 source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding["schema_version"]
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding["content_sha256"]
    ):
        raise PilotContractError("V2.10 source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding["content_sha256"]:
        raise PilotContractError("V2.10 source manifest content hash drifted")


def _validate_v2_10_1_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
    *,
    status: str,
) -> None:
    binding = _mapping(
        amendment.get("source_manifest"),
        "qref_receipt_verifier_retry_amendment.source_manifest",
    )
    expected_binding = _v2_10_1_expected_qref_receipt_verifier_retry_amendment(
        status=status,
    )["source_manifest"]
    if _thaw_json(binding) != expected_binding:
        raise PilotContractError("V2.10.1 source manifest binding drifted")
    if status == "draft":
        if (
            binding.get("file_sha256") is not None
            or binding.get("content_sha256") is not None
        ):
            raise PilotContractError(
                "V2.10.1 draft source-manifest hashes must be null"
            )
        return

    manifest_path = source.parent / "pilot_v2_10_1_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_10_1_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.10.1 source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
        raise PilotContractError("V2.10.1 source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.10.1 source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError(
            "V2.10.1 source manifest is not canonical JSON"
        ) from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.10.1 source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding["schema_version"]
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding["content_sha256"]
    ):
        raise PilotContractError("V2.10.1 source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding["content_sha256"]:
        raise PilotContractError("V2.10.1 source manifest content hash drifted")


def _validate_v2_10_2_source_manifest_file(
    source: Path,
    amendment: Mapping[str, Any],
    *,
    status: str,
) -> None:
    binding = _mapping(
        amendment.get("source_manifest"),
        "p95_consumer_adapter_retry_amendment.source_manifest",
    )
    expected_binding = _v2_10_2_expected_p95_consumer_adapter_retry_amendment(
        status=status,
    )["source_manifest"]
    if _thaw_json(binding) != expected_binding:
        raise PilotContractError("V2.10.2 source manifest binding drifted")
    if status == "draft":
        if (
            binding.get("file_sha256") is not None
            or binding.get("content_sha256") is not None
        ):
            raise PilotContractError(
                "V2.10.2 draft source-manifest hashes must be null"
            )
        return

    manifest_path = source.parent / "pilot_v2_10_2_source_manifest.json"
    if (
        manifest_path.name != "pilot_v2_10_2_source_manifest.json"
        or manifest_path.resolve().parent != source.parent.resolve()
        or not manifest_path.is_file()
    ):
        raise PilotContractError(
            "V2.10.2 source manifest must be the tracked sibling file"
        )
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
        raise PilotContractError("V2.10.2 source manifest file hash drifted")
    try:
        manifest = _mapping(
            json.loads(payload.decode("utf-8")),
            "V2.10.2 source manifest",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError(
            "V2.10.2 source manifest is not canonical JSON"
        ) from exc
    integrity = _mapping(
        manifest.get("integrity"),
        "V2.10.2 source manifest integrity",
    )
    if (
        manifest.get("schema_version") != binding["schema_version"]
        or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
        or integrity.get("content_sha256") != binding["content_sha256"]
    ):
        raise PilotContractError("V2.10.2 source manifest identity drifted")
    content_payload = _json_copy(manifest)
    content_payload["integrity"].pop("content_sha256")
    if canonical_sha256(content_payload) != binding["content_sha256"]:
        raise PilotContractError("V2.10.2 source manifest content hash drifted")


def load_pilot_contract(path: str | Path) -> PilotContract:
    """Load a JSON-compatible YAML pilot contract and verify its declared hash."""

    source = Path(path)

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotContractError(
                    f"pilot contract contains duplicate JSON key: {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PilotContractError(
            f"pilot contract contains non-finite JSON number: {value}"
        )

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise PilotContractError(
            "pilot contract must use JSON-compatible YAML"
        ) from exc
    document = _mapping(value, "pilot contract")
    if document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1:
        document = _expand_v2_1_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_2:
        document = _expand_v2_2_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_3:
        document = _expand_v2_3_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_4:
        document = _expand_v2_4_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_5:
        document = _expand_v2_5_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_6:
        document = _expand_v2_6_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_7:
        document = _expand_v2_7_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_8:
        document = _expand_v2_8_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_9:
        document = _expand_v2_9_overlay(document, source=source)
    elif document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10:
        document = _expand_v2_10_overlay(document, source=source)
    elif (
        document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1
    ):
        document = _expand_v2_10_1_overlay(document, source=source)
    elif (
        document.get("schema_version") == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_2
    ):
        document = _expand_v2_10_2_overlay(document, source=source)
    contract = PilotContract.from_dict(document)
    if contract.contract_id == PILOT_CONTRACT_ID_V2_11:
        if contract.v211_forward_boundary is None:  # pragma: no cover - parser
            raise PilotContractError("V2.11 contract lacks its forward boundary")
        binding = _mapping(
            contract.v211_forward_boundary.get("source_manifest"),
            "V2.11 source manifest binding",
        )
        manifest_path = source.parent / "pilot_v2_11_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_1:
        if contract.v2111_forward_boundary is None:  # pragma: no cover - parser
            raise PilotContractError("V2.11.1 contract lacks its forward boundary")
        binding = _mapping(
            contract.v2111_forward_boundary.get("source_manifest"),
            "V2.11.1 source manifest binding",
        )
        manifest_path = source.parent / "pilot_v2_11_1_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.1 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.1 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.1 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.1 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.1 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_1_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.1 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.1 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_2:
        if contract.v2112_forward_boundary is None:  # pragma: no cover - parser
            raise PilotContractError("V2.11.2 contract lacks its forward boundary")
        binding = _mapping(
            contract.v2112_forward_boundary.get("source_manifest"),
            "V2.11.2 source manifest binding",
        )
        manifest_path = source.parent / "pilot_v2_11_2_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.2 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.2 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.2 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.2 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.2 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_2_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.2 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.2 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_3:
        if contract.v2113_forward_boundary is None:  # pragma: no cover - parser
            raise PilotContractError("V2.11.3 contract lacks its forward boundary")
        binding = _mapping(
            contract.v2113_forward_boundary.get("source_manifest"),
            "V2.11.3 source manifest binding",
        )
        manifest_path = source.parent / "pilot_v2_11_3_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.3 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.3 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.3 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.3 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.3 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_3_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.3 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.3 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_4:
        if contract.v2114_forward_boundary is None:  # pragma: no cover - parser
            raise PilotContractError("V2.11.4 contract lacks its forward boundary")
        binding = _mapping(
            contract.v2114_forward_boundary.get("source_manifest"),
            "V2.11.4 source manifest binding",
        )
        manifest_path = source.parent / "pilot_v2_11_4_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.4 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.4 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.4 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.4 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.4 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_4_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.4 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.4 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_5:
        if contract.v2115_forward_boundary is None:  # pragma: no cover - parser
            raise PilotContractError("V2.11.5 contract lacks its forward boundary")
        binding = _mapping(
            contract.v2115_forward_boundary.get("source_manifest"),
            "V2.11.5 source manifest binding",
        )
        manifest_path = source.parent / "pilot_v2_11_5_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.5 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.5 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.5 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.5 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.5 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_5_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.5 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.5 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_6:
        if contract.v2116_continuation_boundary is None:  # pragma: no cover
            raise PilotContractError("V2.11.6 contract lacks its continuation boundary")
        binding = _mapping(
            contract.v2116_continuation_boundary.get("source_manifest"),
            "V2.11.6 source manifest binding",
        )
        if contract.status == "draft" and (
            binding.get("file_sha256") is None or binding.get("content_sha256") is None
        ):
            return contract
        manifest_path = source.parent / "pilot_v2_11_6_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.6 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.6 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.6 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.6 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.6 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_6_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.6 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.6 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_7:
        if contract.v2117_recovery_boundary is None:  # pragma: no cover
            raise PilotContractError("V2.11.7 contract lacks its recovery boundary")
        binding = _mapping(
            contract.v2117_recovery_boundary.get("source_manifest"),
            "V2.11.7 source manifest binding",
        )
        if contract.status == "draft" and (
            binding.get("file_sha256") is None or binding.get("content_sha256") is None
        ):
            return contract
        manifest_path = source.parent / "pilot_v2_11_7_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.7 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.7 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.7 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.7 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.7 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_7_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.7 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.7 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_8:
        if contract.v2118_recovery_boundary is None:  # pragma: no cover
            raise PilotContractError("V2.11.8 contract lacks its recovery boundary")
        binding = _mapping(
            contract.v2118_recovery_boundary.get("source_manifest"),
            "V2.11.8 source manifest binding",
        )
        if contract.status == "draft" and (
            binding.get("file_sha256") is None or binding.get("content_sha256") is None
        ):
            return contract
        manifest_path = source.parent / "pilot_v2_11_8_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.8 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.8 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.8 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.8 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.8 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_8_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.8 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.8 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_9:
        if contract.v2119_recovery_boundary is None:  # pragma: no cover
            raise PilotContractError("V2.11.9 contract lacks its recovery boundary")
        binding = _mapping(
            contract.v2119_recovery_boundary.get("source_manifest"),
            "V2.11.9 source manifest binding",
        )
        if contract.status == "draft" and (
            binding.get("file_sha256") is None or binding.get("content_sha256") is None
        ):
            return contract
        manifest_path = source.parent / "pilot_v2_11_9_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.9 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.9 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.9 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.9 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.9 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_9_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.9 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.9 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_10:
        if contract.v21110_recovery_boundary is None:  # pragma: no cover
            raise PilotContractError("V2.11.10 contract lacks its recovery boundary")
        binding = _mapping(
            contract.v21110_recovery_boundary.get("source_manifest"),
            "V2.11.10 source manifest binding",
        )
        if contract.status == "draft" and (
            binding.get("file_sha256") is None or binding.get("content_sha256") is None
        ):
            return contract
        manifest_path = source.parent / "pilot_v2_11_10_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.10 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.10 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.10 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.10 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.10 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_10_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.10 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.10 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_11:
        if contract.v21111_fresh_cohort_boundary is None:  # pragma: no cover
            raise PilotContractError(
                "V2.11.11 contract lacks its fresh-cohort boundary"
            )
        binding = _mapping(
            contract.v21111_fresh_cohort_boundary.get("source_manifest"),
            "V2.11.11 source manifest binding",
        )
        if contract.status == "draft" and (
            binding.get("file_sha256") is None or binding.get("content_sha256") is None
        ):
            return contract
        manifest_path = source.parent / "pilot_v2_11_11_source_manifest.json"
        try:
            payload = manifest_path.read_bytes()
        except OSError as exc:
            raise PilotContractError("V2.11.11 source manifest is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != binding["file_sha256"]:
            raise PilotContractError("V2.11.11 source manifest file hash drifted")
        try:
            manifest = _mapping(
                json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_nonfinite,
                ),
                "V2.11.11 source manifest",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotContractError(
                "V2.11.11 source manifest is not canonical JSON"
            ) from exc
        integrity = _mapping(
            manifest.get("integrity"),
            "V2.11.11 source manifest integrity",
        )
        if (
            binding.get("path") != "experiments/pilot_v2_11_11_source_manifest.json"
            or manifest.get("schema_version") != binding["schema_version"]
            or integrity.get("canonicalization") != PILOT_CONTRACT_CANONICALIZATION
            or integrity.get("content_sha256") != binding["content_sha256"]
        ):
            raise PilotContractError("V2.11.11 source manifest identity drifted")
        content_payload = _json_copy(manifest)
        content_payload["integrity"].pop("content_sha256")
        if canonical_sha256(content_payload) != binding["content_sha256"]:
            raise PilotContractError("V2.11.11 source manifest content hash drifted")
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_4:
        if contract.matrix_amendment is None:  # pragma: no cover - parser guard
            raise PilotContractError("V2.4 contract lacks its matrix amendment")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_5:
        if (
            contract.matrix_amendment is None
            or contract.parent_import_retry_amendment is None
        ):  # pragma: no cover - parser guard
            raise PilotContractError("V2.5 contract lacks its amendment chain")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
        _validate_v2_5_source_manifest_file(
            source,
            contract.parent_import_retry_amendment,
        )
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_6:
        if (
            contract.matrix_amendment is None
            or contract.parent_import_retry_amendment is None
            or contract.p95_authority_retry_amendment is None
        ):  # pragma: no cover - parser guard
            raise PilotContractError("V2.6 contract lacks its amendment chain")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
        _validate_v2_5_source_manifest_file(
            source,
            contract.parent_import_retry_amendment,
        )
        _validate_v2_6_source_manifest_file(
            source,
            contract.p95_authority_retry_amendment,
            status=contract.status,
        )
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_7:
        if (
            contract.matrix_amendment is None
            or contract.parent_import_retry_amendment is None
            or contract.p95_authority_retry_amendment is None
            or contract.stage0_evaluator_retry_amendment is None
        ):  # pragma: no cover - parser guard
            raise PilotContractError("V2.7 contract lacks its amendment chain")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
        _validate_v2_5_source_manifest_file(
            source,
            contract.parent_import_retry_amendment,
        )
        _validate_v2_6_source_manifest_file(
            source,
            contract.p95_authority_retry_amendment,
            status="frozen",
        )
        _validate_v2_7_source_manifest_file(
            source,
            contract.stage0_evaluator_retry_amendment,
            status=contract.status,
        )
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_8:
        if (
            contract.matrix_amendment is None
            or contract.parent_import_retry_amendment is None
            or contract.p95_authority_retry_amendment is None
            or contract.stage0_evaluator_retry_amendment is None
            or contract.qref_identity_retry_amendment is None
        ):  # pragma: no cover - parser guard
            raise PilotContractError("V2.8 contract lacks its amendment chain")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
        _validate_v2_5_source_manifest_file(
            source,
            contract.parent_import_retry_amendment,
        )
        _validate_v2_6_source_manifest_file(
            source,
            contract.p95_authority_retry_amendment,
            status="frozen",
        )
        _validate_v2_7_source_manifest_file(
            source,
            contract.stage0_evaluator_retry_amendment,
            status="frozen",
        )
        _validate_v2_8_source_manifest_file(
            source,
            contract.qref_identity_retry_amendment,
            status=contract.status,
        )
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_9:
        if (
            contract.matrix_amendment is None
            or contract.parent_import_retry_amendment is None
            or contract.p95_authority_retry_amendment is None
            or contract.stage0_evaluator_retry_amendment is None
            or contract.qref_identity_retry_amendment is None
            or contract.qref_summary_equivalence_amendment is None
        ):  # pragma: no cover - parser guard
            raise PilotContractError("V2.9 contract lacks its amendment chain")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
        _validate_v2_5_source_manifest_file(
            source,
            contract.parent_import_retry_amendment,
        )
        _validate_v2_6_source_manifest_file(
            source,
            contract.p95_authority_retry_amendment,
            status="frozen",
        )
        _validate_v2_7_source_manifest_file(
            source,
            contract.stage0_evaluator_retry_amendment,
            status="frozen",
        )
        _validate_v2_8_source_manifest_file(
            source,
            contract.qref_identity_retry_amendment,
            status="frozen",
        )
        _validate_v2_9_source_manifest_file(
            source,
            contract.qref_summary_equivalence_amendment,
            status=contract.status,
        )
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_10:
        if (
            contract.matrix_amendment is None
            or contract.parent_import_retry_amendment is None
            or contract.p95_authority_retry_amendment is None
            or contract.stage0_evaluator_retry_amendment is None
            or contract.qref_identity_retry_amendment is None
            or contract.qref_summary_equivalence_amendment is None
            or contract.p95_runner_binding_retry_amendment is None
        ):  # pragma: no cover - parser guard
            raise PilotContractError("V2.10 contract lacks its amendment chain")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
        _validate_v2_5_source_manifest_file(
            source,
            contract.parent_import_retry_amendment,
        )
        _validate_v2_6_source_manifest_file(
            source,
            contract.p95_authority_retry_amendment,
            status="frozen",
        )
        _validate_v2_7_source_manifest_file(
            source,
            contract.stage0_evaluator_retry_amendment,
            status="frozen",
        )
        _validate_v2_8_source_manifest_file(
            source,
            contract.qref_identity_retry_amendment,
            status="frozen",
        )
        _validate_v2_9_source_manifest_file(
            source,
            contract.qref_summary_equivalence_amendment,
            status="frozen",
        )
        _validate_v2_10_source_manifest_file(
            source,
            contract.p95_runner_binding_retry_amendment,
            status=contract.status,
        )
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_10_1:
        if (
            contract.matrix_amendment is None
            or contract.parent_import_retry_amendment is None
            or contract.p95_authority_retry_amendment is None
            or contract.stage0_evaluator_retry_amendment is None
            or contract.qref_identity_retry_amendment is None
            or contract.qref_summary_equivalence_amendment is None
            or contract.p95_runner_binding_retry_amendment is None
            or contract.qref_receipt_verifier_retry_amendment is None
        ):  # pragma: no cover - parser guard
            raise PilotContractError("V2.10.1 contract lacks its amendment chain")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
        _validate_v2_5_source_manifest_file(
            source,
            contract.parent_import_retry_amendment,
        )
        _validate_v2_6_source_manifest_file(
            source,
            contract.p95_authority_retry_amendment,
            status="frozen",
        )
        _validate_v2_7_source_manifest_file(
            source,
            contract.stage0_evaluator_retry_amendment,
            status="frozen",
        )
        _validate_v2_8_source_manifest_file(
            source,
            contract.qref_identity_retry_amendment,
            status="frozen",
        )
        _validate_v2_9_source_manifest_file(
            source,
            contract.qref_summary_equivalence_amendment,
            status="frozen",
        )
        _validate_v2_10_source_manifest_file(
            source,
            contract.p95_runner_binding_retry_amendment,
            status="frozen",
        )
        _validate_v2_10_1_source_manifest_file(
            source,
            contract.qref_receipt_verifier_retry_amendment,
            status=contract.status,
        )
    elif contract.contract_id == PILOT_CONTRACT_ID_V2_10_2:
        if (
            contract.matrix_amendment is None
            or contract.parent_import_retry_amendment is None
            or contract.p95_authority_retry_amendment is None
            or contract.stage0_evaluator_retry_amendment is None
            or contract.qref_identity_retry_amendment is None
            or contract.qref_summary_equivalence_amendment is None
            or contract.p95_runner_binding_retry_amendment is None
            or contract.qref_receipt_verifier_retry_amendment is None
            or contract.p95_consumer_adapter_retry_amendment is None
        ):  # pragma: no cover - parser guard
            raise PilotContractError("V2.10.2 contract lacks its amendment chain")
        _validate_v2_4_parent_source_manifest_file(
            source,
            contract.matrix_amendment,
        )
        _validate_v2_5_source_manifest_file(
            source,
            contract.parent_import_retry_amendment,
        )
        _validate_v2_6_source_manifest_file(
            source,
            contract.p95_authority_retry_amendment,
            status="frozen",
        )
        _validate_v2_7_source_manifest_file(
            source,
            contract.stage0_evaluator_retry_amendment,
            status="frozen",
        )
        _validate_v2_8_source_manifest_file(
            source,
            contract.qref_identity_retry_amendment,
            status="frozen",
        )
        _validate_v2_9_source_manifest_file(
            source,
            contract.qref_summary_equivalence_amendment,
            status="frozen",
        )
        _validate_v2_10_source_manifest_file(
            source,
            contract.p95_runner_binding_retry_amendment,
            status="frozen",
        )
        _validate_v2_10_1_source_manifest_file(
            source,
            contract.qref_receipt_verifier_retry_amendment,
            status="frozen",
        )
        _validate_v2_10_2_source_manifest_file(
            source,
            contract.p95_consumer_adapter_retry_amendment,
            status=contract.status,
        )
    return contract


__all__ = [
    "PILOT_CONTRACT_CANONICALIZATION",
    "PILOT_CONTRACT_SCHEMA_VERSION",
    "PILOT_CONTRACT_SCHEMA_VERSION_V1",
    "PILOT_CONTRACT_SCHEMA_VERSION_V2",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_2",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_3",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_4",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_5",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_6",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_7",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_8",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_9",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_2",
    "PILOT_CONTRACT_ID_V2",
    "PILOT_CONTRACT_ID_V2_1",
    "PILOT_CONTRACT_ID_V2_2",
    "PILOT_CONTRACT_ID_V2_3",
    "PILOT_CONTRACT_ID_V2_4",
    "PILOT_CONTRACT_ID_V2_5",
    "PILOT_CONTRACT_ID_V2_6",
    "PILOT_CONTRACT_ID_V2_7",
    "PILOT_CONTRACT_ID_V2_8",
    "PILOT_CONTRACT_ID_V2_9",
    "PILOT_CONTRACT_ID_V2_10",
    "PILOT_CONTRACT_ID_V2_10_1",
    "PILOT_CONTRACT_ID_V2_10_2",
    "PILOT_CONTRACT_ID_V2_11",
    "PILOT_CONTRACT_ID_V2_11_1",
    "PILOT_CONTRACT_ID_V2_11_2",
    "PILOT_CONTRACT_ID_V2_11_3",
    "PILOT_CONTRACT_ID_V2_11_4",
    "PILOT_CONTRACT_ID_V2_11_5",
    "PILOT_CONTRACT_ID_V2_11_6",
    "PILOT_CONTRACT_ID_V2_11_7",
    "PILOT_CONTRACT_ID_V2_11_8",
    "PILOT_CONTRACT_ID_V2_11_9",
    "PILOT_CONTRACT_ID_V2_11_10",
    "PILOT_CONTRACT_ID_V2_11_11",
    "PILOT_CONTRACT_TAG_V2",
    "PILOT_CONTRACT_TAG_V2_1",
    "PILOT_CONTRACT_TAG_V2_2",
    "PILOT_CONTRACT_TAG_V2_3",
    "PILOT_CONTRACT_TAG_V2_4",
    "PILOT_CONTRACT_TAG_V2_5",
    "PILOT_CONTRACT_TAG_V2_6",
    "PILOT_CONTRACT_TAG_V2_7",
    "PILOT_CONTRACT_TAG_V2_8",
    "PILOT_CONTRACT_TAG_V2_9",
    "PILOT_CONTRACT_TAG_V2_10",
    "PILOT_CONTRACT_TAG_V2_10_1",
    "PILOT_CONTRACT_TAG_V2_10_2",
    "PILOT_CONTRACT_TAG_V2_11",
    "PILOT_CONTRACT_TAG_V2_11_1",
    "PILOT_CONTRACT_TAG_V2_11_2",
    "PILOT_CONTRACT_TAG_V2_11_3",
    "PILOT_CONTRACT_TAG_V2_11_4",
    "PILOT_CONTRACT_TAG_V2_11_5",
    "PILOT_CONTRACT_TAG_V2_11_6",
    "PILOT_CONTRACT_TAG_V2_11_7",
    "PILOT_CONTRACT_TAG_V2_11_8",
    "PILOT_CONTRACT_TAG_V2_11_9",
    "PILOT_CONTRACT_TAG_V2_11_10",
    "PILOT_CONTRACT_TAG_V2_11_11",
    "PILOT_CONTRACT_V2_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_1_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_2_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_3_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_4_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_5_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_6_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_7_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_8_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_9_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_10_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_10_2_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_3_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_5_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_11_6_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_11_7_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_11_8_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256",
    "PILOT_V2_11_1_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_1_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_2_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_3_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_3_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_5_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_5_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_6_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_6_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_7_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_7_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_8_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_8_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_9_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_9_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_10_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_10_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_9_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_9_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_10_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_10_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_10_1_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_10_1_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_10_2_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_10_2_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_V2_9_EVIDENCE_COMMIT",
    "PILOT_V2_9_EVIDENCE_MERGE_COMMIT",
    "PILOT_V2_9_EVIDENCE_PACKAGE_FILE_SHA256",
    "PILOT_V2_9_EVIDENCE_CHECKSUMS_FILE_SHA256",
    "PILOT_V2_8_EVIDENCE_COMMIT",
    "PILOT_V2_8_EVIDENCE_MERGE_COMMIT",
    "PILOT_V2_8_EVIDENCE_PACKAGE_FILE_SHA256",
    "PILOT_V2_8_EVIDENCE_CHECKSUMS_FILE_SHA256",
    "PILOT_V2_8_SOURCE_MANIFEST_FILE_SHA256",
    "PILOT_V2_8_SOURCE_MANIFEST_CONTENT_SHA256",
    "PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256",
    "PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256",
    "DecodingFieldDispatch",
    "DenominatorPolicy",
    "ModelRolePolicy",
    "ParameterDispatchPolicy",
    "PilotContract",
    "PilotContractError",
    "PilotRunSpec",
    "PilotStage",
    "PilotStageCell",
    "PriceSnapshot",
    "ProviderRequestProfile",
    "ReasoningProfile",
    "ReleaseRequirements",
    "TaskOutputContract",
    "canonical_contract_sha256",
    "canonical_sha256",
    "load_pilot_contract",
    "science_design_sha256",
]
