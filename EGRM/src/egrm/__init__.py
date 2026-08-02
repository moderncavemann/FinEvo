"""Evidence-Grounded Rule Memory (EGRM).

The public names below describe the new paper abstraction.  The legacy class
names remain available because the implementation was extracted from FinEvo's
audited memory path and its serialized schemas must remain readable.
"""

from .m3_semantic import VerifiedSemanticRuleTrack
from .system import VerifiedDualTrackMemory

EvidenceGroundedRuleMemory = VerifiedDualTrackMemory
EvidenceGroundedRuleTrack = VerifiedSemanticRuleTrack

__version__ = "0.1.0"

__all__ = [
    "EvidenceGroundedRuleMemory",
    "EvidenceGroundedRuleTrack",
    "VerifiedDualTrackMemory",
    "VerifiedSemanticRuleTrack",
    "__version__",
]
