# Handle RAFT Dependency
# External\RAFT
import sys
from pathlib import Path

RAFT_ROOT = Path(__file__).resolve().parent / "external" / "RAFT"
RAFT_CORE = RAFT_ROOT / "core"

if str(RAFT_CORE) not in sys.path:
    sys.path.insert(0, str(RAFT_CORE))

from .external.RAFT.core.raft import RAFT
from .external.RAFT.core.utils import flow_viz
from .external.RAFT.core.utils.utils import InputPadder
