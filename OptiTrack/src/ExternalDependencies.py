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

# Handle RAFT Models Dependency
# External\RAFT\models
import re
import zipfile
import urllib.request

def DownloadModelsForRaft():
    # Paths
    currentFile = Path(__file__).resolve().parent
    downloadScript = currentFile / "external" / "RAFT" / "download_models.sh"
    download_dir = currentFile / "external"
    
    # Read file and extract wget URL
    text = downloadScript.read_text(encoding="utf-8")
    
    match = re.search(r"wget\s+(https?://\S+)", text)
    if not match:
        raise RuntimeError("No wget URL found in download_models.py")
    
    url = match.group(1)
    zip_path = download_dir / "models.zip"
    
    print(f"Found URL: {url}")
    
    # Download zip
    print("Downloading models...")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract zip
    print("Extracting models...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)
    
    # Optional: cleanup zip
    zip_path.unlink()
    
    print(f"Models extracted to: {download_dir.resolve()}")

from ..Settings import *

if Settings.defaultRaftModelPath is None:
    Settings.defaultRaftModelPath = Path(__file__).resolve().parent / "external" / "models"
    if not Settings.defaultRaftModelPath.exists():
        DownloadModelsForRaft()
