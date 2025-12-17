import argparse
import torch
from ...Settings import *

from ..ExternalDependencies import *

def InitializeRaft(preferredBackend: BackendPreference = Settings.defaultBackendPreference):
    raft_model = RAFT(
        argparse.Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
        )
    )
    
    device = torch.device('cpu');
    
    match preferredBackend:
        case BackendPreference.PROPRIETARY:
            if torch.cuda.is_available():
                device = torch.device('cuda');
        case BackendPreference.VULKAN:
            if torch.is_vulkan_available():
                device = torch.device('vulkan');
        case BackendPreference.OPENCL:
            device = torch.device('opencl');
    
    state_dict = torch.load('models/raft-sintel.pth', map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    
    raft_model.load_state_dict(new_state_dict)
    raft_model.to(device)
    raft_model.eval()

    return raft_model
