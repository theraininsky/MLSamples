import torch
from torch import Tensor
import tensorrt as trt

class RaftTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Cache for buffers
        self.cached_shape = None
        self.inputs = {}
        self.outputs = {}
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

    def allocate_buffers(self, shape):
        self.inputs = {}
        self.outputs = {}
        self.cached_shape = shape
        
        # Update Input Shapes
        self.context.set_input_shape("img1", shape)
        self.context.set_input_shape("img2", shape)

        # Iterate through all tensors to allocate outputs
        for name in self.tensor_names:
            # Skip if it is an input (we bind these dynamically in infer)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                continue

            # Get the shape TensorRT expects for this specific run
            out_shape = self.context.get_tensor_shape(name)
            
            # Sanity check for dynamic dims
            dims = list(out_shape)

            # Note: TensorRT returns -1 for dynamic dims if shape isn't fully propagated yet.
            # For RAFT, output dims usually match input dims or 1/8th of them.
            # If get_tensor_shape returns valid dims (not -1), use them.
            if -1 in dims:
                # Fallback: Inference of output size based on name if TRT hasn't computed it yet
                # (This rarely happens if input shapes are set correctly above)
                if "flow_low" in name:
                    dims = (shape[0], 2, shape[2] // 8, shape[3] // 8)
                else:
                # Assume it's 'flow_up' or the weird '_unsafe_view_2' which usually matches flow_up size
                    dims = (shape[0], 2, shape[2], shape[3])
            
            # Create buffer
            tensor = torch.empty(tuple(dims), dtype=torch.float32, device='cuda')
            self.outputs[name] = tensor
            
            # Set Tensor Addresses (TensorRT 10+ API)
            # Instead of a list of indices, we map the Name -> Memory Pointer directly.
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

    def infer(self, img1_tensor: Tensor, img2_tensor: Tensor):
        # Torch tensor (1, 3, H, W) on GPU

        # Update bindings for dynamic shapes
        # Note: TensorRT expects bindings in order. For RAFT: img1, img2 -> flow_low, flow_up
        # We need to set input shapes explicitly before execution.
        shape = tuple(img1_tensor.shape)
        
        # Check if we need to re-allocate (e.g., first run or resolution changed)
        if shape != self.cached_shape:
            self.allocate_buffers(shape)

        # Bind inputs
        self.context.set_tensor_address("img1", int(img1_tensor.data_ptr()))
        self.context.set_tensor_address("img2", int(img2_tensor.data_ptr()))

        # Execute
        stream_ptr = self.stream.cuda_stream
        if not self.context.execute_async_v3(stream_ptr):
            raise RuntimeError("TensorRT enqueue_v3 failed")

        self.stream.synchronize()

        # Return the main output we care about
        if "flow_up" in self.outputs:
            return self.outputs["flow_up"]
        else:
            # Fallback if names are weird, just return the first output found
            # return list(self.outputs.values())[0]
            raise RuntimeError("flow_up field not found in the outputs")

    def __del__(self):
        del self.context
        del self.engine
        del self.runtime
        del self.logger
        del self.stream

from .RaftTorch import InitializeRaft
from ...Settings import *

def ExportRaftAsOnnx(onnxName: str):
    raft_model = InitializeRaft(Settings.BackendPreference.CPU)

    device = next(raft_model.parameters()).device
    
    # Create Dummy Input
    # Using dynamic axes, so exact size here matters less, but must be divisible by 8
    H, W = 320, 320
    img1 = torch.randn(1, 3, H, W).to(device)
    img2 = torch.randn(1, 3, H, W).to(device)
    
    # ONNX export
    torch.onnx.export(
        raft_model,
        (img1, img2),
        onnxName,
        input_names=["img1", "img2"],
        output_names=["flow_low", "flow_up"],
        opset_version=17,
        dynamic_axes={
            "img1": {0: "batch", 2: "height", 3: "width"},
            "img2": {0: "batch", 2: "height", 3: "width"},
            # The raw flow prediction at 1/8th resolution.
            "flow_low": {0: "batch", 2: "height", 3: "width"},
            # The final flow upsampled to the original image resolution.
            "flow_up": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print("RAFT exported to raft.onnx")

def BuildRaftTRT(onnx_file_path:str, engine_file_path: str, frameHW: tuple):
    # Setup Logger and Builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Create Network (Explicit Batch is required for ONNX)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    # Create Builder Config
    config = builder.create_builder_config()

    # Parse ONNX File
    print(f"Parsing ONNX file: {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Define Optimization Profile
    # Because we exported with dynamic_axes, TensorRT needs to know the 
    # specific range of shapes we intend to use at runtime.
    profile = builder.create_optimization_profile()

    # Define (min_shape, opt_shape, max_shape)
    # Adjust 'max' if you plan to inference on 1080p or 4k images.
    # Format: (Batch, Channel, Height, Width)
    min_shape = (1, 3, 256, 256)
    opt_shape = (1, 3, *frameHW)  # The size you used for tracing
    max_shape = (1, 3, 1280, 720)

    # Set dimensions for both inputs named in our ONNX export
    profile.set_shape("img1", min_shape, opt_shape, max_shape)
    profile.set_shape("img2", min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    # Optional: Enable FP16 (Half Precision) for faster inference on Tensor Cores
    if builder.platform_has_fast_fp16:
        print("Enabling FP16 precision...")
        config.set_flag(trt.BuilderFlag.FP16)

    # Build Serialized Engine
    print("Building TensorRT engine... (this may take a few minutes)")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except AttributeError:
        # Fallback for older TensorRT versions
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize()

    if serialized_engine is None:
        print("Build failed.")
        return None

    # Save Engine to File
    print(f"Saving engine to {engine_file_path}...")
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print("Success! TensorRT engine built.")

    return serialized_engine

from pathlib import Path

def InitializeRaftTRT(frameHW: tuple)->RaftTRT:
    onnxName: str = "raft.onnx"
    if not Path(onnxName).exists():
        ExportRaftAsOnnx(onnxName)

    engine_file_path = 'raftTRT.engine'
    if not Path(engine_file_path).exists():
        BuildRaftTRT(onnxName, engine_file_path,frameHW)

    trt_model = RaftTRT(engine_file_path)

    return trt_model
