import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import argparse
import tensorrt as trt
import os
import sys

def build_engine(model_file, shapes, max_ws=512*1024*1024, fp16=True):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    #builder.fp16_mode = fp16

    config = builder.create_builder_config()
    # config.max_workspace_size = max_ws
    config.max_workspace_size = 1 << 31
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        # config.flags |= 1 << int(trt.BuilderFlag.FP16)
    
    profile = builder.create_optimization_profile()
    for s in shapes:
        profile.set_shape(s['name'], min=s['min'], opt=s['opt'], max=s['max'])
    config.add_optimization_profile(profile)
    #Added for further optimization
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())

        last_layer = network.get_layer(network.num_layers - 1)
        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))
            
        
        for i in range(parser.num_errors):
            print("TensorRT ONNX parser error:", parser.get_error(i))
        engine = builder.build_engine(network, config=config)

        return engine
sys.path.append('./')

#from trt_utils import build_engine
model_name = "bert-base-cased-goemotions-ekman_model"

fp16 = False
engine_prec = "_fp16" if fp16 else "_fp32"
out_folder = "/3" if fp16 else "/1"

model = "./model-sim.onnx"
output =  "./"

shape=[{"name": "input_ids", "min": (1,3), "opt": (1,20), "max": (1,80)},
       {"name": "attention_mask", "min": (1,3), "opt": (1,20), "max": (1,80)},
       {"name": "token_type_ids", "min": (1,3), "opt": (1,20), "max": (1,80)}
        ]
if model != "":
    print("Building model ...")
    model_engine = build_engine(model, shapes = shape ,fp16=fp16)
    if model_engine is not None:
        engine_path = os.path.join(output, "model"+engine_prec+".plan")
        with open(engine_path, 'wb') as f:
            # f.write(model_engine)
            f.write(model_engine.serialize())
    else:
        print("Failed to build engine from", model)
        sys.exit()
