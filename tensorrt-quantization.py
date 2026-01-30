import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pandas as pd

onnx_model = ""
engine_file = ""

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {cls: idx for idx, cls in enumerate(AAMI_CLASSES)}

class ECGCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, batch_size=32):
        super(ECGCalibrator, self).__init__()
        self.data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(int(np.prod(calibration_data[0].shape) * self.batch_size * np.float32().nbytes))


    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.data):
           return None
        batch = self.data[self.current_index:self.current_index + self.batch_size]
        if batch.shape[0] < self.batch_size:
           pad_rows = self.batch_size - batch.shape[0]
           batch = np.vstack([batch, np.tile(batch[-1], (pad_rows,1,1))])

        batch = np.ascontiguousarray(batch.astype(np.float32))
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        return None

def build_int8_engine(onnx_file, engine_file, calibration_data, input_length):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=(1, input_length, 1),
        opt=(32, input_length, 1),
        max=(32, input_length, 1)
    )
    config.add_optimization_profile(profile)
    
    calibrator = ECGCalibrator(calibration_data, batch_size=32)
    config.int8_calibrator = calibrator

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine!")

    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ INT8 engine saved to {engine_file}")
    return serialized_engine


def load_and_preprocess(path):
    df = pd.read_csv(path, low_memory=False)

    if "features_aami" in path:
        print("✅ Detected features_aami dataset")
        df = df.dropna(subset=['lf_hf_ratio', 'lfnu', 'hfnu', 'sd2','ratio_sd2_sd1','csi','cvi','Modified_csi'])
        df['sdsd'] = pd.to_numeric(df['sdsd'], errors='coerce')
        df = df[df['sdsd'].notna()]
        X = df.drop(columns=['label', 'aami_label', 'tinn', 'channel','entropy','record_id', 'min_hr','mean_hr','sample',]).astype(float)

    else:
        print("✅ Detected normal dataset")
        df['sdsd'] = pd.to_numeric(df['sdsd'], errors='coerce')
        df = df.dropna(subset=['sdsd'])
        X = df.drop(columns=['start_partition_idx']).astype(float)

    y = df['aami_label_encoded']
    X = X.loc[y.index]

    X_np = np.expand_dims(X.values, axis=2).astype(np.float32)
    y_np = y.astype(np.int64).values

    return X_np, y_np, AAMI_MAP

def infer(engine_file, seq_len):
    import tensorrt as trt
    import pycuda.driver as cuda
    import numpy as np

    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)

    with open(engine_file, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    input_shape = (1, seq_len, 1)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, input_shape)
    output_shape = (1,5)
    d_input = cuda.mem_alloc(dummy_input.nbytes)
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.float32().nbytes)
    bindings = [int(d_input), int(d_output)]
    cuda.memcpy_htod(d_input, dummy_input)
    context.execute_v2(bindings)
    host_output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(host_output, d_output)

    print("Inference done. Output shape:", host_output.shape)
    return host_output


if __name__ == "__main__":
    X, y, encoder = load_and_preprocess("")
    print(X.shape)
    build_int8_engine(onnx_model, engine_file,X, X.shape[1])
    infer(engine_file, X.shape[1])
