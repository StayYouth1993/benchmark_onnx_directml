import time
import numpy as np
import onnxruntime as rt


# use hd630 igpu*************************************************************************************************
# --------------------------fp32 start----------------------------
ops = rt.SessionOptions()
ops.enable_mem_pattern = False

sess = rt.InferenceSession("bres256_core_sim.onnx", sess_options=ops, providers=[('DmlExecutionProvider', {
    'device_id': 0,
})])

out_names = [each.name for each in sess.get_outputs()]
result = sess.run(out_names, {"img": np.random.random((1, 3, 256, 256)).astype(np.float32)})
data = [np.random.random((1, 3, 256, 256)).astype(np.float32) for _ in range(50)]

start = time.time()
for i in range(50):
    result = sess.run(out_names, {"img": data[i]})
end = time.time()
avg_time = (end - start) / 50.0
print(f'resnet 50 infer with hd630 use onnx directml fps is :{1 / avg_time}')
del sess
# --------------------------fp32 end----------------------------


# --------------------------fp16 start----------------------------
ops = rt.SessionOptions()
ops.enable_mem_pattern = False

sess = rt.InferenceSession("bres256_core_sim_fp16.onnx", sess_options=ops, providers=[('DmlExecutionProvider', {
    'device_id': 0,
})])

out_names = [each.name for each in sess.get_outputs()]
result = sess.run(out_names, {"img": np.random.random((1, 3, 256, 256)).astype(np.float16)})
data = [np.random.random((1, 3, 256, 256)).astype(np.float16) for _ in range(50)]

start = time.time()
for i in range(50):
    result = sess.run(out_names, {"img": data[i]})
end = time.time()
avg_time = (end - start) / 50.0
print(f'resnet 50 infer with hd630 use onnx directml fp16 fps is :{1 / avg_time}')
del sess
# --------------------------fp16 end----------------------------


# use nvidia 940M gpu *************************************************************************************************
# --------------------------fp32 start----------------------------
ops = rt.SessionOptions()
ops.enable_mem_pattern = False

sess = rt.InferenceSession("bres256_core_sim.onnx", sess_options=ops, providers=[('DmlExecutionProvider', {
    'device_id': 1,
})])

out_names = [each.name for each in sess.get_outputs()]
result = sess.run(out_names, {"img": np.random.random((1, 3, 256, 256)).astype(np.float32)})
data = [np.random.random((1, 3, 256, 256)).astype(np.float32) for _ in range(50)]

start = time.time()
for i in range(50):
    result = sess.run(out_names, {"img": data[i]})
end = time.time()
avg_time = (end - start) / 50.0
print(f'resnet 50 infer with nvidia 940M use onnx directml fps is :{1 / avg_time}')
del sess
# --------------------------fp32 end----------------------------


# --------------------------fp16 start----------------------------
ops = rt.SessionOptions()
ops.enable_mem_pattern = False

sess = rt.InferenceSession("bres256_core_sim_fp16.onnx", sess_options=ops, providers=[('DmlExecutionProvider', {
    'device_id': 1,
})])

out_names = [each.name for each in sess.get_outputs()]
result = sess.run(out_names, {"img": np.random.random((1, 3, 256, 256)).astype(np.float16)})
data = [np.random.random((1, 3, 256, 256)).astype(np.float16) for _ in range(50)]

start = time.time()
for i in range(50):
    result = sess.run(out_names, {"img": data[i]})
end = time.time()
avg_time = (end - start) / 50.0
print(f'resnet 50 infer with nvidia 940M use onnx directml fp16 fps is :{1 / avg_time}')
del sess
# --------------------------fp16 end----------------------------

