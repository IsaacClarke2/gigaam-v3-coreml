# Конвертация энкодера GigaAM v3 в CoreML MLProgram fp16 для Neural Engine.
# Ответ на жалобу «такой нагрузки нельзя допускать»: ONNX-энкодер на CPU
# занимал все P-ядра на минуты, а CoreML EP поверх ORT падает на выполнении.
#
# Запуск (venv с torch, coremltools, onnx2torch):
#   python3 scripts/convert-gigaam-ane.py \
#     "~/Library/Application Support/Loud/Models/gigaam-v3/v3_e2e_rnnt_encoder.onnx" \
#     "~/Library/Application Support/Loud/Models/gigaam-v3/encoder-ane.mlpackage"
#
# Итог кладётся рядом с ONNX-моделями; GigaAMANE.swift находит его по имени
# encoder-ane.mlpackage, компилирует в .mlmodelc один раз и дальше энкодер
# считается на ANE (замер 29 июля 2026, встреча 23:45 на M1 Pro:
# 33 c и 52 c CPU-времени против 69 c и 325 c CPU у ONNX).
#
# Окно ФИКСИРОВАННОЕ — 3360 мел-кадров (33.6 c, больше максимального чанка
# 33 c): ANE не умеет динамические формы. Расхождение текста от fp16 —
# ~2.4% слов равноценными вариантами (измерено на 6 кусках живой встречи).

import os, sys
import numpy as np, torch, coremltools as ct
from onnx2torch import convert

# onnx2torch эмитит алиасы torch-опов, которых нет в реестре coremltools —
# сами опы есть под каноничными именами, регистрируем синонимы.
from coremltools.converters.mil.frontend.torch import ops as _t
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    register_torch_op, _TORCH_OPS_REGISTRY)

def _alias(name, target):
    if name in _TORCH_OPS_REGISTRY:
        return
    def _f(context, node):
        target(context, node)
    _f.__name__ = name
    register_torch_op(_f)

_alias("less", _t.lt)
_alias("greater", _t.gt)
_alias("less_equal", _t.le)
_alias("greater_equal", _t.ge)
_alias("not_equal", _t.ne)

ENC = os.path.expanduser(sys.argv[1])
OUT = os.path.expanduser(sys.argv[2])
T = 3360

print("onnx → torch…", flush=True)
model = convert(ENC).eval()

class Wrapper(torch.nn.Module):
    # length приходит int32 (CoreML не любит int64-входы), внутрь — int64.
    # ⚠️ encoded_len у трассы на паддинге на 1 больше честного — Swift
    # считает длину сам по формуле ORT ceil(len/4).
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, audio_signal, length):
        out, out_len = self.inner(audio_signal, length.to(torch.int64))
        return out, out_len.to(torch.int32)

wrapper = Wrapper(model).eval()
x = torch.randn(1, 64, T)
ln = torch.tensor([T], dtype=torch.int32)
print("trace…", flush=True)
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (x, ln))
print("trace готов; ct.convert…", flush=True)

mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS14,
    inputs=[
        ct.TensorType(name="audio_signal", shape=(1, 64, T), dtype=np.float32),
        ct.TensorType(name="length", shape=(1,), dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="encoded", dtype=np.float32),
        ct.TensorType(name="encoded_len", dtype=np.int32),
    ],
)
mlmodel.save(OUT)
print("СОХРАНЕНО:", OUT, flush=True)
