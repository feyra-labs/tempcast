"""Экспорт МАЯК в ONNX + динамическая int8-квантизация"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from mayak.constants import L_MAX, H


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, lat, lon, elev, x_hist, mask_hist,
                doy_hist, hour_hist, doy_fut, hour_fut):
        batch = dict(lat=lat, lon=lon, elev=elev, x_hist=x_hist, mask_hist=mask_hist,
                     doy_hist=doy_hist, hour_hist=hour_hist,
                     doy_fut=doy_fut, hour_fut=hour_fut)
        o = self.model(batch)
        return o["q"], o["mu"], o["sigma_c"]


def dummy_inputs(B=1):
    return (torch.zeros(B), torch.zeros(B), torch.zeros(B),
            torch.zeros(B, L_MAX, 3), torch.ones(B, L_MAX, 3),
            torch.zeros(B, L_MAX), torch.zeros(B, L_MAX),
            torch.zeros(B, H), torch.zeros(B, H))


def main():
    from mayak.lit import LitMayak
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runtime/mayak.onnx")
    args = ap.parse_args()

    lit = LitMayak.load_from_checkpoint(args.ckpt, map_location="cpu")
    wrap = ExportWrapper(lit.model)
    args_in = dummy_inputs(1)

    names_in = ["lat", "lon", "elev", "x_hist", "mask_hist",
                "doy_hist", "hour_hist", "doy_fut", "hour_fut"]
    dyn = {n: {0: "batch"} for n in names_in}
    dyn.update({"q": {0: "batch"}, "mu": {0: "batch"}, "sigma_c": {0: "batch"}})

    torch.onnx.export(
        wrap, args_in, args.out, input_names=names_in,
        output_names=["q", "mu", "sigma_c"], dynamic_axes=dyn,
        opset_version=17, dynamo=False)
    print("Экспортировано:", args.out)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    feed = {n: a.numpy() for n, a in zip(names_in, args_in)}
    q_onnx = sess.run(None, feed)[0]
    with torch.no_grad():
        q_torch = wrap(*args_in)[0].numpy()
    err = float(np.abs(q_onnx - q_torch).max())
    print(f"max|ONNX − PyTorch| по q: {err:.2e}  (норма: доли °C)")

    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        prep = args.out.replace(".onnx", "_prep.onnx")
        quant_pre_process(args.out, prep)
        q8 = args.out.replace(".onnx", "_int8.onnx")
        quantize_dynamic(prep, q8, weight_type=QuantType.QInt8,
                         extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
        import os
        s0 = os.path.getsize(args.out) / 1024
        s8 = os.path.getsize(q8) / 1024
        print(f"Размер: fp32 {s0:.0f} КБ → int8 {s8:.0f} КБ  ({q8})")
    except Exception as e:
        print("int8-квантизация пропущена:", e)


if __name__ == "__main__":
    main()