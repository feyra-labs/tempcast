"""Экспорт МАЯК в ONNX одним графом.

Архитектура и размеры входов берутся из конфига модели, сохранённого в чекпойнте;
тот же конфиг записывается в метаданные ONNX-файла (ключ mayak_model_config).
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn


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


def dummy_inputs(cfg, B=1):
    L, Hh = cfg.max_history, cfg.horizon
    return (torch.zeros(B), torch.zeros(B), torch.zeros(B),
            torch.zeros(B, L, 3), torch.ones(B, L, 3),
            torch.zeros(B, L), torch.zeros(B, L),
            torch.zeros(B, Hh), torch.zeros(B, Hh))


def main():
    ap = argparse.ArgumentParser(description="экспорт МАЯК в ONNX")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runtime/mayak.onnx")
    args = ap.parse_args()
    from mayak.lit import load_model

    model = load_model(args.ckpt)
    wrap = ExportWrapper(model)
    args_in = dummy_inputs(model.cfg, 1)

    names_in = ["lat", "lon", "elev", "x_hist", "mask_hist",
                "doy_hist", "hour_hist", "doy_fut", "hour_fut"]
    dyn = {n: {0: "batch"} for n in names_in}
    dyn.update({"q": {0: "batch"}, "mu": {0: "batch"}, "sigma_c": {0: "batch"}})

    torch.onnx.export(
        wrap, args_in, args.out, input_names=names_in,
        output_names=["q", "mu", "sigma_c"], dynamic_axes=dyn,
        opset_version=17, dynamo=False)
    import onnx
    proto = onnx.load(args.out)
    meta = proto.metadata_props.add()
    meta.key, meta.value = "mayak_model_config", json.dumps(model.cfg.to_dict(), ensure_ascii=False)
    onnx.save(proto, args.out)
    print("Экспортировано:", args.out)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    feed = {n: a.numpy() for n, a in zip(names_in, args_in)}
    q_onnx = sess.run(None, feed)[0]
    with torch.no_grad():
        q_torch = wrap(*args_in)[0].numpy()
    err = float(np.abs(q_onnx - q_torch).max())
    print(f"max|ONNX − PyTorch| по q: {err:.2e}  (норма: доли °C)")


if __name__ == "__main__":
    main()
