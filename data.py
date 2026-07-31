"""
Sourced default inputs. Every figure here is a starting point you should replace
with your own measurements or contract prices. Sources and dates are attached so
you can check them and so they age visibly rather than silently.


TDP figures are manufacturer thermal design power for the SXM form factor unless
noted; real average draw under serving load is lower (that is what `utilization`
is for). PCIe variants draw less than SXM — override if that is what you run.
"""


# NVIDIA data-center GPU: thermal design power (watts) and approximate purchase
# price (USD). TDP sources: NVIDIA datasheets; IntuitionLabs GPU spec comparison
# (2026). Purchase-price sources: IntuitionLabs AI GPU pricing guide (2026) —
# H100 SXM $25-40k ("from ~$27k"), A100 80GB $15-17k. Figures marked "est." are
# rough street estimates, not a cited quote — treat them as an editable input.
GPUS = {
    "H100 SXM (80GB)":   {"tdp_w": 700,  "memory_gb": 80,  "price_usd": 27000, "note": "SXM5; PCIe ~350W. Price: IntuitionLabs 2026"},
    "H200 SXM (141GB)":  {"tdp_w": 700,  "memory_gb": 141, "price_usd": 32000, "note": "up to 700W. Price est."},
    "B200 (192GB)":      {"tdp_w": 1000, "memory_gb": 192, "price_usd": 40000, "note": "Blackwell ~1000W. Price est."},
    "A100 SXM (80GB)":   {"tdp_w": 400,  "memory_gb": 80,  "price_usd": 16000, "note": "SXM4; PCIe ~300W. Price: IntuitionLabs 2026"},
    "A100 PCIe (80GB)":  {"tdp_w": 300,  "memory_gb": 80,  "price_usd": 16000, "note": "PCIe. Price: IntuitionLabs 2026"},
    "L40S (48GB)":       {"tdp_w": 350,  "memory_gb": 48,  "price_usd": 9000,  "note": "Ada 350W. Price est."},
    "L4 (24GB)":         {"tdp_w": 72,   "memory_gb": 24,  "price_usd": 2500,  "note": "72W. Price est."},
}


# GPU depreciation life and duty cycle for the owned-hardware TCO line.
# Life: commonly 3-5 years for AI GPUs (hyperscalers have extended toward 6).
# Duty cycle: fraction of the hardware's life spent serving real load — the
# single biggest swing in buy-vs-rent. Both are assumptions; tune them.
DEFAULT_LIFE_YEARS = 3
DEFAULT_DUTY_CYCLE = 0.5


# Reference throughput presets. Each is a REAL published benchmark, cited — but
# the point they make together is that the SAME 70B model on ONE H100 ranges
# ~460 to ~2,780 tok/s depending on framework, precision, and concurrency. Use
# them to get a first answer, then measure your own. Sources: Spheron (2026) FP8
# concurrency benchmark; Cerebrium Llama-3.1 batch benchmark.
PRESETS = {
    "Custom — measure your own": None,
    "Llama 3.3 70B FP8 · 1×H100 · TensorRT-LLM @100 concurrency": {
        "gpu": "H100 SXM (80GB)", "n_gpus": 1, "throughput": 2780, "src": "Spheron 2026"},
    "Llama 3.3 70B FP8 · 1×H100 · vLLM @100 concurrency": {
        "gpu": "H100 SXM (80GB)", "n_gpus": 1, "throughput": 2400, "src": "Spheron 2026"},
    "Llama 3.1 70B · 1×H100 · SGLang @batch 64": {
        "gpu": "H100 SXM (80GB)", "n_gpus": 1, "throughput": 460, "src": "Cerebrium"},
}


# On-demand cloud rental, USD per GPU-hour. Wide by provider; edit to your rate.
# Source: IntuitionLabs, "H100 Rental Prices Compared" (Nov 2025):
#   H100 low ~$1.49 (specialist clouds), representative ~$3.00-3.90 (AWS/GCP),
#   high ~$6.98 (Azure); A100 now sub-$1/GPU-hr open market.
CLOUD_PRICE_USD_PER_GPU_HR = {
    "H100 – specialist cloud (low)": 1.49,
    "H100 – hyperscaler (representative)": 3.50,
    "H100 – Azure (high)": 6.98,
    "A100 – open market": 0.99,
}


# Electricity price, USD/kWh. Default: US industrial average, 8.71 cents/kWh,
# May 2026 (US EIA, Electricity Monthly Update). Replace with your tariff —
# large data centers often contract well below retail.
DEFAULT_PRICE_PER_KWH = 0.0871


# Power Usage Effectiveness. Google fleet 2025 = 1.09; Uptime Institute 2025
# global average = 1.54. Default 1.2 as a modern-but-not-hyperscale placeholder.
DEFAULT_PUE = 1.20
PUE_REFERENCES = {"Hyperscaler (Google fleet 2025)": 1.09,
                  "Modern enterprise (default)": 1.20,
                  "Industry average (Uptime 2025)": 1.54}


# Average power draw as a fraction of TDP while serving. An assumption, not a
# spec — tune it. 0.7 is a middle-of-the-road placeholder.
DEFAULT_UTILIZATION = 0.70


# Grid carbon intensity, gCO2 per kWh. Default ~ US average grid; varies enormously
# by region and by hour. Replace with your grid's figure for a real number.
DEFAULT_GRID_GCO2_KWH = 380


# Throughput has NO credible default. It depends on model, hardware, batch size,
# sequence length, quantization, and framework. This placeholder exists only so
# the app runs on first load — measure your own and replace it.
PLACEHOLDER_THROUGHPUT_TOK_S = 2000
