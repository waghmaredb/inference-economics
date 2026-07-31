# Inference Economics

**What does a token actually cost — in watts, dollars, and carbon?**

The research tells you what a model can do. It almost never tells you what it costs to run once it's serving real traffic. That gap is where most enterprise AI business cases quietly fall apart: the pilot proves capability, nobody prices the production bill, and the program stalls at the exact moment it's supposed to scale.

This is a transparent calculator for the number the leaderboards skip. No hidden benchmarks, no vendor magic — just the arithmetic that turns throughput, power, and price into cost per million tokens, tokens per watt, and cost-to-serve at scale. Every assumption is on the table so you can argue with it, which is the point.

It ships with an interactive front end (`python app.py`) as well as an importable core you can script into your own capacity models — see **Run it locally** below.

## The one number you have to bring

Everything here is honest arithmetic, but arithmetic needs one input no tool can fake for you: **aggregate throughput** — the output tokens per second your model actually sustains on your hardware, at your batch size, sequence length, and quantization. It swings by an order of magnitude across those choices, so there is no credible default. The app ships with a labelled placeholder; replace it with a measured number from your own serving stack (a short load test against your inference server, reading tokens/sec at your target concurrency). Bring that, and every other figure follows.

## The method

The calculator is a handful of formulas, all in [`core.py`](core.py):

```
GPU power        = n_gpus × TDP × utilization
Facility power   = GPU power × PUE
Tokens per watt  = throughput ÷ GPU power
Energy / 1M tok  = facility power ÷ (throughput × 3600) × 1e6              (→ kWh)
Energy $ / 1M    = energy per 1M × electricity price                       (the floor)
Cloud $ / 1M     = (n_gpus × $/GPU-hr) ÷ (throughput × 3600) × 1e6         (rent)
Owned $ / 1M     = (amortized capex + energy) ÷ (throughput × 3600) × 1e6  (buy)
  where amortized capex/hr = purchase price ÷ (life-years × 8760 × duty-cycle)
```

It reports three costs deliberately: the **cloud list price** (renting the GPU, everything bundled in), the **owned TCO** (amortized capex + energy + facility — the number a buy-vs-rent decision actually turns on), and the **electricity-only floor** (raw power if the hardware were free). The interesting one is the middle.

## The headline it exists to expose

Run any realistic set of inputs and two gaps appear. First: the cloud rental price of a million tokens is *tens of times* the raw electricity cost of producing them — in the shipped example (one H100, ~2,400 tok/s, representative pricing), cloud list price lands around **68× the electricity**. The power is not the expensive part; hardware amortization, facility, idle headroom, and margin are.

Second, and more useful: **whether to buy or rent turns almost entirely on duty cycle.** Own the GPU and keep it hot, and the capex spreads thin and undercuts the cloud. Own it and run it half-idle, and the same box costs *more* per token than renting — you're paying for silicon that isn't serving. Optimising your electricity contract while ignoring both gaps is optimising the rounding error. Tokens-per-watt matters; tokens-per-dollar-at-your-utilization is where the business case is won or lost.

## Presets — and why they disagree

The reference presets are all the *same* 70B-class model on *one* H100 — and they range from ~460 to ~2,780 tok/s depending on framework, precision, and concurrency (Cerebrium; Spheron, 2026). That ~6× spread on identical hardware is the whole reason there's no honest default: pick a preset to get a first answer, then measure your own stack and replace it.

## Sourced defaults

Every default is a starting point with a source and a date attached, so it ages visibly instead of silently. All are editable in the app and live in [`data.py`](data.py).

| Input | Default | Source (as accessed) |
|---|---|---|
| GPU TDP (H100 SXM) | 700 W | NVIDIA datasheets; IntuitionLabs GPU spec comparison, 2026 |
| Cloud rental, H100 | $1.49–$6.98 /GPU-hr (rep. ~$3.50) | IntuitionLabs H100 rental comparison, Nov 2025 |
| Electricity | $0.0871 /kWh (US industrial avg) | US EIA, Electricity Monthly Update, May 2026 |
| PUE | 1.20 (Google fleet 1.09 · industry avg 1.54) | Google Data Centers 2025; Uptime Institute 2025 |
| GPU purchase price | H100 SXM ~$27k · A100 80GB ~$16k | IntuitionLabs AI GPU pricing guide, 2026 (others est.) |
| Preset throughput | 460–2,780 tok/s (70B on 1×H100) | Cerebrium; Spheron, 2026 |
| Utilization, duty cycle, depreciation life, grid carbon, throughput | assumptions — tune them | — |

Prices and specs move. Treat the table as a snapshot, not gospel; the tool is built to be corrected.

## Run it locally

```bash
pip install -r requirements.txt
python app.py          # launches the interactive calculator
python test_core.py    # runs the unit tests
```

`core.py` is importable on its own if you'd rather script it into a capacity model or a spreadsheet export.

## What this is and isn't

It is a reasoning tool for order-of-magnitude economics and for pressure-testing a business case before it reaches a board. It is **not** a billing system or a benchmark of any specific model — the throughput presets it ships are cited published figures offered as a starting point, clearly labelled to replace with your own measurement. Bring that measurement; it does the rest.

---

Built by **Deepak Waghmare**. I write about the economics and resilience of enterprise technology at [deepakwaghmare.com](https://deepakwaghmare.com) — this is the code behind one of those arguments. MIT licensed; corrections and pull requests welcome.
