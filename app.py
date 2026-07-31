"""
Interactive front end for the inference economics calculator.
Runs as a Hugging Face Space (Gradio) or locally: `python app.py`.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

import core
import data

# Deepak Waghmare design system: navy ground, amber + teal accents.
NAVY, INK, MUTED = "#0B1B2B", "#E8EEF5", "#94A3B8"
AMBER, TEAL, SLATE = "#F5A623", "#2DD4BF", "#5B6B7F"


def cost_chart(cloud, owned, floor):
    labels = ["Electricity\nfloor", "Owned TCO\n(amortized)", "Cloud rental\n(list price)"]
    vals = [floor, owned, cloud]
    colors = [SLATE, TEAL, AMBER]

    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=130)
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    bars = ax.barh(labels, vals, color=colors, height=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.015, b.get_y() + b.get_height() / 2,
                f"${v:,.4f}", va="center", ha="left", color=INK,
                fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_title("Cost to serve 1M tokens — the gap the leaderboards skip",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.tick_params(colors=INK, labelsize=10.5, length=0)
    fig.tight_layout()
    return fig


def compute(gpu_name, n_gpus, throughput, utilization, pue,
            cloud_price_choice, custom_price, price_per_kwh, grid_gco2,
            purchase_price, life_years, duty_cycle, tokens_per_day):
    tdp_w = data.GPUS[gpu_name]["tdp_w"]
    price_per_gpu_hr = (custom_price if cloud_price_choice == "Custom…"
                        else data.CLOUD_PRICE_USD_PER_GPU_HR[cloud_price_choice])

    tpw = core.tokens_per_watt(throughput, n_gpus, tdp_w, utilization)
    kwh_fac = core.energy_kwh_per_1m_tokens(throughput, n_gpus, tdp_w, utilization, pue=pue)
    co2 = core.co2_kg_per_1m_tokens(throughput, n_gpus, tdp_w, utilization, grid_gco2, pue=pue)

    floor = core.energy_cost_per_1m_tokens(throughput, n_gpus, tdp_w, utilization, price_per_kwh, pue=pue)
    owned = core.owned_tco_per_1m_tokens(throughput, n_gpus, tdp_w, utilization, purchase_price,
                                         price_per_kwh, life_years, duty_cycle, pue=pue)
    cloud = core.cloud_cost_per_1m_tokens(throughput, n_gpus, price_per_gpu_hr)

    cloud_scale = core.scale_cost(cloud, tokens_per_day)
    owned_scale = core.scale_cost(owned, tokens_per_day)
    floor_scale = core.scale_cost(floor, tokens_per_day)

    headline = (
        f"## {throughput:,.0f} tok/s on {n_gpus:g} × {gpu_name}\n"
        f"**{tpw:,.2f}** tokens/sec per watt · **{kwh_fac*1000:,.1f} Wh** / 1M tokens "
        f"(facility) · **{co2*1000:,.0f} g CO₂** / 1M tokens"
    )

    table = [
        ["Cloud rental (list price)", f"${cloud:,.4f}",
         f"${cloud_scale['per_period']:,.0f}", f"${cloud_scale['per_year']:,.0f}"],
        ["Owned TCO (amortized capex + power)", f"${owned:,.4f}",
         f"${owned_scale['per_period']:,.0f}", f"${owned_scale['per_year']:,.0f}"],
        ["Electricity floor", f"${floor:,.4f}",
         f"${floor_scale['per_period']:,.0f}", f"${floor_scale['per_year']:,.0f}"],
    ]

    buy_vs_rent = core.cloud_vs_energy_multiple(cloud, owned)
    cheaper = "owning undercuts renting" if owned < cloud else "renting beats owning"
    verdict = (
        f"### Buy vs rent, at these inputs\n"
        f"Cloud list price is **{core.cloud_vs_energy_multiple(cloud, floor):,.0f}×** the raw "
        f"electricity, and **{buy_vs_rent:,.1f}×** the fully-amortized cost of hardware you own "
        f"and run at {duty_cycle*100:.0f}% duty over {life_years:g} years. So **{cheaper}** here.\n\n"
        f"The lever is that duty cycle. Own the GPU and keep it hot and the capex spreads thin; "
        f"own it and run it half-idle and the same box costs far more per token than the cloud. "
        f"The electricity — the row everyone points at — is the rounding error."
    )
    return headline, cost_chart(cloud, owned, floor), table, verdict


def apply_preset(name):
    p = data.PRESETS.get(name)
    if not p:
        return gr.update(), gr.update(), gr.update()
    return gr.update(value=p["gpu"]), gr.update(value=p["n_gpus"]), gr.update(value=p["throughput"])


def prefill_capex(gpu_name):
    return gr.update(value=data.GPUS[gpu_name]["price_usd"])


with gr.Blocks(title="Inference Economics") as demo:
    gr.Markdown(
        "# Inference Economics\n"
        "### What does a token *actually* cost — to rent, to own, and in raw power?\n\n"
        "The research tells you what a model can do. Almost nobody prices what it costs to run at "
        "scale. This is transparent arithmetic on inputs you control — no hidden benchmarks. "
        "**The one number you must supply is throughput** (aggregate output tok/s for *your* model, "
        "hardware, and batch). Start from a preset, then measure your own; everything else ships "
        "with a sourced, editable default."
    )

    preset = gr.Dropdown(list(data.PRESETS.keys()), value="Custom — measure your own",
                         label="Reference preset (published benchmark — override with your measurement)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Hardware & workload")
            gpu = gr.Dropdown(list(data.GPUS.keys()), value="H100 SXM (80GB)", label="GPU")
            n_gpus = gr.Number(value=1, label="Number of GPUs", minimum=1)
            throughput = gr.Number(value=data.PLACEHOLDER_THROUGHPUT_TOK_S,
                                   label="Aggregate throughput (output tok/s) — MEASURE THIS", minimum=1)
            utilization = gr.Slider(0.1, 1.0, value=data.DEFAULT_UTILIZATION, step=0.05,
                                    label="Avg power draw as fraction of TDP")
            pue = gr.Slider(1.0, 2.0, value=data.DEFAULT_PUE, step=0.01,
                            label="Data-center PUE (Google 1.09 · industry avg 1.54)")
        with gr.Column():
            gr.Markdown("#### Prices & ownership")
            cloud_choice = gr.Dropdown(
                list(data.CLOUD_PRICE_USD_PER_GPU_HR.keys()) + ["Custom…"],
                value="H100 – hyperscaler (representative)", label="Cloud rental $/GPU-hr")
            custom_price = gr.Number(value=3.50, label="Custom $/GPU-hr (if selected)")
            price_kwh = gr.Number(value=data.DEFAULT_PRICE_PER_KWH,
                                  label="Electricity $/kWh (US industrial avg 0.0871)")
            purchase_price = gr.Number(value=data.GPUS["H100 SXM (80GB)"]["price_usd"],
                                       label="GPU purchase price $ (for owned TCO)")
            life_years = gr.Slider(1, 6, value=data.DEFAULT_LIFE_YEARS, step=1,
                                   label="Depreciation life (years)")
            duty_cycle = gr.Slider(0.1, 1.0, value=data.DEFAULT_DUTY_CYCLE, step=0.05,
                                   label="Duty cycle — share of life spent serving")

    with gr.Row():
        grid = gr.Number(value=data.DEFAULT_GRID_GCO2_KWH, label="Grid carbon (gCO₂/kWh)")
        tokens_day = gr.Number(value=1_000_000_000, label="Tokens/day (for scale projection)")

    go = gr.Button("Calculate", variant="primary")
    headline = gr.Markdown()
    chart = gr.Plot(label="Cost to serve 1M tokens")
    table = gr.Dataframe(headers=["Cost basis", "per 1M tokens", "per month", "per year"],
                         label="Cost to serve", interactive=False, wrap=True)
    verdict = gr.Markdown()

    gr.Markdown(
        "---\n*Defaults are sourced and dated in `data.py` — replace them with your own contract "
        "prices and measured throughput. Sources: NVIDIA & IntuitionLabs (GPU TDP & purchase price, "
        "2026); IntuitionLabs (H100 rental, Nov 2025); US EIA (electricity, May 2026); Uptime "
        "Institute & Google (PUE, 2025); Spheron & Cerebrium (preset throughputs, 2026). "
        "Built by Deepak Waghmare — deepakwaghmare.com*"
    )

    inputs = [gpu, n_gpus, throughput, utilization, pue, cloud_choice, custom_price,
              price_kwh, grid, purchase_price, life_years, duty_cycle, tokens_day]
    outputs = [headline, chart, table, verdict]

    preset.change(apply_preset, preset, [gpu, n_gpus, throughput])
    gpu.change(prefill_capex, gpu, purchase_price)
    go.click(compute, inputs, outputs)
    demo.load(compute, inputs, outputs)


if __name__ == "__main__":
    demo.launch()
