"""
Inference economics: the cost and energy math behind serving LLM tokens.

Everything here is transparent arithmetic on inputs you control. There are no
hidden benchmarks and no vendor magic. The point of the tool is the opposite of
a leaderboard: it makes the assumptions explicit so an enterprise can argue with
them.

Definitions
-----------
throughput_tok_s : aggregate output tokens/second across the whole GPU set at
                   your batch size and sequence length. This is the one number
                   you must measure for your own workload — see README.
n_gpus           : number of GPUs serving the workload.
tdp_w            : per-GPU thermal design power in watts (manufacturer spec).
utilization      : average power actually drawn as a fraction of TDP while
                   serving (0-1). GPUs rarely sit at 100% TDP under real serving
                   load; this is an assumption you should tune.
pue              : data-center Power Usage Effectiveness (total facility power /
                   IT power). 1.0 is a perfect facility; real ones are higher.
price_per_gpu_hr : on-demand rental price per GPU-hour (USD).
price_per_kwh    : electricity price (USD/kWh).
grid_gco2_kwh    : grid carbon intensity (grams CO2 per kWh).
"""

from __future__ import annotations

SECONDS_PER_HOUR = 3600
ONE_MILLION = 1_000_000


def it_power_watts(n_gpus: float, tdp_w: float, utilization: float) -> float:
    """Power drawn by the GPUs themselves (no facility overhead)."""
    _positive(n_gpus=n_gpus, tdp_w=tdp_w)
    _fraction(utilization=utilization)
    return n_gpus * tdp_w * utilization


def facility_power_watts(n_gpus: float, tdp_w: float, utilization: float, pue: float) -> float:
    """Total facility power including cooling and distribution overhead (PUE)."""
    if pue < 1.0:
        raise ValueError("pue must be >= 1.0 (a facility cannot use less than the IT load)")
    return it_power_watts(n_gpus, tdp_w, utilization) * pue


def tokens_per_watt(throughput_tok_s: float, n_gpus: float, tdp_w: float,
                    utilization: float) -> float:
    """
    Output tokens per second per watt of GPU power (equivalently, tokens per
    joule). Higher is better. This is the pure efficiency number, before any
    facility overhead or price.
    """
    _positive(throughput_tok_s=throughput_tok_s)
    return throughput_tok_s / it_power_watts(n_gpus, tdp_w, utilization)


def energy_kwh_per_1m_tokens(throughput_tok_s: float, n_gpus: float, tdp_w: float,
                             utilization: float, pue: float = 1.0) -> float:
    """
    Facility energy in kWh to produce 1,000,000 output tokens. Set pue=1.0 for
    the GPU-only floor, or a real PUE for the full facility draw.
    """
    _positive(throughput_tok_s=throughput_tok_s)
    power_w = facility_power_watts(n_gpus, tdp_w, utilization, pue)
    wh_per_1m = power_w / (throughput_tok_s * SECONDS_PER_HOUR) * ONE_MILLION
    return wh_per_1m / 1000.0


def energy_cost_per_1m_tokens(throughput_tok_s: float, n_gpus: float, tdp_w: float,
                              utilization: float, price_per_kwh: float,
                              pue: float = 1.0) -> float:
    """The raw electricity cost of 1M tokens — the physical cost floor."""
    _positive(price_per_kwh=price_per_kwh)
    return energy_kwh_per_1m_tokens(throughput_tok_s, n_gpus, tdp_w, utilization, pue) * price_per_kwh


def cloud_cost_per_1m_tokens(throughput_tok_s: float, n_gpus: float,
                             price_per_gpu_hr: float) -> float:
    """
    What 1M tokens cost you at a cloud's on-demand GPU rental price. This bundles
    hardware amortization, facility, margin, and everything else into one number.
    """
    _positive(throughput_tok_s=throughput_tok_s, n_gpus=n_gpus, price_per_gpu_hr=price_per_gpu_hr)
    cost_per_hour = n_gpus * price_per_gpu_hr
    tokens_per_hour = throughput_tok_s * SECONDS_PER_HOUR
    return cost_per_hour / tokens_per_hour * ONE_MILLION


def owned_tco_per_1m_tokens(throughput_tok_s: float, n_gpus: float, tdp_w: float,
                            utilization: float, purchase_price_usd: float,
                            price_per_kwh: float, life_years: float = 3.0,
                            duty_cycle: float = 0.5, pue: float = 1.0) -> float:
    """
    Cost per 1M tokens on hardware you *own* — the number cloud-vs-buy actually
    turns on. It is amortized capital + energy + facility, not the electricity
    floor and not the cloud list price.

    life_years  : depreciation life of the GPU (commonly 3-5 years).
    duty_cycle  : fraction of its life the GPU spends serving real load. This is
                  the hidden lever: capex spread over an idle GPU is expensive.
                  A box you own but only use half the time costs ~2x per token.
    """
    _positive(throughput_tok_s=throughput_tok_s, n_gpus=n_gpus,
              purchase_price_usd=purchase_price_usd, price_per_kwh=price_per_kwh,
              life_years=life_years)
    _fraction(duty_cycle=duty_cycle)

    hours_of_service = life_years * 8760 * duty_cycle
    capex_per_gpu_hour = purchase_price_usd / hours_of_service

    facility_power_kw = facility_power_watts(1, tdp_w, utilization, pue) / 1000.0
    energy_per_gpu_hour = facility_power_kw * price_per_kwh

    owned_cost_per_gpu_hour = capex_per_gpu_hour + energy_per_gpu_hour
    tokens_per_hour = throughput_tok_s * SECONDS_PER_HOUR
    return n_gpus * owned_cost_per_gpu_hour / tokens_per_hour * ONE_MILLION


def co2_kg_per_1m_tokens(throughput_tok_s: float, n_gpus: float, tdp_w: float,
                         utilization: float, grid_gco2_kwh: float,
                         pue: float = 1.0) -> float:
    """Operational carbon (kg CO2) per 1M tokens from grid electricity."""
    _positive(grid_gco2_kwh=grid_gco2_kwh)
    kwh = energy_kwh_per_1m_tokens(throughput_tok_s, n_gpus, tdp_w, utilization, pue)
    return kwh * grid_gco2_kwh / 1000.0


def scale_cost(cost_per_1m_tokens: float, tokens_per_day: float, days: int = 30) -> dict:
    """Project a per-1M-token cost out to daily / monthly / annual volume."""
    _positive(tokens_per_day=tokens_per_day)
    per_day = tokens_per_day / ONE_MILLION * cost_per_1m_tokens
    return {"per_day": per_day, "per_period": per_day * days, "per_year": per_day * 365}


def cloud_vs_energy_multiple(cloud_cost: float, energy_cost: float) -> float:
    """
    How many times the raw electricity cost you pay at cloud list price. This is
    the headline the tool exists to expose: the power is almost never the
    expensive part.
    """
    if energy_cost <= 0:
        raise ValueError("energy_cost must be > 0")
    return cloud_cost / energy_cost


# --- validation helpers -----------------------------------------------------

def _positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value is None or value <= 0:
            raise ValueError(f"{name} must be > 0 (got {value!r})")


def _fraction(**kwargs) -> None:
    for name, value in kwargs.items():
        if not (0 < value <= 1):
            raise ValueError(f"{name} must be in (0, 1] (got {value!r})")
