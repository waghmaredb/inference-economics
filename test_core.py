"""Unit tests. Run: python -m pytest test_core.py -q  (or python test_core.py)"""

import math
import core


def approx(a, b, tol=1e-3):
    return math.isclose(a, b, rel_tol=tol)


def test_it_and_facility_power():
    assert approx(core.it_power_watts(1, 700, 0.7), 490.0)
    assert approx(core.facility_power_watts(1, 700, 0.7, 1.2), 588.0)


def test_tokens_per_watt():
    # 2000 tok/s over 490 W of GPU draw
    assert approx(core.tokens_per_watt(2000, 1, 700, 0.7), 2000 / 490)


def test_energy_floor_vs_facility():
    floor = core.energy_kwh_per_1m_tokens(2000, 1, 700, 0.7, pue=1.0)
    facility = core.energy_kwh_per_1m_tokens(2000, 1, 700, 0.7, pue=1.2)
    assert approx(floor, 490 / 7.2 / 1000)          # ~0.06806 kWh
    assert approx(facility, 588 / 7.2 / 1000)        # ~0.08167 kWh
    assert facility > floor


def test_cloud_cost_per_1m():
    # $3.50/GPU-hr, 1 GPU, 2000 tok/s
    assert approx(core.cloud_cost_per_1m_tokens(2000, 1, 3.50), 3.5 / 7.2)  # ~$0.4861


def test_energy_cost_per_1m():
    cost = core.energy_cost_per_1m_tokens(2000, 1, 700, 0.7, price_per_kwh=0.0871, pue=1.2)
    assert approx(cost, 588 / 7.2 / 1000 * 0.0871)   # ~$0.00711


def test_cloud_is_a_large_multiple_of_energy():
    energy = core.energy_cost_per_1m_tokens(2000, 1, 700, 0.7, 0.0871, pue=1.2)
    cloud = core.cloud_cost_per_1m_tokens(2000, 1, 3.50)
    mult = core.cloud_vs_energy_multiple(cloud, energy)
    assert 60 < mult < 75   # the headline: cloud list price dwarfs the electricity


def test_scale_projection():
    per_1m = core.cloud_cost_per_1m_tokens(2000, 1, 3.50)
    s = core.scale_cost(per_1m, tokens_per_day=1_000_000_000, days=30)  # 1B tokens/day
    assert approx(s["per_day"], 1000 * per_1m)
    assert approx(s["per_period"], s["per_day"] * 30)
    assert approx(s["per_year"], s["per_day"] * 365)


def test_owned_tco_value():
    # H100 $27k, 3yr life, 50% duty, 2000 tok/s
    tco = core.owned_tco_per_1m_tokens(2000, 1, 700, 0.7, purchase_price_usd=27000,
                                       price_per_kwh=0.0871, life_years=3, duty_cycle=0.5, pue=1.2)
    # capex/hr 27000/13140 + energy 0.588*0.0871, over 7.2M tok/hr
    expected = (27000 / (3 * 8760 * 0.5) + 0.588 * 0.0871) / 7.2
    assert approx(tco, expected)   # ~$0.2925 / 1M


def test_owned_sits_between_floor_and_cloud():
    floor = core.energy_cost_per_1m_tokens(2000, 1, 700, 0.7, 0.0871, pue=1.2)
    owned = core.owned_tco_per_1m_tokens(2000, 1, 700, 0.7, 27000, 0.0871, 3, 0.5, pue=1.2)
    cloud = core.cloud_cost_per_1m_tokens(2000, 1, 3.50)
    assert floor < owned < cloud   # the whole point of the third line


def test_lower_duty_cycle_raises_owned_cost():
    high = core.owned_tco_per_1m_tokens(2000, 1, 700, 0.7, 27000, 0.0871, 3, 0.8, pue=1.2)
    low = core.owned_tco_per_1m_tokens(2000, 1, 700, 0.7, 27000, 0.0871, 3, 0.2, pue=1.2)
    assert low > high   # idle owned hardware is expensive per token


def test_co2():
    kg = core.co2_kg_per_1m_tokens(2000, 1, 700, 0.7, grid_gco2_kwh=380, pue=1.2)
    assert approx(kg, 588 / 7.2 / 1000 * 380 / 1000)


def test_validation():
    for bad in (0, -1):
        try:
            core.tokens_per_watt(bad, 1, 700, 0.7)
            assert False, "expected ValueError"
        except ValueError:
            pass
    try:
        core.facility_power_watts(1, 700, 0.7, 0.9)  # pue < 1
        assert False
    except ValueError:
        pass
    try:
        core.it_power_watts(1, 700, 1.5)  # utilization > 1
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
