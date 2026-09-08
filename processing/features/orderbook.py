import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

def order_imbalance(bid_qty: list[float], ask_qty: list[float]) -> float:
    total_bid = sum(bid_qty)
    total_ask = sum(ask_qty)
    denom = total_bid + total_ask
    if denom == 0:
        return 0.0
    return (total_bid - total_ask) / denom

def weighted_mid_price(bid_prices: list[float], bid_qtys: list[float],
                       ask_prices: list[float], ask_qtys: list[float]) -> float:
    best_bid = bid_prices[0] if bid_prices else 0
    best_ask = ask_prices[0] if ask_prices else 0
    bid_q    = bid_qtys[0]   if bid_qtys  else 0
    ask_q    = ask_qtys[0]   if ask_qtys  else 0
    denom    = bid_q + ask_q
    if denom == 0:
        return (best_bid + best_ask) / 2
    return (best_bid * ask_q + best_ask * bid_q) / denom

def best_spread(best_bid: float, best_ask: float) -> float:
    return best_ask - best_bid

def depth_weighted_price(prices: list[float], qtys: list[float]) -> float:
    total_qty = sum(qtys)
    if total_qty == 0:
        return prices[0] if prices else 0.0
    return sum(p * q for p, q in zip(prices, qtys)) / total_qty

def ofi_cont_kukanov(row: dict, prev_row: dict) -> float:
    """
    Cont-Kukanov-Stoikov (2014) multi-level weighted OFI.
    Uses changes in bid/ask quantities weighted by inverse distance from mid.
    
    OFI_t = Σ_i w_i × (ΔBid_qi - ΔAsk_qi)
    
    where w_i = 1/(price_distance_from_mid at level i)
    """
    mid = (row.get('bid_p1', 0) + row.get('ask_p1', 0)) / 2
    if mid == 0:
        return 0.0

    ofi = 0.0
    for i in range(1, 6):
        bid_p = row.get(f'bid_p{i}', 0)
        ask_p = row.get(f'ask_p{i}', 0)
        bid_q = row.get(f'bid_q{i}', 0)
        ask_q = row.get(f'ask_q{i}', 0)
        prev_bid_q = prev_row.get(f'bid_q{i}', 0)
        prev_ask_q = prev_row.get(f'ask_q{i}', 0)

        # Weight by inverse distance from mid
        w_bid = 1 / (mid - bid_p + 1e-9) if bid_p > 0 and bid_p < mid else 0
        w_ask = 1 / (ask_p - mid + 1e-9) if ask_p > 0 and ask_p > mid else 0

        # Flow = change in quantity (positive = buy pressure)
        delta_bid = bid_q - prev_bid_q
        delta_ask = ask_q - prev_ask_q

        ofi += w_bid * delta_bid - w_ask * delta_ask

    return float(ofi)