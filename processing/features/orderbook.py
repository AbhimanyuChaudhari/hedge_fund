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