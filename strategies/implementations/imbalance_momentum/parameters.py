from dataclasses import dataclass

@dataclass
class ImbalanceMomentumParameters:
    # ── Signal thresholds ──────────────────────────────────────────
    imbalance_threshold:  float = 0.30   # min imbalance to trigger (TODO: calibrate)
    confirmation_bars:    int   = 3      # consecutive bars above threshold
    
    # ── Position management ────────────────────────────────────────
    max_inventory:        int   = 5      # max lots
    lot_size:             int   = 1000   # USDINR lot size
    
    # ── Exit rules ────────────────────────────────────────────────
    stop_loss_ticks:      int   = 10     # stop loss in ticks
    take_profit_ticks:    int   = 20     # take profit in ticks
    max_hold_seconds:     int   = 300    # max hold time (5 minutes)
    
    # ── Filters ───────────────────────────────────────────────────
    min_spread:           float = 0.0025 # min market spread to trade
    min_bid_qty:          float = 1000   # min depth at best bid
    min_ask_qty:          float = 1000   # min depth at best ask
    
    # ── Instrument ────────────────────────────────────────────────
    tick_size:            float = 0.0025
    instrument_type:      str   = 'currency_futures'