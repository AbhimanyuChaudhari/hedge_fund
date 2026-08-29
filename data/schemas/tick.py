from dataclasses import dataclass
from typing import NamedTuple, Optional
from datetime import datetime


class DepthLevel(NamedTuple):
    price: float
    quantity: int
    order: int

@dataclass
class Tick:
    symbol:str
    bid:tuple[DepthLevel, DepthLevel, DepthLevel, DepthLevel, DepthLevel]
    ask:tuple[DepthLevel, DepthLevel, DepthLevel, DepthLevel, DepthLevel]
    volume:int
    timestamp: int
    ltp: float
    open_interest: int
    expiry: datetime
    instrument_type: str
    strike_price: Optional[float] = None
    option_type: Optional[str] = None

