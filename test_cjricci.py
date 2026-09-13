from strategies.implementations.cartea_jaimungal_ricci.model import CJRicciModel
from strategies.implementations.cartea_jaimungal_ricci.parameters import CJRicciParameters
import time

params = CJRicciParameters(gamma=0.1, sigma=2.0, kappa=1.5)
model  = CJRicciModel(params)

print('=== CJ-Ricci vs CJ comparison ===')
print()

print('Quiet market (no Hawkes excitation):')
for q in [-3, 0, 3]:
    bid, ask = model.optimal_quotes(mid=24000, q=q,
                                     time_remaining=0.5, alpha=0.0)
    diag = model.get_diagnostics()
    keff = diag['kappa_eff']
    print(f'  q={q:+d}  bid={bid:.2f}  ask={ask:.2f}  spread={ask-bid:.2f}  kappa_eff={keff:.4f}')

print()

now = time.time()
for i in range(10):
    model.hawkes.add_event(now - i*0.5)

print('Active market (10 recent orders):')
for q in [-3, 0, 3]:
    bid, ask = model.optimal_quotes(mid=24000, q=q,
                                     time_remaining=0.5, alpha=0.0)
    diag = model.get_diagnostics()
    keff = diag['kappa_eff']
    print(f'  q={q:+d}  bid={bid:.2f}  ask={ask:.2f}  spread={ask-bid:.2f}  kappa_eff={keff:.4f}')
