import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Moments (you can replace with your estimates) ----------
mu = np.array([1.06, 1.04, 1.08, 1.03])   # gross means
n = len(mu)
rng = np.random.default_rng(42)
A = rng.normal(size=(n, n))
Sigma = A @ A.T
Sigma = Sigma / (np.max(np.diag(Sigma)) * 150.0)
one = np.ones(n)
SigInv = np.linalg.inv(Sigma)

# ---------- MV frontier helpers ----------
def mv_frontier(R):
    A = one @ SigInv @ one
    B = one @ SigInv @ mu
    C = mu @ SigInv @ mu
    D = A*C - B**2
    var = (A*R**2 - 2*B*R + C) / D
    return np.sqrt(var)

def tangency_point(Rf):
    z = SigInv @ (mu - Rf*one)
    if np.allclose(z, 0):
        w = (SigInv@one)/(one@SigInv@one)
    else:
        w = z/(one@z)
    mean = w @ mu
    sd   = np.sqrt(w @ Sigma @ w)
    return sd, mean

# ---------- HJ frontier ----------
def hj_sigma_min(mbar):
    v = (one - mbar*mu)
    return np.sqrt(v @ SigInv @ v)

# Grids
R_grid = np.linspace(0.95*min(mu.min(), ((SigInv@one)/(one@SigInv@one))@mu),
                     1.05*max(mu.max(), ((SigInv@one)/(one@SigInv@one))@mu), 200)
mv_sd = mv_frontier(R_grid)

mbar_grid = np.linspace(0.70, 1.30, 51)     # slider values (E[m])
Rf_grid   = 1.0/mbar_grid
hj_sd_grid = np.array([hj_sigma_min(x) for x in np.linspace(0.6,1.4,200)])
hj_mbar_axis = np.linspace(0.6,1.4,200)

# ---------- Build figure with two panels ----------
fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                    subplot_titles=("Mean–Variance Frontier (Top)", "HJ Frontier (Bottom)"))

# static lines
fig.add_trace(go.Scatter(x=mv_sd, y=R_grid, mode='lines', name="MV Frontier"), row=1, col=1)
fig.add_trace(go.Scatter(x=hj_mbar_axis, y=hj_sd_grid, mode='lines', name="HJ Frontier"), row=2, col=1)

# moving markers (initialized)
sd0, er0 = tangency_point(Rf_grid[0])
m0, s0   = mbar_grid[0], hj_sigma_min(mbar_grid[0])

tan_trace = go.Scatter(x=[sd0], y=[er0], mode='markers', name="Tangency", marker=dict(size=9))
hj_trace  = go.Scatter(x=[m0],  y=[s0], mode='markers', name="Current E[m]", marker=dict(size=9))

fig.add_trace(tan_trace, row=1, col=1)
fig.add_trace(hj_trace,  row=2, col=1)

fig.update_xaxes(title_text="Portfolio SD", row=1, col=1)
fig.update_yaxes(title_text="Expected Gross Return", row=1, col=1)
fig.update_xaxes(title_text="E[m]", row=2, col=1)
fig.update_yaxes(title_text="σ(m)", row=2, col=1)

# ---------- Slider frames (update markers as mbar changes) ----------
steps = []
for i, mbar in enumerate(mbar_grid):
    Rf = Rf_grid[i]
    sd_t, er_t = tangency_point(Rf)
    sd_m = hj_sigma_min(mbar)
    step = dict(
        method="update",
        args=[{"data": [
            # 0: MV line, 1: HJ line, 2: Tangency point, 3: HJ point
            fig.data[0],               # unchanged MV line
            fig.data[1],               # unchanged HJ line
            go.Scatter(x=[sd_t], y=[er_t], mode='markers', name="Tangency", marker=dict(size=9)),
            go.Scatter(x=[mbar], y=[sd_m], mode='markers', name="Current E[m]", marker=dict(size=9)),
        ]},
        {"annotations": []}],
        label=f"E[m]={mbar:.3f}\nRf≈{Rf:.3f}"
    )
    steps.append(step)

fig.update_layout(
    sliders=[dict(active=0, steps=steps, x=0.05, xanchor="left", y= -0.08, len=0.9)],
    height=800,
    showlegend=True,
    title="MV–HJ Duality (use slider to change E[m]; Rf = 1/E[m])"
)

# Save a single interactive HTML file
fig.write_html("mv_hj_interactive.html", include_plotlyjs="cdn", full_html=True)
print("Wrote mv_hj_interactive.html")
