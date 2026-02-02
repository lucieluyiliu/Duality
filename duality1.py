import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as sco

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Duality: MV Frontier vs SDF Bounds")

st.markdown(
    """
    <style>
      .stApp { background-color: #ffffff; color: #000000; }
      section[data-testid="stSidebar"] { background-color: #ffffff; color: #000000; }
      header, footer { background-color: #ffffff !important; color: #000000 !important; }
      .stMarkdown, .stText, .stTitle, .stSubheader, .stCaption { color: #000000 !important; }
      label, p, span, div { color: #000000; }
      input, textarea, select { background-color: #ffffff !important; color: #000000 !important; }
      .stNumberInput input, .stTextInput input, .stSelectbox select, .stSlider input {
        background-color: #ffffff !important;
        color: #000000 !important;
      }
      .stSlider div, .stSlider span { color: #000000 !important; }
      .stDataFrame, .stTable, table, th, td { color: #000000 !important; background-color: #ffffff !important; }
      .stMarkdown a { color: #000000 !important; }
      [data-testid="stMainMenu"] { background-color: #ffffff !important; color: #000000 !important; }
      [data-testid="stMainMenu"] * { color: #000000 !important; }
      [data-testid="stMainMenu"] div { background-color: #ffffff !important; }
      [data-testid="stDeployButton"] { background-color: #ffffff !important; color: #000000 !important; }
      [data-testid="stDeployButton"] * { color: #000000 !important; }
      [data-testid="stPopover"] { background-color: #ffffff !important; color: #000000 !important; }
      [data-testid="stPopover"] * { color: #000000 !important; }
      [data-testid="stDialog"] { background-color: #ffffff !important; color: #000000 !important; }
      [data-testid="stDialog"] * { color: #000000 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("The Duality: Mean-Variance Frontier & Hansen-Jagannathan Bound")
st.markdown("""
This app illustrates the mathematical link between the Asset Pricing world (Returns) and the Stochastic Discount Factor world (SDF).
**Key Concept:** The slope of the Capital Market Line (Max Sharpe Ratio) determines the slope of the minimum volatility cone for the SDF.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Market Parameters")
mu1 = st.sidebar.number_input("Asset 1 Exp Return", value=0.06, step=0.01)
mu2 = st.sidebar.number_input("Asset 2 Exp Return", value=0.12, step=0.01)
std1 = st.sidebar.number_input("Asset 1 Volatility", value=0.10, step=0.01)
std2 = st.sidebar.number_input("Asset 2 Volatility", value=0.20, step=0.01)
corr = st.sidebar.slider("Correlation", -0.9, 0.9, 0.3, 0.1)

st.sidebar.header("2. The Bridge (Duality)")
rf_input = st.sidebar.slider("Risk Free Rate (Rf)", 0.0, 0.08, 0.03, 0.005, format="%.3f")

# Derived E[M]
em_target = 1 / (1 + rf_input)
st.sidebar.info(f"Implied E[M] = 1/(1+Rf) = {em_target:.4f}")

# --- CALCULATIONS ---

# 1. Setup Data
mu = np.array([mu1, mu2])
cov_matrix = np.array([
    [std1 ** 2, corr * std1 * std2],
    [corr * std1 * std2, std2 ** 2]
])
inv_cov = np.linalg.inv(cov_matrix)

# 2. Efficient Frontier Generation (Static geometry)
# For two assets, compute the frontier analytically to avoid numerical artifacts.
num_ports = 400
w_min, w_max = -3.0, 3.0  # Allow long-short to display full frontier
r_at_wmin = w_min * mu1 + (1 - w_min) * mu2
r_at_wmax = w_max * mu1 + (1 - w_max) * mu2
frontier_ret = np.linspace(min(r_at_wmin, r_at_wmax), max(r_at_wmin, r_at_wmax), num_ports)

# Solve weights for each return level: r = w*mu1 + (1-w)*mu2
if mu1 != mu2:
    w1 = (frontier_ret - mu2) / (mu1 - mu2)
else:
    w1 = np.full_like(frontier_ret, 0.5)
w2 = 1 - w1
valid = (w1 >= w_min) & (w1 <= w_max) & (w2 >= w_min) & (w2 <= w_max)
w1 = w1[valid]
w2 = w2[valid]
frontier_ret = frontier_ret[valid]
frontier_vol = np.sqrt(
    w1**2 * std1**2 + w2**2 * std2**2 + 2 * w1 * w2 * corr * std1 * std2
)

# Global minimum-variance portfolio (within bounds)
gmv_w1 = np.clip((std2**2 - corr * std1 * std2) /
                 (std1**2 + std2**2 - 2 * corr * std1 * std2), w_min, w_max)
gmv_w2 = 1 - gmv_w1
gmv_ret = gmv_w1 * mu1 + gmv_w2 * mu2

# 3. Tangency Portfolio Calculation (Dynamic based on Rf)
# Analytical solution for Tangency weights: w = Sigma^-1 * (mu - rf) / sum(...)
excess_mu = mu - rf_input
numerator = np.dot(inv_cov, excess_mu)
tangency_weights = numerator / np.sum(numerator)

tangency_ret = np.sum(tangency_weights * mu)
tangency_vol = np.sqrt(np.dot(tangency_weights.T, np.dot(cov_matrix, tangency_weights)))
max_sharpe = (tangency_ret - rf_input) / tangency_vol

# CML Line Points
cml_x = [0, tangency_vol * 1.5]
cml_y = [rf_input, rf_input + max_sharpe * (tangency_vol * 1.5)]

# 4. Hansen-Jagannathan Bound Calculation
# Formula: Var(M) = (1 - E[M]E[R])' Sigma^-1 (1 - E[M]E[R])
# We compute sigma_m for a range of E[M]
em_range = np.linspace(0.8, 1.2, 100)
hj_vol = []

for m in em_range:
    # Pricing error vector: 1 - m * mu
    # Note: strictly we use Gross Returns (1+R), but for small time steps R approx log returns
    # Let's stick to excess return duality logic or standard pricing equation: 1 = E[M(1+R)]
    # Pricing error vector u = 1 - m * (1 + mu)

    # Using Gross Returns for precision in HJ formula
    gross_mu = 1 + mu
    u = 1.0 - m * gross_mu

    # Variance of Projection of M
    var_m = np.dot(u.T, np.dot(inv_cov, u))
    hj_vol.append(np.sqrt(var_m))

# The specific point corresponding to Rf
# If the SDF prices the risk free rate, E[M] = 1/(1+Rf)
# The minimum volatility at this specific E[M] is determined by the max Sharpe Ratio.
# Theoretical Min Vol = E[M] * MaxSharpe (approx for lognormality) or calculated via HJ formula
target_u = 1.0 - em_target * (1 + mu)
target_sdf_var = np.dot(target_u.T, np.dot(inv_cov, target_u))
target_sdf_vol = np.sqrt(target_sdf_var)

# --- PLOTTING ---

col1, col2 = st.columns(2)

# PLOT 1: Mean-Variance
with col1:
    st.subheader("Asset Space: Efficient Frontier")
    fig1 = go.Figure()

    # Minimum-variance frontier (both halves)
    fig1.add_trace(
        go.Scatter(x=frontier_vol, y=frontier_ret, mode='lines',
                   name='Minimum-Variance Frontier', line=dict(color='black')))
    # Efficient (upper) half highlight
    efficient = frontier_ret >= gmv_ret
    fig1.add_trace(
        go.Scatter(x=frontier_vol[efficient], y=frontier_ret[efficient], mode='lines',
                   name='Efficient Half', line=dict(color='black', width=3)))

    # Assets
    fig1.add_trace(
        go.Scatter(x=[std1, std2], y=[mu1, mu2], mode='markers', name='Assets', marker=dict(size=10, color='blue')))

    # CML
    fig1.add_trace(
        go.Scatter(x=cml_x, y=cml_y, mode='lines', name='Capital Market Line', line=dict(color='red', dash='dash')))

    # Tangency
    fig1.add_trace(go.Scatter(x=[tangency_vol], y=[tangency_ret], mode='markers', name='Tangency Portfolio',
                              marker=dict(size=14, color='red', symbol='star')))

    # Rf
    fig1.add_trace(
        go.Scatter(x=[0], y=[rf_input], mode='markers', name='Risk Free Rate', marker=dict(size=10, color='green')))

    fig1.update_layout(template="plotly_white",
                       xaxis_title="Volatility (σ)", yaxis_title="Expected Return (μ)", height=500,
                       xaxis_range=[0, max(std1, std2) * 1.5], yaxis_range=[0, max(mu1, mu2) * 1.5],
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(color="black"),
                       legend=dict(font=dict(color="black")),
                       xaxis=dict(color="black",
                                  gridcolor="rgba(0,0,0,0.1)",
                                  title_font=dict(color="black"),
                                  tickfont=dict(color="black")),
                       yaxis=dict(color="black",
                                  gridcolor="rgba(0,0,0,0.1)",
                                  title_font=dict(color="black"),
                                  tickfont=dict(color="black")))
    st.plotly_chart(fig1, use_container_width=True)

# PLOT 2: Hansen-Jagannathan
with col2:
    st.subheader("SDF Space: HJ Bound")
    fig2 = go.Figure()

    # HJ Curve
    fig2.add_trace(go.Scatter(x=em_range, y=hj_vol, mode='lines', name='HJ Bound', line=dict(color='black', width=3)))

    # Feasible Region shading
    fig2.add_trace(go.Scatter(x=em_range, y=[1.0] * len(em_range), fill='tonexty', fillcolor='rgba(0,0,255,0.08)',
                              line=dict(width=0), showlegend=False, name='Feasible Region'))

    # Specific Point (Duality)
    fig2.add_trace(go.Scatter(x=[em_target], y=[target_sdf_vol], mode='markers+text',
                              name='Min Volatility Kernel',
                              text=['Min Volatility'], textposition="top center",
                              textfont=dict(color='black'),
                              marker=dict(size=14, color='red', symbol='diamond')))

    # Connection Line (Visualizing the wedge/slope)
    fig2.add_trace(go.Scatter(x=[0, em_target], y=[0, target_sdf_vol], mode='lines',
                              name='Sharpe Slope Projection', line=dict(color='gray', dash='dot')))

    fig2.update_layout(template="plotly_white",
                       xaxis_title="Mean of SDF (E[M])", yaxis_title="Volatility of SDF (σ_m)", height=500,
                       yaxis_range=[0, 1.0], xaxis_range=[0.8, 1.2],
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(color="black"),
                       legend=dict(font=dict(color="black")),
                       xaxis=dict(color="black",
                                  gridcolor="rgba(0,0,0,0.1)",
                                  title_font=dict(color="black"),
                                  tickfont=dict(color="black")),
                       yaxis=dict(color="black",
                                  gridcolor="rgba(0,0,0,0.1)",
                                  title_font=dict(color="black"),
                                  tickfont=dict(color="black")))

    st.plotly_chart(fig2, use_container_width=True)

# --- EXPLANATION ---
st.divider()
st.markdown(f"""
### The Mathematics of Duality
1.  **Left Side (CML):** The slope of the red dashed line is the **Sharpe Ratio**:
    $$ SR = \\frac{{E[R_T] - R_f}}{{\\sigma_T}} = \\mathbf{{{max_sharpe:.4f}}} $$

2.  **Right Side (HJ):** The red diamond represents the minimum volatility an SDF must have to price these assets correctly given the risk-free rate.

    The Duality equation is:
    $$ \\frac{{\\sigma_M}}{{E[M]}} \\geq \\text{{Max Sharpe Ratio}} $$

    Checking the values:
    *   $\\sigma_M / E[M]$ = {target_sdf_vol:.4f} / {em_target:.4f} = **{target_sdf_vol / em_target:.4f}**

    *The ratio matches (approximately, subject to gross vs log return nuances).* 

    **Experiment:** Move the $R_f$ slider up.
    *   On the **Left**: The line becomes flatter (Sharpe ratio decreases).
    *   On the **Right**: The cup becomes flatter (Required volatility decreases).
""")
