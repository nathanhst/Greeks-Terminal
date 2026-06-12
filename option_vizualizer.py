import os
import numpy as np
import pandas as pd
import scipy.special as scipy_special
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
# 1. MATHS (HULL, VECTORISÉ)
# ══════════════════════════════════════════════════════════════════
def N(x):  return 0.5 * (1 + scipy_special.erf(x / np.sqrt(2)))
def dN(x): return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def calculate_greeks(S, K, T, r, sig, q, option_type):
    S   = np.atleast_1d(np.asarray(S,   dtype=float))
    K   = float(K);  T = float(T);  r = float(r)
    sig = float(sig); q = float(q)
    T   = max(T,   1e-4)
    sig = max(sig, 1e-4)
    sqrtT = np.sqrt(T)
    d1  = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * sqrtT)
    d2  = d1 - sig * sqrtT

    gamma = np.exp(-q * T) * dN(d1) / (S * sig * sqrtT)
    vega  = S * np.exp(-q * T) * dN(d1) * sqrtT

    if option_type == "call":
        delta = np.exp(-q * T) * N(d1)
        theta = (
            -S * sig * np.exp(-q * T) * dN(d1) / (2 * sqrtT)
            - r * K * np.exp(-r * T) * N(d2)
            + q * S * np.exp(-q * T) * N(d1)
        )
        rho   = K * T * np.exp(-r * T) * N(d2)
        price = S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
    else:
        delta = np.exp(-q * T) * (N(d1) - 1)
        theta = (
            -S * sig * np.exp(-q * T) * dN(d1) / (2 * sqrtT)
            + r * K * np.exp(-r * T) * N(-d2)
            - q * S * np.exp(-q * T) * N(-d1)
        )
        rho   = -K * T * np.exp(-r * T) * N(-d2)
        price = K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365,
        "vega":  vega  / 100,
        "rho":   rho   / 100,
        "d1":    d1,
    }

# ══════════════════════════════════════════════════════════════════
# 2. DONNÉES / ÉCHÉANCES
# ══════════════════════════════════════════════════════════════════
_cache = os.environ.get("OPTIONS_CACHE",
         os.path.join(os.path.dirname(__file__), "aapl_options_cache.csv"))
try:
    _df = pd.read_csv(_cache)
    _df["expiration"] = pd.to_datetime(_df["expiration"])
    _df = _df[_df["expiration"] > pd.Timestamp(datetime.now().date())]
    EXPIRATIONS = sorted(_df["expiration"].dt.strftime("%Y-%m-%d").unique())
except FileNotFoundError:
    today = pd.Timestamp(datetime.now().date())
    EXPIRATIONS = [(today + pd.offsets.MonthEnd(i)).strftime("%Y-%m-%d") for i in range(1, 25)]

# ══════════════════════════════════════════════════════════════════
# 3. DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════
PALETTE = {
    "bg":      "#07080a",
    "surface": "#0e1015",
    "panel":   "#13151c",
    "border":  "#1e2130",
    "muted":   "#4a5068",
    "text":    "#d4d8e8",
    "subtext": "#6b7290",
    "accent":  "#4f8ef7",
}
GREEK_COLORS = {
    "delta": "#4f8ef7",
    "gamma": "#34c9a0",
    "theta": "#f0883e",
    "vega":  "#c97adb",
    "rho":   "#f7c948",
}
# rgba fill versions (7% opacity) — proper rgba strings, no hex tricks
GREEK_FILLS = {
    "delta": "rgba(79,142,247,0.07)",
    "gamma": "rgba(52,201,160,0.07)",
    "theta": "rgba(240,136,62,0.07)",
    "vega":  "rgba(201,122,219,0.07)",
    "rho":   "rgba(247,201,72,0.07)",
}
GREEK_SYMBOLS = {
    "delta": "Δ", "gamma": "Γ", "theta": "Θ", "vega": "ν", "rho": "ρ",
}

X_AXIS_OPTIONS = [
    {"label": "Spot price (S)",     "value": "S"},
    {"label": "Volatility (σ)",     "value": "vol"},
    {"label": "Time to expiry (T)", "value": "T"},
    {"label": "Risk-free rate (r)", "value": "r"},
    {"label": "Dividend yield (q)", "value": "q"},
    {"label": "Strike (K)",         "value": "K"},
]

# ══════════════════════════════════════════════════════════════════
# 4. APP + LAYOUT
# ══════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Greeks Terminal"

SL = dict(marks=None, tooltip={"always_visible": False, "placement": "bottom"})

def param_slider(label, symbol, id_, min_, max_, step, value):
    return html.Div([
        html.Div([
            html.Span(label,  style={"fontSize": "11px", "color": PALETTE["subtext"],
                                     "letterSpacing": ".06em"}),
            html.Span(symbol, style={"fontSize": "13px", "color": PALETTE["muted"],
                                     "marginLeft": "6px"}),
            html.Span(id=f"lbl-{id_}", style={"marginLeft": "auto", "fontSize": "13px",
                                               "fontWeight": "600", "color": PALETTE["text"],
                                               "fontVariantNumeric": "tabular-nums"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
        dcc.Slider(id=f"sl-{id_}", min=min_, max=max_, step=step, value=value, **SL),
    ], style={"marginBottom": "18px"})

def section(title, children):
    return html.Div([
        html.Div(title, style={
            "fontSize": "9px", "fontWeight": "700", "letterSpacing": ".12em",
            "color": PALETTE["muted"], "textTransform": "uppercase",
            "marginBottom": "14px", "paddingBottom": "6px",
            "borderBottom": f"1px solid {PALETTE['border']}",
        }),
        *children,
    ], style={"marginBottom": "24px"})

BTN_BASE = {
    "flex": "1", "padding": "7px 0", "fontSize": "13px", "fontWeight": "600",
    "cursor": "pointer", "transition": "all .15s",
}
BTN_ACTIVE   = {**BTN_BASE, "border": f"1px solid {PALETTE['accent']}",
                "background": PALETTE["accent"], "color": "#fff"}
BTN_INACTIVE = {**BTN_BASE, "border": f"1px solid {PALETTE['border']}",
                "background": "transparent", "color": PALETTE["subtext"]}

sidebar = html.Div([
    html.Div([
        html.Span("Greeks",   style={"fontSize": "16px", "fontWeight": "700",
                                      "color": PALETTE["accent"], "letterSpacing": "-.01em"}),
        html.Span(" Terminal",style={"fontSize": "16px", "fontWeight": "300",
                                      "color": PALETTE["text"]}),
    ], style={"marginBottom": "28px", "paddingBottom": "14px",
              "borderBottom": f"1px solid {PALETTE['border']}"}),

    section("Option", [
        html.Div([
            html.Button("Call", id="btn-call", n_clicks=0,
                        style={**BTN_ACTIVE,   "borderRadius": "5px 0 0 5px"}),
            html.Button("Put",  id="btn-put",  n_clicks=0,
                        style={**BTN_INACTIVE, "borderRadius": "0 5px 5px 0"}),
        ], style={"display": "flex", "marginBottom": "16px"}),
        dcc.Store(id="opt-type", data="call"),

        html.Div([
            html.Div("Échéance", style={"fontSize": "11px", "color": PALETTE["subtext"],
                                         "letterSpacing": ".06em", "marginBottom": "4px"}),
            dcc.Dropdown(id="expiry-dp",
                         options=[{"label": e, "value": e} for e in EXPIRATIONS],
                         value=EXPIRATIONS[-1], clearable=False,
                         style={"fontSize": "13px"}),
        ], style={"marginBottom": "18px"}),
    ]),

    section("Parameters", [
        param_slider("STRIKE",     "K",  "K",    100, 300,  5,   180),
        param_slider("SPOT",       "S",  "S",     50, 400, 10,   180),
        param_slider("VOLATILITY", "σ",  "vol",    5,  80,  1,    25),
        param_slider("RATE",       "r",  "r",      0,  15,  1,     5),
        param_slider("DIVIDENDS",  "q",  "q",      0,  10,  1,     1),
    ]),

    section("X axis", [
        html.Div("Vary across", style={"fontSize": "11px", "color": PALETTE["subtext"],
                                        "letterSpacing": ".06em", "marginBottom": "8px"}),
        dcc.RadioItems(
            id="xaxis-selector",
            options=X_AXIS_OPTIONS,
            value="S",
            labelStyle={"display": "flex", "alignItems": "center", "gap": "8px",
                        "fontSize": "13px", "color": PALETTE["text"],
                        "marginBottom": "8px", "cursor": "pointer"},
            inputStyle={"accentColor": PALETTE["accent"]},
        ),
    ]),

    html.Div(id="atm-block"),

], style={
    "width": "240px", "minWidth": "240px", "height": "100vh", "overflowY": "auto",
    "padding": "24px 20px", "background": PALETTE["panel"],
    "borderRight": f"1px solid {PALETTE['border']}",
    "fontFamily": "'Inter', 'DM Sans', system-ui, sans-serif",
})

def empty_graph():
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig

chart_area = html.Div([
    html.Div([
        html.Div(dcc.Graph(id=f"graph-{g}", figure=empty_graph(),
                           config={"displayModeBar": False}),
                 style={"flex": "1", "minWidth": "0"})
        for g in ["delta", "gamma", "theta"]
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
    html.Div([
        html.Div(dcc.Graph(id=f"graph-{g}", figure=empty_graph(),
                           config={"displayModeBar": False}),
                 style={"flex": "1", "minWidth": "0"})
        for g in ["vega", "rho"]
    ], style={"display": "flex", "gap": "12px"}),
], style={"flex": "1", "padding": "20px", "background": PALETTE["bg"], "overflowY": "auto"})

app.layout = html.Div([sidebar, chart_area],
    style={"display": "flex", "height": "100vh", "overflow": "hidden",
           "fontFamily": "'Inter', 'DM Sans', system-ui, sans-serif"})

# ══════════════════════════════════════════════════════════════════
# 5. CALLBACKS
# ══════════════════════════════════════════════════════════════════

@app.callback(
    Output("opt-type", "data"),
    Output("btn-call", "style"),
    Output("btn-put",  "style"),
    Input("btn-call",  "n_clicks"),
    Input("btn-put",   "n_clicks"),
    prevent_initial_call=False,
)
def toggle_type(nc_call, nc_put):
    triggered = ctx.triggered_id
    t = "put" if triggered == "btn-put" else "call"
    call_style = {**(BTN_ACTIVE  if t == "call" else BTN_INACTIVE), "borderRadius": "5px 0 0 5px"}
    put_style  = {**(BTN_ACTIVE  if t == "put"  else BTN_INACTIVE), "borderRadius": "0 5px 5px 0"}
    return t, call_style, put_style


@app.callback(
    Output("lbl-K",   "children"), Output("lbl-S",   "children"),
    Output("lbl-vol", "children"), Output("lbl-r",   "children"),
    Output("lbl-q",   "children"),
    Input("sl-K",   "value"), Input("sl-S",   "value"),
    Input("sl-vol", "value"), Input("sl-r",   "value"),
    Input("sl-q",   "value"),
)
def update_labels(K, S, vol, r, q):
    return f"${K}", f"${S}", f"{vol}%", f"{r}%", f"{q}%"


@app.callback(
    Output("graph-delta", "figure"), Output("graph-gamma", "figure"),
    Output("graph-theta", "figure"), Output("graph-vega",  "figure"),
    Output("graph-rho",   "figure"), Output("atm-block",   "children"),
    Input("opt-type",       "data"),
    Input("expiry-dp",      "value"),
    Input("sl-K",           "value"),
    Input("sl-S",           "value"),
    Input("sl-vol",         "value"),
    Input("sl-r",           "value"),
    Input("sl-q",           "value"),
    Input("xaxis-selector", "value"),
)
def update_charts(opt_type, expiry, K, S, vol_pct, r_pct, q_pct, x_var):
    vol = vol_pct / 100.0
    r   = r_pct   / 100.0
    q   = q_pct   / 100.0
    T   = max((pd.to_datetime(expiry) - pd.Timestamp(datetime.now().date())).days / 365.0, 1/365)

    N_PTS = 200

    # x-axis sweep definitions: (min, max, label, param-override fn)
    X_META = {
        "S":   (S   * 0.5, S   * 1.5, "Spot (S)",    lambda v: (v,   K,   T,   r,   vol, q  )),
        "vol": (0.05, 0.80,            "Volatility σ",lambda v: (S,   K,   T,   r,   v,   q  )),
        "T":   (1/365, 2.0,            "Time T (yr)", lambda v: (S,   K,   v,   r,   vol, q  )),
        "r":   (0.0,  0.20,            "Rate r",      lambda v: (S,   K,   T,   v,   vol, q  )),
        "q":   (0.0,  0.10,            "Div yield q", lambda v: (S,   K,   T,   r,   vol, v  )),
        "K":   (K   * 0.5, K * 1.5,   "Strike (K)",  lambda v: (S,   v,   T,   r,   vol, q  )),
    }
    x_min, x_max, x_label, params_fn = X_META[x_var]
    xs = np.linspace(x_min, x_max, N_PTS)

    # Vectorised: build parameter arrays and call calculate_greeks once
    s_arr   = np.array([params_fn(v)[0] for v in xs])
    k_arr   = np.array([params_fn(v)[1] for v in xs])
    t_arr   = np.array([params_fn(v)[2] for v in xs])
    r_arr   = np.array([params_fn(v)[3] for v in xs])
    vol_arr = np.array([params_fn(v)[4] for v in xs])
    q_arr   = np.array([params_fn(v)[5] for v in xs])

    # calculate_greeks is vectorised over S; for other sweeps we loop smartly
    # Easiest: just pass all as arrays — numpy will broadcast scalars fine
    T_arr   = np.maximum(t_arr,   1e-4)
    sig_arr = np.maximum(vol_arr, 1e-4)
    sqrtT   = np.sqrt(T_arr)
    d1 = (np.log(s_arr / k_arr) + (r_arr - q_arr + 0.5 * sig_arr**2) * T_arr) / (sig_arr * sqrtT)
    d2 = d1 - sig_arr * sqrtT

    gamma_v = np.exp(-q_arr * T_arr) * dN(d1) / (s_arr * sig_arr * sqrtT)
    vega_v  = s_arr * np.exp(-q_arr * T_arr) * dN(d1) * sqrtT

    if opt_type == "call":
        delta_v = np.exp(-q_arr * T_arr) * N(d1)
        theta_v = (
            -s_arr * sig_arr * np.exp(-q_arr * T_arr) * dN(d1) / (2 * sqrtT)
            - r_arr * k_arr * np.exp(-r_arr * T_arr) * N(d2)
            + q_arr * s_arr * np.exp(-q_arr * T_arr) * N(d1)
        )
        rho_v   = k_arr * T_arr * np.exp(-r_arr * T_arr) * N(d2)
        price_v = s_arr * np.exp(-q_arr * T_arr) * N(d1) - k_arr * np.exp(-r_arr * T_arr) * N(d2)
    else:
        delta_v = np.exp(-q_arr * T_arr) * (N(d1) - 1)
        theta_v = (
            -s_arr * sig_arr * np.exp(-q_arr * T_arr) * dN(d1) / (2 * sqrtT)
            + r_arr * k_arr * np.exp(-r_arr * T_arr) * N(-d2)
            - q_arr * s_arr * np.exp(-q_arr * T_arr) * N(-d1)
        )
        rho_v   = -k_arr * T_arr * np.exp(-r_arr * T_arr) * N(-d2)
        price_v = k_arr * np.exp(-r_arr * T_arr) * N(-d2) - s_arr * np.exp(-q_arr * T_arr) * N(-d1)

    series = {
        "delta": delta_v,
        "gamma": gamma_v,
        "theta": theta_v / 365,
        "vega":  vega_v  / 100,
        "rho":   rho_v   / 100,
    }

    # current parameter value on x axis (for reference line)
    ref_vals = {"S": S, "vol": vol, "T": T, "r": r, "q": q, "K": K}
    ref_x = ref_vals[x_var]

    def make_fig(greek):
        color = GREEK_COLORS[greek]
        fill  = GREEK_FILLS[greek]
        sym   = GREEK_SYMBOLS[greek]
        ys    = series[greek]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            fill="tozeroy", fillcolor=fill,
            line=dict(color=color, width=2),
            hovertemplate=f"{x_label}=%{{x:.3g}}<br>{sym}=%{{y:.4f}}<extra></extra>",
        ))

        # dot at current value
        if x_min <= ref_x <= x_max:
            idx = int(np.argmin(np.abs(xs - ref_x)))
            fig.add_trace(go.Scatter(
                x=[xs[idx]], y=[ys[idx]], mode="markers",
                marker=dict(color=color, size=7, line=dict(color=PALETTE["bg"], width=2)),
                hovertemplate=f"{sym} = %{{y:.4f}}<extra></extra>",
                showlegend=False,
            ))
            fig.add_vline(x=ref_x, line_dash="dot", line_color="rgba(255,255,255,0.12)",
                          annotation_text="current",
                          annotation_font=dict(size=9, color="rgba(255,255,255,0.25)"))

        fig.update_layout(
            title=dict(
                text=(f'<span style="font-size:18px;color:{color}">{sym}</span>'
                      f'<span style="font-size:11px;color:{PALETTE["subtext"]};margin-left:6px">'
                      f'{greek.upper()}</span>'),
                x=0.0, xanchor="left", pad=dict(l=4),
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a0c10",
            margin=dict(l=36, r=16, t=44, b=36),
            xaxis=dict(
                title=dict(text=x_label, font=dict(size=10, color=PALETTE["muted"])),
                showgrid=False, zeroline=False,
                tickfont=dict(size=10, color=PALETTE["muted"]),
                tickformat=".3g",
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#12141a",
                zeroline=True, zerolinecolor="#1e2130", zerolinewidth=1,
                tickfont=dict(size=10, color=PALETTE["muted"]),
                tickformat=".3g",
            ),
            hovermode="x unified",
            showlegend=False,
            height=220,
        )
        return fig

    figs = [make_fig(g) for g in ["delta", "gamma", "theta", "vega", "rho"]]

    # ATM metrics at current S
    atm = calculate_greeks(S=S, K=K, T=T, r=r, sig=vol, q=q, option_type=opt_type)
    def sc(v): return float(np.atleast_1d(v)[0])

    def metric_card(label, value):
        return html.Div([
            html.Div(label, style={"fontSize": "9px", "color": PALETTE["subtext"],
                                    "letterSpacing": ".08em", "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "14px", "fontWeight": "600",
                                    "color": PALETTE["text"],
                                    "fontVariantNumeric": "tabular-nums", "marginTop": "2px"}),
        ], style={"background": PALETTE["surface"], "border": f"1px solid {PALETTE['border']}",
                   "borderRadius": "6px", "padding": "8px 10px", "flex": "1"})

    atm_block = html.Div([
        html.Div("Current point", style={
            "fontSize": "9px", "color": PALETTE["muted"], "letterSpacing": ".1em",
            "textTransform": "uppercase", "marginBottom": "10px",
            "paddingTop": "16px", "borderTop": f"1px solid {PALETTE['border']}",
        }),
        html.Div([metric_card("Price",   f"${sc(atm['price']):.2f}"),
                  metric_card("d₁",      f"{sc(atm['d1']):.3f}")],
                 style={"display": "flex", "gap": "8px", "marginBottom": "8px"}),
        html.Div([metric_card("Δ Delta", f"{sc(atm['delta']):.4f}"),
                  metric_card("Γ Gamma", f"{sc(atm['gamma']):.4f}")],
                 style={"display": "flex", "gap": "8px", "marginBottom": "8px"}),
        html.Div([metric_card("Θ /day",  f"{sc(atm['theta']):.4f}"),
                  metric_card("ν /1%σ",  f"{sc(atm['vega']):.4f}")],
                 style={"display": "flex", "gap": "8px"}),
    ])

    return *figs, atm_block


# ══════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True)
