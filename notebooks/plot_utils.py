import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_emission_ensemble(em_data, scenarios, gas, start_year=2000, n_sample=80, show_band=True, save_name=None):
    """
    Plot baseline trajectory (em_data[gas]) together with scenario trajectories.
    """
    mask = em_data.index >= start_year
    years = em_data.index[mask].to_numpy()
    baseline = em_data.loc[mask, gas].to_numpy()

    # Stack scenario series: shape T x S
    series = [scen["emissions_data"].loc[mask, gas].to_numpy() for scen in scenarios]
    M = np.stack(series, axis=1)

    # Percentiles for fan band
    if show_band:
        q5, q95 = np.percentile(M, [5, 95], axis=1)

    # Sample a subset
    S = M.shape[1]
    rng = np.random.default_rng(0)
    samp = np.arange(S) if S <= n_sample else rng.choice(S, size=n_sample, replace=False)

    # --- Plot ---
    plt.figure(figsize=(6, 3))
    # Scenario lines (light gray)
    plt.plot(years, M[:, samp], color="#999999", linewidth=0.8, alpha=0.15)
    # Percentile band (soft blue)
    if show_band:
        plt.fill_between(years, q5, q95, color="#a6cee3", alpha=0.35, label="5–95% interval")
    # Baseline (strong blue)
    plt.plot(years, baseline, color="#1f78b4", linewidth=2.2, label="Baseline")

    plt.xlabel("Year", fontsize = 14)
    plt.ylabel(f"{gas} emissions (Mt/yr)", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
   # plt.title(f"{gas} — Baseline and scenario ensemble")
    plt.legend(frameon=True, loc="upper right", bbox_to_anchor=(1, 1), fontsize = 12)    
    plt.tight_layout()

    if save_name:
        os.makedirs("plots", exist_ok=True)
        save_path = os.path.join("plots", save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_temperature_ensemble(
    results, 
    baseline_result, 
    start_year=2000, 
    n_sample=80, 
    save_name=None, 
    random_seed=0
):
    """
    Plot baseline temperature trajectory with scenario ensemble (5–95% band + spaghetti).
    Robust to year columns being str or int.
    """
    var_name = "Surface Air Temperature Change"
    np.random.seed(random_seed)

    # Identify all year columns (as ints), sorted
    def _all_years(df):
        ys = []
        for c in df.columns:
            if isinstance(c, (int, np.integer)):
                ys.append(int(c))
            elif isinstance(c, str) and c.isdigit():
                ys.append(int(c))
        return sorted(set(ys))

    # Extract a 1D numpy array from a DataFrame/Series row for the given years
    def _row_year_values(row_like, years_int):
        idx = row_like.index
        cols = []
        for y in years_int:
            if y in idx:
                cols.append(y)
            elif str(y) in idx:
                cols.append(str(y))
            else:
                raise KeyError(f"Year {y} not found in columns (int or str).")
        return row_like[cols].to_numpy(dtype=float)

    # 1) Filter variable
    df_temp = results[results["variable"] == var_name].copy()
    if df_temp.empty:
        raise ValueError(f"`results` has no rows with variable == '{var_name}'.")

    years_all = _all_years(df_temp)
    years_mask = np.array(years_all) >= start_year
    years = np.array(years_all)[years_mask]

    # 2) Baseline series
    base = baseline_result[baseline_result["variable"] == var_name]
    if base.empty:
        raise ValueError(f"`baseline_result` has no rows with variable == '{var_name}'.")
    unit = str(base["unit"].iloc[0]) if "unit" in base.columns else "K"
    base_row = base.iloc[0]
    baseline_series = _row_year_values(base_row, years_all)[years_mask]

    # 3) Stack scenarios into matrix (T x S)
    vals = []
    for _, row in df_temp.iterrows():
        v = _row_year_values(row, years_all)
        vals.append(v)
    M_full = np.stack(vals, axis=1)     # shape: (T_all, S)
    M = M_full[years_mask, :]           # keep >= start_year

    # 4) Percentiles + sample subset
    q5, q95 = np.percentile(M, [5, 95], axis=1)
    S = M.shape[1]
    rng = np.random.default_rng(random_seed)
    samp = np.arange(S) if S <= n_sample else rng.choice(S, size=n_sample, replace=False)

    # 5) Plot
    plt.figure(figsize=(12, 6))
    # Spaghetti ensemble
    plt.plot(years, M[:, samp], color="#999999", lw=0.8, alpha=0.3, zorder=1)
    # Percentile band
    plt.fill_between(years, q5, q95, color="#a6cee3", alpha=0.35, zorder=2)
    # Baseline
    plt.plot(years, baseline_series, color="#247ab3", lw=2.2, zorder=3)
    # Policy start
    if start_year <= 2015 <= years.max():
        plt.axvline(2015, color="#28e008", lw=1, ls="--", zorder=4, alpha=0.7)

    # --- Legend handles ---
    line_ens = plt.Line2D([], [], color="#999999", lw=1.0, alpha=0.6, label="Scenario ensemble")
    band = plt.Line2D([], [], color="#a6cee3", lw=6, alpha=0.35, label="5–95% percentile")
    line_base = plt.Line2D([], [], color="#247ab3", lw=2.2, label="SSP2-4.5 baseline scenario")
    line_policy = plt.Line2D([], [], color="#28e008", lw=1, ls="--", label="Policy start (2015)")

    plt.legend(
        handles=[line_ens, band, line_base, line_policy],
        frameon=True, loc="upper left", fontsize=16
    )

    plt.xlabel("Year", fontsize=20)
    plt.ylabel(f"Global Mean Surface Air Temperature ({unit})", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    if save_name:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", save_name), dpi=300, bbox_inches="tight")
    plt.show()



def plot_temperature_sequences(
    y_true, 
    y_pred, 
    seq_len=36, 
    n_samples=3, 
    seed=0, 
    years=None, 
    savepath=None,
    surrogate_name = None
):
    """
    Reshape flat arrays (N,) -> (num_seq, seq_len), sample n sequences,
    and plot all true vs. predicted curves in one plot.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    num_seq = y_true.size // seq_len
    Y_true = y_true.reshape(num_seq, seq_len)
    Y_pred = y_pred.reshape(num_seq, seq_len)

    rng = np.random.default_rng(seed)
    idx = rng.choice(num_seq, size=min(n_samples, num_seq), replace=False)

    # x-axis
    if years is None:
        x = np.arange(seq_len)
    else:
        years = np.asarray(years)
        if years.size != seq_len:
            raise ValueError("`years` must have length equal to seq_len.")
        x = years

    # Plot
    plt.figure(figsize=(16, 8))
    cmap = plt.get_cmap("ocean")
    color = "lightblue"

    for i, k in enumerate(idx):
        plt.plot(x, Y_true[k], color=color, lw=2.0, alpha=0.5)
        plt.plot(x, Y_pred[k], color=color, lw=1.6, ls="--", alpha=0.8)

    if surrogate_name is None:
        surrogate_name = "RNN-based surrogate"
    # Dummy handles for legend
    true_line = plt.Line2D([], [], color=color, lw=2.0, label="CICERO-SCM")
    pred_line = plt.Line2D([], [], color=color, lw=1.6, ls="--", label=surrogate_name)

    plt.legend(handles=[true_line, pred_line], frameon=True, fontsize=20, loc="upper left")

    plt.xlabel("Year", fontsize=22)
    plt.ylabel("Air Temperature Change (K)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()



def plot_training_time_and_speedup(
    homogenous_scm_training_time,
    homogenous_net_training_time,
    cmap="ocean",
    savefig=None,
    xmax=None,
    figsize=(8, 6),
):
    # Helper to coerce inputs to Series[name] with integer index
    def _to_series(x, name):
        if isinstance(x, pd.DataFrame):
            s = x["value"].copy()
        elif isinstance(x, pd.Series):
            s = x.copy()
        else:
            s = pd.Series(x, copy=True)
        s.name = name
        s.index = pd.Index(s.index).astype(int)
        s = s.sort_index()
        return s

    # Coerce and align
    s_h_scm = _to_series(homogenous_scm_training_time, "SCM (homog.)")
    s_h_net = _to_series(homogenous_net_training_time, "NET (homog.)")
    df = pd.concat([s_h_scm, s_h_net], axis=1).sort_index()

    # Speed-up (guard against divide-by-zero)
    speedup = df["SCM (homog.)"] / df["NET (homog.)"]
    speedup = speedup.replace([np.inf, -np.inf], np.nan)

    # Colors
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    c_scm = cmap_obj(0.2)
    c_net = cmap_obj(0.8)

    # Figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, height_ratios=[1.5, 1]
    )

    # (1) Absolute training times (log)
    ax1.plot(df.index, df["SCM (homog.)"], label="SCM (homog.)", color=c_scm, ls="-", alpha=0.8)
    ax1.plot(df.index, df["NET (homog.)"], label="NET (homog.)", color=c_net, ls="-", alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_ylabel("Training time (s, log)", fontsize=16)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=14, loc="lower right")
    ax1.tick_params(axis="y", labelsize=14)

    # (2) Relative speed-up
    ax2.plot(speedup.index, speedup.values, label="Speed-up (SCM / NET)", color=c_scm, ls="-")
    ax2.set_xlabel("Environment steps", fontsize=16)
    ax2.set_ylabel("Speed-up (×)", fontsize=16)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=14, loc="lower right")
    ax2.tick_params(axis="y", labelsize=14)

    # X axis limits and ticks
    xmin = int(df.index.min())
    auto_xmax = int(df.index.max())
    xmax_use = auto_xmax if xmax is None else int(xmax)
    ax2.set_xlim(xmin, xmax_use)

    # Nice K ticks (about 6–8 ticks)
    n_ticks = 7
    ticks = np.linspace(xmin, xmax_use, num=n_ticks, dtype=int)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{t//1000}K" if t >= 1000 else str(t) for t in ticks], fontsize=12)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2)

def _mean_safe(xs):
    try:
        return float(np.mean(xs)) if len(xs) else np.nan
    except Exception:
        return np.nan

def table_for(greedy_dict: dict, var: str) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        {
            int(iter_k): {
                country: _mean_safe(country_dict.get(var, []))
                for country, country_dict in countries.items()
            }
            for iter_k, countries in greedy_dict.items()
        },
        orient="index",
    )
    return df.sort_index()


from matplotlib.ticker import MaxNLocator, FuncFormatter

def plot_train_reward(
    scm_df: pd.DataFrame | None = None,
    net_df: pd.DataFrame | None = None,
    steps_max: int = 200_000,
    cmap: str = "cividis",
    y_min: float | None = None,
    y_max: float | None = None,
    savefig: str | None = None,
):
    if scm_df is None and net_df is None:
        raise ValueError("Provide at least one of scm_df or net_df.")
    
    def _steps_formatter(x, pos=None):
        x = int(round(x))
        if x >= 1_000_000:
            v = x / 1_000_000
            return f"{v:.1f}M".replace(".0M", "M")
        if x >= 1000:
            v = x / 1000
            return f"{v:.0f}K"
        return str(x)

    # Normalize inputs and intersect columns if both provided
    if scm_df is not None:
        scm = scm_df.copy()
        scm.index = scm.index.astype(int)
    else:
        scm = None

    if net_df is not None:
        net = net_df.copy()
        net.index = net.index.astype(int)
    else:
        net = None

    if scm is not None and net is not None:
        countries = [c for c in scm.columns if c in net.columns]
        if not countries:
            raise ValueError("SCM and NET share no common columns (agents).")
        scm = scm[countries].sort_index()
        net = net[countries].sort_index()
    else:
        # Only one DF: keep all its columns
        one = scm if scm is not None else net
        countries = list(one.columns)
        one.sort_index(inplace=True)

    # Colors
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(x) for x in np.linspace(0.15, 0.85, len(countries))]

    fig, ax = plt.subplots(figsize=(10, 6))

    scm_handles, net_handles = [], []

    # Plot SCM (solid)
    if scm is not None:
        for i, country in enumerate(countries):
            (h1,) = ax.plot(scm.index, scm[country], color=colors[i], linewidth=2.0, alpha=0.7)
            scm_handles.append(h1)

    # Plot NET (dashed)
    if net is not None:
        for i, country in enumerate(countries):
            (h2,) = ax.plot(net.index, net[country], color=colors[i], linewidth=2.0, linestyle="--")
            net_handles.append(h2)

    # Axes cosmetics
    ax.set_xlabel("Environment steps", fontsize=16)
    ax.set_ylabel("Training reward", fontsize=16)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    # X-limits
    ax.set_xlim(0, steps_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(_steps_formatter))
    ax.margins(x=0)

    # Y-limits
    ymins, ymaxs = [], []
    if scm is not None:
        ymins.append(scm.min().min()); ymaxs.append(scm.max().max())
    if net is not None:
        ymins.append(net.min().min());  ymaxs.append(net.max().max())
    if y_min is None: y_min = min(ymins)
    if y_max is None: y_max = max(ymaxs)
    ax.set_ylim(y_min, y_max)

    # Legends
    if scm is not None and net is not None:
        leg1 = ax.legend(
            scm_handles, countries, title="SCM",
            frameon=True, loc="lower left", bbox_to_anchor=(0.59, 0.02), fontsize=14,
        )
        ax.add_artist(leg1)
        ax.legend(
            net_handles, countries, title="NET",
            frameon=True, loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=14,
        )
    else:
        # Single legend
        handles = scm_handles if scm is not None else net_handles
        ax.legend(handles, countries, frameon=True, loc="lower right", fontsize=14)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

    return fig, ax


def _prep(df):
    df = df.copy()
    df.index = df.index.astype(int)
    return df.sort_index()


def plot_lever_consistency(
    lever_data,
    agent_mask,
    action_mask,
    steps_max: int = 200_000,
    cmap: str = "cividis",
    savefig: str = None,
):
    """Plot lever usage for selected agents across SCM and NET runs."""

    if not isinstance(lever_data, dict) or not lever_data:
        raise ValueError("lever_data must map lever names to (scm_df, net_df)")

    canonical_order = ["energy", "methane", "agriculture", "adaptation"]
    lever_order = [name for name in canonical_order if name in lever_data]
    if not lever_order:
        raise ValueError("lever_data must contain at least one recognised lever")

    action_mask = list(action_mask)
    if len(action_mask) < len(lever_order):
        action_mask += [0] * (len(lever_order) - len(action_mask))
    selected_levers = [name for name, flag in zip(lever_order, action_mask) if flag]
    if not selected_levers:
        raise ValueError("action_mask selects no levers to plot")

    first = selected_levers[0]
    scm_df, net_df = lever_data[first]
    if not isinstance(scm_df, pd.DataFrame) or not isinstance(net_df, pd.DataFrame):
        raise TypeError("lever_data values must be (pd.DataFrame, pd.DataFrame)")
    agent_cols = list(scm_df.columns)
    if list(net_df.columns) != agent_cols:
        raise ValueError("SCM and NET DataFrames must share identical agent columns")

    agent_mask = list(agent_mask)
    if len(agent_mask) < len(agent_cols):
        agent_mask += [0] * (len(agent_cols) - len(agent_mask))
    selected_agents = [col for col, flag in zip(agent_cols, agent_mask) if flag]
    if not selected_agents:
        raise ValueError("agent_mask selects no agents to plot")

    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(x) for x in np.linspace(0.15, 0.85, len(selected_agents))]

    def _prep_action(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = out.index.astype(int)
        out = out.sort_index()
        return out[selected_agents]

    n_rows = len(selected_levers)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, lever in zip(axes, selected_levers):
        scm_df, net_df = lever_data[lever]
        scm_use, net_use = _prep_action(scm_df), _prep_action(net_df)

        scm_handles, net_handles = [], []
        for color, agent in zip(colors, selected_agents):
            (h1,) = ax.plot(scm_use.index, scm_use[agent], color=color, linewidth=2.0, alpha=0.85)
            (h2,) = ax.plot(net_use.index, net_use[agent], color=color, linewidth=2.0, linestyle="--", alpha=0.9)
            scm_handles.append(h1)
            net_handles.append(h2)

        ax.set_ylabel(f"{lever.title()} (fraction)", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.5)
    _legend_pair(ax, scm_handles, net_handles, selected_agents)

    xmax = int(steps_max)
    for ax in axes:
        ax.set_xlim(0, xmax)
        ticks = np.arange(0, xmax + 1, 50_000)
        ax.set_xticks(ticks)
        if len(ticks):
            ax.set_xticklabels(["0"] + [f"{t // 1000}K" for t in ticks[1:]], fontsize=11)
        ax.margins(x=0)

    axes[-1].set_xlabel("Environment steps", fontsize=12)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")

    return fig, axes


def _legend_pair(ax, left_handles, right_handles, labels, loc_left=(0.62,0.5), loc_right=(1,0.5)):
    leg1 = ax.legend(left_handles, labels, title="CICERO-SCM", frameon=True,
                     loc="lower left", bbox_to_anchor=loc_left)
    ax.add_artist(leg1)
    ax.legend(right_handles, labels, title="CICERO-NET", frameon=True,
              loc="lower right", bbox_to_anchor=loc_right)

def plot_policy_consistency(
    scm_emission_delta: pd.DataFrame,
    net_emission_delta: pd.DataFrame,
    scm_prevention_rate: pd.DataFrame,
    net_prevention_rate: pd.DataFrame,
    steps_max: int = 200_000,
    cmap: str = "cividis",
    savefig: str = None,
):
    # --- prep & align ---
    s_ed, n_ed = _prep(scm_emission_delta), _prep(net_emission_delta)
    s_pr, n_pr = _prep(scm_prevention_rate), _prep(net_prevention_rate)

    countries = [c for c in s_ed.columns if c in n_ed.columns]
    # keep same ordering across both variables
    s_ed, n_ed = s_ed[countries], n_ed[countries]
    s_pr, n_pr = s_pr[countries], n_pr[countries]

    # color per country (shared between SCM/NET; style distinguishes)
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(x) for x in np.linspace(0.15, 0.85, len(countries))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # ---------- Panel 1: emission_delta ----------
    scm_handles, net_handles = [], []
    for i, c in enumerate(countries):
        (h1,) = ax1.plot(s_ed.index, s_ed[c], color=colors[i], linewidth=2.0)
        (h2,) = ax1.plot(n_ed.index, n_ed[c], color=colors[i], linewidth=2.0, linestyle="--")
        scm_handles.append(h1); net_handles.append(h2)
    ax1.set_ylabel("Emission Δ", fontsize=14)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.tick_params(axis="y", labelsize=14)
    _legend_pair(ax1, scm_handles, net_handles, countries)

    # ---------- Panel 2: prevention_rate ----------
    scm_handles, net_handles = [], []
    for i, c in enumerate(countries):
        (h1,) = ax2.plot(s_pr.index, s_pr[c], color=colors[i], linewidth=2.0, alpha = 0.7)
        (h2,) = ax2.plot(n_pr.index, n_pr[c], color=colors[i], linewidth=2.0, linestyle="--")
        scm_handles.append(h1); net_handles.append(h2)
    ax2.set_xlabel("Environment steps", fontsize=14)
    ax2.set_ylabel("Prevention rate", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # x-axis formatting
    for ax in (ax1, ax2):
        ax.set_xlim(0, steps_max)
        ticks = np.arange(0, steps_max + 1, 50_000)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0"] + [f"{t//1000}K" for t in ticks[1:]], fontsize=14)
        ax.margins(x=0)

    plt.tight_layout()
    if savefig != None:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    plt.show()



# Function to plot schematic with clean style and updated legend
def plot_schematic(end_year, save_path):
    np.random.seed(0)
    years = np.arange(1965, 2051)
    n_years = len(years)
    
    # Simulated gas data
    gas1 = np.cumsum(np.random.randn(n_years)) * 0.02 + 1.0
    gas2 = np.cumsum(np.random.randn(n_years)) * 0.015 + 0.5
    temp = 0.02 * (years - 1965) + np.sin(0.1 * (years - 1965)) * 0.2  # base-like temperature
    
    start_window = end_year - 50
    mask_window = (years >= start_window) & (years <= end_year)
    
    fig, ax1 = plt.subplots(figsize=(6, 3))
    
    # Plot gases
    ax1.plot(years, gas1, label="Gas 1", color="tab:blue")
    ax1.plot(years, gas2, label="Gas 2", color="tab:green")
    
    # Highlight window
    ax1.axvspan(start_window, end_year, color="gray", alpha=0.2, label="Input window (only gases)")
    ax1.set_ylabel("Illustrative emissions")
    
    # Plot temperature on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(years, temp, "--", color="black", label=r"$\Delta T$ (not part of input window)")
    ax2.set_ylabel("Illustrative temperature")
    
    # Mark target year
    target_year = end_year + 1
    target_temp = temp[years == target_year][0]
    ax2.scatter(target_year, target_temp, color="red", zorder=5, label="Target ΔT")
    
    # Legend combining both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8)
    
    ax1.set_xlim(1965, 2050)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



def _extract_lever_table(greedy_policy, lever_name):
    rows = {}
    for step_key, countries in greedy_policy.items():
        row = {}
        for country, metrics in countries.items():
            series = metrics["lever_effort_fraction"].get(lever_name, [])
            row[country] = _mean_safe(series)
        rows[int(step_key)] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()

def _extract_adaptation_table(greedy_policy):
    rows = {}
    for step_key, countries in greedy_policy.items():
        row = {}
        for country, metrics in countries.items():
            series = metrics["adaptation_investment_fraction"]
            row[country] = _mean_safe(series)
        rows[int(step_key)] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()



def plot_lever_consistency_mean(
    lever_data: dict,
    agent_mask,
    action_mask,
    steps_max: int = 200_000,
    cmap: str = "cividis",
    savefig: str | None = None,
):
    """
    Plot mean (across selected agents) lever effort for SCM vs NET on a single axis.
    """
    if not isinstance(lever_data, dict) or not lever_data:
        raise ValueError("lever_data must map lever names to (scm_df, net_df)")

    # Canonical lever order and selection
    canonical_order = ["energy", "methane", "agriculture", "adaptation"]
    lever_order = [name for name in canonical_order if name in lever_data]
    if not lever_order:
        raise ValueError("lever_data must contain at least one recognised lever")

    action_mask = list(action_mask)
    if len(action_mask) < len(lever_order):
        action_mask += [0] * (len(lever_order) - len(action_mask))
    selected_levers = [name for name, flag in zip(lever_order, action_mask) if flag]
    if not selected_levers:
        raise ValueError("action_mask selects no levers to plot")

    # Infer agent columns from the first lever, and validate SCM/NET shapes
    first = selected_levers[0]
    scm_df0, net_df0 = lever_data[first]
    if not isinstance(scm_df0, pd.DataFrame) or not isinstance(net_df0, pd.DataFrame):
        raise TypeError("lever_data values must be (pd.DataFrame, pd.DataFrame)")
    agent_cols = list(scm_df0.columns)
    if list(net_df0.columns) != agent_cols:
        raise ValueError("SCM and NET DataFrames must share identical agent columns")

    # Agent selection
    agent_mask = list(agent_mask)
    if len(agent_mask) < len(agent_cols):
        agent_mask += [0] * (len(agent_cols) - len(agent_mask))
    selected_agents = [col for col, flag in zip(agent_cols, agent_mask) if flag]
    if not selected_agents:
        raise ValueError("agent_mask selects no agents to plot")

    def _prep(df: pd.DataFrame) -> pd.Series:
        s = df.copy()
        s.index = s.index.astype(int)
        s = s.sort_index()
        s = s[selected_agents].mean(axis=1)
        return s

    # Colors per lever
    cmap_obj = plt.get_cmap(cmap)
    colors = {lev: cmap_obj(x) for lev, x in zip(selected_levers, np.linspace(0.15, 0.85, len(selected_levers)))}

    fig, ax = plt.subplots(figsize=(9, 4.8))

    scm_handles, net_handles = [], []
    scm_labels,  net_labels  = [], []

    for lev in selected_levers:
        scm_df, net_df = lever_data[lev]
        scm_mean = _prep(scm_df)
        net_mean = _prep(net_df)

        # Plot SCM (solid)
        h1, = ax.plot(scm_mean.index, scm_mean.values,
                      color=colors[lev], lw=2.2, alpha=0.9)
        # Plot NET (dashed)
        h2, = ax.plot(net_mean.index, net_mean.values,
                      color=colors[lev], lw=2.2, ls="--", alpha=0.95)

        scm_handles.append(h1); scm_labels.append(lev.title())
        net_handles.append(h2); net_labels.append(lev.title())

    # Axes cosmetics
    ax.set_xlabel("Environment steps", fontsize=14)
    ax.set_ylabel("Mean effort (fraction)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.5)

    # X ticks as 0, 50K, 100K, ...
    xmax = int(steps_max)
    ax.set_xlim(0, xmax)
    ticks = np.arange(0, xmax + 1, 50_000)
    ax.set_xticks(ticks)
    if len(ticks):
        ax.set_xticklabels(["0"] + [f"{t//1000}K" for t in ticks[1:]])
    ax.margins(x=0)

    # --- Separate legends inside the plot (like plot_train_reward) ---
    leg1 = ax.legend(
        scm_handles, scm_labels,
        title="SCM",
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.62, 0.02),  # tweak if overlapping
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        net_handles, net_labels,
        title="NET",
        frameon=True,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
    )

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")

    return fig, ax



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

def plot_lever_consistency_mean(
    lever_data: dict,
    agent_mask,
    action_mask,
    steps_max: int = 200_000,
    cmap: str = "cividis",
    savefig: str | None = None,
    figsize: tuple[float, float] = (9, 4.8),
    y_min: float | None = None,
    y_max: float | None = None,
    primary_label: str = "SCM",
):
    """
    Plot mean (across selected agents) lever effort for one or two engines on a single axis.

    lever_data maps lever_name ->
      - DataFrame (single engine), or
      - (first_df, second_df) tuple (two engines; first legend titled `primary_label`, second "NET").

    Each df: index = environment steps (ints), columns = agent names.
    """
    if not isinstance(lever_data, dict) or not lever_data:
        raise ValueError("lever_data must be a non-empty dict")

    # Canonical lever order and selection
    canonical_order = ["energy", "methane", "agriculture", "adaptation"]
    lever_order = [name for name in canonical_order if name in lever_data]
    if not lever_order:
        raise ValueError(f"lever_data must contain at least one recognised lever from {canonical_order}")

    action_mask = list(action_mask)
    if len(action_mask) < len(lever_order):
        action_mask += [0] * (len(lever_order) - len(action_mask))
    selected_levers = [name for name, flag in zip(lever_order, action_mask) if flag]
    if not selected_levers:
        raise ValueError("action_mask selects no levers to plot")

    # Unpack helper: return (first_df or None, second_df or None)
    def _unpack(val):
        if isinstance(val, pd.DataFrame):
            return (val, None)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return val[0], val[1]
        raise TypeError("Each lever value must be a DataFrame, or a (first_df, second_df) tuple/list.")

    # Determine agent columns from the first selected lever
    first = selected_levers[0]
    df1, df2 = _unpack(lever_data[first])
    ref_df = df1 if isinstance(df1, pd.DataFrame) else df2
    if not isinstance(ref_df, pd.DataFrame):
        raise ValueError(f"No DataFrame found for lever '{first}'")
    agent_cols = list(ref_df.columns)

    # Agent selection
    agent_mask = list(agent_mask)
    if len(agent_mask) < len(agent_cols):
        agent_mask += [0] * (len(agent_cols) - len(agent_mask))
    selected_agents = [col for col, flag in zip(agent_cols, agent_mask) if flag]
    if not selected_agents:
        raise ValueError("agent_mask selects no agents to plot")

    # Series prep: mean over selected agents, sorted integer index
    def _prep(df: pd.DataFrame) -> pd.Series:
        s = df.copy()
        s.index = pd.Index(s.index).astype(int)
        s = s.sort_index()
        cols = [c for c in selected_agents if c in s.columns]
        if not cols:
            raise ValueError("Selected agents not present in a provided DataFrame.")
        return s[cols].mean(axis=1)

    # Colors per lever
    cmap_obj = plt.get_cmap(cmap)
    colors = {lev: cmap_obj(x) for lev, x in zip(selected_levers, np.linspace(0.15, 0.85, len(selected_levers)))}

    fig, ax = plt.subplots(figsize=figsize)

    first_handles, second_handles = [], []
    first_labels,  second_labels  = [], []
    any_first, any_second = False, False

    ymins, ymaxs = [], []

    for lev in selected_levers:
        df_first, df_second = _unpack(lever_data[lev])

        if isinstance(df_first, pd.DataFrame):
            ser_first = _prep(df_first)
            (h1,) = ax.plot(ser_first.index, ser_first.values,
                            color=colors[lev], lw=2.2, alpha=0.9)
            first_handles.append(h1); first_labels.append(lev.title())
            ymins.append(ser_first.min()); ymaxs.append(ser_first.max())
            any_first = True

        if isinstance(df_second, pd.DataFrame):
            ser_second = _prep(df_second)
            (h2,) = ax.plot(ser_second.index, ser_second.values,
                            color=colors[lev], lw=2.2, ls="--", alpha=0.95)
            second_handles.append(h2); second_labels.append(lev.title())
            ymins.append(ser_second.min()); ymaxs.append(ser_second.max())
            any_second = True

    # Axes cosmetics
    ax.set_xlabel("Environment steps", fontsize=14)
    ax.set_ylabel("Mean effort (fraction)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.5)

    # X: limits and adaptive ticks (K/M)
    ax.set_xlim(0, int(steps_max))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: f"{x/1_000_000:.1f}M".replace(".0M", "M") if x >= 1_000_000
        else (f"{int(x/1000)}K" if x >= 1000 else str(int(x)))
    ))
    ax.margins(x=0)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    # Y limits
    if y_min is None or y_max is None:
        if ymins and ymaxs:
            auto_ymin, auto_ymax = float(min(ymins)), float(max(ymaxs))
            if y_min is None: y_min = auto_ymin
            if y_max is None: y_max = auto_ymax
    ax.set_ylim(y_min, y_max)

    # Legends
    if any_first and any_second:
        leg1 = ax.legend(
            first_handles, first_labels, title=primary_label,
            frameon=True, loc="lower left", bbox_to_anchor=(0.53, 0.05), fontsize=14, title_fontsize=14,
        )
        ax.add_artist(leg1)
        ax.legend(
            second_handles, second_labels, title="NET",
            frameon=True, loc="lower right", bbox_to_anchor=(1.00, 0.05), fontsize=14, title_fontsize=14,
        )
    else:
        handles = first_handles if any_first else second_handles
        labels  = first_labels  if any_first else second_labels
        title   = primary_label
        ax.legend(handles, labels, title=title, frameon=True, loc="lower right", fontsize=14, title_fontsize=14)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig, ax


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

def plot_lever_consistency(
    lever_data: dict,
    agent_mask,
    action_mask,
    steps_max: int = 200_000,
    cmap: str = "cividis",
    savefig: str | None = None,
    figsize: tuple[float, float] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    max_legend_labels: int | None = 12,
    primary_label: str = "SCM",
):
    """
    Plot PER-AGENT lever usage for one or two engines, one subplot per lever.

    lever_data maps lever_name ->
      - DataFrame (single engine), or
      - (first_df, second_df) tuple (two engines; first legend titled `primary_label`, second "NET").

    Each df: index = environment steps (ints), columns = agent names.
    """
    if not isinstance(lever_data, dict) or not lever_data:
        raise ValueError("lever_data must be a non-empty dict")

    # Canonical lever order and selection
    canonical_order = ["energy", "methane", "agriculture", "adaptation"]
    lever_order = [name for name in canonical_order if name in lever_data]
    if not lever_order:
        raise ValueError(f"lever_data must contain at least one recognised lever from {canonical_order}")

    action_mask = list(action_mask)
    if len(action_mask) < len(lever_order):
        action_mask += [0] * (len(lever_order) - len(action_mask))
    selected_levers = [name for name, flag in zip(lever_order, action_mask) if flag]
    if not selected_levers:
        raise ValueError("action_mask selects no levers to plot")

    # Unpack helper: return (first_df or None, second_df or None)
    def _unpack(val):
        if isinstance(val, pd.DataFrame):
            return (val, None)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return val[0], val[1]
        raise TypeError("Each lever value must be a DataFrame, or a (first_df, second_df) tuple/list.")

    # Determine agent columns from the first selected lever
    df1, df2 = _unpack(lever_data[selected_levers[0]])
    ref_df = df1 if isinstance(df1, pd.DataFrame) else df2
    if not isinstance(ref_df, pd.DataFrame):
        raise ValueError(f"No DataFrame found for lever '{selected_levers[0]}'")
    agent_cols = list(ref_df.columns)

    # Agent selection
    agent_mask = list(agent_mask)
    if len(agent_mask) < len(agent_cols):
        agent_mask += [0] * (len(agent_cols) - len(agent_mask))
    selected_agents = [col for col, flag in zip(agent_cols, agent_mask) if flag]
    if not selected_agents:
        raise ValueError("agent_mask selects no agents to plot")

    # Prep helper: returns df with selected agents, sorted integer index
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = pd.Index(out.index).astype(int)
        out = out.sort_index()
        cols = [c for c in selected_agents if c in out.columns]
        if not cols:
            raise ValueError("Selected agents not present in a provided DataFrame.")
        return out[cols]

    # Colors: one per agent (consistent across subplots)
    cmap_obj = plt.get_cmap(cmap)
    colors = {agent: cmap_obj(x) for agent, x in zip(selected_agents,
                                                     np.linspace(0.15, 0.85, len(selected_agents)))}

    n_rows = len(selected_levers)
    if figsize is None:
        figsize = (12.0, max(5, 3.5 * n_rows))  # scale with number of levers

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]

    # For Y autoscale (per panel if user doesn't fix)
    panel_ymins, panel_ymaxs = [], []

    for ax, lever in zip(axes, selected_levers):
        df_first, df_second = _unpack(lever_data[lever])
        any_first = isinstance(df_first, pd.DataFrame)
        any_second = isinstance(df_second, pd.DataFrame)

        if any_first:
            data_first = _prep(df_first)
        if any_second:
            data_second = _prep(df_second)

        first_pairs, second_pairs = [], []

        for agent in selected_agents:
            if any_first:
                (h1,) = ax.plot(data_first.index, data_first[agent], color=colors[agent], lw=2.0, alpha=0.85)
                first_pairs.append((h1, agent))
            if any_second:
                (h2,) = ax.plot(data_second.index, data_second[agent], color=colors[agent], lw=2.0, ls="--", alpha=0.9)
                second_pairs.append((h2, agent))

        # y-range collection
        yvals = []
        if any_first:
            yvals.extend([data_first[agent].min() for agent in selected_agents])
            yvals.extend([data_first[agent].max() for agent in selected_agents])
        if any_second:
            yvals.extend([data_second[agent].min() for agent in selected_agents])
            yvals.extend([data_second[agent].max() for agent in selected_agents])
        if yvals:
            panel_ymins.append(min(yvals)); panel_ymaxs.append(max(yvals))

        ax.set_ylabel(f"{lever.title()} (fraction)", fontsize=14)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

        # Legend helpers
        def _truncate(pairs, maxn):
            if maxn is None or len(pairs) <= maxn:
                return pairs
            shown = pairs[:maxn]
            # add "+N more" dummy
            dummy = plt.Line2D([], [], color='none')
            shown.append((dummy, f"+{len(pairs)-maxn} more"))
            return shown

        if any_first and any_second:
            pairs1 = _truncate(first_pairs,  max_legend_labels)
            pairs2 = _truncate(second_pairs, max_legend_labels)

            leg1 = ax.legend([h for h, _ in pairs1],
                             [lab for _, lab in pairs1],
                             title=primary_label, frameon=True,
                             loc="lower left", bbox_to_anchor=(0.67, 0.05),
                             fontsize=14, title_fontsize=14)
            ax.add_artist(leg1)
            ax.legend([h for h, _ in pairs2],
                      [lab for _, lab in pairs2],
                      title="NET", frameon=True,
                      loc="lower right", bbox_to_anchor=(1.00, 0.05),
                      fontsize=14, title_fontsize=14)
        else:
            pairs = first_pairs if any_first else second_pairs
            pairs = _truncate(pairs, max_legend_labels)
            ax.legend([h for h, _ in pairs],
                      [lab for _, lab in pairs],
                      title=primary_label, frameon=True,
                      loc="lower right", fontsize=14, title_fontsize=14)

    # X axis formatting
    for ax in axes:
        ax.set_xlim(0, int(steps_max))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos=None: f"{x/1_000_000:.1f}M".replace(".0M", "M") if x >= 1_000_000
            else (f"{int(x/1000)}K" if x >= 1000 else str(int(x)))
        ))
        ax.margins(x=0)

    axes[-1].set_xlabel("Environment steps", fontsize=14)

    # Y limits
    if y_min is not None and y_max is not None:
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    else:
        for ax, ymin, ymax in zip(axes, panel_ymins, panel_ymaxs):
            ax.set_ylim(y_min if y_min is not None else ymin,
                        y_max if y_max is not None else ymax)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig, axes