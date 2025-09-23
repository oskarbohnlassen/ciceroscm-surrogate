import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_temperature_sequences(
    y_true, 
    y_pred, 
    seq_len=36, 
    n_samples=3, 
    seed=0, 
    years=None, 
    savepath=None
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
    plt.figure(figsize=(6, 3))
    cmap = plt.get_cmap("tab:cyan")
    for i, k in enumerate(idx):
        col = cmap(i)
        plt.plot(x, Y_true[k], color=col, lw=2.0, alpha = 0.5, label=f"S{i+1} — True")
        plt.plot(x, Y_pred[k], color=col, lw=1.6, ls="--", label=f"S{i+1} — Pred")

    plt.xlabel("Year", fontsize = 14)
    plt.ylabel("Air Temperature Change (°K)", fontsize = 14)
    plt.legend(frameon=True, fontsize=12, ncol=3, loc = "upper left")
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()



def plot_training_time_and_speedup(
    homogenous_scm_training_time,
    homogenous_net_training_time,
    heterogenous_scm_training_time,
    heterogenous_net_training_time,
    cmap="ocean",
    savefig=None,
    xmax=None,
    figsize=(8, 6),
):
    # Helper to coerce inputs to Series[name] with integer index
    def _to_series(x, name):
        if isinstance(x, pd.DataFrame):
            s = x["value"].copy()
        else:
            s = pd.Series(x, copy=True)
        s.name = name
        s.index = s.index.astype(int)
        return s

    # Coerce and align
    s_h_scm  = _to_series(homogenous_scm_training_time,  "Homog-SCM")
    s_h_net  = _to_series(homogenous_net_training_time,  "Homog-NET")
    s_he_scm = _to_series(heterogenous_scm_training_time, "Hetero-SCM")
    s_he_net = _to_series(heterogenous_net_training_time, "Hetero-NET")

    df = pd.concat([s_h_scm, s_h_net, s_he_scm, s_he_net], axis=1).sort_index()

    # Speed-ups
    speedup_homog  = df["Homog-SCM"]  / df["Homog-NET"]
    speedup_hetero = df["Hetero-SCM"] / df["Hetero-NET"]

    # Colors from cmap
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    c_scm = cmap_obj(0.2)
    c_net = cmap_obj(0.8)

    # Build figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[1.5, 1])

    # (1) Absolute training times (log)
    ax1.plot(df.index, df["Homog-SCM"],  label="SCM (homog.)",   color=c_scm, linestyle="-", alpha=0.6)
    ax1.plot(df.index, df["Hetero-SCM"], label="SCM (heterog.)", color=c_scm, linestyle="--", alpha=0.6)
    ax1.plot(df.index, df["Homog-NET"],  label="NET (homog.)",   color=c_net, linestyle="-", alpha=0.6)
    ax1.plot(df.index, df["Hetero-NET"], label="NET (heterog.)", color=c_net, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")
    ax1.set_ylabel("Training time (s, log)", fontsize=14)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=10, ncol=2, loc="lower right")
    ax1.tick_params(axis="y", labelsize=14)


    # (2) Relative speed-up
    ax2.plot(speedup_homog.index,  speedup_homog.values,  label="Speed-up (homog.)",  color=c_scm, linestyle="-")
    ax2.plot(speedup_hetero.index, speedup_hetero.values, label="Speed-up (heterog.)", color=c_scm, linestyle="--")
    ax2.set_xlabel("Environment steps", fontsize=14)
    ax2.set_ylabel("Speed-up (SCM / NET)", fontsize=14)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.tick_params(axis="y", labelsize=14)


    # X axis limits and ticks
    xmin = int(df.index.min())
    auto_xmax = int(df.index.max())
    if xmax is None:
        xmax_use = auto_xmax
    else:
        xmax_use = int(xmax)
        # Trim plotted range (optional; just limit view)
    ax2.set_xlim(xmin, xmax_use)

    # Nice K ticks
    # Choose about 6–8 ticks in range
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


def plot_train_reward(
    scm_df: pd.DataFrame,
    net_df: pd.DataFrame,
    steps_max: int = 200_000,
    cmap: str = "cividis",
):
    # Ensure numeric x
    scm = scm_df.copy(); net = net_df.copy()
    scm.index = scm.index.astype(int); net.index = net.index.astype(int)

    # Common countries
    countries = [c for c in scm.columns if c in net.columns]
    scm = scm[countries].sort_index()
    net = net[countries].sort_index()

    # Palette: one shade per country
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(x) for x in np.linspace(0.15, 0.85, len(countries))]

    fig, ax = plt.subplots(figsize=(10, 6))

    scm_handles, net_handles = [], []
    for i, country in enumerate(countries):
        (h1,) = ax.plot(scm.index, scm[country], color=colors[i], linewidth=2.0, alpha=0.7)
        (h2,) = ax.plot(net.index, net[country], color=colors[i], linewidth=2.0, linestyle="--")
        scm_handles.append(h1); net_handles.append(h2)

    # Axes cosmetics
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Training reward")
    ax.set_title("Training reward per agent: SCM vs NET")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, steps_max)
    ticks = np.arange(0, steps_max + 1, 50_000)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [f"{t//1000}K" for t in ticks[1:]])
    ax.margins(x=0)

    # --- Separate legends inside the plot ---
    leg1 = ax.legend(
        scm_handles,
        countries,
        title="SCM",
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.7, 0.02),
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        net_handles,
        countries,
        title="NET",
        frameon=True,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
    )

    plt.tight_layout()
    plt.show()


def _prep(df):
    df = df.copy()
    df.index = df.index.astype(int)
    return df.sort_index()

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

