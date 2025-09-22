import matplotlib.pyplot as plt
import numpy as np

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