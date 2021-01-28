import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def backdiff_operator(step, size):
    a = np.ones(size) / step
    d = np.diag(a, 0) - np.diag(a[1:], -1)
    return d


def integral_operator(step, size):
    s = np.tri(size, k=-1) + 0.5 * np.eye(size)
    s[:, 0] -= 0.5
    s[0, 0] = 0.0
    s *= step
    return s


def reg_deriv(y, step, alpha=0.1):
    s = integral_operator(step, y.size)
    d = backdiff_operator(step, y.size)
    u = np.linalg.solve(s.T.dot(s) + alpha * d.T.dot(d), s.T.dot(y - y[0]))
    f = np.cumsum(u) * step + y[0]
    return u, f


def feature_trends(df, cname_x, cnames_y, alpha):
    # Subselect on non-NA values to ensure contiguous data for reg_deriv.
    df_nona = df.dropna(subset=[cname_x] + cnames_y)

    # Assume df is sorted by cname_x in increasing order
    assert df_nona[
        cname_x
    ].is_monotonic_increasing, "cname_x must by monotonically increasing"

    # Don't allow any duplicates in the independent variable
    assert (
        not df_nona[cname_x].duplicated().any()
    ), "No duplicate values in cname_x allowed"

    if df_nona.size:
        df_out = pd.DataFrame(index=df_nona.index)
        df_out[cname_x] = df_nona[cname_x]

        for c_y in cnames_y:
            xy = df_nona[[cname_x, c_y]].values
            x = xy[:, 0]
            y = xy[:, 1]
            # Poor man's estimate of the grid step size dx
            dx = np.median(x[1:] - x[:-1])
            u, f = reg_deriv(y, step=dx, alpha=alpha)
            df_out[c_y + "_trend"] = u
            df_out[c_y + "_smooth"] = f
        return df_out
    else:
        return None


# plot styling
# plt.style.use("ggplot")
plt.rcParams["font.size"] = 14
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["scatter.edgecolors"] = "black"
plt.rcParams["xtick.color"] = "k"
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.color"] = "k"
plt.rcParams["ytick.labelsize"] = "medium"
plt.rcParams["figure.figsize"] = 12, 8
plt.rcParams["image.cmap"] = "RdBu_r"  # "coolwarm"
plt.rcParams["savefig.edgecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.format"] = "svg"

SECS_IN_DAY = 24 * 60 * 60

df = pd.read_csv("tesla.csv")

df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["time"] = df["date"].astype("int64") // 1e9
df["time"] = df["time"] - df["time"][0]
df["time_days"] = df["time"] / SECS_IN_DAY

# time difference between readings
df["delta_time"] = df["time"].diff()
df["delta_time_days"] = df["delta_time"] / SECS_IN_DAY

# compute gradients and smoothed values of readings
df_trends = feature_trends(df, "time_days", ["Close"], alpha=0.1)
df = pd.merge(df, df_trends)
window = 50
df["Close_rolling"] = df["Close"].rolling(window=window).mean()

# only take every Nth item otherwise it is too slow
offset = 400
n = 5
df = df.iloc[max(offset, window) :: n]

x = df.index
y = df["Close"]
y_smooth = df["Close_rolling"]
y_trend = df["Close_trend"].fillna(0).replace(np.inf, 0)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

(line,) = ax1.plot(x, y, color="k", linewidth=1, alpha=0.8)
(line2,) = ax1.plot(x, y_smooth, color="g", linewidth=4, alpha=0.4)
# ax1.set_xlabel("Time", fontsize=22)
ax1.set_ylabel("Value")

(line3,) = ax2.plot(x, y_trend, color="r")
ax2.set_xlabel("Time", fontsize=22)
ax2.set_ylabel("Trend")

# Turn off tick labels
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
# for minor ticks
ax1.set_xticks([], minor=False)
ax2.set_xticks([], minor=True)
# plt.gca().axes.get_yaxis().set_visible(False)

ax1.set_ylim([0, max(y)])
ax2.set_ylim([min(y_trend), max(y_trend)])

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
# ax1.spines["left"].set_position(("data", 200))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
# ax2.spines["left"].set_position(("data", 200))
ax2.spines["right"].set_color("none")
ax2.spines["bottom"].set_position(("data", -40))
ax2.spines["top"].set_color("none")
ax2.spines["left"].set_smart_bounds(True)
ax2.spines["bottom"].set_smart_bounds(True)
ax2.spines["bottom"].set_linewidth(2)
ax2.spines["bottom"].set_alpha(0.6)
ax1.spines["left"].set_alpha(0.6)
ax2.spines["left"].set_alpha(0.6)
ax1.spines["left"].set_linewidth(2)
ax2.spines["left"].set_linewidth(2)


def update(num, x, y, y_smooth, y_trend, line, line2, line3):
    line.set_data(x[:num], y[:num])
    line2.set_data(x[:num], y_smooth[:num])
    line3.set_data(x[:num], y_trend[:num])

    return (line, line2, line3)


ani = animation.FuncAnimation(
    fig,
    update,
    len(x),
    fargs=[x, y, y_smooth, y_trend, line, line2, line3],
    interval=1,
    blit=False,
)
#ani.save("test.gif", fps=20)
plt.show()
