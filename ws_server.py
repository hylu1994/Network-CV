# Standard Library
import asyncio
import concurrent.futures
import functools
import json
import signal
import sys
from enum import IntEnum

# Other Libraries
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ks_2samp
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import seaborn as sns
import six

sys.modules["sklearn.externals.six"] = six
# to use matplotlib for the backend (https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa)
matplotlib.use("agg")

import websockets

from two_class_density_scatterplot import density_and_class_ratio, polar_colormap


def to_scatterplot_info(xs, ys, labels, adjust_density=False):
    uniq_labels = np.unique(labels)
    if len(uniq_labels) > 2:
        print("labels should have only two classes")

    two_class_labels = np.zeros_like(labels)
    for i, label in enumerate(uniq_labels):
        two_class_labels[labels == label] = i

    xy = np.zeros((xs.shape[0], 2))
    xy[:, 0] = xs
    xy[:, 1] = ys

    density, class1_ratio = density_and_class_ratio(xy, two_class_labels)
    order = np.argsort(density)

    if adjust_density:
        density = density * 0.8 + 0.2  # adujstment for small scatterplot

    hex_colors = np.array(
        [
            to_hex(polar_colormap(radius, ratio1), keep_alpha=True)
            for radius, ratio1 in zip(density, class1_ratio)
        ]
    )
    # hex_colors = np.array(
    #     ["#fb694a" if label == 0 else "#6aaed6" for label in two_class_labels]
    # )

    return hex_colors, order


class Message(IntEnum):
    passData = 0

    @property
    def key(self):
        if self == Message.passData:
            return "passData"

    @property
    def label(self):
        if self == Message.passData:
            return "passData"


async def _send(event_loop, executor, ws, args, func):
    buf = await event_loop.run_in_executor(executor, func, args)
    await ws.send(buf)


async def _serve(event_loop, executor, stop, host="0.0.0.0", port=9000):
    bound_handler = functools.partial(
        _handler, event_loop=event_loop, executor=executor
    )

    async with websockets.serve(bound_handler, host, port):
        await stop


async def _handler(ws, event_loop, executor):
    try:
        while True:
            recv_msg = await ws.recv()
            asyncio.ensure_future(_handle_message(event_loop, executor, ws, recv_msg))
    except websockets.ConnectionClosed as e:
        print(f"Connection closed: {ws.remote_address}")
    except Exception as e:
        print(f"Unexpected exception {e}: {sys.exc_info()[0]}")


def _prepare_data(args):
    with open("./data/case1.json") as f:
        data = json.load(f)

    if args["type"] == "shap":
        attr_names = data["attr_names"]
        attr_values = np.array(data["attr_values"])
        shap_values = data["shap_values"]
        labels = (
            pd.DataFrame(data["nodes"])
            .iloc[data["processed_node_indices"]]["label"]
            .tolist()
        )

        json_by_attr = {}
        domain = {
            "x": [np.finfo(float).max, -np.finfo(float).max],
            "y": [np.finfo(float).max, -np.finfo(float).max],
        }
        for name in shap_values.keys():
            xs = np.array(shap_values[name])
            ys = attr_values[:, attr_names.index(name)]
            cs, order = to_scatterplot_info(xs, ys, labels, adjust_density=True)

            df = pd.DataFrame({"x": xs, "y": ys, "c": cs, "order": order})

            if df["x"].min() < domain["x"][0]:
                domain["x"][0] = df["x"].min()
            if df["x"].max() > domain["x"][1]:
                domain["x"][1] = df["x"].max()
            if df["y"].min() < domain["y"][0]:
                domain["y"][0] = df["y"].min()
            if df["y"].max() > domain["y"][1]:
                domain["y"][1] = df["y"].max()

            json_by_attr[name] = df.to_json(orient="records")

            # if name == "netaddict":
            #     df = pd.DataFrame({"x_pos": xs, "y_pos": ys, "label": labels})
            #     df.to_csv("coords.csv", index=False)

        return json.dumps(
            {
                "action": Message.passData,
                "type": "shap",
                "content": json_by_attr,
                "labels": labels,
                "xy_domain": domain,
            }
        )
    elif args["type"] == "composite":
        target_correlation = args["target_correlation"]
        attr_names = data["attr_names"]
        attr_values = np.array(data["attr_values"])
        lda_coords = np.array(data["representative_values"])
        labels = np.array(
            pd.DataFrame(data["nodes"]).iloc[data["processed_node_indices"]]["label"]
        )

        # Note: if this part is slow, we can precompute swarmplot's y-coords
        ax = sns.swarmplot(data=pd.DataFrame({"x": lda_coords}), x="x", size=1)
        ys = ax.collections[0].get_offsets().data[:, 1]
        plt.clf()  # clean up ax

        attr_indices = []
        best_weights = np.array([])
        corr = 0
        pval = 0

        attr_name_to_index = {}
        for i, attr_name in enumerate(attr_names):
            attr_name_to_index[attr_name] = i

        if len(args["selected_attr_names"]) > 0:
            attr_indices = [
                attr_name_to_index[name] for name in args["selected_attr_names"]
            ]
            inputs = scale(attr_values[:, attr_indices])

            reg = LinearRegression().fit(inputs, lda_coords)
            best_weights = reg.coef_ / np.linalg.norm(reg.coef_)  # unit vector
            corr, pval = pearsonr(inputs @ best_weights, lda_coords)
            if pearsonr(inputs @ best_weights, lda_coords)[0] < 0:
                best_weights = -best_weights

            if target_correlation == "spearman":
                # tested COBYLA and Nelder-Mead (COBYLA showed better results)
                constraints = []
                for i in range(len(best_weights)):
                    constraints.append({"type": "ineq", "fun": lambda x: 1 - x[i]})
                    constraints.append({"type": "ineq", "fun": lambda x: x[i] + 1})

                # use optimized for pearson as initial weights
                res = minimize(
                    lambda w: -spearmanr(inputs @ w, lda_coords)[0],
                    best_weights,
                    method="COBYLA",
                    constraints=constraints,
                    tol=1e-10,
                    options={"maxiter": 1000},
                )

                w_signs = np.sign(best_weights)  # peason's wight signs
                vec_norm = np.linalg.norm(res.x)
                if vec_norm > 0:
                    best_weights = res.x / vec_norm
                else:
                    n_weights = len(res.x)
                    unit_vec = np.array([1 / n_weights**0.5] * n_weights)
                    best_weights = unit_vec * w_signs

                corr, pval = spearmanr(inputs @ best_weights, lda_coords)

            ys = inputs @ best_weights

        cs, order = to_scatterplot_info(lda_coords, ys, labels)
        comp_json = json.dumps(
            {"x": lda_coords.tolist(), "y": ys.tolist(), "c": cs.tolist()}
        )

        # df = pd.DataFrame({"x_pos": lda_coords, "y_pos": ys, "label": labels})
        # df.to_csv("coords.csv", index=False)

        return json.dumps(
            {
                "action": Message.passData,
                "type": "composite",
                "content": comp_json,
                "attr": [attr_names[idx] for idx in attr_indices],
                "weight": best_weights.tolist(),
                "correlation": corr,
                "correlation_measure": target_correlation,
                "pval": pval,
                "order": order.tolist(),
            }
        )
    elif args["type"] == "network":
        # reorder nodes for linking in UI
        nodeidx2shapidx = {}
        shapidx2nodeidx = {}
        for shap_idx, node_idx in enumerate(data["processed_node_indices"]):
            nodeidx2shapidx[node_idx] = shap_idx
            shapidx2nodeidx[shap_idx] = node_idx

        # allocate dummy shap index if node is not in processed_node_indices
        for node_idx in range(len(data["nodes"])):
            if not node_idx in nodeidx2shapidx:
                dummy_shap_idx = len(nodeidx2shapidx)
                nodeidx2shapidx[node_idx] = dummy_shap_idx
                shapidx2nodeidx[dummy_shap_idx] = node_idx

        reordered_nodes = []
        for shap_idx in shapidx2nodeidx.keys():
            reordered_nodes.append(data["nodes"][shapidx2nodeidx[shap_idx]])

        reordered_links = []
        for link in data["links"]:
            reordered_links.append(
                {
                    "source": nodeidx2shapidx[link["source"]],
                    "target": nodeidx2shapidx[link["target"]],
                }
            )

        return json.dumps(
            {
                "action": Message.passData,
                "type": "network",
                "node": reordered_nodes,
                "link": reordered_links,
            }
        )
    elif args["type"] == "hist":
        attr_names = data["attr_names"]

        # input is selected index [ 0, 0, 1, 1, 0, 1, 0, ...]
        selected = np.array(args["selected"], dtype=bool)
        k = int(args["topk"])

        # compute KS score
        X = np.array(data["attr_values"])
        X_scaled = scale(X)
        n_insts = X.shape[0]
        X_s = X_scaled[selected[:n_insts], :]  # selected
        X_o = X_scaled[~selected[:n_insts], :]  # others

        # compute K-S scores
        scores = np.array(
            [ks_2samp(X_s[:, i], X_o[:, i])[0] for i in range(X.shape[1])]
        )

        # compute total scores
        topk_attr_idxs = np.argsort(-scores)[:k]

        # prepare histogram info
        plots = []
        for attr_idx in topk_attr_idxs:
            x = X_scaled[:, attr_idx]
            x_s = X_s[:, attr_idx]
            x_o = X_o[:, attr_idx]

            uniq_vals = np.unique(x)
            if len(uniq_vals) < 50:  # no histogram/aggregation
                bins = np.array(list(uniq_vals) + [uniq_vals[-1] + 1])
            else:  # histogram
                bin_width = 3.49 * x.var() ** 0.5 / len(x) ** (1 / 3)
                bins = np.arange(x.min(), x.max(), bin_width)

            hist_s, _ = np.histogram(x_s, bins=bins)
            hist_o, _ = np.histogram(x_o, bins=bins)
            # relative frequency
            hist_s = hist_s / len(x_s)
            hist_o = hist_o / len(x_o)

            if len(uniq_vals) < 50:  # no histogram/aggregation
                x = bins[:-1].tolist() * 2
            else:  # histogram
                x = (bins[:-1] + bin_width / 2).tolist() * 2

            plots.append(
                {
                    "x": x,
                    "y": hist_s.tolist() + hist_o.tolist(),
                    "z": [0] * len(hist_s) + [1] * len(hist_o),
                    "attr_name": attr_names[attr_idx],
                }
            )

        return json.dumps(
            {"action": Message.passData, "type": "hist", "content": json.dumps(plots)}
        )
    elif args["type"] == "auxiliary":
        # print(data.keys())
        info = data["classification_info"]
        classification_info = json.dumps(
            {
                "targetVariable": info["target_variable"],
                "class0": info["class_0"],
                "class1": info["class_1"],
                "accuracy": info["accuracy"],
            }
        )
        return json.dumps(
            {
                "action": Message.passData,
                "type": "auxiliary",
                "content": classification_info,
            }
        )


async def _handle_message(event_loop, executor, ws, recv_msg):
    m = json.loads(recv_msg)
    m_action = m["action"]

    if m_action == Message.passData:
        await _send(event_loop, executor, ws, m["content"], _prepare_data)


async def start_websocket_server(host="0.0.0.0", port=9000, max_workers=4):
    if not sys.platform.startswith("win"):
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

        event_loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # The stop condition is set when receiving SIGINT.
        stop = asyncio.Future()

        event_loop.add_signal_handler(signal.SIGINT, stop.set_result, True)

        # Run the server until the stop condition is met.
        event_loop.run_until_complete(
            await _serve(event_loop, executor, stop, host, port)
        )
    else:  # windows
        # Windows cannot use uvloop library and signals
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

        event_loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        stop = asyncio.Future()

        try:
            await _serve(event_loop, executor, stop, host, port)
            # event_loop.run_until_complete(
            #     _serve(event_loop, executor, stop, host, port)
            # )
        finally:
            executor.shutdown(wait=True)
            # event_loop.close()


asyncio.run(start_websocket_server(host="0.0.0.0", port=9000, max_workers=4))
