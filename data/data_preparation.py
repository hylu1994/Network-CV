import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

# library required for network representation learning
# graph-tool installation: see https://graph-tool.skewed.de/ (only supports macOS and Linux)
# multilens installation: see https://github.com/takanori-fujiwara/multilens
import graph_tool.all as gt
from multilens import MultiLens

# pytorch is required for classification
# instllation: pip3 install torch
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

# ulca is required for regularized LDA for 1D represnetaiton learning
# ulca installation: see https://github.com/takanori-fujiwara/ulca
from manopt_dr.core import gen_ldr
from manopt_dr.predefined_func_generator import (
    gen_cost_regularized_lda,
    gen_default_proj,
)

# shap is required for SHAP value computation
# shap installation: see https://github.com/slundberg/shap
# Note: current version of shap (as of May 26, 2023) has bugs related to depreciation of np.bool and np.int
# To fix this, download shap package and replace all 'np.bool' with 'bool' and 'np.int' with 'int',
# then, manually install by running "pip3 install ." at Terminal.
import shap

##
## 1. Loading and preprocessing of graph data
##

# download data
g = gt.collection.ns["faculty_hiring_us/academia"]
inst_stats = pd.read_csv(
    "https://raw.githubusercontent.com/LarremoreLab/us-faculty-hiring-networks/main/data/institution-stats.csv"
)

# process data
inst_stats["InstitutionId"] = inst_stats["InstitutionId"].astype("int")

g.ep["weight"] = g.ep["total"]
g.vp["production_rank"] = g.new_vertex_property("int")
g.vp["prestige_rank"] = g.new_vertex_property("double")
g.vp["ordinal_prestige_rank"] = g.new_vertex_property("int")
g.vp["us"] = g.new_vertex_property("int")

taxonomy_value = "Academia"
for v in g.vertices():
    inst_id = g.vp["id"][v]
    stat = inst_stats[
        (inst_stats["InstitutionId"] == inst_id)
        * (inst_stats["TaxonomyValue"] == taxonomy_value)
    ]
    if len(stat) == 0:
        g.vp["us"][v] = 0
        g.vp["production_rank"][v] = -1
        g.vp["prestige_rank"][v] = -1
        g.vp["ordinal_prestige_rank"][v] = -1
    else:
        g.vp["us"][v] = 1
        g.vp["production_rank"][v] = stat["ProductionRank"].iloc[0]
        g.vp["prestige_rank"][v] = stat["PrestigeRank"].iloc[0]
        g.vp["ordinal_prestige_rank"][v] = stat["OrdinalPrestigeRank"].iloc[0]

# node positions are already precomputed
# if needed, graph-tool's graph drawing method can be used (e.g., sfdp)
node_positions = np.array(list(g.vp["_pos"]))

node_labels = np.array([-1] * g.num_vertices())
# prestige_rank: 0 is better, 1 is worse
node_labels[(g.vp["prestige_rank"].a < 0.25) * (g.vp["prestige_rank"].a >= 0.0)] = 0
node_labels[g.vp["prestige_rank"].a > 0.5] = 1

nodes = pd.DataFrame(
    {"x": node_positions[:, 0], "y": node_positions[:, 1], "label": node_labels}
)

##
## 2. Network representation learing using DeepGL/MultiLens
##

multilens = MultiLens(
    ego_dist=2,
    base_feat_defs=[
        "us",  # 'total_degree', 'eigenvector', 'betweenness',
        "w_total_degree",
        "w_eigenvector",
        "w_betweenness",
    ],
    use_nbr_for_hist=False,
    rel_feat_ops=["minimum", "maximum", "mean", "variance"],
    nbr_types=["all"],
    log_binning_alpha=0.1,
)

_, X = multilens.fit_transform(g, return_hist=True)

X_ = scale(X)[node_labels >= 0, :]
y_ = node_labels[node_labels >= 0]
feat_names = multilens.get_feat_defs()


def simplify_feat_names(names):
    replacer_list = [
        ["all-", ""],
        ["variance", "var"],
        ["maximum", "max"],
        ["minimum", "min"],
        ["total_degree", "degree"],
        ["^", "-"],
    ]
    simplified_names = []
    for name in names:
        for replacer in replacer_list:
            name = name.replace(replacer[0], replacer[1])
        simplified_names.append(name)
    return simplified_names


feat_names = simplify_feat_names(feat_names)

##
## 3. Classification using neural networks
##


class Classifier(nn.Module):

    def __init__(self, dim, hidden_size=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.output(x)


def arr_to_tensor(X):
    return torch.from_numpy(X.astype(np.float32))


dataloader = DataLoader(
    TensorDataset(
        torch.tensor(X_, dtype=torch.float32),
        torch.tensor(y_, dtype=torch.float32).reshape(-1, 1),
    ),
    batch_size=30,
    shuffle=True,
)
epochs = 1000
model = Classifier(X_.shape[1])
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    running_loss = 0
    acc = 0
    for i, (X_batch, y_batch) in enumerate(dataloader):
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc_batch = (y_pred.round() == y_batch).float().mean()

        if i == 0:
            acc = acc_batch
        else:
            acc = (acc * i + acc_batch) / (i + 1)
    print("accuracy: ", float(acc))

predicted = model(torch.tensor(X_, dtype=torch.float32))
acc = (predicted.reshape(-1).detach().numpy().round() == y_).mean()
print("accuracy:", acc)

##
## 4. Learning 1D representation from the nD representation using (regularized LDA)
##
representation = model.encoder(arr_to_tensor(X_)).detach().numpy()

LDA = gen_ldr(gen_cost_regularized_lda, gen_default_proj)
lda = LDA(n_components=1)
Z_ = lda.fit_transform(representation, y=y_, gamma=0)

# import matplotlib.pyplot as plt
# plt.scatter(Z_[:, 0], np.random.rand(Z_.shape[0]), c=y_)
# plt.show()

##
## 5. Computing SHAP values (contributiosn to the 1D representation)
##

# preparing the function that combine the transformation of NN and LDA's
combined_transform = lambda X: lda.transform(
    model.encoder(arr_to_tensor(X)).detach().numpy()
)

explainer = shap.Explainer(combined_transform, X_, feature_names=feat_names)
shap_values = explainer(X_)
# shap.plots.beeswarm(shap_values, max_display=16)

##
## 6. Saving all information as json
##
data = {}
data["nodes"] = json.loads(nodes.to_json(orient="records"))
data["links"] = []
for e in g.edges():
    data["links"].append({"source": int(e.source()), "target": int(e.target())})

# feature info
data["processed_node_indices"] = np.where(node_labels >= 0)[0].tolist()
data["attr_values"] = X_.tolist()
data["attr_names"] = feat_names
data["representative_values"] = Z_[:, 0].tolist()
topk = 10
data["shap_values"] = {}
for feat_idx in np.argsort(-np.abs(shap_values.values).mean(0))[:topk]:
    data["shap_values"][feat_names[feat_idx]] = shap_values.values[:, feat_idx].tolist()

# classification info
data["classification_info"] = {}
data["classification_info"]["target_variable"] = "Prestige Rank"
data["classification_info"]["class_0"] = "Score < 0.25 (better)"
data["classification_info"]["class_1"] = "Score > 0.50 (worse)"
data["classification_info"]["accuracy"] = acc

with open("./data.json", "w") as f:
    json.dump(data, f)
