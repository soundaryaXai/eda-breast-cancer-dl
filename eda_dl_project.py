# %% [markdown]
# # EDA, Visualization and Deep Learning on the Breast Cancer Wisconsin (Diagnostic) Dataset
# Course: U21ADP05 – Exploratory Data Analysis and Visualization | Assignment II (Case Study)
#
# Pipeline: 1) Load  2) EDA  3) Preprocess  4) Visualize  5) Model (MLP)  6) Evaluate

# %% Imports and global settings
import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, log_loss, classification_report)
from sklearn.inspection import permutation_importance
import plotly.express as px

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")
FIG = "figures/"
R = {}                                   # results collected for the report

def save(name):
    plt.tight_layout(); plt.savefig(FIG + name, dpi=160, bbox_inches="tight"); plt.close()

# %% 1. DATA LOADING  (Stage 1-2 of visualization: ACQUIRE + PARSE)
raw = load_breast_cancer(as_frame=True)          # UCI Breast Cancer Wisconsin (Diagnostic)
df = raw.frame.copy()
# In sklearn, target 0 = malignant, 1 = benign. Re-label for readability.
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
features = list(raw.feature_names)
print("Shape:", df.shape)
R["shape"] = list(df.shape); R["n_features"] = len(features)

# %% 2. EDA
print(df[features].describe().T.head(10))
R["missing_total"] = int(df.isna().sum().sum())
R["duplicates"] = int(df.duplicated().sum())
R["class_counts"] = df["diagnosis"].value_counts().to_dict()
R["dtypes"] = df[features].dtypes.astype(str).value_counts().to_dict()
desc = df[features].describe().T
desc["skew"] = df[features].skew()
desc.round(3).to_csv("outputs/summary_statistics.csv")
R["desc_sample"] = desc.loc[["mean radius","mean texture","mean area","mean concavity","worst perimeter"],
                            ["mean","std","min","50%","max","skew"]].round(3).reset_index().values.tolist()
print("Missing:", R["missing_total"], "Duplicates:", R["duplicates"])

# Outlier detection with the IQR rule
q1, q3 = df[features].quantile(.25), df[features].quantile(.75)
iqr = q3 - q1
out_mask = (df[features] < q1 - 1.5*iqr) | (df[features] > q3 + 1.5*iqr)
out_counts = out_mask.sum().sort_values(ascending=False)
R["outlier_top"] = out_counts.head(5).to_dict()
R["outlier_rows_any"] = int(out_mask.any(axis=1).sum())
R["highly_skewed"] = int((df[features].skew().abs() > 1).sum())

# %% 3. PREPROCESSING  (Stage 3 of visualization: FILTER)
X = df[features].values; y = df["target"].values            # 1 = benign, 0 = malignant
# Stratified 70 / 15 / 15 split
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
R["split"] = [len(X_tr), len(X_va), len(X_te)]
# Outlier handling: winsorize at the 1st/99th percentile (limits learnt on TRAIN only -> no leakage)
lo, hi = np.percentile(X_tr, 1, axis=0), np.percentile(X_tr, 99, axis=0)
clip = lambda A: np.clip(A, lo, hi)
# Standardization (z-score), fitted on TRAIN only
scaler = StandardScaler().fit(clip(X_tr))
Xs_tr, Xs_va, Xs_te = (scaler.transform(clip(A)) for A in (X_tr, X_va, X_te))
print("Split sizes:", R["split"])

# %% 4. VISUALIZATION
# V1 – Class distribution (bar + pie)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
order = ["Benign", "Malignant"]; pal = {"Benign": "#2a9d8f", "Malignant": "#e63946"}
sns.countplot(data=df, x="diagnosis", order=order, palette=pal, ax=ax[0])
for p in ax[0].patches: ax[0].annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
ax[0].set(title="Class distribution of diagnosis", xlabel="Diagnosis", ylabel="Number of patients")
ax[1].pie(df["diagnosis"].value_counts()[order], labels=order, autopct="%1.1f%%", colors=[pal[o] for o in order], startangle=90)
ax[1].set_title("Class proportion"); save("v1_class_distribution.png")

# V2 – Histograms + KDE by class (univariate)
sel = ["mean radius", "mean texture", "mean concavity", "mean smoothness"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for a, f in zip(axes.ravel(), sel):
    sns.histplot(data=df, x=f, hue="diagnosis", kde=True, stat="density", common_norm=False, palette=pal, ax=a, alpha=.45)
    a.set_title(f"Distribution of {f}")
fig.suptitle("Univariate distributions by diagnosis", y=1.01); save("v2_histograms_kde.png")

# V3 – Boxplots (outliers + class separation)
sel2 = ["mean radius", "mean area", "mean concavity", "worst perimeter", "worst concave points", "mean texture"]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for a, f in zip(axes.ravel(), sel2):
    sns.boxplot(data=df, x="diagnosis", y=f, order=order, palette=pal, ax=a); a.set_title(f)
    a.set_xlabel("")
fig.suptitle("Boxplots of key features by diagnosis", y=1.01); save("v3_boxplots.png")

# V4 – Correlation heatmap (mean features)
mean_cols = [c for c in features if c.startswith("mean")]
corr = df[mean_cols + ["target"]].rename(columns={"target": "benign(1)"}).corr()
plt.figure(figsize=(9, 7.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, annot_kws={"size": 7}, cbar_kws={"label": "Pearson r"})
plt.title("Correlation heatmap – mean features and diagnosis"); save("v4_correlation_heatmap.png")
full_corr = df[features].corr()
upper = full_corr.where(np.triu(np.ones(full_corr.shape), 1).astype(bool)).stack()
R["pairs_gt_95"] = int((upper.abs() > 0.95).sum())
R["top_pairs"] = [(a, b, round(v, 3)) for (a, b), v in upper.abs().sort_values(ascending=False).head(4).items()]
tcorr = df[features].corrwith(df["target"]).sort_values()
R["top_neg_target"] = tcorr.head(4).round(3).to_dict()
R["top_pos_target"] = tcorr.tail(3).round(3).to_dict()

# V5 – Scatter plot (bivariate)
plt.figure(figsize=(7, 5.5))
sns.scatterplot(data=df, x="mean radius", y="mean concave points", hue="diagnosis", palette=pal, alpha=.75, edgecolor="w")
plt.title("Mean radius vs mean concave points"); save("v5_scatter.png")
R["r_radius_area"] = round(df["mean radius"].corr(df["mean area"]), 3)
R["r_radius_cp"] = round(df["mean radius"].corr(df["mean concave points"]), 3)

# V6 – Pairplot of 4 features
pp = sns.pairplot(df[["mean radius", "mean texture", "mean concavity", "mean smoothness", "diagnosis"]],
                  hue="diagnosis", palette=pal, corner=True, plot_kws={"alpha": .6, "s": 18}, height=2.2)
pp.fig.suptitle("Pairplot of selected mean features", y=1.02); pp.savefig(FIG + "v6_pairplot.png", dpi=150); plt.close("all")

# V7 – Continuous distribution fitting (Normal vs Log-normal) + Q-Q plots
x = df["mean area"].values
mu, sd = stats.norm.fit(x); s, loc, sc = stats.lognorm.fit(x, floc=0)
ks_n = stats.kstest(x, "norm", args=(mu, sd)); ks_l = stats.kstest(x, "lognorm", args=(s, loc, sc))
R["fit"] = {"mu": round(mu, 1), "sd": round(sd, 1), "skew": round(float(stats.skew(x)), 3),
            "ks_norm": [round(ks_n.statistic, 3), float(f"{ks_n.pvalue:.2e}")],
            "ks_logn": [round(ks_l.statistic, 3), round(ks_l.pvalue, 3)]}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
g = np.linspace(x.min(), x.max(), 300)
ax[0].hist(x, bins=35, density=True, alpha=.45, color="#457b9d", label="Data")
ax[0].plot(g, stats.norm.pdf(g, mu, sd), "r-", label="Normal fit")
ax[0].plot(g, stats.lognorm.pdf(g, s, loc, sc), "g--", label="Log-normal fit")
ax[0].set(title="Mean area: histogram with fitted PDFs", xlabel="Mean area", ylabel="Density"); ax[0].legend()
stats.probplot(x, dist="norm", plot=ax[1]); ax[1].set_title("Q-Q plot vs Normal")
stats.probplot(np.log(x), dist="norm", plot=ax[2]); ax[2].set_title("Q-Q plot of log(mean area) vs Normal")
save("v7_distribution_fit.png")

# V8 – Discrete distribution: Binomial model of malignant cases in random samples of n = 20
p_mal = (df["target"] == 0).mean(); n = 20
rng = np.random.default_rng(SEED)
draws = [(rng.choice(df["target"].values, n, replace=True) == 0).sum() for _ in range(5000)]
k = np.arange(0, 17); emp = np.bincount(draws, minlength=21)[:17] / 5000
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.bar(k - .2, emp, .4, label="Empirical (5000 samples)", color="#e9c46a")
ax.bar(k + .2, stats.binom.pmf(k, n, p_mal), .4, label=f"Binomial(n={n}, p={p_mal:.3f})", color="#264653")
ax.set(title="Number of malignant tumours in a random sample of 20 patients", xlabel="Malignant cases (k)", ylabel="Probability"); ax.legend()
save("v8_binomial.png")
R["binom"] = {"p": round(p_mal, 4), "mean": round(n*p_mal, 2), "sd": round(float(np.sqrt(n*p_mal*(1-p_mal))), 2),
              "emp_mean": round(float(np.mean(draws)), 2)}

# V9 – Interactive (Stage 7: INTERACT) – Plotly HTML scatter
fig = px.scatter(df, x="mean radius", y="mean concave points", color="diagnosis", color_discrete_map=pal,
                 size="mean area", hover_data=["mean texture", "mean smoothness"], title="Interactive: radius vs concave points (bubble = area)")
fig.write_html("outputs/interactive_scatter.html", include_plotlyjs="cdn")

# %% 5. MODEL – MLP (feed-forward deep network) trained epoch by epoch
def train_mlp(hidden, alpha, lr, epochs=120, seed=SEED):
    m = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=alpha,
                      learning_rate_init=lr, batch_size=32, random_state=seed, warm_start=True, max_iter=1)
    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    for ep in range(epochs):
        m.partial_fit(Xs_tr, y_tr, classes=[0, 1])
        hist["tr_loss"].append(log_loss(y_tr, m.predict_proba(Xs_tr)))
        hist["va_loss"].append(log_loss(y_va, m.predict_proba(Xs_va)))
        hist["tr_acc"].append(m.score(Xs_tr, y_tr)); hist["va_acc"].append(m.score(Xs_va, y_va))
    return m, hist

# Basic hyper-parameter tuning on the VALIDATION set
rows = []
for hidden in [(16,), (32, 16), (64, 32), (64, 32, 16)]:
    for alpha in [1e-4, 1e-2]:
        for lr in [1e-3, 1e-2]:
            m, h = train_mlp(hidden, alpha, lr, epochs=60)
            rows.append({"hidden": str(hidden), "alpha": alpha, "lr": lr,
                         "val_acc": round(h["va_acc"][-1], 4), "val_loss": round(h["va_loss"][-1], 4)})
tune = pd.DataFrame(rows).sort_values(["val_loss"]).reset_index(drop=True)
tune.to_csv("outputs/hyperparameter_tuning.csv", index=False)
best = tune.iloc[0]; print(tune.head())
R["tune_top"] = tune.head(5).values.tolist(); R["n_configs"] = len(tune)
best_hidden = eval(best["hidden"]); best_alpha, best_lr = float(best["alpha"]), float(best["lr"])
R["best"] = {"hidden": best["hidden"], "alpha": best_alpha, "lr": best_lr}

# Final model with best hyper-parameters
model, hist = train_mlp(best_hidden, best_alpha, best_lr, epochs=100)
n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
R["n_params"] = int(n_params)

# %% 6. EVALUATION
prob = model.predict_proba(Xs_te)[:, 1]; pred = (prob >= .5).astype(int)
R["test"] = {"acc": round(accuracy_score(y_te, pred), 4), "prec": round(precision_score(y_te, pred), 4),
             "rec": round(recall_score(y_te, pred), 4), "f1": round(f1_score(y_te, pred), 4),
             "auc": round(roc_auc_score(y_te, prob), 4), "loss": round(log_loss(y_te, prob), 4)}
R["cm"] = confusion_matrix(y_te, pred).tolist()
R["final_train_acc"] = round(hist["tr_acc"][-1], 4); R["final_val_acc"] = round(hist["va_acc"][-1], 4)
R["final_train_loss"] = round(hist["tr_loss"][-1], 4); R["final_val_loss"] = round(hist["va_loss"][-1], 4)
R["min_val_loss_epoch"] = int(np.argmin(hist["va_loss"]) + 1)
print(classification_report(y_te, pred, target_names=["Malignant", "Benign"]))

ep = np.arange(1, len(hist["tr_loss"]) + 1)
plt.figure(figsize=(7, 4.3)); plt.plot(ep, hist["tr_loss"], label="Training loss"); plt.plot(ep, hist["va_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Binary cross-entropy loss"); plt.title("Loss vs Epoch"); plt.legend(); save("r1_loss_curve.png")
plt.figure(figsize=(7, 4.3)); plt.plot(ep, hist["tr_acc"], label="Training accuracy"); plt.plot(ep, hist["va_acc"], label="Validation accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch"); plt.legend(); save("r2_accuracy_curve.png")
fig, ax = plt.subplots(figsize=(4.8, 4.3))
ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["Malignant", "Benign"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix (test set)"); save("r3_confusion_matrix.png")
fpr, tpr, _ = roc_curve(y_te, prob)
plt.figure(figsize=(5.5, 4.8)); plt.plot(fpr, tpr, lw=2, label=f"MLP (AUC = {R['test']['auc']:.3f})"); plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve (test set)"); plt.legend(loc="lower right"); save("r4_roc_curve.png")

# Feature importance (permutation importance on the validation set)
pi = permutation_importance(model, Xs_va, y_va, n_repeats=30, random_state=SEED, scoring="accuracy")
imp = pd.Series(pi.importances_mean, index=features).sort_values(ascending=False)
plt.figure(figsize=(7, 5)); imp.head(10)[::-1].plot.barh(color="#457b9d")
plt.xlabel("Mean decrease in validation accuracy"); plt.title("Top-10 features (permutation importance)"); save("r5_feature_importance.png")
R["top_feats"] = imp.head(5).round(4).to_dict()

json.dump(R, open("outputs/results.json", "w"), indent=1, default=str)
print(json.dumps(R["test"]))
