"""后处理：融合 → 边缘平衡 → 排版约束 → 稠密簇匹配。

只依赖 numpy 与 scipy。给一份「成员 → logit 矩阵」的字典即可运行：

    from src.postprocess import load_recipe, fuse, postprocess
    cfg = load_recipe()
    logp, pred = postprocess(fuse(member_logits), texts, groups, cfg)

member_logits: {成员名: (权重, ndarray[n_samples, 18])}
texts:         list[str]，用于排版规则
groups:        ndarray[n_samples]，同一提示词的样本同组；None 表示不做簇匹配
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs" / "recipe.json"


def load_recipe(path: str | Path | None = None) -> dict:
    p = Path(path) if path else RECIPE
    return json.loads(p.read_text(encoding="utf-8"))


def log_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.exp(log_softmax(x, axis=axis))


def fuse(member_logits: dict[str, tuple[float, np.ndarray]]) -> np.ndarray:
    """按权重求和 log 概率，再除以权重和。

    加权在 log 域做，避免概率相乘时被单个成员主导。
    """
    total = None
    wsum = 0.0
    for name, (w, logits) in member_logits.items():
        if w == 0:
            continue
        lp = log_softmax(logits) * float(w)
        total = lp if total is None else total + lp
        wsum += abs(float(w))
    if total is None or wsum == 0:
        raise ValueError("没有任何成员参与融合")
    return total / wsum


def blend(prob: np.ndarray, other: np.ndarray, w: float) -> np.ndarray:
    """把另一路概率按权重并入（用于互补堆叠 / ATLAS）。"""
    if w <= 0:
        return prob
    out = (1.0 - w) * prob + w * other
    return out / out.sum(axis=1, keepdims=True)


def sinkhorn_balance(prob: np.ndarray, iters: int = 100) -> np.ndarray:
    """行列交替归一，把 18 类的预测频次压向均匀。

    balanced accuracy 对类别先验敏感，这一步直接对齐指标口径。
    """
    q = prob.copy()
    k = prob.shape[1]
    for _ in range(iters):
        q *= (len(q) / k) / np.maximum(q.sum(axis=0, keepdims=True), 1e-12)
        q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
    return q


def apply_guards(logp: np.ndarray, texts, cfg: dict) -> tuple[np.ndarray, int]:
    """排版硬约束：出现某标点 → 只允许特定模型。

    语义是单向蕴含（出现则必属该集合），反向不成立。
    """
    labels = cfg["labels"]
    idx = {l: i for i, l in enumerate(labels)}
    out = logp.copy()
    n_fixed = 0
    for rule in cfg["guards"]:
        mark = rule["mark"]
        allowed = {idx[l] for l in rule["allowed"]}
        forbidden = np.ones(len(labels), dtype=bool)
        forbidden[list(allowed)] = False
        for i, text in enumerate(texts):
            if mark in text and int(out[i].argmax()) not in allowed:
                out[i, forbidden] = -1e9
                n_fixed += 1
    return out, n_fixed


def dense_cluster_assign(logp: np.ndarray, groups, cfg: dict) -> tuple[np.ndarray, int]:
    """对规模在 [lo, hi] 的簇做一对一分配，消掉同题重复预测。"""
    g = cfg["grouping"]
    lo, hi = g["dense_cluster_lo"], g["dense_cluster_hi"]
    out = logp.copy()
    sizes = np.bincount(groups)[groups]      # 每行所属簇的规模
    n_changed = 0
    for gid in np.unique(groups[(sizes >= lo) & (sizes <= hi)]):
        rows = np.where(groups == gid)[0]
        before = out[rows].argmax(axis=1)
        r, c = linear_sum_assignment(-out[rows])
        if not np.array_equal(before[r], c):
            n_changed += int((before[r] != c).sum())
        out[rows] = -1e9
        out[rows[r], c] = 0.0
    return out, n_changed


def postprocess(
    logp: np.ndarray,
    texts,
    groups=None,
    cfg: dict | None = None,
    balance: bool = True,
    cluster: bool = True,
    guards: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (最终 log 概率, 预测索引, 诊断信息)。"""
    cfg = cfg or load_recipe()
    k = len(cfg["labels"])
    n = len(logp)

    prob = softmax(logp)
    info: dict = {}

    if balance and n >= k:
        q = sinkhorn_balance(prob)
        lam = float(cfg["fusion"]["sinkhorn"])
        logp = (1.0 - lam) * np.log(prob + 1e-12) + lam * np.log(q + 1e-12)
        info["sinkhorn"] = True
    else:
        logp = np.log(prob + 1e-12)
        info["sinkhorn"] = False

    if cluster and groups is not None:
        logp, info["cluster_changed"] = dense_cluster_assign(logp, np.asarray(groups), cfg)

    # 硬约束放在最后：排版规则是事实性约束，不应被簇匹配启发式覆盖
    if guards and texts is not None:
        logp, info["guard_fixed"] = apply_guards(logp, texts, cfg)

    pred = logp.argmax(axis=1)
    if verbose:
        print("[post]", info)
    return logp, pred, info


def to_labels(pred, cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_recipe()
    return [cfg["labels"][int(i)] for i in pred]


def _demo() -> None:
    """合成数据自检：4 个簇，每簇恰好含全部 18 类（模拟"同题 18 模型各一条"）。"""
    cfg = load_recipe()
    k = len(cfg["labels"])
    n_groups = 4
    n = n_groups * k
    groups = np.repeat(np.arange(n_groups), k)

    rng = np.random.default_rng(0)
    truth = np.concatenate([rng.permutation(k) for _ in range(n_groups)])
    logits = rng.normal(0, 1.0, size=(n, k))
    logits[np.arange(n), truth] += 2.5

    # 让文本与标签一致：只有允许 em dash 的类才带这个符号
    em_dash_ok = {cfg["labels"].index(l) for l in cfg["guards"][0]["allowed"]}
    texts = []
    for i in range(n):
        t = f"sample passage {i}, ordinary punctuation."
        if int(truth[i]) in em_dash_ok and i % 2 == 0:
            t += " He paused — then continued."
        texts.append(t)

    logp = fuse({"synthetic": (1.0, logits)})

    def acc(p):
        return float(np.mean(p == truth))

    raw = logp.argmax(axis=1)
    _, full, info = postprocess(logp, texts, groups, cfg, verbose=True)
    _, no_bal, _ = postprocess(logp, texts, groups, cfg, balance=False)
    _, no_clus, _ = postprocess(logp, texts, groups, cfg, cluster=False)

    print(f"原始 argmax      acc={acc(raw):.3f}")
    print(f"完整后处理       acc={acc(full):.3f}   {info}")
    print(f"关闭 Sinkhorn    acc={acc(no_bal):.3f}")
    print(f"关闭簇匹配       acc={acc(no_clus):.3f}")

    # 不变式：被排版规则禁止的类不应再出现在 argmax 中
    for rule in cfg["guards"]:
        allowed = set(rule["allowed"])
        for i, t in enumerate(texts):
            if rule["mark"] in t:
                assert cfg["labels"][int(full[i])] in allowed, "guard 未生效"
    print("排版约束不变式   通过")

    # 排版规则的独立验证：强行把带 em dash 的文本判成不允许的类
    i0 = next(i for i, t in enumerate(texts) if " — " in t)
    bad = np.zeros((1, k))
    bad[0, cfg["labels"].index("gemini-3.1-pro-preview")] = 5.0
    fixed, nfix = apply_guards(bad, [texts[i0]], cfg)
    new = cfg["labels"][int(fixed[0].argmax())]
    print(f"排版规则单元检查 修正 {nfix} 行 → 新标签 {new}")
    assert new in set(cfg["guards"][0]["allowed"])

    # Sinkhorn 只调整类别先验，不改变排序信息；本数据本身均匀，故为中性
    print("说明：本数据每簇已是 18 类各一条，Sinkhorn 预期为中性；"
          "增益来自稠密簇匹配。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="后处理参考实现（合成数据自检）")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--no-guards", action="store_true")
    args = ap.parse_args()
    _demo()
