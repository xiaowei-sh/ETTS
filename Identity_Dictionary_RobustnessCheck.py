# ==========================================
# Robustness Check: Identity Dictionary
# with Different Cosine Similarity Thresholds
# min_score = 0.70, 0.80, 0.90
# ==========================================
import pandas as pd
import numpy as np
import re
import os
import json

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 参数设置
# ==========================================
INPUT_FILE = "F:\\研究数据\\论文过程数据\\Eu_Think_Tank\\out\\JCMS\\think_tank.xlsx"          # ← 修改为你的 Excel 路径
TEXT_COL = "text"                                  # ← 文本列名
OUTPUT_FOLDER = "F:\\研究数据\\论文过程数据\\Eu_Think_Tank\\out\\JCMS\\"                       # ← 输出文件夹
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

THRESHOLDS = [0.70, 0.80, 0.90]                    # 稳健性检验阈值
TOPN = 20                                          # 与原程序一致

# 与原程序完全一致的种子词
normative_seed = ["democracy", "freedom", "human rights", "multilateralism", "norms"]
market_seed = ["trade", "market", "investment", "market economy",
               "fair trade", "level playing field", "liberalization"]
strategic_seed = ["strategic autonomy", "security", "autonomy",
                  "geopolitical", "sovereignty", "resilience"]

# ==========================================
# 2. 读取数据与预处理（与原程序一致）
# ==========================================
df = pd.read_excel(INPUT_FILE)
df = df.dropna(subset=[TEXT_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str)
print(f"Loaded {len(df)} documents")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

df['tokens'] = df[TEXT_COL].apply(clean_text)

# Bigram
phrases = Phrases(df['tokens'], min_count=5, threshold=10)
bigram = Phraser(phrases)
df['tokens'] = df['tokens'].apply(lambda x: bigram[x])

# ==========================================
# 3. 训练 Word2Vec（与原程序一致）
# ==========================================
sentences = df['tokens'].tolist()

w2v_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1
)
w2v_model.save(os.path.join(OUTPUT_FOLDER, "w2v_model_robustness.model"))
print("Word2Vec model trained and saved.")

# ==========================================
# 4. 检查 seed word 是否在词表中
#    并自动识别 bigram 形式（如 human_rights）
# ==========================================
def resolve_seed(word, model):
    """
    若原始 seed 在词表中，返回其本身；
    否则尝试将空格替换为下划线（bigram 形式）。
    若都不在词表中，返回 None。
    """
    if word in model.wv:
        return word
    bigram_form = word.replace(" ", "_")
    if bigram_form in model.wv:
        return bigram_form
    return None

print("\n=== Seed word 词表检查 ===")
seed_report = []
for category, seeds in [
    ("normative", normative_seed),
    ("market", market_seed),
    ("strategic", strategic_seed)
]:
    for w in seeds:
        resolved = resolve_seed(w, w2v_model)
        seed_report.append({
            "category": category,
            "seed": w,
            "resolved": resolved,
            "in_vocab": resolved is not None
        })
        print(f"[{category}] {w:25s} -> {resolved}")

seed_report_df = pd.DataFrame(seed_report)
seed_report_df.to_excel(
    os.path.join(OUTPUT_FOLDER, "seed_word_check.xlsx"),
    index=False
)

# ==========================================
# 5. 相似度分布：打印每个 seed 的 topn 相似词及分数
#    用于判断 0.80 在分布中的位置
# ==========================================
sim_records = []
for category, seeds in [
    ("normative", normative_seed),
    ("market", market_seed),
    ("strategic", strategic_seed)
]:
    for w in seeds:
        resolved = resolve_seed(w, w2v_model)
        if resolved is None:
            continue
        similar = w2v_model.wv.most_similar(resolved, topn=TOPN)
        for rank, (word, score) in enumerate(similar, start=1):
            sim_records.append({
                "category": category,
                "seed": w,
                "resolved_seed": resolved,
                "rank": rank,
                "similar_word": word,
                "score": score
            })

sim_df = pd.DataFrame(sim_records)
sim_df.to_excel(
    os.path.join(OUTPUT_FOLDER, "seed_similarity_distribution.xlsx"),
    index=False
)

# 每个 seed 的相似度分位数（用于论文报告）
quantile_df = (
    sim_df.groupby(["category", "seed"])["score"]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .reset_index()
)
quantile_df.to_excel(
    os.path.join(OUTPUT_FOLDER, "seed_similarity_quantiles.xlsx"),
    index=False
)

print("\n=== 相似度分位数（每类 seed 汇总） ===")
print(sim_df.groupby("category")["score"].describe(
    percentiles=[0.25, 0.5, 0.75]
))

# ==========================================
# 6. expand_seed 函数（与原程序一致，加入 min_score 参数）
# ==========================================
def expand_seed(seed_list, model, topn=TOPN, min_score=0.6):
    expanded = set()
    for word in seed_list:
        resolved = resolve_seed(word, model)
        if resolved is None:
            # 保留 seed 本身（若可用 bigram 形式则用 bigram）
            bigram_form = word.replace(" ", "_")
            expanded.add(bigram_form)
            continue
        # 先加入 seed 本身
        expanded.add(resolved)
        similar = model.wv.most_similar(resolved, topn=topn)
        for w, score in similar:
            if score >= min_score:
                expanded.add(w)
    return sorted(expanded)

# ==========================================
# 7. 分别以 0.70 / 0.80 / 0.90 构建词典
# ==========================================
results = {}

for t in THRESHOLDS:
    norm_dict = expand_seed(normative_seed, w2v_model, topn=TOPN, min_score=t)
    market_dict = expand_seed(market_seed, w2v_model, topn=TOPN, min_score=t)
    strat_dict = expand_seed(strategic_seed, w2v_model, topn=TOPN, min_score=t)

    results[t] = {
        "normative": norm_dict,
        "market": market_dict,
        "strategic": strat_dict
    }

    print(f"\n===== Threshold = {t} =====")
    print(f"Normative ({len(norm_dict)}): {norm_dict}")
    print(f"Market    ({len(market_dict)}): {market_dict}")
    print(f"Strategic ({len(strat_dict)}): {strat_dict}")

# ==========================================
# 8. 词典规模对比
# ==========================================
size_rows = []
for t in THRESHOLDS:
    for cat in ["normative", "market", "strategic"]:
        size_rows.append({
            "Threshold": t,
            "Category": cat,
            "NumWords": len(results[t][cat])
        })
size_df = pd.DataFrame(size_rows)
size_pivot = size_df.pivot(index="Category", columns="Threshold", values="NumWords")

print("\n=== 词典规模对比 ===")
print(size_pivot)

size_df.to_excel(
    os.path.join(OUTPUT_FOLDER, "dictionary_size_comparison.xlsx"),
    index=False
)

# ==========================================
# 9. 类别重叠分析
# ==========================================
overlap_rows = []
for t in THRESHOLDS:
    n = set(results[t]["normative"])
    m = set(results[t]["market"])
    s = set(results[t]["strategic"])

    overlap_rows.append({
        "Threshold": t,
        "Norm_Market_overlap": len(n & m),
        "Norm_Strat_overlap": len(n & s),
        "Market_Strat_overlap": len(m & s),
        "Norm_Market_words": "; ".join(sorted(n & m)),
        "Norm_Strat_words": "; ".join(sorted(n & s)),
        "Market_Strat_words": "; ".join(sorted(m & s)),
        "Total_unique_words": len(n | m | s)
    })

overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_excel(
    os.path.join(OUTPUT_FOLDER, "dictionary_overlap_comparison.xlsx"),
    index=False
)

print("\n=== 类别重叠对比 ===")
print(overlap_df[[
    "Threshold",
    "Norm_Market_overlap",
    "Norm_Strat_overlap",
    "Market_Strat_overlap",
    "Total_unique_words"
]])

# ==========================================
# 10. 不同阈值下词典的 Jaccard 相似度
#     衡量阈值变化是否显著改变词典内容
# ==========================================
def jaccard(a, b):
    a, b = set(a), set(b)
    if len(a | b) == 0:
        return np.nan
    return len(a & b) / len(a | b)

jaccard_rows = []
for cat in ["normative", "market", "strategic"]:
    for i in range(len(THRESHOLDS)):
        for j in range(i + 1, len(THRESHOLDS)):
            t1, t2 = THRESHOLDS[i], THRESHOLDS[j]
            j_val = jaccard(results[t1][cat], results[t2][cat])
            jaccard_rows.append({
                "Category": cat,
                "Threshold_pair": f"{t1} vs {t2}",
                "Jaccard": j_val,
                "Size_t1": len(results[t1][cat]),
                "Size_t2": len(results[t2][cat]),
                "Shared_words": len(set(results[t1][cat]) & set(results[t2][cat]))
            })

jaccard_df = pd.DataFrame(jaccard_rows)
jaccard_df.to_excel(
    os.path.join(OUTPUT_FOLDER, "dictionary_jaccard.xlsx"),
    index=False
)

print("\n=== 词典 Jaccard 相似度 ===")
print(jaccard_df)

# ==========================================
# 11. 保存每个阈值下的完整词典（JSON）
# ==========================================
for t in THRESHOLDS:
    dict_out = {
        "threshold": t,
        "topn": TOPN,
        "normative": results[t]["normative"],
        "market": results[t]["market"],
        "strategic": results[t]["strategic"]
    }
    with open(
        os.path.join(OUTPUT_FOLDER, f"identity_dictionary-score={t}.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(dict_out, f, indent=4, ensure_ascii=False)

print(f"\n词典已保存至 {OUTPUT_FOLDER}（每阈值一个 JSON 文件）")

# ==========================================
# 12. 可视化
# ==========================================
# (1) 词典规模随阈值变化图
plt.figure(figsize=(8, 5))
for cat in ["normative", "market", "strategic"]:
    sizes = [len(results[t][cat]) for t in THRESHOLDS]
    plt.plot(THRESHOLDS, sizes, marker='o', label=cat)
plt.xlabel("Cosine similarity threshold")
plt.ylabel("Dictionary size (number of words)")
plt.title("Identity Dictionary Size by Threshold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "dict_size_by_threshold.png"))
plt.close()

# (2) 类别重叠随阈值变化图
plt.figure(figsize=(8, 5))
plt.plot(THRESHOLDS, overlap_df["Norm_Market_overlap"], marker='o', label="Norm ∩ Market")
plt.plot(THRESHOLDS, overlap_df["Norm_Strat_overlap"], marker='o', label="Norm ∩ Strategic")
plt.plot(THRESHOLDS, overlap_df["Market_Strat_overlap"], marker='o', label="Market ∩ Strategic")
plt.xlabel("Cosine similarity threshold")
plt.ylabel("Number of overlapping words")
plt.title("Cross-category Overlap by Threshold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "dict_overlap_by_threshold.png"))
plt.close()

# (3) 相似度分布图（每个 seed 的 top-20 相似度）
plt.figure(figsize=(10, 5))
sns.boxplot(data=sim_df, x="category", y="score")
plt.axhline(0.70, color='red', linestyle='--', label='0.70')
plt.axhline(0.80, color='orange', linestyle='--', label='0.80')
plt.axhline(0.90, color='green', linestyle='--', label='0.90')
plt.title("Distribution of Seed Word Similarity Scores (Top-20)")
plt.ylabel("Cosine similarity")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "similarity_score_distribution.png"))
plt.close()

# (4) Jaccard 热力图
jaccard_pivot = jaccard_df.pivot(index="Category", columns="Threshold_pair", values="Jaccard")
plt.figure(figsize=(8, 4))
sns.heatmap(jaccard_pivot, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Jaccard Similarity of Dictionaries across Thresholds")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "dictionary_jaccard_heatmap.png"))
plt.close()

print("\n===== 稳健性检验完成 =====")
print(f"所有结果已保存至：{OUTPUT_FOLDER}")