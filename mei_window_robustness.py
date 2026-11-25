# mei_window_robustness.py
import pandas as pd
import numpy as np
import re
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 参数
INPUT_FILE = "F:\\研究数据\\论文过程数据\\Eu_Think_Tank\\out\\JCMS\\think_tank.xlsx"
TEXT_COL = "text"
DATE_COL = "date"
DICTIONARY_JSON = "F:\\研究数据\\论文过程数据\\Eu_Think_Tank\\out\\JCMS\\identity_dictionary-score=0.8.json"
OUTPUT_FOLDER = "F:\\研究数据\\论文过程数据\\Eu_Think_Tank\\out\\JCMS\\"
WINDOWS = [3, 5, 10]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 读取数据
df = pd.read_excel(INPUT_FILE)
df = df.dropna(subset=[TEXT_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str)

# 预处理
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

df["tokens"] = df[TEXT_COL].apply(clean_text)

# Bigram
phrases = Phrases(df["tokens"], min_count=5, threshold=10)
bigram = Phraser(phrases)
df["tokens"] = df["tokens"].apply(lambda x: bigram[x])

# 读取 identity dictionary
with open(DICTIONARY_JSON, "r", encoding="utf-8") as f:
    identity_dict = json.load(f)
normative = set(identity_dict["normative"])
market = set(identity_dict["market"])
strategic = set(identity_dict["strategic"])

china_words = set(["china", "chinese", "beijing"])

def count_words(tokens, vocab):
    return sum(1 for w in tokens if w in vocab)

df["length"] = df["tokens"].apply(len)
df["norm"] = df["tokens"].apply(lambda x: count_words(x, normative))
df["market"] = df["tokens"].apply(lambda x: count_words(x, market))
df["strategic"] = df["tokens"].apply(lambda x: count_words(x, strategic))
df["norm_ias"] = df["norm"] / df["length"]
df["market_ias"] = df["market"] / df["length"]
df["strat_ias"] = df["strategic"] / df["length"]

df["china_count"] = df["tokens"].apply(lambda x: sum(1 for w in x if w in china_words))

def co_occurrence(tokens, vocab, window=5):
    count = 0
    for i, w in enumerate(tokens):
        if w in china_words:
            start = max(0, i - window)
            end = min(len(tokens), i + window)  # 与原代码一致
            window_words = tokens[start:end]
            if any(v in window_words for v in vocab):
                count += 1
    return count

# 计算不同窗口下的 MEI
for w in WINDOWS:
    df[f"mei_norm_{w}"] = df["tokens"].apply(lambda x: co_occurrence(x, normative, window=w))
    df[f"mei_market_{w}"] = df["tokens"].apply(lambda x: co_occurrence(x, market, window=w))
    df[f"mei_strat_{w}"] = df["tokens"].apply(lambda x: co_occurrence(x, strategic, window=w))
    df[f"mei_norm_{w}"] /= (df["china_count"] + 1)
    df[f"mei_market_{w}"] /= (df["china_count"] + 1)
    df[f"mei_strat_{w}"] /= (df["china_count"] + 1)

# 描述统计
desc_vars = [f"mei_norm_{w}" for w in WINDOWS] + [f"mei_market_{w}" for w in WINDOWS] + [f"mei_strat_{w}" for w in WINDOWS]
desc = df[desc_vars].describe()

# 相关性
corr_norm_pearson = df[[f"mei_norm_{w}" for w in WINDOWS]].corr(method='pearson')
corr_norm_spearman = df[[f"mei_norm_{w}" for w in WINDOWS]].corr(method='spearman')
corr_market_pearson = df[[f"mei_market_{w}" for w in WINDOWS]].corr(method='pearson')
corr_strat_pearson = df[[f"mei_strat_{w}" for w in WINDOWS]].corr(method='pearson')

# 回归比较
reg_results = []
for w in WINDOWS:
    X = df[[f"mei_norm_{w}", f"mei_market_{w}", f"mei_strat_{w}"]]
    X = sm.add_constant(X)
    y = df["norm_ias"]
    model = sm.OLS(y, X).fit()
    for var in X.columns:
        reg_results.append({
            "Window": w,
            "Variable": var,
            "Coef": model.params[var],
            "Std.Err": model.bse[var],
            "P-value": model.pvalues[var],
            "R-squared": model.rsquared,
            "N": int(model.nobs)
        })
reg_df = pd.DataFrame(reg_results)

# 时间趋势
df["year"] = pd.to_datetime(df[DATE_COL]).dt.year
yearly_norm = df.groupby("year")[[f"mei_norm_{w}" for w in WINDOWS]].mean()
yearly_market = df.groupby("year")[[f"mei_market_{w}" for w in WINDOWS]].mean()
yearly_strat = df.groupby("year")[[f"mei_strat_{w}" for w in WINDOWS]].mean()

# 画图
plt.figure(figsize=(10,6))
for w in WINDOWS:
    plt.plot(yearly_norm.index, yearly_norm[f"mei_norm_{w}"], marker='o', label=f"±{w} words")
plt.title("Normative MEI by Window Size")
plt.xlabel("Year")
plt.ylabel("Mean MEI")
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "mei_norm_window_trends.png"))
plt.close()

# 保存结果到 Excel
with pd.ExcelWriter(os.path.join(OUTPUT_FOLDER, "mei_window_robustness.xlsx")) as writer:
    desc.to_excel(writer, sheet_name="Descriptive")
    corr_norm_pearson.to_excel(writer, sheet_name="Pearson_Norm")
    corr_norm_spearman.to_excel(writer, sheet_name="Spearman_Norm")
    corr_market_pearson.to_excel(writer, sheet_name="Pearson_Market")
    corr_strat_pearson.to_excel(writer, sheet_name="Pearson_Strategic")
    reg_df.to_excel(writer, sheet_name="Regression_Comparison", index=False)
    yearly_norm.to_excel(writer, sheet_name="Yearly_Norm")
    yearly_market.to_excel(writer, sheet_name="Yearly_Market")
    yearly_strat.to_excel(writer, sheet_name="Yearly_Strategic")

print("稳健性检验完成。结果保存在：", os.path.join(OUTPUT_FOLDER, "mei_window_robustness.xlsx"))
print("\n回归系数比较：")
print(reg_df.pivot(index="Variable", columns="Window", values="Coef"))
print("\nNormative MEI 相关系数（Pearson）：")
print(corr_norm_pearson)