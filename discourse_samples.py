import pandas as pd
import numpy as np
import re
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("D:\\temp\\processed_results.xlsx")

# =========================
# 2. 选取高强度文本（Top 10%）
# =========================
def select_top(df, col, q=0.9):
    threshold = df[col].quantile(q)
    return df[df[col] >= threshold]

df_norm = select_top(df, 'mei_norm')
df_strat = select_top(df, 'mei_strat')
df_market = select_top(df, 'mei_market')

# =========================
# 3. 定义“身份话语特征词”
# =========================
self_ref_words = [
    "europe", "eu", "european union", "we", "our","Europen","Europea"
]

normative_words = [
    "democracy", "human rights", "rule of law", "values",
    "freedom", "norms"
]

strategic_words = [
    "security", "strategy", "geopolitics", "autonomy",
    "defense", "threat"
]

# =========================
# 4. 句子评分函数
# =========================
def score_sentence(sent, word_list):
    sent_lower = sent.lower()
    return sum(1 for w in word_list if w in sent_lower)

def extract_sentences(df_subset, label, word_list):
    results = []

    for text in df_subset['text'].dropna():
        sentences = sent_tokenize(text)
        for sent in sentences:
            score_self = score_sentence(sent, self_ref_words)
            score_topic = score_sentence(sent, word_list)

            total_score = score_self + score_topic

            if total_score >= 2:  # 阈值可调
                results.append({
                    "type": label,
                    "sentence": sent,
                    "self_score": score_self,
                    "topic_score": score_topic,
                    "total_score": total_score
                })

    return pd.DataFrame(results)

# =========================
# 5. 抽取三类话语
# =========================
df_sent_norm = extract_sentences(df_norm, "normative", normative_words)
df_sent_strat = extract_sentences(df_strat, "strategic", strategic_words)
df_sent_market = extract_sentences(df_market, "market", ["trade", "investment", "economy", "market"])

# =========================
# 6. 选Top句子（每类前20条）
# =========================
def top_sentences(df_sent, n=20):
    return df_sent.sort_values(by="total_score", ascending=False).head(n)

top_norm = top_sentences(df_sent_norm)
top_strat = top_sentences(df_sent_strat)
top_market = top_sentences(df_sent_market)

# =========================
# 7. 保存结果
# =========================
with pd.ExcelWriter("D:\\temp\\discourse_samples.xlsx") as writer:
    top_norm.to_excel(writer, sheet_name="Normative", index=False)
    top_strat.to_excel(writer, sheet_name="Strategic", index=False)
    top_market.to_excel(writer, sheet_name="Market", index=False)

print("完成：已生成 discourse_samples.xlsx")