import pandas as pd
import numpy as np
import re
import os
import json  # 新增：导入json模块，用于读取json文件

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 替换：删除gensim相关导入，新增Bertopic相关导入
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

# 新增：导入openpyxl相关模块，用于规避Excel单元格字符长度限制
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# ==========================================
# 1. 参数设置（只需要改这里）
# ==========================================

INPUT_FILE = "F:\\研究数据\\论文过程数据\\Eu_Think_Tank\\out\\20261023\\think_tank.xlsx"   # ← 修改这里
TEXT_COL = "text"              # ← 修改为你的列名
DATE_COL = "date"              # ← 修改为你的列名
# 新增：json字典文件路径（请确认路径正确，若与Excel同目录可直接写文件名）
DICTIONARY_JSON = "F:\\研究数据\\论文过程数据\\Eu_Think_Tank\\out\\20261023\\identity_dictionary-score=last.json"  # ← 修改为你的json文件路径
# 新增：Bertopic参数（可根据需求调整）
NUM_TOPICS = 23  # 与原LDA主题数一致，保持结果可比性
MIN_TOPIC_SIZE = 17  # 最小主题文档数，避免过小主题

OUTPUT_FOLDER = "F:\\研究数据\\论文过程数据\\Eu_Think_Tank\\out\\20261023"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. 读取数据
# ==========================================

df = pd.read_excel(INPUT_FILE)
df = df.dropna(subset=[TEXT_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str)

print(f"Loaded {len(df)} documents")

# ==========================================
# 3. 文本预处理（保留原有逻辑，适配Bertopic输入）
# ==========================================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    # 调整：Bertopic接收字符串输入，将预处理后的列表转为空格分隔字符串
    return " ".join(words)

# 调整：预处理后得到字符串，用于Bertopic建模
df["cleaned_text"] = df[TEXT_COL].apply(clean_text)
# 保留tokens列，用于后续Identity Dictionary计数（不影响原有逻辑）
df["tokens"] = df["cleaned_text"].apply(lambda x: x.split())

# Bigram（保留，用于tokens列优化，不影响Bertopic建模）
from gensim.models.phrases import Phrases, Phraser  # 单独导入，避免多余gensim依赖
phrases = Phrases(df["tokens"], min_count=5, threshold=10)
bigram = Phraser(phrases)
df["tokens"] = df["tokens"].apply(lambda x: bigram[x])

# ==========================================
# 4. Bertopic主题模型（替换原有LDA模型，修复参数报错）
# ==========================================

# 初始化Bertopic组件（默认使用all-MiniLM-L6-v2嵌入模型，轻量且高效）
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# 初始化UMAP（降维）和HDBSCAN（聚类），可调整参数优化主题效果
umap_model = UMAP(n_neighbors=17, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# 初始化Bertopic模型（删除random_state参数，适配低版本BERTopic；将随机种子转移至UMAP保证稳定性）
bertopic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True  # 打印建模过程日志
)

# 执行主题建模（输入预处理后的字符串文本）
topics, probabilities = bertopic_model.fit_transform(df["cleaned_text"])

# 新增：动态调整主题数为19个（适配低版本，替代初始化时的num_topics参数）
# 修复：将DataFrame的Series转为列表，适配embedding_model.encode的输入要求
if len(bertopic_model.get_topic_info()) > NUM_TOPICS:
    bertopic_model = bertopic_model.reduce_topics(df["cleaned_text"].tolist(), nr_topics=NUM_TOPICS)
    # 重新获取调整后的主题（Series转列表，解决Unsupported input type: Series报错）
    topics = bertopic_model.transform(df["cleaned_text"].tolist())[0]

# 将主题结果写入DataFrame，与原LDA逻辑一致（topic列格式不变，便于后续分析）
df["topic"] = topics

# 打印主题结果（格式与原LDA一致，显示主题ID和关键词）
print("\n=== Bertopic Topics ===")
topic_info = bertopic_model.get_topic_info()
# 过滤掉未分类主题（-1），仅显示有效主题
valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:NUM_TOPICS]:  # 确保只显示19个主题
    # 获取主题关键词及权重，取前15个（与原LDA一致）
    topic_keywords = bertopic_model.get_topic(topic_id)
    keywords_str = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_keywords[:15]])
    print(f"({topic_id}, '{keywords_str}')")

# ==========================================
# 5. Identity Dictionary（修改：从json文件读取normative、market、strategic）
# ==========================================

# 读取json文件，获取三个类别对应的词汇列表
try:
    with open(DICTIONARY_JSON, "r", encoding="utf-8") as f:
        identity_dict = json.load(f)
    # 提取三个类别词汇，转为集合（保持原有逻辑，便于后续计数）
    normative = set(identity_dict["normative"])
    market = set(identity_dict["market"])
    strategic = set(identity_dict["strategic"])
    print(normative)
    print(f"\nSuccessfully loaded identity dictionary from {DICTIONARY_JSON}")
except Exception as e:
    print(f"\nError loading identity dictionary: {str(e)}")
    print("Please check the json file path and format.")
    exit()  # 若读取失败，终止程序，避免后续报错

def count_words(tokens, vocab):
    return sum(1 for w in tokens if w in vocab)

df["length"] = df["tokens"].apply(len)

df["norm"] = df["tokens"].apply(lambda x: count_words(x, normative))
df["market"] = df["tokens"].apply(lambda x: count_words(x, market))
df["strategic"] = df["tokens"].apply(lambda x: count_words(x, strategic))

df["norm_ias"] = df["norm"] / df["length"]
df["market_ias"] = df["market"] / df["length"]
df["strat_ias"] = df["strategic"] / df["length"]

# ==========================================
# 6. Mirror Effect（MEI）
# ==========================================

china_words = set(["china","chinese","beijing","shanghai","taiwan","hongkong"])

def co_occurrence(tokens, vocab, window=5): #共现距离
    count = 0
    for i, w in enumerate(tokens):
        if w in china_words:
            start = max(0, i-window)
            end = min(len(tokens), i+window)
            window_words = tokens[start:end]
            if any(v in window_words for v in vocab):
                count += 1
    return count

df["china_count"] = df["tokens"].apply(lambda x: sum(1 for w in x if w in china_words))

df["mei_norm"] = df["tokens"].apply(lambda x: co_occurrence(x, normative))
df["mei_market"] = df["tokens"].apply(lambda x: co_occurrence(x, market))
df["mei_strat"] = df["tokens"].apply(lambda x: co_occurrence(x, strategic))

df["mei_norm"] /= (df["china_count"] + 1)
df["mei_market"] /= (df["china_count"] + 1)
df["mei_strat"] /= (df["china_count"] + 1)

# ==========================================
# 7. 回归分析
# ==========================================
#norm_ies因变量
X = df[["mei_norm","mei_market","mei_strat"]]
y = df["norm_ias"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 保存summary
with open(os.path.join(OUTPUT_FOLDER,"regression_summary_norm.txt"),"w") as f:
    f.write(model.summary().as_text())

# 表格
table = pd.DataFrame({
    "Variable": model.params.index,
    "Coef": model.params.values,
    "StdErr": model.bse.values,
    "t": model.tvalues.values,
    "p": model.pvalues.values
}).round(3)

table.to_excel(os.path.join(OUTPUT_FOLDER,"table_regression_results_norm.xlsx"), index=False)

print("\n=== Regression Results ===")
print(model.summary())

#============================================
#strat_ies因变量
X = df[["mei_norm","mei_market","mei_strat"]]
y = df["strat_ias"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 保存summary
with open(os.path.join(OUTPUT_FOLDER,"regression_summary_strat.txt"),"w") as f:
    f.write(model.summary().as_text())

# 表格
table = pd.DataFrame({
    "Variable": model.params.index,
    "Coef": model.params.values,
    "StdErr": model.bse.values,
    "t": model.tvalues.values,
    "p": model.pvalues.values
}).round(3)

table.to_excel(os.path.join(OUTPUT_FOLDER,"table_regression_results_strat.xlsx"), index=False)

print("\n=== Regression Results ===")
print(model.summary())

#=================================================
#market_ies因变量
X = df[["mei_norm","mei_market","mei_strat"]]
y = df["market_ias"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 保存summary
with open(os.path.join(OUTPUT_FOLDER,"regression_summary_market.txt"),"w") as f:
    f.write(model.summary().as_text())

# 表格
table = pd.DataFrame({
    "Variable": model.params.index,
    "Coef": model.params.values,
    "StdErr": model.bse.values,
    "t": model.tvalues.values,
    "p": model.pvalues.values
}).round(3)

table.to_excel(os.path.join(OUTPUT_FOLDER,"table_regression_results_market.xlsx"), index=False)

print("\n=== Regression Results ===")
print(model.summary())

# ==========================================
# 7-2. 整体身份回归模型
# ==========================================
# 确保三个身份维度使用同一批完整观测
cols = [
    "norm_ias", "market_ias", "strat_ias",
    "mei_norm", "mei_market", "mei_strat"
]
data = df[cols].dropna().copy()

# 构建综合欧洲身份表达指标
data["overall_ias"] = data[
    ["norm_ias", "market_ias", "strat_ias"]
].mean(axis=1)

# 建立整体身份回归模型
model = smf.ols(
    "overall_ias ~ mei_norm + mei_market + mei_strat",
    data=data
).fit()

# 使用 HC3 异方差稳健标准误
robust_model = model.get_robustcov_results(cov_type="HC3")

print(robust_model.summary())


# ==========================================
# 8. 时间处理
# ==========================================

df["year"] = pd.to_datetime(df[DATE_COL]).dt.year

# ==========================================
# 9. 图表输出（保留原有图表，新增Bertopic专属主题可视化图表）
# ==========================================

# Topic distribution（与原图表一致，适配Bertopic主题）
sns.countplot(x="topic", data=df)
plt.title("Topic Distribution (Bertopic)")
plt.savefig(os.path.join(OUTPUT_FOLDER, "topic_distribution.png"))
plt.close()

# Identity trends（保持原有逻辑不变）
yearly = df.groupby("year")[["norm_ias","market_ias","strat_ias"]].mean()
yearly.plot()
plt.title("Identity Trends")
plt.savefig(os.path.join(OUTPUT_FOLDER, "identity_trends.png"))
plt.close()

# Mirror effect trends（保持原有逻辑不变）
yearly_mei = df.groupby("year")[["mei_norm","mei_market","mei_strat"]].mean()
yearly_mei.plot()
plt.title("Mirror Effect Trends")
plt.savefig(os.path.join(OUTPUT_FOLDER, "mirror_effect.png"))
plt.close()

# 新增：Bertopic主题可视化（词云图+主题相似度图，提升分析维度）
# 主题词云图（每个主题一张图）
bertopic_model.visualize_topics().write_html(os.path.join(OUTPUT_FOLDER, "bertopic_topics_visualization.html"))
# 主题关键词条形图
bertopic_model.visualize_barchart(top_n_topics=NUM_TOPICS).write_html(os.path.join(OUTPUT_FOLDER, "bertopic_keywords_barchart.html"))

# ==========================================
# 10. 导出结果（修改部分：用openpyxl保存，规避32767字符限制+解决列表无法写入问题）
# ==========================================

output_file = os.path.join(OUTPUT_FOLDER, "processed_results.xlsx")
# 替换原有df.to_excel，用openpyxl保存，保留所有数据且无字符截断
wb = Workbook()
ws = wb.active

# 核心修改：处理列表类型数据（tokens列），转为字符串后再写入Excel
# 先复制DataFrame，避免修改原数据影响后续分析
df_export = df.copy()
# 将tokens列（列表）转为空格分隔的字符串，openpyxl可正常写入
df_export["tokens"] = df_export["tokens"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)

# 将处理后的DataFrame数据写入工作表，保留表头和所有内容
for r in dataframe_to_rows(df_export, index=False, header=True):
    ws.append(r)
# 保存文件
wb.save(output_file)

# 保存Bertopic模型（便于后续复用、调参）
bertopic_model.save(os.path.join(OUTPUT_FOLDER, "bertopic_model"))

print(f"\nAll results saved to: {OUTPUT_FOLDER}")
print(f"Bertopic model saved to: {os.path.join(OUTPUT_FOLDER, 'bertopic_model')}")