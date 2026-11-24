# ==========================================
# Pipeline: Construct Identity Dictionary using Word2Vec
# ==========================================
import pandas as pd
import re
import os

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

# ==========================================
# 1. 参数设置
# ==========================================
INPUT_FILE = "D:\\temp\\think_tank.xlsx"  # 修改为你的Excel路径
TEXT_COL = "text"
OUTPUT_FOLDER = "D:\\temp\\"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. 读取数据
# ==========================================
df = pd.read_excel(INPUT_FILE)
df = df.dropna(subset=[TEXT_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str)
print(f"Loaded {len(df)} documents")

# ==========================================
# 3. 文本清洗
# ==========================================
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

# Bigram识别
phrases = Phrases(df['tokens'], min_count=5, threshold=10)
bigram = Phraser(phrases)
df['tokens'] = df['tokens'].apply(lambda x: bigram[x])

# ==========================================
# 4. 训练 Word2Vec
# ==========================================
sentences = df['tokens'].tolist()

w2v_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1  # skip-gram
)
w2v_model.save(os.path.join(OUTPUT_FOLDER, "w2v_model.model"))
print("Word2Vec model trained and saved.")

# ==========================================
# 5. Seed Words
# ==========================================
normative_seed = ["democracy", "freedom", "human rights","multilateralism","norms"]
market_seed = ["trade", "market", "investment","market economy","fair trade","level playing field","liberalization"]
strategic_seed = ["strategic autonomy","security", "autonomy", "geopolitical","sovereignty","resilience"]

# ==========================================
# 6. 扩展 Seed Words 构建 Identity Dictionary
# ==========================================
def expand_seed(seed_list, model, topn=20, min_score=0.8):
    expanded = set(seed_list)
    for word in seed_list:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=topn)
            for w, score in similar:
                if score >= min_score:
                    expanded.add(w)
    return list(expanded)

normative_dict = expand_seed(normative_seed, w2v_model)
market_dict = expand_seed(market_seed, w2v_model)
strategic_dict = expand_seed(strategic_seed, w2v_model)

print("\n=== Identity Dictionary ===")
print(f"Normative ({len(normative_dict)} words): {normative_dict}")
print(f"Market ({len(market_dict)} words): {market_dict}")
print(f"Strategic ({len(strategic_dict)} words): {strategic_dict}")

# ==========================================
# 7. 保存 Identity Dictionary
# ==========================================
import json
identity_dict = {
    "normative": normative_dict,
    "market": market_dict,
    "strategic": strategic_dict
}

#with open(os.path.join(OUTPUT_FOLDER, "identity_dictionary.json"), "w") as f:
with open(os.path.join(OUTPUT_FOLDER, "eu_china.json"), "w") as f:
    json.dump(identity_dict, f, indent=4)

print(f"Identity Dictionary saved to {os.path.join(OUTPUT_FOLDER, 'identity_dictionary.json')}")