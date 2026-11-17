# -*- coding: utf-8 -*-
"""
课程实验文档检索问答工具（本地运行，零第三方依赖）

用法：
    python ask.py                                 交互模式，只做检索
    python ask.py --ask "VLAN Mapping 怎么配"      单次提问
    python ask.py --llm                           启用大模型生成答案（需免费 API Key）
    python ask.py --docs "D:/其他目录"             指定文档目录

设计要点（面试可以照着讲）：
    1. 切分：按空行分段，长段再切，块间保留 80 字重叠，避免答案被切断
    2. 检索：字符 bigram + 自写 BM25，不依赖分词库，中文召回稳定
    3. 生成：把 top-k 片段拼进 Prompt，强制要求标注来源，无依据时拒答
"""
import os
import re
import sys
import math
import json
import argparse
import urllib.request
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-4-flash"
CHUNK = 400          # 每块最大字符数
OVERLAP = 80         # 块间重叠字符数
TOP_K = 4            # 召回片段数


# ---------------- 文档切分 ----------------

def load_chunks(root):
    """遍历目录下所有 txt，切成带来源标注的文本块。"""
    chunks = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            path = os.path.join(dirpath, name)
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            rel = os.path.relpath(path, root).replace("\\", "/")
            for piece in split_text(text):
                if is_clean(piece):
                    chunks.append({"text": piece, "src": rel})
    return chunks


def split_text(text):
    """按空行分段，超长段再切，带重叠。"""
    text = re.sub(r"[ \t]+", " ", text)
    paras, buf = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if buf:
                paras.append("\n".join(buf))
                buf = []
        else:
            buf.append(s)
    if buf:
        paras.append("\n".join(buf))

    out = []
    for p in paras:
        if len(p) <= CHUNK:
            if len(p) >= 20:          # 丢太短的（标题、页码）
                out.append(p)
        else:
            start = 0
            while start < len(p):
                out.append(p[start:start + CHUNK])
                if start + CHUNK >= len(p):
                    break
                start += CHUNK - OVERLAP
    return out


# ---------------- 检索 ----------------

def is_clean(text):
    """剔除 .doc 二进制解析产生的乱码块：中文占比过低、生僻字过多的一律不要。"""
    if len(text) < 30:
        return False
    cn = re.findall(r"[\u4e00-\u9fa5]", text)
    if len(cn) < 10 or len(cn) / len(text) < 0.40:
        return False
    common = 0
    for ch in cn:
        try:
            ch.encode("gb2312")
            common += 1
        except UnicodeEncodeError:
            pass
    return common / len(cn) >= 0.90


def tokenize(text):
    """中文按字符 bigram，英文数字按词。不依赖分词库。"""
    text = text.lower()
    tokens = []
    for m in re.finditer(r"[a-z0-9\.\-]{2,}", text):
        tokens.append(m.group())
    cn = re.sub(r"[^\u4e00-\u9fa5]+", " ", text)
    for seg in cn.split():
        for i in range(len(seg) - 1):
            tokens.append(seg[i:i + 2])
    return tokens


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.n = len(docs)
        self.avg = sum(self.len) / max(self.n, 1)
        self.k1, self.b = k1, b
        df = Counter()
        for d in docs:
            for t in set(d):
                df[t] += 1
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
            for t, c in df.items()
        }

    def score(self, query):
        q = [t for t in query if t in self.idf]
        scores = [0.0] * self.n
        for i in range(self.n):
            tf, dl = self.tf[i], self.len[i]
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                s += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avg)
                )
            scores[i] = s
        return scores


def phrase_bonus(question, text):
    """查询里的连续词块若在原文中原样出现，说明更像精确命中，给加权。"""
    cn = re.sub(r"[^\u4e00-\u9fa5a-z0-9]+", " ", question.lower())
    low = text.lower()
    subs = set()
    for seg in cn.split():
        for L in (4, 3, 2):
            for i in range(len(seg) - L + 1):
                subs.add((seg[i:i + L], L))
    bonus = 0.0
    for s, L in subs:
        if s in low:
            bonus += (L - 1) * 0.25
    return min(bonus, 2.0)


def search(chunks, bm25, question, top_k=TOP_K, pool=30):
    """两阶段：BM25 先粗排 30 条，再用短语命中重排，取前 k。"""
    q = tokenize(question)
    scores = bm25.score(q)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]
    cands = []
    for i in order:
        if scores[i] <= 0:
            continue
        cands.append({
            "text": chunks[i]["text"],
            "src": chunks[i]["src"],
            "score": scores[i] * (1 + phrase_bonus(question, chunks[i]["text"])),
        })
    cands.sort(key=lambda x: x["score"], reverse=True)
    return [
        {"text": c["text"], "src": c["src"], "score": round(c["score"], 2)}
        for c in cands[:top_k]
    ]


# ---------------- 生成 ----------------

def call_llm(api_key, question, hits):
    ctx = "\n\n".join(
        f"[资料{i + 1}] 来源：{h['src']}\n{h['text']}" for i, h in enumerate(hits)
    )
    system = (
        "你是网络工程实验文档助手。只能依据用户提供的资料回答，"
        "资料里没有的内容必须明确说「资料里没有这部分内容」，"
        "不得编造设备命令、VLAN ID、IP 地址等参数。"
        "回答格式：先一句话结论，再分步骤列操作要点，最后列不超过 3 条易错点，"
        "每个关键结论后用 [资料N] 标注来源。"
    )
    user = f"资料：\n{ctx}\n\n问题：{question}"
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ---------------- 主流程 ----------------

def answer(chunks, bm25, question, api_key=None):
    hits = search(chunks, bm25, question)
    if not hits:
        print("\n没有检索到相关内容。换个说法，或确认文档目录里有没有相关实验。\n")
        return

    if api_key:
        try:
            print("\n" + call_llm(api_key, question, hits) + "\n")
        except Exception as e:
            print(f"\n调用大模型失败：{e}\n降级为纯检索结果。\n")
    else:
        print("\n【检索到的片段】")

    if not api_key:
        for i, h in enumerate(hits, 1):
            print(f"\n--- 资料{i}  来源：{h['src']}  相关度 {h['score']} ---")
            print(h["text"][:400])

    print("\n【引用来源】")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h['src']}（{h['score']}）")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=DEFAULT_DOCS, help="文档目录")
    ap.add_argument("--ask", help="单个问题，不给则进入交互模式")
    ap.add_argument("--llm", action="store_true", help="启用大模型生成")
    ap.add_argument("--key", default=os.getenv("ZHIPU_API_KEY"), help="API Key")
    args = ap.parse_args()

    if not os.path.isdir(args.docs):
        print(f"文档目录不存在：{args.docs}")
        return

    print(f"正在读取文档：{args.docs}")
    chunks = load_chunks(args.docs)
    if not chunks:
        print("没有读到任何 txt 文档。")
        return
    print(f"共 {len(chunks)} 个文本块，正在建立索引…")
    bm25 = BM25([tokenize(c["text"]) for c in chunks])
    mode = "大模型生成" if (args.llm and args.key) else "仅检索"
    print(f"就绪（{mode}）。\n")

    if args.ask:
        answer(chunks, bm25, args.ask, args.key if args.llm else None)
        return

    while True:
        try:
            q = input("问题（回车退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer(chunks, bm25, q, args.key if args.llm else None)


if __name__ == "__main__":
    main()
