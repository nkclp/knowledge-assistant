import streamlit as st
import json
import os
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import glob
from datetime import datetime

# ========== 1. 配置 ==========
st.set_page_config(page_title="个人知识库助手", page_icon="📚")
st.title("📚 个人知识库 + 智能推荐助手")

API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

# ========== 2. 加载 embedding 模型 ==========
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', local_files_only=True)

try:
    model = load_model()
    st.success("✅ 模型加载成功")
except:
    st.warning("⚠️ 模型加载失败，将使用关键词匹配模式")
    model = None

# ========== 3. 笔记存储路径 ==========
NOTES_DIR = "my_notes"
if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)

# ========== 4. 保存笔记 ==========
def save_note(content):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{NOTES_DIR}/note_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

# ========== 5. 读取所有笔记 ==========
def load_all_notes():
    notes = []
    note_files = glob.glob(f"{NOTES_DIR}/*.txt")
    for file_path in note_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            notes.append({
                'file': file_path,
                'content': content
            })
    return notes

# ========== 6. 检索相关笔记（RAG） ==========
def search_notes(query, top_k=3):
    notes = load_all_notes()
    if not notes:
        return []
    
    if model is not None:
        # 向量检索
        query_vec = model.encode([query])
        note_vecs = model.encode([n['content'] for n in notes])
        similarities = np.dot(note_vecs, query_vec.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': notes[idx]['content'],
                'file': notes[idx]['file'],
                'score': float(similarities[idx])
            })
        return results
    else:
        # 关键词检索（备用）
        results = []
        for note in notes:
            if query.lower() in note['content'].lower():
                results.append(note)
        return results[:top_k]

# ========== 7. AI 问答 ==========
def ask_question(question):
    related_notes = search_notes(question, top_k=3)
    
    if not related_notes:
        prompt = f"用户问：{question}\n你是一个知识库助手，但目前没有相关笔记。请礼貌地告诉用户，并建议他先记录一些学习内容。"
    else:
        context = "\n\n---\n\n".join([n['content'] for n in related_notes])
        prompt = f"""根据用户的学习笔记回答下面的问题。

【用户的学习笔记】
{context}

【问题】
{question}

要求：
1. 只根据笔记内容回答
2. 如果笔记中找不到答案，诚实说"笔记中没有相关信息"
3. 回答要简洁有帮助
"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ========== 8. 生成推荐 ==========
def get_recommendation():
    notes = load_all_notes()
    if not notes:
        return "你还没有记录任何学习笔记。试着记录一些学习内容，我就能给你推荐了！"
    
    all_content = "\n".join([n['content'] for n in notes])
    prompt = f"""根据用户的学习历史，推荐接下来可以学习什么。

【用户的学习历史】
{all_content}

要求：
1. 分析用户的学习方向（比如 Python、AI、前端等）
2. 推荐 2-3 个下一步可以学习的主题
3. 每个推荐附带简单的理由

输出格式：
推荐1：XXX（理由：...）
推荐2：XXX（理由：...）
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ========== 9. 界面布局 ==========
tab1, tab2, tab3 = st.tabs(["📝 写笔记", "💬 问问题", "🎯 推荐"])

with tab1:
    st.subheader("记录你的学习")
    note_content = st.text_area("今天学到了什么？", height=200)
    if st.button("保存笔记"):
        if note_content.strip():
            filename = save_note(note_content)
            st.success(f"✅ 笔记已保存！")
        else:
            st.warning("内容不能为空")

with tab2:
    st.subheader("问你的知识库")
    question = st.text_input("输入你的问题", placeholder="例如：我上周学了什么？")
    if st.button("提问"):
        if question:
            with st.spinner("思考中..."):
                answer = ask_question(question)
            st.write("🤖 **回答：**")
            st.write(answer)
        else:
            st.warning("请输入问题")

with tab3:
    st.subheader("智能推荐")
    if st.button("给我推荐"):
        with st.spinner("分析中..."):
            recommendation = get_recommendation()
        st.write("📌 **推荐：**")
        st.write(recommendation)
    
    # 显示已有笔记
    st.subheader("📋 已有笔记")
    notes = load_all_notes()
    if notes:
        for note in notes[-5:]:  # 只显示最近5条
            with st.expander(f"📄 {note['file']}"):
                st.write(note['content'])
    else:
        st.info("还没有笔记，快记录一条吧！")