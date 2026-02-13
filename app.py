"""
MedSearch AI - Modern Minimalist Interface
Inspired by professional medical search design
Author: BOUCHAMA Sarra
ESI 2025-2026
"""

import streamlit as st
import time
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from config import Config, Model1Config, Model2Config
import plotly.graph_objects as go

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="MedSearch AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ULTRA-MODERN CSS ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* ===== MAIN LAYOUT ===== */
    .main {
         background: #f5f1eb;
        background-attachment: fixed;
    }
            
           
    
    .block-container {
        padding: 2rem 2rem 3rem 2rem;
        max-width: 1600px;
    }
    
    /* ===== HEADER STYLE GOOGLE ===== */
    .search-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .logo {
        font-family: 'Product Sans', sans-serif;
        font-size: 3.5rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        letter-spacing: -1px;
    }
    
    .logo-medical {
        color: #1a73e8;
    }
    
    .logo-search {
        color: #ea4335;
    }
    
    .tagline {
        color: #5f6368;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    /* ===== SEARCH BOX GOOGLE STYLE ===== */
    .search-container {
        max-width: 650px;
        margin: 0 auto 2rem auto;
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #dfe1e5;
        border-radius: 24px;
        padding: 0.75rem 3rem 0.75rem 3rem;
        font-size: 1rem;
        box-shadow: 0 1px 6px rgba(32,33,36,.28);
        transition: box-shadow 0.2s;
    }
    
    .stTextInput > div > div > input:hover {
        box-shadow: 0 1px 6px rgba(32,33,36,.40);
        border-color: rgba(223,225,229,0);
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 1px 6px rgba(32,33,36,.40);
        border-color: rgba(223,225,229,0);
        outline: none;
    }
    
    /* Search icon */
    .stTextInput > label {
        display: none;
    }
    
    /* ===== BUTTONS GOOGLE STYLE ===== */
    .search-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1.5rem 0 2rem 0;
        flex-wrap: wrap;
    }
    
    .stButton > button {
        background-color: #f8f9fa;
        border: 1px solid #f8f9fa;
        border-radius: 4px;
        color: #3c4043;
        font-size: 0.8125rem;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        transition: all 0.1s;
        box-shadow: none;
        min-width: 160px;
    }
    
    .stButton > button:hover {
        box-shadow: 0 1px 1px rgba(0,0,0,.1);
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        color: #202124;
    }
    
    .stButton > button:active {
        background-color: #f1f3f4;
    }
    
    
    /* ===== MAIN SEARCH AREA ===== */
    .search-section {
        background: white;
        border-radius: 20px;
        padding: 3rem 3rem 2.5rem 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .search-title {
        font-size: 2rem;
        font-weight: 500;
        color: #2c3e50;
        margin-bottom: 0.75rem;
        line-height: 1.3;
    }
    
    /* ===== SEARCH INPUT ===== */
    .search-container {
        max-width: 800px;
        margin: 2rem auto 2.5rem auto;
        position: relative;
    }
    
    .stTextInput > div > div > input {
        border: 2px solid #e8ecef;
        border-radius: 50px;
        padding: 1rem 4rem 1rem 3.5rem;
        font-size: 1rem;
        background: #f8f9fa;
        transition: all 0.3s ease;
        box-shadow: none;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: #3498db;
        background: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3498db;
        background: white;
        box-shadow: 0 4px 12px rgba(52,152,219,0.15);
        outline: none;
    }
    
    .stTextInput label {
        display: none;
    }
    
    /* ===== MODE BUTTONS ===== */
    .mode-buttons-container {
        display: flex;
        gap: 0.875rem;
        justify-content: center;
        max-width: 900px;
        margin: 0 auto 2rem auto;
        flex-wrap: wrap;
    }
    
    .stButton > button {
        background: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-size: 0.8125rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(52,152,219,0.2);
        min-width: 160px;
        line-height: 1.3;
    }
    
    .stButton > button:hover {
        background: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(52,152,219,0.3);
    }
    
    /* ===== STATS BAR ===== */
    .stats-container {
        display: flex;
        gap: 3rem;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #3498db;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* ===== COMPARISON BUTTONS ===== */
    .comparison-section {
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #ecf0f1;
    }
    
    .comparison-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .comparison-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }

    /* style comparison buttons by aria-label (global) to ensure coloring */
    button[aria-label="Comparer Rapide vs Médicale"],
    button[aria-label='Comparer Rapide vs Médicale'] {
        background: linear-gradient(90deg,#a855f7,#9333ea) !important;
        color: white !important;
        border: none !important;
        padding: 0.32rem 0.6rem !important;
        font-size: 0.72rem !important;
        min-width: 100px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(168,85,247,0.2) !important;
    }
    button[aria-label="Comparer Rapide vs Médicale"]:hover { transform: translateY(-2px) !important; }

    button[aria-label="Comparer Mots-Clés vs Médicale"],
    button[aria-label='Comparer Mots-Clés vs Médicale'] {
        background: linear-gradient(90deg,#06b6d4,#0891b2) !important;
        color: white !important;
        border: none !important;
        padding: 0.32rem 0.6rem !important;
        font-size: 0.72rem !important;
        min-width: 100px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(6,182,212,0.2) !important;
    }
    button[aria-label="Comparer Mots-Clés vs Médicale"]:hover { transform: translateY(-2px) !important; }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(200,200,200,0.4);
            min-width: 220px;
            max-width: 260px;
    }
    
    [data-testid="stSidebar"] > div:first-child {
            padding: 0.5rem 0.5rem;
    }
    
    .sidebar-section {
            background: transparent;
            border-radius: 4px;
            padding: 0.35rem 0.4rem;
            margin-bottom: 0.4rem;
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }
    
    .sidebar-content {
            font-size: 0.66rem;
            color: #334e57;
            line-height: 1.25;
    }
    
    .model-info {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: white;
        border-radius: 6px;
        margin: 0.375rem 0;
    }
    
    .model-dot {
            width: 5px;
            height: 5px;
        background: #3498db;
        border-radius: 50%;
    }
    
    .model-name {
        font-size: 0.75rem;
        font-weight: 500;
        color: #2c3e50;
    }
    
    .model-desc {
        font-size: 0.6875rem;
        color: #7f8c8d;
    }
    
    /* ===== RESULTS SECTION ===== */
    .results-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 2rem;
    }
    
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .results-info {
        font-size: 0.9rem;
        color: #5a6c7d;
    }
    
    .method-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* ===== RESULT CARD ===== */
    .result-card {
        background: #fafbfc;
        border: 1px solid #e8ecef;
        border-radius: 10px;
        padding: 0.7rem;
        margin: 0.4rem auto 0.6rem auto;
        max-width: 700px;
        transition: all 0.18s ease;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transform: scaleY(0);
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        border-color: #3498db;
        box-shadow: 0 4px 16px rgba(52,152,219,0.12);
        transform: translateY(-2px);
    }

    .result-card.overlap {
        border-color: #d2b48c;
        background: linear-gradient(180deg, #fffaf0, #fff8ec);
    }
    
    .overlap-badge {
        position: absolute;
        right: 10px;
        top: 10px;
        background: #ffd8a8;
        color: #3a2b12;
        padding: 0.15rem 0.4rem;
        border-radius: 5px;
        font-size: 0.65rem;
        font-weight: 600;
    }
    
    .result-card:hover::before {
        transform: scaleY(1);
    }
    
    .result-top {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .result-rank {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.95rem;
        flex-shrink: 0;
    }
    
    .result-meta-top {
        display: flex;
        gap: 1rem;
        align-items: center;
        flex-grow: 1;
    }
    
    .result-category {
        background: #e8f4fd;
        color: #2980b9;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .result-score-badge {
        margin-left: auto;
        background: #d5f4e6;
        color: #27ae60;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .result-question {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2c3e50;
        line-height: 1.3;
        margin: 0.4rem 0;
    }
    
    .result-answer {
        font-size: 0.82rem;
        color: #5a6c7d;
        line-height: 1.45;
        margin: 0.5rem 0;
    }
    
    .result-footer {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e8ecef;
        font-size: 0.8rem;
        color: #95a5a6;
    }
    
    .footer-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    /* ===== COMPARISON COLUMNS ===== */
    .comparison-wrapper {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
        justify-items: center;
    }
    
    .comparison-col {
        background: white;
        border-radius: 12px;
        padding: 0.9rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        max-width: 600px;
    }
    
    .comparison-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2c3e50;
        padding-bottom: 0.6rem;
        margin-bottom: 0.9rem;
        border-bottom: 2px solid #a855f7;
    }
    
    .comparison-header.secondary {
        border-bottom-color: #06b6d4;
    }
    
    /* ===== METRICS GRID ===== */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #e9eef2;
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
        transition: all 0.18s ease;
        max-width: 160px;
        margin: 0 auto;
    }
    
    .metric-card:hover {
        border-color: #20c997;
        box-shadow: 0 2px 8px rgba(32,204,151,0.1);
        transform: translateY(-1px);
    }
    
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a855f7 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.65rem;
        color: #7f8c8d;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        font-weight: 600;
    }
    
    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #ecf0f1;
    }
    
    .footer-text {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin: 0.25rem 0;
    }
    
    .footer-tech {
        font-size: 0.8rem;
        color: #95a5a6;
        margin-top: 0.5rem;
    }
    
    /* ===== STREAMLIT OVERRIDES ===== */
    .stSlider > div > div > div > div {
        background: #3498db;
    }
    
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 500;
        color: #2c3e50;
    }
    
    .streamlit-expanderHeader:hover {
        background: #ecf0f1;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .comparison-wrapper {
            grid-template-columns: 1fr;
        }
        
        .search-title {
            font-size: 1.5rem;
        }
        
        .mode-buttons-container {
            flex-direction: column;
        }
        
        .stats-container {
            flex-direction: column;
            gap: 1.5rem;
        }
    }
    
    /* ===== HIDE STREAMLIT ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ==================== CACHE FUNCTIONS ====================

@st.cache_resource
def load_models():
    model1 = SentenceTransformer(Model1Config.NAME)
    model2 = SentenceTransformer(Model2Config.NAME)
    return model1, model2


@st.cache_resource
def get_db_connection():
    try:
        if "database" in st.secrets:
            return psycopg2.connect(
                host=st.secrets["database"]["DB_HOST"],
                port=int(st.secrets["database"]["DB_PORT"]),
                dbname=st.secrets["database"]["DB_NAME"],
                user=st.secrets["database"]["DB_USER"],
                password=st.secrets["database"]["DB_PASSWORD"]
            )
    except:
        pass
    
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )


# ==================== SEARCH FUNCTIONS ====================

def semantic_search(query, model, table_name, top_k=5):
    start = time.time()
    embedding = model.encode(query, convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT id, question, answer, category, qtype,
               1 - (embedding <=> %s::vector) as similarity
        FROM {table_name}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (embedding.tolist(), embedding.tolist(), top_k))
    
    results = cursor.fetchall()
    cursor.close()
    
    return results, (time.time() - start) * 1000


def keyword_search(query, table_name, top_k=5):
    conn = get_db_connection()
    cursor = conn.cursor()
    start = time.time()
    
    cursor.execute(f"""
        SELECT id, question, answer, category, qtype,
               ts_rank(to_tsvector('english', combined_text),
                      plainto_tsquery('english', %s)) as rank
        FROM {table_name}
        WHERE to_tsvector('english', combined_text) @@ 
              plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s;
    """, (query, query, top_k))
    
    results = cursor.fetchall()
    cursor.close()
    
    return results, (time.time() - start) * 1000


def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {Model1Config.TABLE_NAME};")
    total = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(DISTINCT category) FROM {Model1Config.TABLE_NAME};")
    categories = cursor.fetchone()[0]
    cursor.close()
    return total, categories


# ==================== DISPLAY FUNCTIONS ====================

def display_result(result, index, search_type="semantic", highlight=False):
    doc_id, question, answer, category, qtype, score = result
    
    answer_preview = answer[:280] + "..." if len(answer) > 280 else answer
    score_display = f"{score:.0%}" if search_type == "semantic" else f"{score:.3f}"
    
    card_class = 'result-card'
    badge = ''
    if highlight:
        card_class += ' overlap'
        badge = '<div class="overlap-badge">Chevauchement</div>'

    st.markdown(
        f'<div class="{card_class}">'
        f'{badge}'
        f'<div class="result-top">'
        f'<div class="result-rank">{index}</div>'
        f'<div class="result-meta-top">'
        f'<div class="result-category">{category}</div>'
        f'<div class="result-score-badge">{score_display}</div>'
        f'</div></div>'
        f'<div class="result-question">{question}</div>'
        f'<div class="result-answer">{answer_preview}</div>'
        f'<div class="result-footer">'
        f'<div class="footer-item">Type: {qtype}</div>'
        f'<div class="footer-item">ID: {doc_id}</div>'
        f'</div></div>',
        unsafe_allow_html=True)
    
    if len(answer) > 280:
        with st.expander("Read full answer"):
            st.write(answer)


def create_performance_chart(data_dict):
    fig = go.Figure()
    
    colors = {
        'Fast': '#667eea',
        'Medical': '#764ba2',
        'Keyword': '#e74c3c'
    }
    
    for name, value in data_dict.items():
        fig.add_trace(go.Bar(
            name=name,
            x=['Time'],
            y=[value],
            marker_color=colors.get(name, '#3498db'),
            text=[f"{value:.0f}ms"],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Performance Comparison',
        barmode='group',
        height=350,
        template='plotly_white',
        font=dict(family="Inter", size=12),
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig


def create_similarity_chart(results1, results2, label1, label2):
    fig = go.Figure()
    
    indices = list(range(1, len(results1) + 1))
    
    fig.add_trace(go.Scatter(
        x=indices,
        y=[r[5] for r in results1],
        mode='lines+markers',
        name=label1,
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=indices,
        y=[r[5] for r in results2],
        mode='lines+markers',
        name=label2,
        line=dict(color='#764ba2', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Similarity Scores',
        xaxis_title='Rank',
        yaxis_title='Score',
        height=350,
        template='plotly_white',
        font=dict(family="Inter", size=12),
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig


# ==================== MAIN APP ====================

def main():
    
    # ===== HEADER GOOGLE STYLE =====
    st.markdown("""
    <div class="search-header">
        <div class="logo">
            <span class="logo-medical">Medical</span><span class="logo-search">Search</span>
        </div>
        <div class="tagline">16,407 verified medical Q&A from NIH, CDC & FDA</div>
                
    </div>
    """, unsafe_allow_html=True)
    
  
    # Sidebar
    with st.sidebar:
        total_docs, total_cats = get_stats()
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">Dataset Oeuf</div>
            <div class="sidebar-content">
                <div style="font-size: 1.5rem; font-weight: 600; color: #2c3e50;">{:,}</div>
                <div style="font-size: 0.75rem; color: #7f8c8d; margin-bottom: 0.75rem;">Documents</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #2c3e50;">{}</div>
                <div style="font-size: 0.75rem; color: #7f8c8d;">Catégories</div>
                <div style="margin-top: 0.75rem; font-size: 0.6875rem; color: #95a5a6;">
                    Source vérifiée de documents médicaux
                </div>
            </div>
        </div>
        """.format(total_docs, total_cats), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">Models</div>
            <div class="model-info">
                <div class="model-dot"></div>
                <div>
                    <div class="model-name">Rapide:</div>
                    <div class="model-desc">MiniLM-L6-v2 (384D)</div>
                </div>
            </div>
            <div class="model-info">
                <div class="model-dot" style="background: #9b59b6;"></div>
                <div>
                    <div class="model-name">Médical:</div>
                    <div class="model-desc">PubMedBERT (768D)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">Search Config</div>
        </div>
        """, unsafe_allow_html=True)
        
        top_k = st.slider("Nombre de résultats", 1, 10, 5)
    
    
    # Search input
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        query = st.text_input(
            "search",
            placeholder="quels sont les symptômes de la maladie cardiaque",
            label_visibility="collapsed"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mode selection
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Recherche Sémantique\nMiniLM-L6-v2 (384D)"):
            st.session_state.mode = 'fast'
    
    with col2:
        if st.button("Recherche Médicale IA\nPubMedBert (768D)"):
            st.session_state.mode = 'medical'
    
    with col3:
        if st.button("Recherche par Mots-Clés\nFull-Text"):
            st.session_state.mode = 'keyword'
    
    
    
    # Comparison section
    st.markdown("""
    <div class="comparison-section">
        <div class="comparison-title">Comparer les modèles</div>
        <div class="comparison-buttons">
    """, unsafe_allow_html=True)
    
    # center the two comparison buttons using an outer three-column layout
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        inner1, inner2 = st.columns([1, 1])
        with inner1:
            if st.button("Comparer Rapide vs Médicale"):
                st.session_state.mode = 'compare_semantic'
        with inner2:
            if st.button("Comparer Mots-Clés vs Médicale"):
                st.session_state.mode = 'compare_keyword'
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Execute search
    if query and st.session_state.mode:
        mode = st.session_state.mode
        
        # Fast semantic
        if mode == 'fast':
            with st.spinner("Loading AI model..."):
                model1, model2 = load_models()
            
            results, search_time = semantic_search(query, model1, Model1Config.TABLE_NAME, top_k)
            
            if results:
                avg_score = np.mean([r[5] for r in results])
                avg_pct = f"{avg_score*100:.1f}%"
                st.markdown(
                    '<div class="results-container">'
                    '<div class="results-header">'
                    f'<div class="results-info">{len(results)} results ({search_time:.0f}ms) • Avg: {avg_pct}</div>'
                    '<div class="method-badge">Fast Semantic</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                
                for i, result in enumerate(results, 1):
                    display_result(result, i, "semantic")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No results found")
        
        # Medical semantic
        elif mode == 'medical':
            with st.spinner("Loading medical AI..."):
                model1, model2 = load_models()
            
            results, search_time = semantic_search(query, model2, Model2Config.TABLE_NAME, top_k)
            
            if results:
                avg_score = np.mean([r[5] for r in results])
                avg_pct = f"{avg_score*100:.1f}%"
                st.markdown(
                    '<div class="results-container">'
                    '<div class="results-header">'
                    f'<div class="results-info">{len(results)} results ({search_time:.0f}ms) • Avg: {avg_pct}</div>'
                    '<div class="method-badge">Medical AI</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                
                for i, result in enumerate(results, 1):
                    display_result(result, i, "semantic")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No results found")
        
        # Keyword search
        elif mode == 'keyword':
            results, search_time = keyword_search(query, Model1Config.TABLE_NAME, top_k)
            
            if results:
                avg_rank = np.mean([r[5] for r in results])
                avg_rank_str = f"{avg_rank:.3f}"
                st.markdown(
                    '<div class="results-container">'
                    '<div class="results-header">'
                    f'<div class="results-info">{len(results)} results ({search_time:.0f}ms) • Avg rank: {avg_rank_str}</div>'
                    '<div class="method-badge">Keyword Search</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                
                for i, result in enumerate(results, 1):
                    display_result(result, i, "keyword")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No keyword results found")
        
        # Compare semantic
        elif mode == 'compare_semantic':
            with st.spinner("Comparing models..."):
                model1, model2 = load_models()
                results1, time1 = semantic_search(query, model1, Model1Config.TABLE_NAME, top_k)
                results2, time2 = semantic_search(query, model2, Model2Config.TABLE_NAME, top_k)
            
            # Metrics + chevauchement
            ids1 = {r[0] for r in results1} if results1 else set()
            ids2 = {r[0] for r in results2} if results2 else set()
            overlap = ids1 & ids2
            overlap_count = len(overlap)
            overlap_pct = (overlap_count / top_k) if top_k else 0

            st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{time1:.0f}ms</div>
                    <div class="metric-label">Fast Time</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{time2:.0f}ms</div>
                    <div class="metric-label">Medical Time</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                avg1 = np.mean([r[5] for r in results1]) if results1 else 0
                avg1_pct = f"{avg1*100:.0f}%"
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{avg1_pct}</div>'
                    '<div class="metric-label">Fast Score</div>'
                    '</div>', unsafe_allow_html=True)

            with col4:
                avg2 = np.mean([r[5] for r in results2]) if results2 else 0
                avg2_pct = f"{avg2*100:.0f}%"
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{avg2_pct}</div>'
                    '<div class="metric-label">Medical Score</div>'
                    '</div>', unsafe_allow_html=True)

            with col5:
                overlap_pct_str = f"{overlap_pct*100:.0f}%"
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{overlap_count} ({overlap_pct_str})</div>'
                    '<div class="metric-label">Chevauchement</div>'
                    '</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_performance_chart({'Fast': time1, 'Medical': time2})
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if results1 and results2:
                    fig2 = create_similarity_chart(results1, results2, 'Fast', 'Medical')
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Side by side
            st.markdown('<div class="comparison-wrapper">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="comparison-col">', unsafe_allow_html=True)
                st.markdown('<div class="comparison-header">Fast Semantic</div>', unsafe_allow_html=True)
                if results1:
                    for i, result in enumerate(results1, 1):
                        is_overlap = result[0] in overlap
                        display_result(result, i, "semantic", highlight=is_overlap)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="comparison-col">', unsafe_allow_html=True)
                st.markdown('<div class="comparison-header secondary">Medical Semantic</div>', unsafe_allow_html=True)
                if results2:
                    for i, result in enumerate(results2, 1):
                        is_overlap = result[0] in overlap
                        display_result(result, i, "semantic", highlight=is_overlap)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Compare keyword vs medical
        elif mode == 'compare_keyword':
            with st.spinner("Comparing..."):
                model1, model2 = load_models()
                results_kw, time_kw = keyword_search(query, Model1Config.TABLE_NAME, top_k)
                results_med, time_med = semantic_search(query, model2, Model2Config.TABLE_NAME, top_k)
            
            # Metrics + chevauchement
            ids_kw = {r[0] for r in results_kw} if results_kw else set()
            ids_med = {r[0] for r in results_med} if results_med else set()
            overlap = ids_kw & ids_med
            overlap_count = len(overlap)
            overlap_pct = (overlap_count / top_k) if top_k else 0

            st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{time_kw:.0f}ms</div>
                    <div class="metric-label">Keyword Time</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{time_med:.0f}ms</div>
                    <div class="metric-label">Medical Time</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                avg_kw = np.mean([r[5] for r in results_kw]) if results_kw else 0
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{avg_kw:.3f}</div>'
                    '<div class="metric-label">Keyword Rank</div>'
                    '</div>', unsafe_allow_html=True)

            with col4:
                avg_med = np.mean([r[5] for r in results_med]) if results_med else 0
                avg_med_pct = f"{avg_med*100:.0f}%"
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{avg_med_pct}</div>'
                    '<div class="metric-label">Medical Score</div>'
                    '</div>', unsafe_allow_html=True)

            with col5:
                overlap_pct_str = f"{overlap_pct*100:.0f}%"
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{overlap_count} ({overlap_pct_str})</div>'
                    '<div class="metric-label">Chevauchement</div>'
                    '</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
            
            # Chart
            fig = create_performance_chart({'Keyword': time_kw, 'Medical': time_med})
            st.plotly_chart(fig, use_container_width=True)
            
            # Side by side
            st.markdown('<div class="comparison-wrapper">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="comparison-col">', unsafe_allow_html=True)
                st.markdown('<div class="comparison-header secondary">Keyword Search</div>', unsafe_allow_html=True)
                if results_kw:
                    for i, result in enumerate(results_kw, 1):
                        is_overlap = result[0] in overlap
                        display_result(result, i, "keyword", highlight=is_overlap)
                else:
                    st.info("No keyword results")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="comparison-col">', unsafe_allow_html=True)
                st.markdown('<div class="comparison-header">Medical Semantic</div>', unsafe_allow_html=True)
                if results_med:
                    for i, result in enumerate(results_med, 1):
                        is_overlap = result[0] in overlap
                        display_result(result, i, "semantic", highlight=is_overlap)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="app-footer">
        <div class="footer-text">Développé par Sarra BOUCHAMA • ESI 2025-2026</div>
        <div class="footer-tech">PostgreSQL • pgvector • Sentence Transformers • Streamlit</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()