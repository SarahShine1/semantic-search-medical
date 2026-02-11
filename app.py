"""
Professional Medical Search Interface - Refined Design
Clean, compact, and professional medical search system
"""
import streamlit as st
import time
import numpy as np
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from config import Config, Model1Config, Model2Config
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Medical Search System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refined professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #fafafa;
    }
    
    /* Compact header */
    .app-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    .app-header h1 {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.3px;
    }
    
    .app-header p {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }
    
    /* Compact search section */
    .search-section {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .search-section h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 0.75rem 0;
    }
    
    .stTextInput input {
        border-radius: 6px;
        border: 1.5px solid #e2e8f0;
        padding: 0.6rem 1rem;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    
    .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Compact buttons */
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #2563eb;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Compact result cards */
    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s;
    }
    
    .result-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .result-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #f1f5f9;
        color: #475569;
        width: 24px;
        height: 24px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.65rem;
        flex-shrink: 0;
    }
    
    .result-question {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.4;
        margin: 0.4rem 0;
    }
    
    .result-answer {
        color: #64748b;
        line-height: 1.6;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .result-meta {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid #f1f5f9;
        flex-wrap: wrap;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .badge-category {
        background-color: #dbeafe;
        color: #1e40af;
    }
    
    .badge-score {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .badge-type {
        background-color: #f1f5f9;
        color: #64748b;
    }
    
    /* Compact stats */
    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.9rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 0.4rem;
        font-weight: 500;
    }
    
    /* Compact section headers */
    .section-header {
        background: #f8fafc;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 1rem 0 0.75rem 0;
        border-left: 3px solid #3b82f6;
    }
    
    .section-header h3 {
        margin: 0;
        color: #0f172a;
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Compact sidebar */
    .sidebar-section {
        margin-bottom: 1.25rem;
    }
    
    .sidebar-section h4 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .info-box h5 {
        color: #0f172a;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
    }
    
    .info-box p {
        color: #64748b;
        font-size: 0.8rem;
        line-height: 1.5;
        margin: 0;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.8rem;
    }
    
    .info-row:last-child {
        border-bottom: none;
    }
    
    .info-label {
        color: #64748b;
        font-weight: 500;
    }
    
    .info-value {
        color: #0f172a;
        font-weight: 600;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .app-footer p {
        margin: 0.25rem 0;
    }
    
    .credit {
        color: #64748b;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Reduce spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    /* Compact expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load both semantic models"""
    model1 = SentenceTransformer(Model1Config.NAME)
    model2 = SentenceTransformer(Model2Config.NAME)
    return model1, model2


@st.cache_resource
def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )


def get_database_stats():
    """Get database statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    for model_config in [Model1Config, Model2Config]:
        table_name = model_config.TABLE_NAME
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        total = cursor.fetchone()[0]
        
        cursor.execute(f"""
            SELECT category, COUNT(*) 
            FROM {table_name}
            GROUP BY category 
            ORDER BY COUNT(*) DESC;
        """)
        categories = dict(cursor.fetchall())
        
        stats[table_name] = {
            'total': total,
            'categories': categories
        }
    
    cursor.close()
    return stats


def semantic_search(query, model, table_name, top_k=5, category_filter=None):
    """Perform semantic search"""
    start_time = time.time()
    embedding = model.encode(query, convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding)
    encode_time = time.time() - start_time
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    search_start = time.time()
    
    if category_filter and category_filter != "All Categories":
        sql = f"""
            SELECT id, question, answer, category, qtype,
                   1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            WHERE category = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (embedding.tolist(), category_filter, embedding.tolist(), top_k)
    else:
        sql = f"""
            SELECT id, question, answer, category, qtype,
                   1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (embedding.tolist(), embedding.tolist(), top_k)
    
    cursor.execute(sql, params)
    results = cursor.fetchall()
    search_time = time.time() - search_start
    
    cursor.close()
    
    return results, {
        'encode_time': encode_time * 1000,
        'search_time': search_time * 1000,
        'total_time': (encode_time + search_time) * 1000
    }


def display_result(result, index):
    """Display a compact search result"""
    doc_id, question, answer, category, qtype, similarity = result
    
    answer_preview = answer[:280] + "..." if len(answer) > 280 else answer
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; align-items: start;">
            <span class="result-number">{index}</span>
            <div style="flex: 1;">
                <div class="result-question">{question}</div>
                <div class="result-answer">{answer_preview}</div>
                <div class="result-meta">
                    <span class="badge badge-category">{category}</span>
                    <span class="badge badge-score">{similarity:.1%}</span>
                    <span class="badge badge-type">{qtype}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if len(answer) > 280:
        with st.expander("Read full answer"):
            st.write(answer)
            st.caption(f"Document ID: {doc_id}")


def create_comparison_chart(results1, results2, time1, time2):
    """Create compact comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Fast Model',
        x=['Encode', 'Search', 'Total'],
        y=[time1['encode_time'], time1['search_time'], time1['total_time']],
        marker_color='#3b82f6',
        text=[f"{time1['encode_time']:.0f}", 
              f"{time1['search_time']:.0f}", 
              f"{time1['total_time']:.0f}"],
        textposition='outside',
        texttemplate='%{text}ms'
    ))
    
    fig.add_trace(go.Bar(
        name='Medical Model',
        x=['Encode', 'Search', 'Total'],
        y=[time2['encode_time'], time2['search_time'], time2['total_time']],
        marker_color='#8b5cf6',
        text=[f"{time2['encode_time']:.0f}", 
              f"{time2['search_time']:.0f}", 
              f"{time2['total_time']:.0f}"],
        textposition='outside',
        texttemplate='%{text}ms'
    ))
    
    fig.update_layout(
        title={
            'text': 'Performance Comparison',
            'font': {'size': 14, 'family': 'Inter'}
        },
        barmode='group',
        height=300,
        template='plotly_white',
        showlegend=True,
        margin=dict(t=40, b=40, l=40, r=40),
        font=dict(family="Inter, sans-serif", size=11)
    )
    
    return fig


def main():
    # Compact header
    st.markdown("""
    <div class="app-header">
        <h1>Medical Knowledge Search System</h1>
        <p>AI-powered semantic search across medical Q&A database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h4>Search Configuration</h4></div>', 
                   unsafe_allow_html=True)
        
        search_mode = st.selectbox(
            "Search Mode",
            ["Fast Model", "Medical Model", "Compare Both"]
        )
        
        top_k = st.slider("Results", 1, 15, 5)
        
        db_stats = get_database_stats()
        first_table = list(db_stats.keys())[0]
        categories = ["All Categories"] + sorted(db_stats[first_table]['categories'].keys())
        category_filter = st.selectbox("Category", categories)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-section"><h4>Database Stats</h4></div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{db_stats[first_table]['total']:,}</div>
                <div class="stat-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(db_stats[first_table]['categories'])}</div>
                <div class="stat-label">Categories</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-section"><h4>Dataset Info</h4></div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h5>MedQuAD</h5>
            <p>Medical Q&A from NIH, CDC, FDA</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem;">
            <div class="info-row">
                <span class="info-label">Total Pairs</span>
                <span class="info-value">16,407</span>
            </div>
            <div class="info-row">
                <span class="info-label">Language</span>
                <span class="info-value">English</span>
            </div>
            <div class="info-row">
                <span class="info-label">Quality</span>
                <span class="info-value">Verified</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-section"><h4>Models</h4></div>', 
                   unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <h5>Fast Model</h5>
            <p>{Model1Config.NAME.split('/')[-1]}<br>
            Dim: {Model1Config.DIMENSIONS}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <h5>Medical Model</h5>
            <p>{Model2Config.NAME.split('/')[-1]}<br>
            Dim: {Model2Config.DIMENSIONS}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Search interface
    #st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown('<h3>Enter your medical question</h3>', unsafe_allow_html=True)
    query = st.text_input(
        "",
        placeholder="e.g., How to treat diabetes? What are symptoms of hypertension?",
        label_visibility="collapsed"
    )
    
    search_btn = st.button("Search", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if search_btn and query:
        
        with st.spinner("Loading models..."):
            model1, model2 = load_models()
        
        if search_mode == "Compare Both":
            st.markdown('<div class="section-header"><h3>Model Comparison</h3></div>', 
                       unsafe_allow_html=True)
            
            with st.spinner("Searching..."):
                results1, time1 = semantic_search(query, model1, Model1Config.TABLE_NAME, top_k, category_filter)
                results2, time2 = semantic_search(query, model2, Model2Config.TABLE_NAME, top_k, category_filter)
            
            # Compact metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{time1['total_time']:.0f}ms</div>
                    <div class="stat-label">Fast Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{time2['total_time']:.0f}ms</div>
                    <div class="stat-label">Medical Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg1 = np.mean([r[5] for r in results1]) if results1 else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{avg1:.1%}</div>
                    <div class="stat-label">Fast Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg2 = np.mean([r[5] for r in results2]) if results2 else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{avg2:.1%}</div>
                    <div class="stat-label">Medical Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Chart
            if results1 and results2:
                st.markdown("### Performance Analysis")
                fig = create_comparison_chart(results1, results2, time1, time2)
                st.plotly_chart(fig, use_container_width=True)
                
                # Overlap
                ids1 = {r[0] for r in results1}
                ids2 = {r[0] for r in results2}
                overlap = len(ids1 & ids2)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Common", f"{overlap}/{top_k}")
                with col2:
                    st.metric("Fast Only", len(ids1 - ids2))
                with col3:
                    st.metric("Medical Only", len(ids2 - ids1))
            
            # Results
            st.markdown("### Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="section-header"><h3>Fast Model</h3></div>', 
                           unsafe_allow_html=True)
                for i, result in enumerate(results1, 1):
                    display_result(result, i)
            
            with col2:
                st.markdown('<div class="section-header"><h3>Medical Model</h3></div>', 
                           unsafe_allow_html=True)
                for i, result in enumerate(results2, 1):
                    display_result(result, i)
        
        else:
            st.markdown('<div class="section-header"><h3>Search Results</h3></div>', 
                       unsafe_allow_html=True)
            
            if "Fast" in search_mode:
                model = model1
                table_name = Model1Config.TABLE_NAME
            else:
                model = model2
                table_name = Model2Config.TABLE_NAME
            
            with st.spinner("Searching..."):
                results, timing = semantic_search(query, model, table_name, top_k, category_filter)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{timing['total_time']:.0f}ms</div>
                    <div class="stat-label">Search Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{len(results)}</div>
                    <div class="stat-label">Results</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_score = np.mean([r[5] for r in results]) if results else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{avg_score:.1%}</div>
                    <div class="stat-label">Avg Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            if results:
                for i, result in enumerate(results, 1):
                    display_result(result, i)
            else:
                st.info("No results found. Try a different query or category.")
    
    elif search_btn:
        st.warning("Please enter a search query")
    
    # Footer with credit
    st.markdown("""
    <div class="app-footer">
        <p>Medical Knowledge Search System</p>
        <p>PostgreSQL + pgvector • Sentence Transformers</p>
        <p class="credit">Developed by Sarra</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()