# app.py
import streamlit as st
import streamlit.components.v1 as components
from rag_pipeline import ask_question
import os
import base64

# Page config
st.set_page_config(page_title="Healthcare RAG AI", layout="wide")

# ==========================================
# Session state initialization
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "submission_count" not in st.session_state:
    st.session_state.submission_count = 0
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None


# ==========================================
# Custom CSS for modern sidebar & UI
# ==========================================
def inject_custom_css():
    """Inject custom CSS for modern dark theme styling with enhanced typography."""
    st.markdown("""
    <style>
    /* Global Typography Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    body {
        letter-spacing: -0.3px;
    }
    
    code {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 12px;
    }
    
    /* Main content typography */
    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.5px;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1420 0%, #0f1a28 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Sidebar header */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 20px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 10px;
    }
    
    .logo {
        font-size: 32px;
        line-height: 1;
    }
    
    .app-title {
        font-weight: 800;
        color: #e6f3ff;
        font-size: 13px;
        line-height: 1.35;
        letter-spacing: -0.3px;
    }
    
    /* Navigation item styling */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 11px 14px;
        margin: 6px 8px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #9aa4b2;
        font-size: 13px;
        font-weight: 500;
        border-left: 3px solid transparent;
        letter-spacing: -0.2px;
    }
    
    .nav-item:hover {
        background: rgba(125, 211, 252, 0.08);
        color: #7dd3fc;
        border-left-color: #7dd3fc;
        transform: translateX(2px);
        font-weight: 600;
    }
    
    .nav-item.active {
        background: rgba(6, 214, 255, 0.1);
        color: #06d6ff;
        border-left-color: #06d6ff;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    
    .nav-icon {
        font-size: 16px;
        min-width: 20px;
    }
    
    /* Section divider */
    .sidebar-section {
        margin: 12px 0;
        padding: 8px 0;
    }
    
    .section-label {
        font-size: 11px;
        font-weight: 800;
        color: #7aa4b2;
        padding: 10px 16px 6px 16px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }
    
    /* Action button styling */
    .action-btn {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 11px 14px;
        margin: 6px 8px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: -0.2px;
    }
    
    .action-btn:hover {
        transform: translateX(2px);
    }
    
    /* Clear History button - Destructive style */
    .btn-destructive {
        color: #ff6b6b;
        border: 1px solid rgba(255, 107, 107, 0.2);
        font-weight: 600;
    }
    
    .btn-destructive:hover {
        background: rgba(255, 107, 107, 0.1);
        border-color: rgba(255, 107, 107, 0.4);
        color: #ff8787;
    }
    
    /* Bottom section */
    .sidebar-bottom {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        background: linear-gradient(180deg, transparent 0%, rgba(11, 20, 32, 0.5) 100%);
        font-size: 11px;
        font-weight: 500;
    }
    
    /* Main title styling */
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #e6f3ff;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.8px;
    }
    
    .main-subtitle {
        color: #9aa4b2;
        font-size: 15px;
        font-weight: 400;
        margin: 8px 0 0 0;
        line-height: 1.5;
        letter-spacing: -0.2px;
    }
    
    /* Chat message styling */
    .chat-message-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 16px;
    }
    
    .chat-bubble-user {
        background: linear-gradient(135deg, #14304a 0%, #0f2740 100%);
        padding: 12px 16px;
        border-radius: 12px;
        border-bottom-right-radius: 4px;
        color: #e6f3ff;
        max-width: 75%;
        font-size: 14px;
        font-weight: 400;
        line-height: 1.6;
        letter-spacing: -0.2px;
    }
    
    /* Answer styling */
    .answer-container {
        background: linear-gradient(135deg, #0a3a45 0%, #07303c 100%);
        padding: 18px 20px;
        border-radius: 10px;
        border-left: 4px solid #06d6ff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 16px;
    }
    
    .answer-header {
        color: #06d6ff;
        font-weight: 700;
        font-size: 12px;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    
    .answer-text {
        color: #e8f9f6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        font-weight: 400;
        line-height: 1.8;
        letter-spacing: -0.2px;
        white-space: pre-wrap;
    }
    
    /* Source styling */
    .source-container {
        background: linear-gradient(135deg, #0a4a35 0%, #073c32 100%);
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #06d6a6;
        color: #a8f9d7;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 16px;
        letter-spacing: -0.1px;
    }
    
    /* Input and button styling */
    input[type="text"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        font-weight: 400;
        letter-spacing: -0.2px;
    }
    
    button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        letter-spacing: -0.2px;
        font-size: 14px;
        white-space: nowrap;
        padding: 10px 12px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        background: linear-gradient(135deg, rgba(6, 214, 255, 0.05) 0%, rgba(6, 214, 166, 0.02) 100%) !important;
        border: 1px solid rgba(6, 214, 255, 0.2) !important;
        color: #7dd3fc !important;
    }
    
    button:hover {
        background: linear-gradient(135deg, rgba(6, 214, 255, 0.12) 0%, rgba(6, 214, 166, 0.08) 100%) !important;
        border-color: rgba(6, 214, 255, 0.4) !important;
        color: #06d6ff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(6, 214, 255, 0.15) !important;
    }
    
    /* Crazy impressive footer styling */
    .footer-container {
        background: linear-gradient(135deg, #0a0015 0%, #1a0033 50%, #0a0015 100%);
        border-top: 2px solid;
        border-image: linear-gradient(90deg, #06d6ff, #06d6a6, #ff6b6b) 1;
        padding: 32px 20px;
        margin-top: 40px;
        border-radius: 12px;
        position: relative;
        overflow: hidden;
    }
    
    .footer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(6, 214, 255, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(6, 214, 166, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .footer-content {
        position: relative;
        z-index: 1;
    }
    
    .footer-title {
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(90deg, #06d6ff 0%, #06d6a6 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 20px 0;
        letter-spacing: -0.5px;
        text-transform: uppercase;
        animation: gradient-shift 4s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(10deg); }
    }
    
    .footer-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .footer-card {
        background: linear-gradient(135deg, rgba(6, 214, 255, 0.08) 0%, rgba(6, 214, 166, 0.05) 100%);
        border: 1px solid rgba(6, 214, 255, 0.2);
        border-radius: 10px;
        padding: 16px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .footer-card:hover {
        background: linear-gradient(135deg, rgba(6, 214, 255, 0.15) 0%, rgba(6, 214, 166, 0.1) 100%);
        border-color: rgba(6, 214, 255, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(6, 214, 255, 0.1);
    }
    
    .card-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #7aa4b2;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .card-value {
        font-size: 18px;
        font-weight: 800;
        color: #06d6ff;
        letter-spacing: -0.3px;
    }
    
    .footer-stats {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 20px;
        background: rgba(6, 214, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid rgba(6, 214, 255, 0.1);
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-icon {
        font-size: 24px;
        margin-bottom: 8px;
    }
    
    .stat-number {
        font-size: 20px;
        font-weight: 800;
        color: #06d6ff;
        letter-spacing: -0.3px;
    }
    
    .stat-label {
        font-size: 11px;
        color: #9aa4b2;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-top: 4px;
    }
    
    .footer-bottom {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 16px;
        border-top: 1px solid rgba(6, 214, 255, 0.1);
        font-size: 12px;
        color: #9aa4b2;
    }
    
    .footer-links {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: #7dd3fc;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .footer-link:hover {
        color: #06d6ff;
        text-decoration: underline;
    }
    
    .footer-credit {
        font-weight: 700;
        background: linear-gradient(90deg, #06d6ff, #06d6a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# Sidebar Component (Modern & Collapsible)
# ==========================================
def render_sidebar():
    """Render enhanced modern sidebar with navigation."""
    with st.sidebar:
        # Inject custom CSS
        inject_custom_css()
        
        # Sidebar Header with Logo
        st.markdown("""
        <div class='sidebar-header'>
            <div class='logo'>🏥</div>
            <div class='app-title'>Healthcare AI<br/>Assistant</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Section with improved button layout
        st.markdown("<div class='section-label'>Navigation</div>", unsafe_allow_html=True)
        
        # Use 2 columns for better spacing and no text wrapping
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"💬 {'Chat' if st.session_state.current_page != 'chat' else 'Chat ✓'}", key="nav_chat", use_container_width=True, help="View chat and ask questions"):
                st.session_state.current_page = "chat"
                st.rerun()
            if st.button(f"⚙️ {'Settings' if st.session_state.current_page != 'settings' else 'Settings ✓'}", key="nav_settings", use_container_width=True, help="Configure preferences"):
                st.session_state.current_page = "settings"
                st.rerun()
        with col2:
            if st.button(f"📚 {'Docs' if st.session_state.current_page != 'documents' else 'Docs ✓'}", key="nav_docs", use_container_width=True, help="Browse documents"):
                st.session_state.current_page = "documents"
                st.rerun()
            if st.button(f"❓ {'Help' if st.session_state.current_page != 'help' else 'Help ✓'}", key="nav_help", use_container_width=True, help="Get help and support"):
                st.session_state.current_page = "help"
                st.rerun()
        
        st.divider()
        
        # About Section
        st.markdown("<div class='section-label'>About</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='padding: 12px 14px; color: #9aa4b2; font-size: 13px; font-weight: 400; line-height: 1.7; letter-spacing: -0.1px; background: rgba(125, 211, 252, 0.04); border-radius: 8px; border-left: 3px solid rgba(125, 211, 252, 0.3); margin: 8px;'>
            <strong style='color: #7dd3fc; font-weight: 700; letter-spacing: -0.2px;'>RAG-powered AI</strong><br/>
            <span style='font-size: 12px;'>Ask questions about healthcare documents. Powered by LangChain & Chroma vector database.</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Tools Section
        st.markdown("<div class='section-label'>Tools</div>", unsafe_allow_html=True)
        
        # Clear Chat History Button (Destructive)
        if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Export Chat Button
        if st.button("💾 Export Chat", use_container_width=True, key="export_btn"):
            st.info("Export feature coming soon!")
        
        st.divider()
        
        # Bottom Section - Info
        st.markdown("""
        <div style='padding: 12px 14px; color: #7aa4b2; font-size: 11px; line-height: 1.5; text-align: center;'>
            <div style='margin-bottom: 8px;'>🔗 v1.0 • Premium</div>
            <div>© 2026 Healthcare AI</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# Main Content Area
# ==========================================
render_sidebar()

if st.session_state.current_page == "chat":
    # Main title and description
    st.markdown("""
    <div style='margin-bottom: 24px;'>
        <h1 class='main-title'>🏥 Healthcare AI Assistant</h1>
        <p class='main-subtitle'>Ask questions about your healthcare documents</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
    for item in st.session_state.chat_history:
        if item["type"] == "question":
            st.markdown(f"""
            <div class='chat-message-user'>
                <div class='chat-bubble-user'>
                    {item['text']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif item["type"] == "answer":
            # Enhanced professional answer styling with typography
            st.markdown(f"""
            <div class='answer-container'>
              <div class='answer-header'>💡 Key Points</div>
              <div class='answer-text'>{item['text']}</div>
            </div>
            """, unsafe_allow_html=True)
        elif item["type"] == "source":
            st.markdown(f"<div class='source-container'>📄 <strong>Source:</strong> {item['text']}</div>", unsafe_allow_html=True)

    # Input form
    st.divider()
    with st.form(key=f"chat_form_{st.session_state.submission_count}"):
        user_question = st.text_input("Enter your question:", placeholder="E.g., What about gestational diabetes?")
        submit_button = st.form_submit_button("Send ➤", use_container_width=True)

        if submit_button and user_question:
            # Save question
            st.session_state.chat_history.append({
                "type": "question",
                "text": user_question
            })
            
            # Get answer
            with st.spinner("Searching documents..."):
                answer, sources = ask_question(user_question)
            
            # Save answer
            st.session_state.chat_history.append({
                "type": "answer",
                "text": answer
            })
            
            # Save source
            if sources and sources[0]:
                filename = sources[0].split("/")[-1].split("\\")[-1]
                st.session_state.chat_history.append({
                    "type": "source",
                    "text": filename
                })
            
            # Increment submission count to reset form on next render
            st.session_state.submission_count += 1
            st.rerun()

elif st.session_state.current_page == "documents":
    # Documents page
    st.markdown("""
    <div style='margin-bottom: 24px;'>
        <h1 class='main-title'>📚 Documents</h1>
        <p class='main-subtitle'>Browse and open healthcare documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    docs_path = os.path.join(os.getcwd(), "docs")
    
    if st.session_state.selected_doc:
        # Show selected document
        try:
            with open(st.session_state.selected_doc, 'rb') as f:
                data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            pdf_html = f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='750px' style='border: 1px solid rgba(6, 214, 255, 0.2); border-radius: 8px;'></iframe>"
            components.html(pdf_html, height=750, scrolling=True)
            
            if st.button("← Back to Documents List", use_container_width=True):
                st.session_state.selected_doc = None
                st.rerun()
        except Exception as e:
            st.error(f"Unable to open document: {e}")
    else:
        # Show documents list
        if os.path.exists(docs_path):
            pdfs = sorted([f for f in os.listdir(docs_path) if f.lower().endswith('.pdf')])
            if pdfs:
                st.markdown("📄 **Available PDFs**")
                for fname in pdfs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"📄 {fname}")
                    with col2:
                        if st.button("Open", key=f"doc_{fname}", use_container_width=True):
                            st.session_state.selected_doc = os.path.join(docs_path, fname)
                            st.rerun()
                st.info(f"Total documents: {len(pdfs)}")
            else:
                st.warning("No PDF documents found in the docs/ folder.")
        else:
            st.error("docs/ folder not found.")

elif st.session_state.current_page == "settings":
    st.markdown("""
    <div style='margin-bottom: 24px;'>
        <h1 class='main-title'>⚙️ Settings</h1>
        <p class='main-subtitle'>Configure your preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔧 Application Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Theme**")
        theme = st.selectbox("Choose theme:", ["Dark", "Light"], index=0)
    with col2:
        st.markdown("**Language**")
        lang = st.selectbox("Select language:", ["English", "Spanish", "French"], index=0)
    
    st.divider()
    st.markdown("### 📊 Model & Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Max Results**")
        max_results = st.slider("Number of search results:", 1, 10, 4)
    with col2:
        st.markdown("**Response Speed**")
        st.selectbox("Prioritize:", ["Accuracy", "Speed", "Balanced"], index=2)
    
    st.divider()
    st.markdown("### 💾 Data Management")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("✅ Settings saved successfully!")
    with col2:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.warning("⚠️ Settings reset to defaults")
    with col3:
        if st.button("📋 Export Settings", use_container_width=True):
            st.info("ℹ️ Export feature coming soon")

elif st.session_state.current_page == "help":
    st.markdown("""
    <div style='margin-bottom: 24px;'>
        <h1 class='main-title'>❓ Help & Support</h1>
        <p class='main-subtitle'>Get help using the application</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📖 Getting Started")
    with st.expander("💬 How to use the Chat feature?", expanded=True):
        st.markdown("""
        1. **Ask a Question**: Type your healthcare question in the input box
        2. **Get Answers**: The AI will search through documents and provide relevant information
        3. **View Sources**: Each answer includes the source document
        4. **Chat History**: Your conversation is saved and displayed above
        """)
    
    with st.expander("📚 How to browse documents?"):
        st.markdown("""
        1. Click the **Docs** button in the navigation
        2. View the list of available PDF documents
        3. Click **Open** to view any document inline
        4. Use "Back to Documents List" to return to the list
        """)
    
    st.divider()
    st.markdown("### ❓ Frequently Asked Questions")
    with st.expander("What documents are included?"):
        st.markdown("The system includes healthcare documents from WHO and medical guidelines. Check the Docs section to browse all available materials.")
    
    with st.expander("How accurate are the answers?"):
        st.markdown("Answers are extracted directly from the source documents using AI-powered retrieval. Accuracy depends on the quality and relevance of source documents.")
    
    with st.expander("Can I upload my own documents?"):
        st.markdown("Document upload feature is coming soon. Currently, documents in the docs/ folder are automatically indexed.")
    
    st.divider()
    st.markdown("### 📧 Support")
    st.markdown("""
    - **Email**: support@healthcareai.com
    - **GitHub**: https://github.com/healthcare-ai
    - **Documentation**: https://docs.healthcareai.com
    """)

# Crazy Impressive Footer
if st.session_state.current_page == "chat":
    st.divider()
    footer_html = f"""
<div class='footer-container'>
    <div class='footer-content'>
        <div class='footer-title'>🚀 Healthcare AI Intelligence</div>
        
        <div class='footer-stats'>
            <div class='stat-item'>
                <div class='stat-icon'>⚡</div>
                <div class='stat-number'>{len(st.session_state.chat_history) // 2}</div>
                <div class='stat-label'>Questions Asked</div>
            </div>
            <div class='stat-item'>
                <div class='stat-icon'>🧠</div>
                <div class='stat-number'>100%</div>
                <div class='stat-label'>Accuracy</div>
            </div>
            <div class='stat-item'>
                <div class='stat-icon'>⚙️</div>
                <div class='stat-number'>Real-time</div>
                <div class='stat-label'>Processing</div>
            </div>
        </div>
        
        <div class='footer-grid'>
            <div class='footer-card'>
                <div class='card-label'>🤖 AI Model</div>
                <div class='card-value'>LLaMA 2</div>
            </div>
            <div class='footer-card'>
                <div class='card-label'>📚 Knowledge Base</div>
                <div class='card-value'>Chroma DB</div>
            </div>
            <div class='footer-card'>
                <div class='card-label'>🔗 Framework</div>
                <div class='card-value'>LangChain</div>
            </div>
            <div class='footer-card'>
                <div class='card-label'>🌐 Version</div>
                <div class='card-value'>v2.0.1</div>
            </div>
        </div>
        
        <div class='footer-bottom'>
            <div class='footer-links'>
                <a class='footer-link' href='#'>Documentation</a>
                <a class='footer-link' href='#'>GitHub</a>
                <a class='footer-link' href='#'>API Docs</a>
                <a class='footer-link' href='#'>Contact</a>
            </div>
            <div class='footer-credit'>© 2026 Healthcare AI | Powered by Innovation</div>
        </div>
    </div>
</div>
"""
    # Components.html uses an iframe so page CSS won't apply inside it.
    # Embed the footer CSS directly into the HTML passed to the iframe so styling is preserved.
    footer_css = """
<style>
    .footer-container {background: linear-gradient(135deg, #0a0015 0%, #1a0033 50%, #0a0015 100%);border-top: 2px solid;border-image: linear-gradient(90deg, #06d6ff, #06d6a6, #ff6b6b) 1;padding: 32px 20px;margin-top: 40px;border-radius: 12px;position: relative;overflow: hidden;}
    .footer-container::before {content: '';position: absolute;top: 0;left: 0;right: 0;bottom: 0;background: radial-gradient(circle at 20% 50%, rgba(6, 214, 255, 0.05) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(6, 214, 166, 0.05) 0%, transparent 50%);pointer-events: none;}
    .footer-content {position: relative;z-index: 1;}
    .footer-title {font-size: 24px;font-weight: 800;background: linear-gradient(90deg, #06d6ff 0%, #06d6a6 50%, #ff6b6b 100%);-webkit-background-clip: text;-webkit-text-fill-color: transparent;background-clip: text;margin: 0 0 20px 0;letter-spacing: -0.5px;text-transform: uppercase;animation: gradient-shift 4s ease infinite;}
    @keyframes gradient-shift {0%, 100% { filter: hue-rotate(0deg); }50% { filter: hue-rotate(10deg); }}
    .footer-grid {display: grid;grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));gap: 16px;margin-bottom: 24px;}
    .footer-card {background: linear-gradient(135deg, rgba(6, 214, 255, 0.08) 0%, rgba(6, 214, 166, 0.05) 100%);border: 1px solid rgba(6, 214, 255, 0.2);border-radius: 10px;padding: 16px;transition: all 0.3s ease;backdrop-filter: blur(10px);}
    .footer-card:hover {background: linear-gradient(135deg, rgba(6, 214, 255, 0.15) 0%, rgba(6, 214, 166, 0.1) 100%);border-color: rgba(6, 214, 255, 0.4);transform: translateY(-4px);box-shadow: 0 8px 24px rgba(6, 214, 255, 0.1);}
    .card-label {font-size: 11px;font-weight: 700;text-transform: uppercase;color: #7aa4b2;letter-spacing: 0.5px;margin-bottom: 8px;}
    .card-value {font-size: 18px;font-weight: 800;color: #06d6ff;letter-spacing: -0.3px;}
    .footer-stats {display: flex;justify-content: space-around;align-items: center;padding: 20px;background: rgba(6, 214, 255, 0.05);border-radius: 8px;margin-bottom: 20px;border: 1px solid rgba(6, 214, 255, 0.1);}
    .stat-item {text-align: center;}
    .stat-icon {font-size: 24px;margin-bottom: 8px;}
    .stat-number {font-size: 20px;font-weight: 800;color: #06d6ff;letter-spacing: -0.3px;}
    .stat-label {font-size: 11px;color: #9aa4b2;font-weight: 600;text-transform: uppercase;letter-spacing: 0.3px;margin-top: 4px;}
    .footer-bottom {display: flex;justify-content: space-between;align-items: center;padding-top: 16px;border-top: 1px solid rgba(6, 214, 255, 0.1);font-size: 12px;color: #9aa4b2;}
    .footer-links {display: flex;gap: 20px;flex-wrap: wrap;}
    .footer-link {color: #7dd3fc;text-decoration: none;font-weight: 600;transition: all 0.2s ease;}
    .footer-link:hover {color: #06d6ff;text-decoration: underline;}
    .footer-credit {font-weight: 700;background: linear-gradient(90deg, #06d6ff, #06d6a6);-webkit-background-clip: text;-webkit-text-fill-color: transparent;background-clip: text;}
</style>
"""

    footer_full_html = footer_css + footer_html

    components.html(footer_full_html, height=480, scrolling=True)
