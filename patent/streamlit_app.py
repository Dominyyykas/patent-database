import streamlit as st
from datetime import datetime
import csv
from io import StringIO
import time

from src.core.patent_engine import (
    PatentChatbot,
    analyze_patent_impact,
    generate_article_titles,
    generate_article_angles
)
from src.utils.token_tracker import token_tracker
from src.utils.function_cache import journalist_cache, chat_cache
from src.utils.rate_limiter import rate_limiter

# Page configuration
st.set_page_config(
    page_title="Patent RAG Chatbot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS for clarity
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .patent-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 1.08rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .journalist-section {
        margin-top: 1.5rem;
        padding: 1.2rem;
        background: #f7faff;
        border-radius: 10px;
        border: 1px solid #e0e6ef;
    }
    .journalist-button {
        margin: 0.5rem 0.5rem 0.5rem 0;
        font-size: 1.1rem !important;
        padding: 0.7rem 1.5rem !important;
        border-radius: 8px !important;
    }
    .token-usage-card {
        background-color: #f0f8ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .cost-highlight {
        color: #d63384;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = PatentChatbot()
    st.session_state.conversation_history = []
    st.session_state.current_patent = None
    # Ensure token tracker is initialized
    if not hasattr(token_tracker, '_initialized'):
        token_tracker._initialized = True

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Patent RAG Chatbot", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📤 Export", unsafe_allow_html=True)
    export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"patent_conversation_{export_timestamp}.csv"
    def get_export_csv():
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "role", "content"])
        for msg in st.session_state.conversation_history:
            writer.writerow([msg.get("timestamp", ""), msg.get("role", ""), msg.get("content", "")])
        return output.getvalue()
    st.download_button(
        label="Download as CSV",
        data=get_export_csv() if st.session_state.conversation_history else "",
        file_name=csv_filename,
        mime="text/csv",
        disabled=not st.session_state.conversation_history
    )
    # Log viewing
    if st.button("View Log"):
        log_path = "logs/patent_chatbot.log"
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                last_lines = lines[-100:] if len(lines) > 100 else lines
                log_content = "".join(last_lines)
        except FileNotFoundError:
            log_content = "Log file not found. No logs available yet."
        except Exception as e:
            log_content = f"Error reading log file: {e}"
        with st.expander("Patent Chatbot Log (last 100 lines)", expanded=True):
            st.text(log_content)
    

    
    # Token Usage and Cost Tracking
    usage_summary = token_tracker.get_session_summary()
    st.markdown(f"""
    <div class="token-usage-card">
        <strong>💰 Token Usage & Costs</strong><br/>
        Total Tokens: {usage_summary['total_tokens']:,}<br/>
        <span class="cost-highlight">Total Cost: {usage_summary['total_cost_formatted']}</span>
    </div>
    """, unsafe_allow_html=True)
    


# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h1 class="main-header">Patent RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### 💬 Chat Interface")
    user_input = st.chat_input("Ask about patents...")
    if user_input:
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create progress bar for patent search
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update progress and status
            progress_bar.progress(25)
            status_text.text("🔍 Searching patent database...")
            
            result = st.session_state.chatbot.chat(user_input)
            
            # Update progress
            progress_bar.progress(75)
            status_text.text("📝 Generating response...")
            
            response = result["response"]
            patents = result.get("patents", [])
            if patents:
                # Use the first (most relevant) patent for display
                st.session_state.current_patent = {
                    'title': patents[0]['metadata'].get('title', 'Unknown'),
                    'abstract': patents[0]['content'],
                    'patent_id': patents[0]['metadata'].get('patent_id', 'unknown')
                }
            
            # Complete progress
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Clear progress indicators after a short delay
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            # Clear progress on error
            progress_bar.empty()
            status_text.empty()
            
            error_msg = f"Error: {str(e)}"
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().isoformat()
            })
    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-message bot-message"><strong>Bot:</strong></div>',
                unsafe_allow_html=True
            )
            st.markdown(message["content"], unsafe_allow_html=True)

with col2:
    st.markdown("### <span style='font-size:1.3rem;'>📄 Current Patent</span>", unsafe_allow_html=True)
    if st.session_state.current_patent:
        patent = st.session_state.current_patent
        st.markdown(f"""
        <div class="patent-card">
            <span style='font-weight:600;'>Abstract:</span><br/>
            <span>{patent.get('abstract', 'No abstract available')}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='journalist-section'>", unsafe_allow_html=True)
        st.markdown("### <span style='font-size:1.2rem;'>📝 Journalist Functions</span>", unsafe_allow_html=True)
        col_impact, col_title, col_angles = st.columns(3)
        with col_impact:
            if st.button("Analyze Impact", key="impact_btn", help="Analyze patent impact", use_container_width=True):
                # Create progress bar for impact analysis
                impact_progress = st.progress(0)
                impact_status = st.empty()
                
                try:
                    # Update progress
                    impact_progress.progress(25)
                    impact_status.text("🔍 Analyzing patent impact...")
                    
                    impact_result = analyze_patent_impact(patent.get('abstract', ''), patent.get('patent_id', 'unknown'))
                    
                    # Update progress
                    impact_progress.progress(75)
                    impact_status.text("📊 Processing analysis results...")
                    
                    # Complete progress
                    impact_progress.progress(100)
                    impact_status.text("✅ Impact analysis completed!")
                    
                    st.success("Impact analysis completed!")
                    if isinstance(impact_result, dict):
                        st.markdown("**Impact Summary:**")
                        st.markdown(impact_result.get("impact_summary", "N/A"))
                        st.markdown("**Affected Industries:**")
                        st.markdown(", ".join(impact_result.get("affected_industries", [])))
                        st.markdown("**Predicted Timeline:**")
                        st.markdown(impact_result.get("predicted_timeline", "N/A"))
                    else:
                        st.error("Could not analyze impact.")
                    
                    # Clear progress indicators
                    time.sleep(0.5)
                    impact_progress.empty()
                    impact_status.empty()
                    
                except Exception as e:
                    # Clear progress on error
                    impact_progress.empty()
                    impact_status.empty()
                    st.error(f"Error analyzing impact: {str(e)}")
        with col_title:
            if st.button("Generate Title", key="title_btn", help="Generate article titles", use_container_width=True):
                # Create progress bar for title generation
                title_progress = st.progress(0)
                title_status = st.empty()
                
                try:
                    # Update progress
                    title_progress.progress(25)
                    title_status.text("✍️ Generating article titles...")
                    
                    title_result = generate_article_titles(patent.get('abstract', ''), patent.get('patent_id', 'unknown'))
                    
                    # Update progress
                    title_progress.progress(75)
                    title_status.text("📝 Processing title suggestions...")
                    
                    # Complete progress
                    title_progress.progress(100)
                    title_status.text("✅ Titles generated!")
                    
                    st.success("Title generated!")
                    if isinstance(title_result, list):
                        st.markdown("**Suggested Article Titles:**")
                        for i, title in enumerate(title_result, 1):
                            st.markdown(f"{i}. {title}")
                    else:
                        st.error("Could not generate article titles.")
                    
                    # Clear progress indicators
                    time.sleep(0.5)
                    title_progress.empty()
                    title_status.empty()
                    
                except Exception as e:
                    # Clear progress on error
                    title_progress.empty()
                    title_status.empty()
                    st.error(f"Error generating title: {str(e)}")
        with col_angles:
            if st.button("Generate Angles", key="angles_btn", help="Generate article angles", use_container_width=True):
                # Create progress bar for angles generation
                angles_progress = st.progress(0)
                angles_status = st.empty()
                
                try:
                    # Update progress
                    angles_progress.progress(25)
                    angles_status.text("🎯 Generating article angles...")
                    
                    angles_result = generate_article_angles(patent.get('abstract', ''), patent.get('patent_id', 'unknown'))
                    
                    # Update progress
                    angles_progress.progress(75)
                    angles_status.text("📊 Processing angle analysis...")
                    
                    # Complete progress
                    angles_progress.progress(100)
                    angles_status.text("✅ Angles generated!")
                    
                    st.success("Angles generated!")
                    if isinstance(angles_result, dict):
                        st.markdown("**Implementation Timeline:**")
                        st.markdown(angles_result.get("implementation_timeline", "N/A"))
                        st.markdown("**Market Disruption Assessment:**")
                        st.markdown(angles_result.get("market_disruption_assessment", "N/A"))
                        st.markdown("**Widespread Adoption Likelihood:**")
                        wal = angles_result.get("widespread_adoption_likelihood", "N/A")
                        if isinstance(wal, dict):
                            st.markdown(f"- **Likelihood:** {wal.get('likelihood', 'N/A')}")
                            st.markdown(f"- **Explanation:** {wal.get('explanation', 'N/A')}")
                            st.markdown(f"- **Confidence Level:** {wal.get('confidence_level', 'N/A')}")
                        else:
                            st.markdown(wal)
                    else:
                        st.error("Could not generate article angles.")
                    
                    # Clear progress indicators
                    time.sleep(0.5)
                    angles_progress.empty()
                    angles_status.empty()
                    
                except Exception as e:
                    # Clear progress on error
                    angles_progress.empty()
                    angles_status.empty()
                    st.error(f"Error generating angles: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No patent selected. Start a conversation to see patents here!")