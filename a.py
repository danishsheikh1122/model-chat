from groq import Groq
import streamlit as st
from dotenv import load_dotenv
import os
import shelve
import time
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phoenix-ai")

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "llama3-70b-8192"
MODELS = {
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Claude 3 Opus": "claude-3-opus-20240229"
}
DATA_STORE = "chat_history"

# Custom CSS for professional dark theme
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: #0A0A0A;
    color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #222222;
    width: 320px !important;
}

.stChatInput {
    position: fixed;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    width: calc(65% - 160px);
    background: #111111;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #222222;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

.stChatMessage {
    max-width: 75%;
    margin: 16px 0;
    border-radius: 14px;
    padding: 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

[data-testid="stChatMessageUser"] {
    background: #1A1A1A;
    margin-left: auto;
    margin-right: 0;
    border-radius: 14px 14px 0 14px;
    border: 1px solid #222222;
}

[data-testid="stChatMessageAssistant"] {
    background: #111111;
    margin-right: auto;
    margin-left: 0;
    border-radius: 14px 14px 14px 0;
    border: 1px solid #222222;
}

button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.sidebar-btn {
    margin: 4px 0 !important;
    padding: 0.6rem !important;
}

.stTitle {
    font-weight: 600 !important;
    font-size: 26px !important;
    color: #FFFFFF !important;
}

.stMarkdown h1 {
    border-bottom: 2px solid #222222;
    padding-bottom: 0.75rem;
    font-weight: 600;
}

.stMarkdown h3 {
    font-weight: 600;
    font-size: 16px;
    margin-top: 1.5rem;
}

/* Loading spinner style */
.loading-indicator {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 60px;
}

/* Improved styling for select boxes and inputs */
[data-testid="stSelectbox"] {
    background-color: #111111 !important;
    border-radius: 8px !important;
    border: 1px solid #222222 !important;
}

/* Footer styling */
.footer {
    position: fixed;
    bottom: 10px;
    left: 20px;
    font-size: 12px;
    color: #666666;
}

/* Improved scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #111111;
}

::-webkit-scrollbar-thumb {
    background: #333333;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #444444;
}
</style>
"""


class PhoenixAI:
    """Phoenix AI Chat Application Class"""
    
    def __init__(self):
        """Initialize the Phoenix AI application"""
        self.setup_state()
        self.setup_api_client()
        self.load_chat_history()
    
    def setup_state(self):
        """Set up session state variables"""
        if "groq_model" not in st.session_state:
            st.session_state.groq_model = DEFAULT_MODEL
        if "conversations" not in st.session_state:
            st.session_state.conversations = {}
        if "current_chat" not in st.session_state:
            st.session_state.current_chat = None
        if "api_error" not in st.session_state:
            st.session_state.api_error = None
    
    def setup_api_client(self):
        """Set up the Groq API client"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.error("Missing GROQ_API_KEY environment variable")
                st.session_state.api_error = "Missing API key. Please check your .env file."
                self.client = None
            else:
                self.client = Groq(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            st.session_state.api_error = f"API client initialization failed: {str(e)}"
            self.client = None
    
    def load_chat_history(self):
        """Load conversation history from storage"""
        try:
            with shelve.open(DATA_STORE) as db:
                st.session_state.conversations = db.get("conversations", {})
            
            # Set current chat if not set but conversations exist
            if not st.session_state.current_chat and st.session_state.conversations:
                st.session_state.current_chat = next(iter(st.session_state.conversations))
        except Exception as e:
            logger.error(f"Failed to load chat history: {str(e)}")
            st.error("Failed to load previous conversations. Starting with a fresh session.")
            st.session_state.conversations = {}
    
    def save_chat_history(self):
        """Save conversation history to storage"""
        try:
            with shelve.open(DATA_STORE) as db:
                db["conversations"] = st.session_state.conversations
        except Exception as e:
            logger.error(f"Failed to save chat history: {str(e)}")
            st.warning("Failed to save this conversation for future sessions.", icon="⚠️")
    
    def create_new_chat(self):
        """Create a new conversation"""
        new_chat_name = f"Chat {len(st.session_state.conversations) + 1}"
        st.session_state.conversations[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        self.save_chat_history()
        return new_chat_name
    
    def rename_chat(self, old_name: str, new_name: str):
        """Rename an existing conversation"""
        if new_name and new_name != old_name and new_name not in st.session_state.conversations:
            st.session_state.conversations[new_name] = st.session_state.conversations.pop(old_name)
            st.session_state.current_chat = new_name
            self.save_chat_history()
            return True
        return False
    
    def delete_chat(self, chat_name: str):
        """Delete a conversation"""
        if chat_name in st.session_state.conversations:
            del st.session_state.conversations[chat_name]
            if st.session_state.current_chat == chat_name:
                st.session_state.current_chat = next(iter(st.session_state.conversations)) if st.session_state.conversations else None
            self.save_chat_history()
            return True
        return False
    
    def generate_response(self, messages: List[Dict]):
        """Generate a response using the Groq API"""
        if not self.client:
            return "API client not initialized. Please check your API key."
        
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=st.session_state.groq_model,
                messages=messages
            )
            end_time = time.time()
            logger.info(f"Response generated in {end_time - start_time:.2f} seconds")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def render_sidebar(self):
        """Render the application sidebar"""
        with st.sidebar:
            st.markdown("## Phoenix AI")
            st.markdown("---")
            
            # Model selection
            st.selectbox(
                "Model",
                options=list(MODELS.keys()),
                index=list(MODELS.values()).index(st.session_state.groq_model) if st.session_state.groq_model in MODELS.values() else 0,
                key="model_selector",
                on_change=self.update_model
            )
            
            # New chat button
            if st.button("+ New Conversation", use_container_width=True, type="primary"):
                self.create_new_chat()
            
            st.markdown("### Your Conversations")
            st.markdown("---")
            
            # Chat history list
            if not st.session_state.conversations:
                st.info("No conversations yet. Start one by sending a message!")
            
            for chat_name in list(st.session_state.conversations.keys()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        chat_name,
                        key=f"btn_{chat_name}",
                        type="primary" if chat_name == st.session_state.current_chat else "secondary",
                        use_container_width=True
                    ):
                        st.session_state.current_chat = chat_name
                
                with col2:
                    if st.button("×", key=f"del_{chat_name}"):
                        if st.session_state.current_chat == chat_name:
                            self.delete_chat(chat_name)
                            st.rerun()
            
            # Footer
            st.markdown("""<div class="footer">Phoenix AI v1.2.0</div>""", unsafe_allow_html=True)
    
    def update_model(self):
        """Update the model based on selection"""
        selected_model_name = st.session_state.model_selector
        st.session_state.groq_model = MODELS[selected_model_name]
    
    def render_main_interface(self):
        """Render the main chat interface"""
        st.title("Phoenix AI")
        st.markdown("---")
        
        # Show API error if any
        if st.session_state.api_error:
            st.error(st.session_state.api_error)
        
        # Chat container with fixed height
        chat_container = st.container(height=680)
        input_container = st.container()
        
        # Display messages if conversation exists
        if st.session_state.current_chat:
            messages = st.session_state.conversations[st.session_state.current_chat]
            with chat_container:
                for msg in messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
        
        # Chat input
        with input_container:
            if prompt := st.chat_input("Type your message here..."):
                if not st.session_state.current_chat:
                    self.create_new_chat()
                
                # Add user message
                messages = st.session_state.conversations[st.session_state.current_chat]
                messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with chat_container.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display assistant response
                with chat_container.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        full_response = self.generate_response(messages)
                        st.markdown(full_response)
                
                # Save conversation
                messages.append({"role": "assistant", "content": full_response})
                st.session_state.conversations[st.session_state.current_chat] = messages
                self.save_chat_history()
    
    def run(self):
        """Run the Phoenix AI application"""
        # Apply custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
        # Render the sidebar and main interface
        self.render_sidebar()
        self.render_main_interface()
        
        # Add auto-scroll functionality
        st.components.v1.html("""
        <script>
        const container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
        container.scrollTop = container.scrollHeight;
        
        // Add keyboard shortcut (Ctrl+Enter) for sending messages
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                const chatInput = window.parent.document.querySelector('.stChatInput input');
                const sendButton = window.parent.document.querySelector('.stChatInput button');
                if (chatInput && chatInput.value.trim() && sendButton) {
                    sendButton.click();
                }
            }
        });
        </script>
        """)


# Run the application
if __name__ == "__main__":
    phoenix = PhoenixAI()
    phoenix.run()