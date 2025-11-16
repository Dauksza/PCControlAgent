"""
Streamlit UI for Mistral Agent Platform
"""
import streamlit as st
import requests
import json
from typing import Dict, Any

# Configure page
st.set_page_config(
    page_title="Mistral AI Agent Platform",
    page_icon="🚀",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8000/api"

def get_models():
    """Fetch available models from API"""
    try:
        response = requests.get(f"{API_BASE}/models")
        response.raise_for_status()
        return response.json()["models"]
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        return []

def execute_task(task: str, model: str, max_iterations: int) -> Dict[str, Any]:
    """Execute a task via API"""
    try:
        response = requests.post(
            f"{API_BASE}/execute",
            json={
                "task": task,
                "model": model,
                "max_iterations": max_iterations
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Task execution failed: {e}")
        return None

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🚀 Mistral AI Agent Platform")
    st.markdown("### Advanced Autonomous Agent with Full Mistral API Capabilities")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load models
        with st.spinner("Loading models..."):
            models = get_models()
        
        if models:
            model_names = [m.get("id", m.get("name", "Unknown")) for m in models]
            selected_model = st.selectbox(
                "Select Model",
                options=model_names,
                index=0 if model_names else None
            )
        else:
            selected_model = st.text_input("Model ID", value="mistral-large-2407")
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=5,
            max_value=100,
            value=50,
            step=5
        )
        
        st.divider()
        
        st.header("📊 Model Info")
        if models and selected_model:
            selected_model_info = next((m for m in models if m.get("id") == selected_model), None)
            if selected_model_info:
                st.json(selected_model_info)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Task Input")
        
        # Task examples
        example_tasks = {
            "Web Search & Summary": "Search for the latest developments in AI and provide a summary of the top 3 findings.",
            "Code Generation": "Write a Python function that calculates the Fibonacci sequence and includes error handling.",
            "Data Analysis": "Analyze the concept of machine learning and explain it in simple terms with examples.",
            "Multi-Step Task": "1. Search for Python best practices\n2. Summarize the top 5 practices\n3. Provide code examples"
        }
        
        selected_example = st.selectbox(
            "Example Tasks (Optional)",
            options=["Custom Task"] + list(example_tasks.keys())
        )
        
        if selected_example != "Custom Task":
            default_task = example_tasks[selected_example]
        else:
            default_task = ""
        
        task_input = st.text_area(
            "Task Description",
            value=default_task,
            height=150,
            placeholder="Enter your task here..."
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            execute_button = st.button("🚀 Execute", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
    
    with col2:
        st.header("ℹ️ About")
        st.markdown("""
        **Features:**
        - 🔍 Web search capabilities
        - 💻 Code execution
        - 🧮 Mathematical calculations
        - 🤖 Autonomous task completion
        - 🔄 Multi-step workflows
        
        **How to use:**
        1. Select a model from the sidebar
        2. Enter your task description
        3. Click Execute
        4. View results below
        """)
    
    # Execute task
    if execute_button and task_input:
        st.divider()
        st.header("📋 Execution Results")
        
        with st.spinner(f"Executing task with {selected_model}..."):
            result = execute_task(task_input, selected_model, max_iterations)
        
        if result:
            # Display status
            status = result.get("status", "unknown")
            if status == "completed":
                st.success(f"✅ Task completed successfully!")
            else:
                st.warning(f"⚠️ Task status: {status}")
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Iterations", result.get("total_iterations", 0))
            with col_m2:
                st.metric("Subtasks", len(result.get("subtasks", [])))
            with col_m3:
                st.metric("Tool Calls", len(result.get("tool_calls", [])))
            with col_m4:
                confidence = result.get("completion_confidence", 0)
                st.metric("Confidence", f"{confidence:.0%}")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔧 Tool Calls", "📊 Iterations", "📄 Full Result"])
            
            with tab1:
                st.subheader("Subtasks")
                for subtask in result.get("subtasks", []):
                    status_icon = "✅" if subtask.get("status") == "completed" else "⏳"
                    st.markdown(f"{status_icon} {subtask.get('description')}")
            
            with tab2:
                st.subheader("Tool Executions")
                for tool_call in result.get("tool_calls", []):
                    with st.expander(f"{tool_call.get('tool_name')} - {tool_call.get('status')}"):
                        st.json(tool_call)
            
            with tab3:
                st.subheader("Iteration Details")
                for iteration in result.get("iterations", []):
                    with st.expander(f"Iteration {iteration.get('number')}"):
                        st.json(iteration)
            
            with tab4:
                st.json(result)

if __name__ == "__main__":
    main()
