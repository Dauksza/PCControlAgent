"""
Gradio UI for Mistral Agent Platform
"""
import gradio as gr
import requests
import json
from typing import Dict, Any, Tuple

# API base URL
API_BASE = "http://localhost:8000/api"

def get_models():
    """Fetch available models from API"""
    try:
        response = requests.get(f"{API_BASE}/models")
        response.raise_for_status()
        models = response.json()["models"]
        return [m.get("id", "Unknown") for m in models]
    except Exception as e:
        print(f"Failed to fetch models: {e}")
        return ["mistral-large-2407", "mistral-medium-2312", "pixtral-12b"]

def execute_task(
    task: str,
    model: str,
    max_iterations: int,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Execute a task via API
    
    Returns:
        Tuple of (status, summary, full_result)
    """
    if not task:
        return "⚠️ Please enter a task", "", ""
    
    try:
        progress(0, desc="Starting task execution...")
        
        response = requests.post(
            f"{API_BASE}/execute",
            json={
                "task": task,
                "model": model,
                "max_iterations": max_iterations
            },
            timeout=300  # 5 minute timeout
        )
        response.raise_for_status()
        result = response.json()
        
        progress(1.0, desc="Task completed!")
        
        # Format status
        status = result.get("status", "unknown")
        if status == "completed":
            status_msg = f"✅ Task completed successfully!\n\n"
        else:
            status_msg = f"⚠️ Task status: {status}\n\n"
        
        status_msg += f"**Iterations:** {result.get('total_iterations', 0)}\n"
        status_msg += f"**Subtasks:** {len(result.get('subtasks', []))}\n"
        status_msg += f"**Tool Calls:** {len(result.get('tool_calls', []))}\n"
        status_msg += f"**Confidence:** {result.get('completion_confidence', 0):.0%}"
        
        # Format summary
        summary = "## 📋 Subtasks\n\n"
        for subtask in result.get("subtasks", []):
            status_icon = "✅" if subtask.get("status") == "completed" else "⏳"
            summary += f"{status_icon} {subtask.get('description')}\n"
        
        summary += "\n## 🔧 Tool Calls\n\n"
        for tool_call in result.get("tool_calls", []):
            summary += f"- **{tool_call.get('tool_name')}** ({tool_call.get('status')})\n"
        
        # Full result as JSON
        full_result = json.dumps(result, indent=2)
        
        return status_msg, summary, full_result
        
    except requests.Timeout:
        return "❌ Task execution timed out (5 minutes)", "", ""
    except Exception as e:
        return f"❌ Task execution failed: {str(e)}", "", ""

def create_interface():
    """Create Gradio interface"""
    
    # Load models
    models = get_models()
    
    with gr.Blocks(title="Mistral AI Agent Platform", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Mistral AI Agent Platform
        ### Advanced Autonomous Agent with Full Mistral API Capabilities
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Task input
                task_input = gr.Textbox(
                    label="Task Description",
                    placeholder="Enter your task here...",
                    lines=5
                )
                
                # Example tasks
                gr.Examples(
                    examples=[
                        ["Search for the latest developments in AI and provide a summary of the top 3 findings."],
                        ["Write a Python function that calculates the Fibonacci sequence and includes error handling."],
                        ["Explain machine learning in simple terms with examples."],
                        ["1. Search for Python best practices\n2. Summarize the top 5 practices\n3. Provide code examples"]
                    ],
                    inputs=task_input,
                    label="Example Tasks"
                )
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=models,
                        value=models[0] if models else "mistral-large-2407",
                        label="Model",
                        interactive=True
                    )
                    
                    max_iterations = gr.Slider(
                        minimum=5,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Max Iterations"
                    )
                
                execute_btn = gr.Button("🚀 Execute Task", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### ℹ️ About
                
                **Features:**
                - 🔍 Web search capabilities
                - 💻 Code execution
                - 🧮 Mathematical calculations
                - 🤖 Autonomous task completion
                - 🔄 Multi-step workflows
                
                **How to use:**
                1. Enter your task description
                2. Select a model
                3. Adjust max iterations if needed
                4. Click Execute Task
                5. View results below
                """)
        
        gr.Markdown("---")
        gr.Markdown("## 📊 Results")
        
        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                lines=6,
                interactive=False
            )
        
        with gr.Tab("Summary"):
            summary_output = gr.Markdown()
        
        with gr.Tab("Full Result"):
            full_result_output = gr.Code(
                language="json",
                label="Full Result JSON"
            )
        
        # Connect execute button
        execute_btn.click(
            fn=execute_task,
            inputs=[task_input, model_dropdown, max_iterations],
            outputs=[status_output, summary_output, full_result_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
