"""MCP Server Manager for auto-installation and integration"""
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any

class MCPServerManager:
    def __init__(self):
        self.servers = []
        self.config_path = Path.home() / ".mcp" / "servers.json"
    
    def install_server(self, server_name: str, config: Dict[str, Any]):
        """Install an MCP server"""
        try:
            # Install using npx
            cmd = f"npx -y @modelcontextprotocol/create-server {server_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.servers.append({"name": server_name, "config": config})
                self.save_config()
                return True
            return False
        except Exception as e:
            print(f"Error installing {server_name}: {e}")
            return False
    
    def save_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump({"servers": self.servers}, f, indent=2)
    
    def get_available_tools(self):
        """Get all available tools from MCP servers"""
        tools = []
        for server in self.servers:
            # Query server for available tools
            tools.extend(server.get("tools", []))
        return tools
