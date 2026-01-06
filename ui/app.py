"""
TKLocalAI - Desktop UI Application
===================================
Native desktop wrapper using PyWebView for the web UI.
"""

import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional
import webview
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TKLocalAIApp:
    """
    Desktop application wrapper for TKLocalAI.
    Uses PyWebView to create a native window with the web UI.
    """
    
    def __init__(
        self,
        title: str = "TKLocalAI Assistant",
        width: int = 1200,
        height: int = 800,
        min_width: int = 800,
        min_height: int = 600,
        server_host: str = "127.0.0.1",
        server_port: int = 8000,
    ):
        """
        Initialize the desktop application.
        
        Args:
            title: Window title
            width: Initial window width
            height: Initial window height
            min_width: Minimum window width
            min_height: Minimum window height
            server_host: Backend server host
            server_port: Backend server port
        """
        self.title = title
        self.width = width
        self.height = height
        self.min_width = min_width
        self.min_height = min_height
        self.server_host = server_host
        self.server_port = server_port
        
        self._server_thread: Optional[threading.Thread] = None
        self._server_started = threading.Event()
        self._window = None
    
    def _start_server(self) -> None:
        """Start the FastAPI server in a background thread."""
        import uvicorn
        
        def run_server():
            logger.info(f"Starting server on {self.server_host}:{self.server_port}")
            
            uvicorn.run(
                "src.server:app",
                host=self.server_host,
                port=self.server_port,
                log_level="warning",
                access_log=False,
            )
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
        # Wait for server to start
        self._wait_for_server()
    
    def _wait_for_server(self, timeout: float = 30.0) -> bool:
        """Wait for the server to be ready."""
        import httpx
        
        start_time = time.time()
        url = f"http://{self.server_host}:{self.server_port}/health"
        
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(url, timeout=1.0)
                if response.status_code == 200:
                    logger.info("Server is ready")
                    self._server_started.set()
                    return True
            except Exception:
                pass
            
            time.sleep(0.5)
        
        logger.error("Server failed to start within timeout")
        return False
    
    def _get_ui_path(self) -> str:
        """Get the URL to load the UI from the server."""
        # Always load from server to avoid CORS issues with file:// protocol
        return f"http://{self.server_host}:{self.server_port}"
    
    def run(self, start_server: bool = True) -> None:
        """
        Run the desktop application.
        
        Args:
            start_server: Whether to start the backend server
        """
        logger.info("Starting TKLocalAI Desktop Application")
        
        # Start the backend server
        if start_server:
            self._start_server()
        
        # Get UI path
        ui_path = self._get_ui_path()
        logger.info(f"Loading UI from: {ui_path}")
        
        # Create the window
        self._window = webview.create_window(
            title=self.title,
            url=ui_path,
            width=self.width,
            height=self.height,
            min_size=(self.min_width, self.min_height),
            resizable=True,
            frameless=False,
            easy_drag=False,
            text_select=True,
        )
        
        # Expose Python API to JavaScript
        self._window.expose(self.get_system_info)
        self._window.expose(self.open_external_link)
        
        # Start the webview
        webview.start(debug=False)
        
        logger.info("Application closed")
    
    def get_system_info(self) -> dict:
        """Get system information (exposed to JS)."""
        import platform
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        except ImportError:
            cuda_available = False
            gpu_name = None
        
        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
        }
    
    def open_external_link(self, url: str) -> None:
        """Open a link in the default browser (exposed to JS)."""
        webbrowser.open(url)


def run_desktop_app():
    """Entry point for the desktop application."""
    from src.config import init_config
    
    # Load configuration
    settings = init_config()
    
    # Create and run the app
    app = TKLocalAIApp(
        title=settings.ui.title,
        width=settings.ui.width,
        height=settings.ui.height,
        min_width=settings.ui.min_width,
        min_height=settings.ui.min_height,
        server_host=settings.server.host,
        server_port=settings.server.port,
    )
    
    app.run()


if __name__ == "__main__":
    run_desktop_app()
