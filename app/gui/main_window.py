import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QLabel, QStatusBar, QMessageBox,
    QMenuBar, QMenu, QFileDialog, QApplication
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon

from app.config import AppConfig
from app.engine import InferenceEngine
from app.gui.components.compute_selector import ComputeSelector
from app.gui.components.mode_selector import ModeSelector
from app.gui.components.prompt_input import PromptInput
from app.gui.components.image_preview import ImagePreview
from app.gui.components.media_preview import MediaPreview
from app.gui.components.generation_controls import GenerationControls
from app.gui.workers.generation_worker import GenerationWorker
from app.gui.styles.theme import MAIN_STYLE

class MainWindow(QMainWindow):
    """Main application window for Cosmos3 Inference Studio."""
    
    def __init__(self):
        super().__init__()
        self.app_config = AppConfig()
        self.engine = InferenceEngine()
        self.generated_data = None
        self.generated_type = None
        self.generated_fps = 24.0
        
        self.init_ui()
        self.init_engine()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"{self.app_config.APP_NAME} v{self.app_config.VERSION}")
        self.setMinimumSize(1400, 900)
        
        # Apply styles
        self.setStyleSheet(MAIN_STYLE)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title = QLabel("🎨 Cosmos3 Studio")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)
        
        # Compute Selector (Local vs Remote GPU)
        self.compute_selector = ComputeSelector(
            default_target=self.app_config.DEFAULT_INFERENCE_TARGET,
            default_url=self.app_config.DEFAULT_REMOTE_URL
        )
        self.compute_selector.set_engine(self.engine)
        self.compute_selector.target_changed.connect(self._on_compute_target_changed)
        left_layout.addWidget(self.compute_selector)
        
        # Mode selector
        self.mode_selector = ModeSelector()
        self.mode_selector.mode_changed.connect(self._on_mode_changed)
        left_layout.addWidget(self.mode_selector)
        
        # Prompt input
        self.prompt_input = PromptInput()
        left_layout.addWidget(self.prompt_input)
        
        # Image preview (for I2V)
        self.image_preview = ImagePreview()
        self.image_preview.image_loaded.connect(self._on_image_loaded)
        left_layout.addWidget(self.image_preview)
        
        # Generation controls
        self.gen_controls = GenerationControls()
        self.gen_controls.generate_clicked.connect(self._on_generate)
        left_layout.addWidget(self.gen_controls)
        
        left_layout.addStretch()
        
        # Right panel - Output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Media preview
        self.media_preview = MediaPreview()
        self.media_preview.download_requested.connect(self._on_download)
        right_layout.addWidget(self.media_preview)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])
        
        main_layout.addWidget(splitter)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initial mode setup
        self._on_mode_changed("t2i")
    
    def _create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.image_preview._browse_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Output", self)
        save_action.triggered.connect(self._on_download)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Model menu
        model_menu = menubar.addMenu("Model")
        
        reload_action = QAction("Initialize / Reload Compute", self)
        reload_action.triggered.connect(self.init_engine)
        model_menu.addAction(reload_action)
        
        unload_action = QAction("Unload Local Model", self)
        unload_action.triggered.connect(self._unload_model)
        model_menu.addAction(unload_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def init_engine(self):
        """Initialize the inference engine based on selected target."""
        target = self.compute_selector.get_target()
        url = self.compute_selector.get_url()
        
        self.engine.set_inference_target(target, url)
        
        if target == "remote":
            self.status_bar.showMessage(f"Connecting to remote server: {url}...")
            self.gen_controls.set_progress(20, "Connecting to server")
        else:
            self.status_bar.showMessage("Loading local model...")
            self.gen_controls.set_progress(0, "Loading local weights")
        
        def progress_callback(progress, message):
            self.gen_controls.set_progress(progress, message)
            self.status_bar.showMessage(message)
        
        try:
            success = self.engine.initialize(progress_callback)
            
            if success:
                if self.engine.mock_mode:
                    self.status_bar.showMessage("Initialized in Mock Mode (ML libraries not loaded)")
                    QMessageBox.information(
                        self,
                        "Mock Mode Activated",
                        "The application loaded in Mock Mode because local machine lacks CUDA/PyTorch.\n"
                        "You can test the interface locally with generated gradients, or connect to a remote GPU server."
                    )
                else:
                    self.status_bar.showMessage(f"Compute initialized successfully ({target.upper()})")
                self.gen_controls.hide_progress()
            else:
                self.status_bar.showMessage("Initialization failed")
                self.gen_controls.hide_progress()
        except Exception as e:
            self.status_bar.showMessage("Initialization failed")
            self.gen_controls.hide_progress()
            QMessageBox.warning(
                self, 
                "Compute Init Alert", 
                f"Failed to initialize target:\n\n{str(e)}\n\n"
                "Please configure settings and try again."
            )
    
    def _on_compute_target_changed(self, target, url):
        """Handle target compute changes from the GUI selector."""
        # Clean current output/states on target switch
        self.media_preview.clear()
        
        # Trigger re-initialization of engine
        self.init_engine()
        
    def _on_mode_changed(self, mode: str):
        """Handle generation mode change."""
        self.gen_controls.set_mode(mode)
        
        # Update image preview visibility
        self.image_preview.setVisible(mode == "i2v")
        
        # Update default negative prompt
        self.prompt_input.set_negative_prompt_default(mode)
        
        # Clear previous output
        self.media_preview.clear()
    
    def _on_image_loaded(self, image):
        """Handle image loaded event."""
        self.status_bar.showMessage(
            f"Image loaded: {image.size[0]}x{image.size[1]}"
        )
    
    def _on_generate(self, params: dict):
        """Handle generate button click."""
        mode = self.mode_selector.get_current_mode()
        prompts = self.prompt_input.get_prompts()
        
        # Validate inputs
        if not prompts["positive"].strip():
            QMessageBox.warning(self, "Warning", "Please enter a positive prompt.")
            return
        
        if mode == "i2v" and not self.image_preview.has_image():
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please provide a reference image for image-to-video generation."
            )
            return
        
        # Disable UI during generation
        self.gen_controls.set_generating(True)
        self.media_preview.clear()
        
        # Create and start worker thread
        self.worker = GenerationWorker(
            self.engine,
            params,
            mode,
            prompts,
            self.image_preview.get_image() if mode == "i2v" else None
        )
        
        self.worker.progress.connect(self._on_generation_progress)
        self.worker.finished.connect(self._on_generation_finished)
        self.worker.error.connect(self._on_generation_error)
        
        self.worker.start()
    
    def _on_generation_progress(self, progress: int, message: str):
        """Handle generation progress updates."""
        self.gen_controls.set_progress(progress, message)
        self.status_bar.showMessage(message)
    
    def _on_generation_finished(self, data, data_type: str, fps: float):
        """Handle generation completion."""
        self.generated_data = data
        self.generated_type = data_type
        self.generated_fps = fps
        
        # Display in preview
        if data_type == "image":
            self.media_preview.display_image(data)
            self.status_bar.showMessage("Image generated successfully")
        elif data_type == "video":
            self.media_preview.display_video(data, fps)
            self.status_bar.showMessage(f"Video generated successfully ({len(data)} frames)")
        
        # Re-enable UI
        self.gen_controls.set_generating(False)
        self.gen_controls.hide_progress()
    
    def _on_generation_error(self, error_msg: str):
        """Handle generation errors."""
        self.gen_controls.set_generating(False)
        self.gen_controls.hide_progress()
        
        QMessageBox.critical(
            self,
            "Generation Error",
            f"An error occurred during generation:\n\n{error_msg}"
        )
        self.status_bar.showMessage("Generation failed")
    
    def _on_download(self):
        """Handle download/save request."""
        if self.generated_data is None:
            QMessageBox.warning(self, "Warning", "No generated output to save.")
            return
        
        # Determine file format
        if self.generated_type == "image":
            default_name = "generated_image.jpg"
            file_filter = "JPEG (*.jpg);;PNG (*.png);;All Files (*)"
        else:
            default_name = "generated_video.mp4"
            file_filter = "MP4 (*.mp4);;AVI (*.avi);;All Files (*)"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated Output",
            str(self.app_config.OUTPUTS_DIR / default_name),
            file_filter
        )
        
        if filepath:
            try:
                self.engine.save_output(
                    self.generated_data,
                    filepath,
                    self.generated_fps
                )
                
                # Save metadata side-by-side
                try:
                    from app.utils.file_handler import FileHandler
                    params = {
                        "mode": self.mode_selector.get_current_mode(),
                        "prompts": self.prompt_input.get_prompts(),
                        "compute_target": self.compute_selector.get_target(),
                        "server_url": self.compute_selector.get_url() if self.compute_selector.get_target() == "remote" else "local",
                        "fps": self.generated_fps
                    }
                    FileHandler.save_metadata(params, filepath)
                except Exception as e:
                    logger.warning(f"Could not save generation metadata: {e}")
                    
                self.status_bar.showMessage(f"Saved to: {filepath}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save output:\n\n{str(e)}"
                )
    
    def _unload_model(self):
        """Unload the model to free memory."""
        if self.compute_selector.get_target() == "remote":
            QMessageBox.information(self, "Info", "Compute target is remote. No local model loaded.")
            return
            
        reply = QMessageBox.question(
            self,
            "Unload Model",
            "Are you sure you want to unload the local model?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.shutdown()
            self.status_bar.showMessage("Local model unloaded")
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Cosmos3 Inference Studio",
            f"""
            <h2>Cosmos3 Inference Studio</h2>
            <p>Version: {self.app_config.VERSION}</p>
            <p>A powerful desktop application for NVIDIA's Cosmos3 model.</p>
            <p>Features:</p>
            <ul>
                <li>Text-to-Image generation</li>
                <li>Image-to-Video generation</li>
                <li>Drag-and-drop interface</li>
                <li>Real-time preview</li>
                <li>Advanced generation controls</li>
                <li><b>Remote GPU inference over local networks</b></li>
            </ul>
            <p>Model ID: {self.app_config.ModelConfig.MODEL_ID}</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        reply = QMessageBox.question(
            self,
            "Exit",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.shutdown()
            event.accept()
        else:
            event.ignore()
