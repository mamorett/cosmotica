from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QLineEdit, QPushButton, QGroupBox
)
from PyQt6.QtCore import pyqtSignal

class ComputeSelector(QWidget):
    """
    Component for selecting and testing the compute target (Local GPU vs Remote GPU).
    """
    
    # Emits (target, url) when updated
    target_changed = pyqtSignal(str, str)
    
    def __init__(self, parent=None, default_target="local", default_url="http://localhost:8000"):
        super().__init__(parent)
        self.default_target = default_target
        self.default_url = default_url
        self.engine = None  # Will be set by main window
        self.init_ui()
        
    def set_engine(self, engine):
        self.engine = engine
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Compute Target Settings")
        group_layout = QVBoxLayout()
        
        # Selection row
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Compute Location:"))
        
        self.target_combo = QComboBox()
        self.target_combo.addItems([
            "🖥️ Local GPU (CUDA / Fallback)",
            "🌐 Remote GPU (Network Server)"
        ])
        
        # Pre-select based on default target
        if self.default_target == "remote":
            self.target_combo.setCurrentIndex(1)
        else:
            self.target_combo.setCurrentIndex(0)
            
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        select_layout.addWidget(self.target_combo)
        select_layout.addStretch()
        group_layout.addLayout(select_layout)
        
        # Server URL row
        self.url_widget = QWidget()
        url_layout = QHBoxLayout(self.url_widget)
        url_layout.setContentsMargins(0, 5, 0, 0)
        
        url_layout.addWidget(QLabel("Server URL:"))
        self.url_input = QLineEdit(self.default_url)
        self.url_input.setPlaceholderText("http://<ip-or-host>:<port>")
        self.url_input.textChanged.connect(self._on_url_changed)
        self.url_input.returnPressed.connect(self._test_connection)
        url_layout.addWidget(self.url_input)
        
        self.test_btn = QPushButton("🔌 Test Connection")
        self.test_btn.setObjectName("testConnectionButton")
        self.test_btn.clicked.connect(self._test_connection)
        url_layout.addWidget(self.test_btn)
        
        group_layout.addWidget(self.url_widget)
        
        # Status Label
        self.status_label = QLabel("Mode: Local")
        self.status_label.setStyleSheet("color: #00d2ff; font-style: italic;")
        group_layout.addWidget(self.status_label)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Initial state update
        self._update_visibility()
        
    def _on_target_changed(self, index):
        self._update_visibility()
        self._notify_change()
        
    def _on_url_changed(self, text):
        if self.engine:
            self.engine.remote_url = text.strip()
        self.status_label.setText("Status: URL changed (click 'Test Connection' to verify)")
        self.status_label.setStyleSheet("color: #ffaa00; font-style: italic;")

        
    def _update_visibility(self):
        is_remote = self.get_target() == "remote"
        self.url_widget.setVisible(is_remote)
        if is_remote:
            self.status_label.setText("Status: Connection not tested")
            self.status_label.setStyleSheet("color: #ffaa00;")
        else:
            self.status_label.setText("Mode: Local CPU/GPU execution")
            self.status_label.setStyleSheet("color: #00d2ff;")
            
    def _notify_change(self):
        self.target_changed.emit(self.get_target(), self.get_url())
        
    def get_target(self) -> str:
        """Get 'local' or 'remote'."""
        return "remote" if self.target_combo.currentIndex() == 1 else "local"
        
    def get_url(self) -> str:
        return self.url_input.text().strip()
        
    def _test_connection(self):
        if not self.engine:
            return
            
        self.status_label.setText("Testing connection...")
        self.status_label.setStyleSheet("color: #e0e0e0;")
        self.test_btn.setEnabled(False)
        
        # We test in a simple non-blocking way, but since it's a quick connection test
        # we can do it directly. In production, QThread is safer, but self.engine.test_remote_connection()
        # has a small timeout of 5 seconds.
        # Let's run it.
        # To avoid blocking GUI at all, let's import QTimer or use a simple worker or just run it since 5s is max.
        # Let's do it directly for simplicity, or we can use QTimer to defer it.
        # Actually direct is fine as long as timeout is short.
        
        # We temporary set engine settings to test
        old_target = self.engine.inference_target
        old_url = self.engine.remote_url
        
        self.engine.set_inference_target(self.get_target(), self.get_url())
        success, message = self.engine.test_remote_connection()
        
        # Restore target settings
        self.engine.set_inference_target(old_target, old_url)
        
        self.test_btn.setEnabled(True)
        if success:
            self.status_label.setText(f"Success: {message}")
            self.status_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
