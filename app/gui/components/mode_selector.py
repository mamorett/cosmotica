from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QGroupBox
)
from PyQt6.QtCore import pyqtSignal

class ModeSelector(QWidget):
    """Mode selection component for choosing between T2I and I2V."""
    
    mode_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Mode selection group
        mode_group = QGroupBox("Generation Mode")
        mode_layout = QHBoxLayout()
        
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "🎨 Text to Image",
            "🎬 Image to Video"
        ])
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
    
    def _on_mode_changed(self, text):
        """Emit signal when mode changes."""
        mode = "t2i" if "Text" in text else "i2v"
        self.mode_changed.emit(mode)
    
    def get_current_mode(self) -> str:
        """Get the current generation mode."""
        return "t2i" if "Text" in self.mode_combo.currentText() else "i2v"
