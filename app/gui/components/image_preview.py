from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, 
    QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PIL import Image

class ImagePreview(QWidget):
    """Image preview component with drag-and-drop support."""
    
    image_loaded = pyqtSignal(object)  # Emits PIL Image object
    image_cleared = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.init_ui()
        self.setAcceptDrops(True)
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Image group
        image_group = QGroupBox("Reference Image (for Image-to-Video)")
        image_layout = QVBoxLayout()
        
        # Drop zone
        self.drop_zone = QLabel()
        self.drop_zone.setObjectName("dropZone")
        self.drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_zone.setMinimumSize(300, 200)
        self.drop_zone.setMaximumSize(600, 300)
        self.drop_zone.setText(
            "🖼️\n\nDrag & Drop Image Here\n\n"
            "Supported formats: PNG, JPG, JPEG, BMP, WebP"
        )
        self.drop_zone.setWordWrap(True)
        self.drop_zone.setStyleSheet(self._get_drop_style(False))
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.browse_btn = QPushButton("📁 Browse")
        self.browse_btn.clicked.connect(self._browse_image)
        
        self.clear_btn = QPushButton("❌ Clear")
        self.clear_btn.clicked.connect(self._clear_image)
        self.clear_btn.setEnabled(False)
        
        controls_layout.addWidget(self.browse_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addStretch()
        
        image_layout.addWidget(self.drop_zone)
        image_layout.addLayout(controls_layout)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
    
    def _get_drop_style(self, active: bool) -> str:
        """Get stylesheet for drop zone based on state."""
        base = """
            QLabel#dropZone {
                border: 3px dashed #0f3460;
                border-radius: 15px;
                background-color: #16213e;
                color: #e0e0e0;
                font-size: 14px;
                padding: 20px;
            }
        """
        if active:
            base += """
                QLabel#dropZone {
                    border-color: #00d2ff;
                    background-color: #1a1a4e;
                    border-style: solid;
                }
            """
        return base
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_zone.setStyleSheet(self._get_drop_style(True))
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.drop_zone.setStyleSheet(self._get_drop_style(False))
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            self._load_image(filepath)
        self.drop_zone.setStyleSheet(self._get_drop_style(False))
    
    def _browse_image(self):
        """Open file dialog to browse for image."""
        from PyQt6.QtWidgets import QFileDialog
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if filepath:
            self._load_image(filepath)
    
    def _load_image(self, filepath: str):
        """Load and display image."""
        try:
            # Load with PIL for processing
            pil_image = Image.open(filepath)
            self.current_image = pil_image
            
            # Display in drop zone
            pixmap = QPixmap(filepath)
            scaled_pixmap = pixmap.scaled(
                self.drop_zone.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.drop_zone.setPixmap(scaled_pixmap)
            
            self.clear_btn.setEnabled(True)
            self.image_loaded.emit(pil_image)
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")
    
    def _clear_image(self):
        """Clear the current image."""
        self.current_image = None
        self.drop_zone.setPixmap(QPixmap())
        self.drop_zone.setText(
            "🖼️\n\nDrag & Drop Image Here\n\n"
            "Supported formats: PNG, JPG, JPEG, BMP, WebP"
        )
        self.clear_btn.setEnabled(False)
        self.image_cleared.emit()
    
    def get_image(self) -> Image.Image:
        """Get the current PIL Image."""
        return self.current_image
    
    def has_image(self) -> bool:
        """Check if an image is loaded."""
        return self.current_image is not None
