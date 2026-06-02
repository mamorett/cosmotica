from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np

class MediaPreview(QWidget):
    """Preview component for generated images and videos."""
    
    download_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.video_frames = []
        self.current_frame_idx = 0
        self.is_video = False
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._next_frame)
        self.fps = 24.0
        self.is_playing = False
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Preview group
        preview_group = QGroupBox("Generated Output Preview")
        preview_layout = QVBoxLayout()
        
        # Preview display
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setMaximumHeight(500)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #0a0a1e;
                border: 2px solid #0f3460;
                border-radius: 10px;
            }
        """)
        self.preview_label.setText("Generated output will appear here")
        
        # Video controls (initially hidden)
        self.video_controls = QWidget()
        video_controls_layout = QHBoxLayout(self.video_controls)
        video_controls_layout.setContentsMargins(0, 5, 0, 5)
        
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self._seek_frame)
        
        self.frame_label = QLabel("Frame: 0/0")
        
        video_controls_layout.addWidget(self.play_btn)
        video_controls_layout.addWidget(self.frame_slider)
        video_controls_layout.addWidget(self.frame_label)
        
        self.video_controls.setVisible(False)
        
        # Download button
        self.download_btn = QPushButton("💾 Download Generated Output")
        self.download_btn.clicked.connect(self.download_requested.emit)
        self.download_btn.setEnabled(False)
        self.download_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00ff88, stop:1 #00d2ff);
                color: #1a1a2e;
                font-size: 14px;
                padding: 12px;
            }
        """)
        
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.video_controls)
        preview_layout.addWidget(self.download_btn)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
    
    def resizeEvent(self, event):
        """Update scale of current preview on resize."""
        super().resizeEvent(event)
        if self.is_video and self.video_frames:
            self._show_frame(self.current_frame_idx)
        elif self.current_image:
            self.display_image(self.current_image)
            
    def display_image(self, image: Image.Image):
        """Display a generated image."""
        self.is_video = False
        self.current_image = image
        self.video_frames = []
        
        # Convert PIL to QPixmap
        pixmap = self._pil_to_pixmap(image)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
        
        self.video_controls.setVisible(False)
        self.download_btn.setEnabled(True)
        self._stop_playback()
    
    def display_video(self, frames: list, fps: float = 24.0):
        """Display generated video frames."""
        self.is_video = True
        self.video_frames = frames
        self.fps = fps
        self.current_frame_idx = 0
        
        if frames:
            self._show_frame(0)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(len(frames) - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(f"Frame: 1/{len(frames)}")
        
        self.video_controls.setVisible(True)
        self.download_btn.setEnabled(True)
        self.play_btn.setText("▶️ Play")
        self.is_playing = False
    
    def _show_frame(self, idx: int):
        """Display a specific frame from video."""
        if 0 <= idx < len(self.video_frames):
            frame = self.video_frames[idx]
            if isinstance(frame, Image.Image):
                pixmap = self._pil_to_pixmap(frame)
            else:
                # Assume numpy array
                pixmap = self._array_to_pixmap(frame)
            
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
            self.frame_label.setText(f"Frame: {idx + 1}/{len(self.video_frames)}")
    
    def _next_frame(self):
        """Advance to next frame for playback."""
        if not self.is_video or not self.video_frames:
            self._stop_playback()
            return
        
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.video_frames)
        self._show_frame(self.current_frame_idx)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_idx)
        self.frame_slider.blockSignals(False)
    
    def _toggle_playback(self):
        """Toggle video playback."""
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()
    
    def _start_playback(self):
        """Start video playback."""
        if self.video_frames:
            self.is_playing = True
            self.play_btn.setText("⏸️ Pause")
            interval = int(1000 / self.fps)
            self.video_timer.start(interval)
    
    def _stop_playback(self):
        """Stop video playback."""
        self.is_playing = False
        self.play_btn.setText("▶️ Play")
        self.video_timer.stop()
    
    def _seek_frame(self, value: int):
        """Seek to specific frame."""
        if self.video_frames:
            self.current_frame_idx = value
            self._show_frame(value)
    
    def _pil_to_pixmap(self, pil_image: Image.Image) -> QPixmap:
        """Convert PIL Image to QPixmap."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        r, g, b = pil_image.split()
        pil_image = Image.merge("RGB", (b, g, r))
        
        data = pil_image.tobytes("raw", "RGB")
        qimage = QImage(
            data, 
            pil_image.width, 
            pil_image.height,
            QImage.Format.Format_RGB888
        )
        return QPixmap.fromImage(qimage)
    
    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        
        if len(array.shape) == 2:
            h, w = array.shape
            qimage = QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, c = array.shape
            if c == 3:
                qimage = QImage(array.data, w, h, w * 3, QImage.Format.Format_RGB888)
            else:
                qimage = QImage(array.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        
        return QPixmap.fromImage(qimage)
    
    def clear(self):
        """Clear the preview."""
        self.current_image = None
        self.video_frames = []
        self.is_video = False
        self.preview_label.setText("Generated output will appear here")
        self.video_controls.setVisible(False)
        self.download_btn.setEnabled(False)
        self._stop_playback()
    
    def get_current_data(self):
        """Get the current preview data for download."""
        if self.is_video:
            return self.video_frames, "video", self.fps
        elif self.current_image:
            return self.current_image, "image", None
        return None, None, None
