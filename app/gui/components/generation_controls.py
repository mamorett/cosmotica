from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QGroupBox, QPushButton,
    QProgressBar
)
from PyQt6.QtCore import pyqtSignal

class GenerationControls(QWidget):
    """Controls for generation parameters and execution."""
    
    generate_clicked = pyqtSignal(dict)
    cancel_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Parameters group
        params_group = QGroupBox("Generation Parameters")
        params_layout = QVBoxLayout()
        
        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 1920)
        self.width_spin.setValue(1280)
        self.width_spin.setSingleStep(64)
        self.width_spin.setSuffix(" px")
        
        res_layout.addWidget(QLabel("W:"))
        res_layout.addWidget(self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 1080)
        self.height_spin.setValue(720)
        self.height_spin.setSingleStep(64)
        self.height_spin.setSuffix(" px")
        
        res_layout.addWidget(QLabel("H:"))
        res_layout.addWidget(self.height_spin)
        res_layout.addStretch()
        
        # Video-specific controls
        video_params_layout = QHBoxLayout()
        
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 60.0)
        self.fps_spin.setValue(24.0)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setSuffix(" FPS")
        
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 250)
        self.frames_spin.setValue(189)
        self.frames_spin.setSuffix(" frames")
        
        video_params_layout.addWidget(QLabel("FPS:"))
        video_params_layout.addWidget(self.fps_spin)
        video_params_layout.addWidget(QLabel("Frames:"))
        video_params_layout.addWidget(self.frames_spin)
        video_params_layout.addStretch()
        
        self.video_params_widget = QWidget()
        self.video_params_widget.setLayout(video_params_layout)
        self.video_params_widget.setVisible(False)
        
        # Advanced parameters
        adv_layout = QHBoxLayout()
        
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(5, 100)  # Lower minimum step for faster test iterations
        self.steps_spin.setValue(50)
        self.steps_spin.setSuffix(" steps")
        
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(7.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setSuffix(" CFG")
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("Random")
        
        adv_layout.addWidget(QLabel("Steps:"))
        adv_layout.addWidget(self.steps_spin)
        adv_layout.addWidget(QLabel("Guidance:"))
        adv_layout.addWidget(self.guidance_spin)
        adv_layout.addWidget(QLabel("Seed:"))
        adv_layout.addWidget(self.seed_spin)
        adv_layout.addStretch()
        
        # Add all to params layout
        params_layout.addLayout(res_layout)
        params_layout.addWidget(self.video_params_widget)
        params_layout.addLayout(adv_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Generate button
        self.generate_btn = QPushButton("🚀 Generate")
        self.generate_btn.setObjectName("generateButton")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setMinimumHeight(50)
        layout.addWidget(self.generate_btn)
    
    def _on_generate(self):
        """Collect parameters and emit generate signal."""
        params = {
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
            "seed": self.seed_spin.value() if self.seed_spin.value() != -1 else None,
            "fps": self.fps_spin.value(),
            "num_frames": self.frames_spin.value(),
        }
        self.generate_clicked.emit(params)
    
    def set_mode(self, mode: str):
        """Update UI based on generation mode."""
        self.video_params_widget.setVisible(mode == "i2v")
    
    def set_progress(self, value: int, message: str = ""):
        """Update progress bar."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        if message:
            self.progress_bar.setFormat(f"{message} - %p%")
    
    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.setVisible(False)
    
    def set_generating(self, is_generating: bool):
        """Enable/disable controls during generation."""
        self.generate_btn.setEnabled(not is_generating)
        self.generate_btn.setText("⏳ Generating..." if is_generating else "🚀 Generate")
        self.width_spin.setEnabled(not is_generating)
        self.height_spin.setEnabled(not is_generating)
        self.fps_spin.setEnabled(not is_generating)
        self.frames_spin.setEnabled(not is_generating)
        self.steps_spin.setEnabled(not is_generating)
        self.guidance_spin.setEnabled(not is_generating)
        self.seed_spin.setEnabled(not is_generating)
