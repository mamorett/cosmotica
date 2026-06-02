from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QTextEdit, QGroupBox
)
from PyQt6.QtCore import pyqtSignal

class PromptInput(QWidget):
    """Prompt input component with positive and negative prompts."""
    
    prompts_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Prompts group
        prompts_group = QGroupBox("Prompt Settings")
        prompts_layout = QVBoxLayout()
        
        # Positive prompt
        pos_label = QLabel("Positive Prompt:")
        pos_label.setStyleSheet("font-weight: bold; color: #00ff88;")
        
        self.positive_prompt = QTextEdit()
        self.positive_prompt.setPlaceholderText(
            "Describe what you want to generate in detail..."
        )
        self.positive_prompt.setMaximumHeight(120)
        self.positive_prompt.setMinimumHeight(80)
        self.positive_prompt.textChanged.connect(self._on_prompts_changed)
        
        # Negative prompt
        neg_label = QLabel("Negative Prompt:")
        neg_label.setStyleSheet("font-weight: bold; color: #ff6b6b;")
        
        self.negative_prompt = QTextEdit()
        self.negative_prompt.setPlaceholderText(
            "Describe what you want to avoid in the generation..."
        )
        self.negative_prompt.setMaximumHeight(120)
        self.negative_prompt.setMinimumHeight(80)
        self.negative_prompt.textChanged.connect(self._on_prompts_changed)
        
        # Add to layout
        prompts_layout.addWidget(pos_label)
        prompts_layout.addWidget(self.positive_prompt)
        prompts_layout.addWidget(neg_label)
        prompts_layout.addWidget(self.negative_prompt)
        
        prompts_group.setLayout(prompts_layout)
        layout.addWidget(prompts_group)
    
    def _on_prompts_changed(self):
        """Emit signal when prompts change."""
        self.prompts_changed.emit(self.get_prompts())
    
    def get_prompts(self) -> dict:
        """Get current prompt values."""
        return {
            "positive": self.positive_prompt.toPlainText(),
            "negative": self.negative_prompt.toPlainText()
        }
    
    def set_negative_prompt_default(self, mode: str):
        """Set default negative prompt based on mode."""
        if mode == "i2v":
            default_neg = (
                "The video captures a series of frames showing macroblocking artifacts, "
                "chromatic aberration, high-frequency noise, and rolling shutter distortion. "
                "It includes static with no motion, motion blur, over-saturation, shaky footage, "
                "low resolution, grainy texture, pixelated images, poorly lit areas, "
                "underexposed and overexposed scenes, poor color balance, washed out colors, "
                "choppy sequences, jerky movements, low frame rate, bit-depth compression artifacts, "
                "color banding, unnatural transitions, outdated special effects, fake elements, "
                "unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
                "Avoid moiré patterns, edge halos, and temporal aliasing."
            )
            self.negative_prompt.setText(default_neg)
        else:
            # Clear or set mild default for text-to-image
            self.negative_prompt.setText("deformed, distorted, disfigured, poorly drawn, bad anatomy, low quality, pixelated, blurry")
