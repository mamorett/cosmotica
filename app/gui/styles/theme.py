"""
Modern, beautiful theme for Cosmos3 Studio
Dark theme with vibrant accent colors
"""

MAIN_STYLE = """
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}

QLabel {
    color: #e0e0e0;
    font-size: 13px;
}

QLabel#titleLabel {
    font-size: 24px;
    font-weight: bold;
    color: #00d2ff;
    padding: 10px;
    background: transparent;
}

QLabel#sectionLabel {
    font-size: 16px;
    font-weight: bold;
    color: #00d2ff;
    padding: 5px;
    background: transparent;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #667eea, stop:1 #764ba2);
    color: white;
    border: none;
    padding: 8px 15px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #764ba2, stop:1 #667eea);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5a6fd6, stop:1 #6a4192);
}

QPushButton:disabled {
    background: #3a3a4e;
    color: #666;
}

QPushButton#generateButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00d2ff, stop:1 #3a7bd5);
    font-size: 16px;
    padding: 15px 40px;
    border-radius: 8px;
}

QPushButton#generateButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3a7bd5, stop:1 #00d2ff);
}

QPushButton#testConnectionButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00ff88, stop:1 #00d2ff);
    color: #1a1a2e;
    font-weight: bold;
    padding: 6px 12px;
}

QPushButton#testConnectionButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00d2ff, stop:1 #00ff88);
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #16213e;
    border: 2px solid #0f3460;
    border-radius: 8px;
    padding: 8px;
    color: #e0e0e0;
    font-size: 13px;
    selection-background-color: #533483;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #00d2ff;
}

QComboBox {
    background-color: #16213e;
    border: 2px solid #0f3460;
    border-radius: 8px;
    padding: 6px 12px;
    color: #e0e0e0;
    font-size: 13px;
    min-width: 150px;
}

QComboBox:hover {
    border-color: #00d2ff;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 8px solid #00d2ff;
    margin-right: 5px;
}

QSpinBox, QDoubleSpinBox {
    background-color: #16213e;
    border: 2px solid #0f3460;
    border-radius: 8px;
    padding: 6px;
    color: #e0e0e0;
    font-size: 13px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #00d2ff;
}

QProgressBar {
    border: 2px solid #0f3460;
    border-radius: 8px;
    text-align: center;
    color: white;
    font-weight: bold;
    background-color: #16213e;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2ff, stop:1 #3a7bd5);
    border-radius: 6px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QGroupBox {
    border: 2px solid #0f3460;
    border-radius: 10px;
    margin-top: 15px;
    padding-top: 15px;
    background-color: #16213e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 10px;
    color: #00d2ff;
    font-weight: bold;
    font-size: 13px;
}

QSlider::groove:horizontal {
    border: 1px solid #0f3460;
    height: 8px;
    background: #16213e;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #00d2ff;
    border: none;
    width: 18px;
    margin: -5px 0;
    border-radius: 9px;
}

QCheckBox {
    color: #e0e0e0;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #0f3460;
    border-radius: 4px;
    background-color: #16213e;
}

QCheckBox::indicator:checked {
    background-color: #00d2ff;
    border-color: #00d2ff;
}

QTabWidget::pane {
    border: 2px solid #0f3460;
    border-radius: 10px;
    background-color: #16213e;
}

QTabBar::tab {
    background-color: #1a1a2e;
    color: #e0e0e0;
    padding: 10px 20px;
    margin-right: 5px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QTabBar::tab:selected {
    background-color: #16213e;
    color: #00d2ff;
    border-bottom: 3px solid #00d2ff;
}

QTabBar::tab:hover {
    background-color: #0f3460;
}

QMenuBar {
    background-color: #16213e;
    color: #e0e0e0;
    border-bottom: 1px solid #0f3460;
}

QMenuBar::item:selected {
    background-color: #0f3460;
}

QMenu {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
}

QMenu::item:selected {
    background-color: #533483;
}

QStatusBar {
    background-color: #16213e;
    color: #00d2ff;
    border-top: 1px solid #0f3460;
}

QToolTip {
    background-color: #16213e;
    color: #00d2ff;
    border: 1px solid #0f3460;
    padding: 5px;
    border-radius: 4px;
}
"""

DRAG_DROP_STYLE = """
QLabel#dropZone {
    border: 3px dashed #0f3460;
    border-radius: 15px;
    background-color: #16213e;
    color: #e0e0e0;
    font-size: 14px;
    padding: 30px;
}

QLabel#dropZone:hover {
    border-color: #00d2ff;
    background-color: #1a1a3e;
}

QLabel#dropZone[dragActive="true"] {
    border-color: #00d2ff;
    background-color: #1a1a4e;
    border-style: solid;
}
"""
