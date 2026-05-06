"""
SignWave — Premium Luxury Theme · Fossil Watch Edition

Color palette, typography, and QSS stylesheets for the entire application.
Inspired by the craftsmanship of a $500M luxury timepiece: deep obsidian
blacks, polished 18k gold accents, brushed metal textures, and ivory dials.

This is NOT futuristic.  It is timeless, refined, and authoritative.
"""

# ── Color Constants ─────────────────────────────────────────────────────

# Backgrounds — Obsidian & Leather
BG_DEEP       = "#060504"          # Near-black with warm undertone
BG_CARD       = "#0F0D0A"          # Card surface — dark leather
BG_SURFACE    = "#171310"          # Raised surface — ebony wood
BG_GLASS      = "rgba(15, 13, 10, 0.92)"
BG_GLASS_HOVER = "rgba(23, 19, 16, 0.95)"
BG_INPUT      = "#0B0908"          # Input fields — deep well

# Gold — Primary Accent (18k Polished)
GOLD          = "#D4AF37"          # Classic 18k gold
GOLD_BRIGHT   = "#F0D060"          # Highlight shine / hover state
GOLD_DIM      = "#8B7536"          # Brushed / recessed gold
GOLD_GLOW     = "rgba(212, 175, 55, 0.35)"
GOLD_SUBTLE   = "rgba(212, 175, 55, 0.12)"

# Ivory — Secondary accent (watch dial)
IVORY         = "#F5F0E1"          # Warm cream / dial face
IVORY_DIM     = "#C8BFA8"          # Aged parchment

# Text
TEXT_PRIMARY   = "#F2EDE3"         # Warm off-white (cream)
TEXT_SECONDARY = "#9C9486"         # Muted warm grey
TEXT_DIM       = "#5A5347"         # Engraved / subtle

# Borders
BORDER_GOLD    = "rgba(212, 175, 55, 0.40)"
BORDER_SUBTLE  = "rgba(212, 175, 55, 0.12)"
BORDER_DARK    = "rgba(90, 83, 71, 0.25)"

# Status
SUCCESS        = "#7A9C5D"         # Muted olive green
DANGER         = "#C0392B"         # Deep burgundy red
WARNING        = "#D4AF37"         # Gold as warning

# ── Shared Dimensions ──────────────────────────────────────────────────

RADIUS        = "8px"
RADIUS_SM     = "4px"
RADIUS_LG     = "14px"
RADIUS_PILL   = "999px"

PADDING       = "20px"
PADDING_SM    = "12px"
PADDING_LG    = "32px"

# Fonts — Classic, refined, not techy
FONT_FAMILY   = "'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif"
FONT_SERIF    = "'Georgia', 'Palatino Linotype', 'Times New Roman', serif"
FONT_MONO     = "'Consolas', 'Cascadia Code', monospace"


# ── Global Application QSS ─────────────────────────────────────────────

def get_app_stylesheet() -> str:
    """Return the master QSS stylesheet for the entire application."""
    return f"""
    /* ═══════════ GLOBAL RESETS ═══════════ */

    * {{
        font-family: {FONT_FAMILY};
        color: {TEXT_PRIMARY};
        outline: none;
    }}

    QMainWindow {{
        background-color: {BG_DEEP};
    }}

    QWidget {{
        background-color: transparent;
    }}

    /* ═══════════ SCROLL BARS ═══════════ */

    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {GOLD_DIM};
        min-height: 30px;
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {GOLD};
    }}
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: transparent;
    }}

    /* ═══════════ LABELS ═══════════ */

    QLabel {{
        background: transparent;
        padding: 0;
    }}

    /* ═══════════ BUTTONS ═══════════ */

    QPushButton {{
        background-color: {BG_SURFACE};
        color: {GOLD};
        border: 1px solid {BORDER_GOLD};
        border-radius: {RADIUS_SM};
        padding: 10px 22px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
    }}
    QPushButton:hover {{
        background-color: {BG_GLASS_HOVER};
        border-color: {GOLD};
        color: {GOLD_BRIGHT};
    }}
    QPushButton:pressed {{
        background-color: rgba(212, 175, 55, 0.10);
    }}
    QPushButton:checked {{
        color: {GOLD};
        border-bottom: 2px solid {GOLD};
        background-color: rgba(212, 175, 55, 0.06);
    }}
    QPushButton:disabled {{
        color: {TEXT_DIM};
        border-color: {BORDER_SUBTLE};
    }}

    /* ═══════════ INPUTS ═══════════ */

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {BG_INPUT};
        color: {IVORY};
        border: 1px solid {BORDER_SUBTLE};
        border-radius: {RADIUS_SM};
        padding: 8px 12px;
        font-size: 13px;
        selection-background-color: {GOLD_DIM};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {GOLD};
    }}

    /* ═══════════ GROUP BOX ═══════════ */

    QGroupBox {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER_SUBTLE};
        border-radius: {RADIUS};
        margin-top: 14px;
        padding: 20px 16px 16px 16px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        color: {GOLD_DIM};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 18px;
        padding: 2px 10px;
        background-color: {BG_CARD};
        color: {GOLD};
        font-size: 11px;
        letter-spacing: 2px;
    }}

    /* ═══════════ TOGGLE / CHECKBOX ═══════════ */

    QCheckBox {{
        spacing: 8px;
        font-size: 13px;
    }}
    QCheckBox::indicator {{
        width: 32px;
        height: 18px;
        border-radius: 9px;
        background-color: {BG_SURFACE};
        border: 1px solid {BORDER_SUBTLE};
    }}
    QCheckBox::indicator:checked {{
        background-color: {GOLD_DIM};
        border-color: {GOLD};
    }}

    /* ═══════════ STATUS BAR ═══════════ */

    QStatusBar {{
        background-color: {BG_DEEP};
        color: {GOLD_DIM};
        font-size: 11px;
        letter-spacing: 1px;
        border-top: 1px solid {BORDER_SUBTLE};
        padding: 4px 16px;
        font-family: {FONT_SERIF};
    }}
    """


# ── Glass Panel Mixin QSS ──────────────────────────────────────────────

GLASS_PANEL_QSS = f"""
    background-color: {BG_GLASS};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: {RADIUS};
"""
