"""
BrainBit EEG Viewer v4.3
- Исправлены NaN ошибки
- Метрики отсортированы по достоверности
- Оптимизированная структура

Электроды BrainBit Black: O1, O2 (затылок), T3, T4 (виски)
"""

import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
from collections import deque
from enum import Enum

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                              QWidget, QLabel, QGroupBox, QGridLayout, QProgressBar,
                              QDialog, QPushButton)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QCursor
import pyqtgraph as pg
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy
from pylsl import StreamInlet, resolve_streams

# NumPy 2.0+
try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================
def safe_val(v, default=0.0):
    """Безопасное значение — заменяет None/NaN/Inf на default"""
    if v is None:
        return default
    try:
        if np.isnan(v) or np.isinf(v):
            return default
    except (TypeError, ValueError):
        return default
    return float(v)


def safe_mean(values, default=0.0):
    """Безопасное среднее"""
    filtered = [v for v in values if v is not None]
    try:
        filtered = [v for v in filtered if not np.isnan(v) and not np.isinf(v)]
    except:
        pass
    if not filtered:
        return default
    return float(np.mean(filtered))


def safe_div(a, b, default=0.0):
    """Безопасное деление"""
    a, b = safe_val(a), safe_val(b)
    if b == 0:
        return default
    result = a / b
    return safe_val(result, default)


# ============================================================================
# КОНСТАНТЫ
# ============================================================================
SRATE = 250
BUFFER_SEC = 4
HISTORY_SEC = 120
CHANNELS = ['O1', 'O2', 'T3', 'T4']

CHANNEL_REGIONS = {
    'occipital': ['O1', 'O2'],
    'temporal': ['T3', 'T4'],
    'left': ['O1', 'T3'],
    'right': ['O2', 'T4'],
}


# ============================================================================
# ДОСТОВЕРНОСТЬ МЕТРИК
# ============================================================================
class Reliability(Enum):
    HIGH = "high"        # ✅ Достоверные
    MEDIUM = "medium"    # ⚠️ Упрощённые  
    LOW = "low"          # ❓ Приблизительные


# ============================================================================
# ЧАСТОТНЫЕ ДИАПАЗОНЫ
# ============================================================================
@dataclass
class BandConfig:
    name: str
    range: Tuple[float, float]
    color: str
    desc: str


BANDS = {
    'Delta': BandConfig('Delta', (0.5, 4), '#8B5CF6', '💤 Сон'),
    'Theta': BandConfig('Theta', (4, 8), '#06B6D4', '🧘 Медитация'),
    'Alpha': BandConfig('Alpha', (8, 13), '#10B981', '😌 Расслабление'),
    'Beta': BandConfig('Beta', (13, 30), '#F59E0B', '💭 Фокус'),
    'Gamma': BandConfig('Gamma', (30, 45), '#EF4444', '⚡ Гиперфокус'),
}

SUB_BANDS = {
    'Theta_low': BandConfig('θ low', (4, 6), '#06B6D4', 'Сонливость'),
    'Theta_high': BandConfig('θ high', (6, 8), '#0891B2', 'Память'),
    'Alpha_low': BandConfig('α low', (8, 10), '#10B981', 'Внимание'),
    'Alpha_high': BandConfig('α high', (10, 13), '#059669', 'Покой'),
    'SMR': BandConfig('SMR', (12, 15), '#84CC16', 'Спокойный фокус'),
    'Beta_low': BandConfig('β low', (15, 20), '#F59E0B', 'Мышление'),
    'Beta_high': BandConfig('β high', (20, 30), '#DC2626', 'Тревога'),
}


# ============================================================================
# КОНФИГУРАЦИЯ МЕТРИК
# ============================================================================
@dataclass
class MetricConfig:
    key: str
    name: str
    color: str
    formula: Callable
    reliability: Reliability
    desc: str = ""
    formula_str: str = ""
    min_val: float = 0
    max_val: float = 100
    unit: str = "%"
    source: str = ""


# ============================================================================
# ФОРМУЛЫ МЕТРИК
# ============================================================================

# === ДОСТОВЕРНЫЕ (HIGH) ===

def calc_engagement(d: Dict) -> float:
    """β / (α + θ) — NASA Engagement Index"""
    a, b, t = d.get('Alpha', 0), d.get('Beta', 0), d.get('Theta', 0)
    raw = safe_div(b, a + t + 0.1)
    return np.clip(raw / 2.0 * 100, 0, 100)


def calc_attention(d: Dict) -> float:
    """100 - norm(θ/β) — Theta/Beta ratio"""
    t, b = d.get('Theta', 0), d.get('Beta', 0)
    tbr = safe_div(t, b + 0.1)
    normalized = np.clip((tbr - 1.0) / 3.0, 0, 1)
    return (1 - normalized) * 100


def calc_relaxation(d: Dict) -> float:
    """Затылочный Alpha"""
    occ_alpha = d.get('occ_alpha', d.get('Alpha', 0))
    alpha_high = d.get('Alpha_high', safe_val(occ_alpha) * 0.6)
    return np.clip(safe_val(alpha_high) / 20 * 100, 0, 100)


def calc_drowsiness(d: Dict) -> float:
    """(θ_low + δ×0.5) / (α + β + SMR)"""
    delta = d.get('Delta', 0)
    theta_low = d.get('Theta_low', d.get('Theta', 0) * 0.5)
    a, b = d.get('Alpha', 0), d.get('Beta', 0)
    smr = d.get('SMR', 0)
    
    num = safe_val(theta_low) + safe_val(delta) * 0.5
    denom = safe_val(a) + safe_val(b) + safe_val(smr) + 1
    
    return np.clip(safe_div(num, denom) * 80, 0, 100)


def calc_arousal(d: Dict) -> float:
    """(β + γ) / (δ + θ) — Корковая активация"""
    delta, theta = d.get('Delta', 0), d.get('Theta', 0)
    b, g = d.get('Beta', 0), d.get('Gamma', 0)
    
    num = safe_val(b) + safe_val(g)
    denom = safe_val(delta) + safe_val(theta) + 1
    
    return np.clip(safe_div(num, denom) * 40, 0, 100)


def calc_spectral_entropy(d: Dict) -> float:
    """Энтропия Шеннона спектра"""
    psd = d.get('psd')
    if psd is None or len(psd) < 2:
        return 0.5
    
    psd_sum = np.sum(psd)
    if psd_sum <= 0:
        return 0.5
    
    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]
    
    if len(psd_norm) < 2:
        return 0.5
    
    se = scipy_entropy(psd_norm, base=2)
    max_ent = np.log2(len(psd))
    
    return np.clip(safe_div(se, max_ent, 0.5), 0, 1)


def calc_alpha_peak(d: Dict) -> float:
    """Пик Alpha (IAF)"""
    freqs, psd = d.get('freqs'), d.get('psd')
    if freqs is None or psd is None:
        return 10.0
    
    mask = (freqs >= 8) & (freqs <= 13)
    if not np.any(mask):
        return 10.0
    
    alpha_psd = psd[mask]
    alpha_freqs = freqs[mask]
    
    if len(alpha_psd) == 0:
        return 10.0
    
    return float(alpha_freqs[np.argmax(alpha_psd)])


def calc_symmetry(d: Dict) -> float:
    """Симметрия полушарий"""
    left = safe_val(d.get('power_left', 0))
    right = safe_val(d.get('power_right', 0))
    
    total = left + right
    if total == 0:
        return 100
    
    asym = abs(left - right) / total
    return np.clip((1 - asym) * 100, 0, 100)


def calc_hjorth_activity(d: Dict) -> float:
    """Hjorth Activity = var(signal)"""
    sig = d.get('signal')
    if sig is None or len(sig) < 2:
        return 0
    return float(np.var(sig))


def calc_hjorth_mobility(d: Dict) -> float:
    """Hjorth Mobility"""
    sig = d.get('signal')
    if sig is None or len(sig) < 2:
        return 0
    
    var_sig = np.var(sig)
    if var_sig == 0:
        return 0
    
    var_d = np.var(np.diff(sig))
    return float(np.sqrt(var_d / var_sig))


def calc_hjorth_complexity(d: Dict) -> float:
    """Hjorth Complexity"""
    sig = d.get('signal')
    if sig is None or len(sig) < 3:
        return 0
    
    def mob(s):
        if len(s) < 2:
            return 0
        v = np.var(s)
        if v == 0:
            return 0
        return np.sqrt(np.var(np.diff(s)) / v)
    
    m_sig = mob(sig)
    if m_sig == 0:
        return 0
    
    return float(mob(np.diff(sig)) / m_sig)


# === УПРОЩЁННЫЕ (MEDIUM) ===

def calc_focus(d: Dict) -> float:
    """(β + SMR) × (1 - α/40) × (1 - θ/30)"""
    a, b, t = d.get('Alpha', 0), d.get('Beta', 0), d.get('Theta', 0)
    smr = d.get('SMR', 0)
    
    a, b, t, smr = safe_val(a), safe_val(b), safe_val(t), safe_val(smr)
    
    alpha_supp = max(0, 1 - a / 40)
    theta_pen = max(0, 1 - t / 30)
    
    raw = (b + smr) * alpha_supp * theta_pen
    return np.clip(raw / 20 * 100, 0, 100)


def calc_meditation(d: Dict) -> float:
    """(α + θ) × (1 - β/40)"""
    a, t, b = d.get('Alpha', 0), d.get('Theta', 0), d.get('Beta', 0)
    a, t, b = safe_val(a), safe_val(t), safe_val(b)
    
    beta_supp = max(0, 1 - b / 40)
    raw = (a + t) * beta_supp
    return np.clip(raw / 30 * 100, 0, 100)


def calc_cognitive_load(d: Dict) -> float:
    """θ_high × (1 + (40-α)/40)"""
    a = safe_val(d.get('Alpha', 0))
    th = safe_val(d.get('Theta_high', d.get('Theta', 0) * 0.5))
    
    alpha_factor = max(0.5, 1 + (40 - a) / 40)
    raw = th * alpha_factor
    return np.clip(raw / 15 * 100, 0, 100)


def calc_fatigue(d: Dict) -> float:
    """(θ + α) / β"""
    a, t, b = d.get('Alpha', 0), d.get('Theta', 0), d.get('Beta', 0)
    a, t, b = safe_val(a), safe_val(t), safe_val(b)
    
    raw = safe_div(t + a, b + 0.1)
    return np.clip(raw * 25, 0, 100)


def calc_flow(d: Dict) -> float:
    """α_optimal × (β/θ)"""
    a, t, b = d.get('Alpha', 0), d.get('Theta', 0), d.get('Beta', 0)
    a, t, b = safe_val(a), safe_val(t), safe_val(b)
    
    alpha_opt = max(0, 1 - abs(a - 15) / 15)
    bt = safe_div(b, t + 0.1)
    
    return np.clip(alpha_opt * bt * 40, 0, 100)


def calc_creativity(d: Dict) -> float:
    """α × θ / (β² + 1)"""
    a, t, b = d.get('Alpha', 0), d.get('Theta', 0), d.get('Beta', 0)
    a, t, b = safe_val(a), safe_val(t), safe_val(b)
    
    raw = (a * t) / (b ** 2 + 1)
    return np.clip(raw * 15, 0, 100)


def calc_memory(d: Dict) -> float:
    """Височный Theta"""
    temp_theta = safe_val(d.get('temp_theta', d.get('Theta', 0)))
    return np.clip(temp_theta / 25 * 100, 0, 100)


# === ПРИБЛИЗИТЕЛЬНЫЕ (LOW) ===

def calc_stress(d: Dict) -> float:
    """β_high/(α+1)×20 + max(0, β-α)"""
    a, b = safe_val(d.get('Alpha', 0)), safe_val(d.get('Beta', 0))
    bh = safe_val(d.get('Beta_high', b * 0.4))
    
    c1 = bh / (a + 1) * 20
    c2 = max(0, b - a)
    
    return np.clip((c1 + c2) / 25 * 100, 0, 100)


def calc_valence(d: Dict) -> float:
    """α_right - α_left (нужны F3/F4)"""
    a_left = safe_val(d.get('Alpha_left', 0))
    a_right = safe_val(d.get('Alpha_right', 0))
    
    asym = a_right - a_left
    return np.clip(50 + asym * 2.5, 0, 100)


# ============================================================================
# СЛОВАРЬ МЕТРИК (отсортирован по достоверности)
# ============================================================================
METRICS: Dict[str, MetricConfig] = {
    # === ✅ ДОСТОВЕРНЫЕ ===
    'engagement': MetricConfig(
        'engagement', '🔥 Вовлечённость', '#06B6D4',
        calc_engagement, Reliability.HIGH,
        'Активность обработки информации', 'β/(α+θ)',
        source='Pope et al., 1995'
    ),
    'attention': MetricConfig(
        'attention', '👁️ Внимание', '#3B82F6',
        calc_attention, Reliability.HIGH,
        'Устойчивость внимания', '100-norm(θ/β)',
        source='Lubar, 1991'
    ),
    'relaxation': MetricConfig(
        'relaxation', '😌 Расслабление', '#10B981',
        calc_relaxation, Reliability.HIGH,
        'Уровень расслабления', 'Occipital α',
        source='Berger, 1929'
    ),
    'drowsiness': MetricConfig(
        'drowsiness', '😴 Сонливость', '#8B5CF6',
        calc_drowsiness, Reliability.HIGH,
        'Уровень сонливости', '(θ+δ)/(α+β)',
        source='Santamaria, 1987'
    ),
    'arousal': MetricConfig(
        'arousal', '⚡ Активация', '#EF4444',
        calc_arousal, Reliability.HIGH,
        'Корковая активация', '(β+γ)/(δ+θ)',
        source='Klimesch, 1999'
    ),
    'symmetry': MetricConfig(
        'symmetry', '⚖️ Симметрия', '#6B7280',
        calc_symmetry, Reliability.HIGH,
        'Баланс полушарий', '1-|L-R|/(L+R)'
    ),
    'spectral_entropy': MetricConfig(
        'spectral_entropy', '🎲 Энтропия', '#EC4899',
        calc_spectral_entropy, Reliability.HIGH,
        'Сложность сигнала', 'Shannon(PSD)',
        min_val=0, max_val=1, unit='',
        source='Inouye, 1991'
    ),
    'alpha_peak': MetricConfig(
        'alpha_peak', '📊 Пик α', '#10B981',
        calc_alpha_peak, Reliability.HIGH,
        'Индивидуальная частота α', 'argmax(8-13Hz)',
        min_val=8, max_val=13, unit='Hz',
        source='Klimesch, 1999'
    ),
    
    # === ⚠️ УПРОЩЁННЫЕ ===
    'focus': MetricConfig(
        'focus', '🎯 Фокус', '#F59E0B',
        calc_focus, Reliability.MEDIUM,
        'Концентрация внимания', '(β+SMR)×(1-α/40)'
    ),
    'meditation': MetricConfig(
        'meditation', '🧘 Медитация', '#14B8A6',
        calc_meditation, Reliability.MEDIUM,
        'Медитативное состояние', '(α+θ)×(1-β/40)'
    ),
    'cognitive_load': MetricConfig(
        'cognitive_load', '🧠 Нагрузка', '#8B5CF6',
        calc_cognitive_load, Reliability.MEDIUM,
        'Когнитивная нагрузка', 'θ_high×(1+(40-α)/40)',
        source='Gevins, 1997'
    ),
    'fatigue': MetricConfig(
        'fatigue', '🔋 Усталость', '#F97316',
        calc_fatigue, Reliability.MEDIUM,
        'Ментальная усталость', '(θ+α)/β',
        source='Eoh, 2005'
    ),
    'flow': MetricConfig(
        'flow', '🌊 Поток', '#06B6D4',
        calc_flow, Reliability.MEDIUM,
        'Состояние потока', 'α_opt×(β/θ)',
        source='Katahira, 2018'
    ),
    'creativity': MetricConfig(
        'creativity', '💡 Креативность', '#A855F7',
        calc_creativity, Reliability.MEDIUM,
        'Творческое мышление', 'α×θ/(β²+1)',
        source='Fink, 2014'
    ),
    'memory': MetricConfig(
        'memory', '📝 Память', '#06B6D4',
        calc_memory, Reliability.MEDIUM,
        'Кодирование памяти', 'Temporal θ',
        source='Klimesch, 1999'
    ),
    
    # === ❓ ПРИБЛИЗИТЕЛЬНЫЕ ===
    'stress': MetricConfig(
        'stress', '😰 Стресс*', '#EF4444',
        calc_stress, Reliability.LOW,
        'Стресс (нужны F3/F4)', 'β_high/α'
    ),
    'valence': MetricConfig(
        'valence', '🙂 Настрой*', '#EC4899',
        calc_valence, Reliability.LOW,
        'Валентность (нужны F3/F4)', 'α_R-α_L',
        source='Davidson, 2004'
    ),
}

# Группы для UI
METRICS_HIGH = [k for k, v in METRICS.items() if v.reliability == Reliability.HIGH]
METRICS_MEDIUM = [k for k, v in METRICS.items() if v.reliability == Reliability.MEDIUM]
METRICS_LOW = [k for k, v in METRICS.items() if v.reliability == Reliability.LOW]


# ============================================================================
# ДЕТЕКТОР МОРГАНИЙ
# ============================================================================
class BlinkDetector:
    def __init__(self, srate=250):
        self.srate = srate
        self.threshold = 75
        self.min_dur = int(0.08 * srate)
        self.max_dur = int(0.4 * srate)
        self.refractory = int(0.3 * srate)
        
        self.in_blink = False
        self.blink_start = 0
        self.blink_peak = 0
        self.sample_idx = 0
        self.last_blink_end = 0
        
        self.blink_times = deque(maxlen=500)
        self.blink_count = 0
        self.recent_blinks = deque(maxlen=30)
        self.is_blinking = False
    
    def process(self, samples):
        detected = []
        
        for sample in samples:
            self.sample_idx += 1
            sample = np.array(sample)
            
            max_amp = np.max(np.abs(sample))
            mean_amp = np.mean(np.abs(sample))
            
            if self.sample_idx - self.last_blink_end < self.refractory:
                self.is_blinking = False
                continue
            
            if not self.in_blink:
                if max_amp > self.threshold and mean_amp > self.threshold * 0.6:
                    self.in_blink = True
                    self.is_blinking = True
                    self.blink_start = self.sample_idx
                    self.blink_peak = max_amp
            else:
                self.blink_peak = max(self.blink_peak, max_amp)
                self.is_blinking = True
                
                if max_amp < self.threshold * 0.4:
                    dur = self.sample_idx - self.blink_start
                    
                    if self.min_dur <= dur <= self.max_dur:
                        self.blink_count += 1
                        self.blink_times.append(self.sample_idx)
                        self.last_blink_end = self.sample_idx
                        
                        dur_ms = dur * 1000 / self.srate
                        detected.append({
                            'sample': self.blink_start,
                            'duration_ms': dur_ms,
                            'amplitude': self.blink_peak
                        })
                        self.recent_blinks.append((self.blink_start, self.blink_peak, dur_ms))
                    
                    self.in_blink = False
                    self.is_blinking = False
                    
                elif self.sample_idx - self.blink_start > self.max_dur:
                    self.in_blink = False
                    self.is_blinking = False
        
        return detected
    
    def get_rate(self):
        if len(self.blink_times) < 1:
            return 0.0
        
        one_min = self.sample_idx - 60 * self.srate
        recent = sum(1 for t in self.blink_times if t > one_min)
        elapsed = min(self.sample_idx / self.srate, 60)
        
        return recent * 60 / elapsed if elapsed > 5 else 0.0
    
    def get_artifact_mask(self, buf_len, margin_ms=100):
        mask = np.zeros(buf_len, dtype=bool)
        buf_start = self.sample_idx - buf_len
        margin = int(margin_ms * self.srate / 1000)
        
        for bs, amp, dur_ms in self.recent_blinks:
            dur_samp = int(dur_ms * self.srate / 1000)
            start = max(0, bs - margin - buf_start)
            end = min(buf_len, bs + dur_samp + margin - buf_start)
            if start < buf_len and end > 0:
                mask[start:end] = True
        
        return mask
    
    def get_blink_positions(self, buf_len):
        positions = []
        buf_start = self.sample_idx - buf_len
        
        for bs, amp, _ in self.recent_blinks:
            if bs > buf_start:
                pos = bs - buf_start
                if 0 <= pos < buf_len:
                    positions.append((int(pos), amp))
        
        return positions


# ============================================================================
# ДЕТЕКТОР АРТЕФАКТОВ
# ============================================================================
class ArtifactType(Enum):
    BLINK = "blink"
    EYE = "eye"
    MUSCLE = "muscle"
    ELECTRODE = "electrode"
    MOVEMENT = "movement"


@dataclass
class ArtifactInfo:
    has_artifact: bool = False
    types: List[ArtifactType] = None
    description: str = ""
    
    def __post_init__(self):
        if self.types is None:
            self.types = []


class ArtifactDetector:
    def __init__(self, srate=250):
        self.srate = srate
    
    def detect(self, data: Dict, blink_detector=None) -> ArtifactInfo:
        types = []
        descs = []
        
        win = self.srate // 2
        signals = {}
        
        for ch, sig in data.items():
            if len(sig) >= win:
                signals[ch] = np.array(sig[-win:])
        
        if len(signals) < 4:
            return ArtifactInfo()
        
        sig_list = [signals[ch] for ch in CHANNELS if ch in signals]
        
        # Моргание
        if blink_detector and blink_detector.is_blinking:
            types.append(ArtifactType.BLINK)
            descs.append("👁️")
        
        # Движение глаз
        max_amps = [np.max(np.abs(s)) for s in sig_list]
        if np.mean(max_amps) > 60 and not (blink_detector and blink_detector.is_blinking):
            corrs = []
            for i in range(len(sig_list)):
                for j in range(i+1, len(sig_list)):
                    try:
                        c = np.corrcoef(sig_list[i], sig_list[j])[0, 1]
                        if not np.isnan(c):
                            corrs.append(abs(c))
                    except:
                        pass
            
            if corrs and np.mean(corrs) > 0.7:
                types.append(ArtifactType.EYE)
                descs.append("👀")
        
        # Мышцы
        for ch, sig in signals.items():
            try:
                freqs, psd = scipy_signal.welch(sig, fs=self.srate, nperseg=len(sig))
                hf = np.sum(psd[(freqs >= 25) & (freqs <= 45)])
                lf = np.sum(psd[(freqs >= 1) & (freqs <= 25)])
                if lf > 0 and hf / lf > 0.8:
                    if ArtifactType.MUSCLE not in types:
                        types.append(ArtifactType.MUSCLE)
                        descs.append("💪")
                    break
            except:
                pass
        
        # Электрод
        for ch, sig in signals.items():
            if np.std(sig) < 2:
                types.append(ArtifactType.ELECTRODE)
                descs.append(f"⚡{ch}")
                break
        
        # Движение
        for sig in sig_list:
            if np.sum(np.abs(np.diff(sig)) > 50) > len(sig) * 0.1:
                if ArtifactType.MOVEMENT not in types:
                    types.append(ArtifactType.MOVEMENT)
                    descs.append("🏃")
                break
        
        return ArtifactInfo(
            has_artifact=len(types) > 0,
            types=types,
            description=" ".join(descs)
        )


# ============================================================================
# КАЛЬКУЛЯТОР МЕТРИК
# ============================================================================
class MetricCalculator:
    def __init__(self, srate=250):
        self.srate = srate
        self.ema = 0.3
        self.smoothed = {}
    
    def smooth(self, key, val):
        val = safe_val(val)
        if key not in self.smoothed:
            self.smoothed[key] = val
        else:
            self.smoothed[key] = self.ema * val + (1 - self.ema) * self.smoothed[key]
        return self.smoothed[key]
    
    def prepare(self, powers, sub_powers, signals):
        avg = {b: safe_mean([powers[ch].get(b, 0) for ch in powers if powers[ch]])
               for b in BANDS}
        
        avg_sub = {b: safe_mean([sub_powers[ch].get(b, 0) for ch in sub_powers if sub_powers[ch]])
                   for b in SUB_BANDS}
        
        occ_alpha = safe_mean([powers.get(ch, {}).get('Alpha', 0) for ch in CHANNEL_REGIONS['occipital']])
        temp_theta = safe_mean([powers.get(ch, {}).get('Theta', 0) for ch in CHANNEL_REGIONS['temporal']])
        
        left_power = sum(safe_val(powers.get(ch, {}).get('Alpha', 0)) + 
                        safe_val(powers.get(ch, {}).get('Beta', 0))
                        for ch in CHANNEL_REGIONS['left'])
        right_power = sum(safe_val(powers.get(ch, {}).get('Alpha', 0)) + 
                         safe_val(powers.get(ch, {}).get('Beta', 0))
                         for ch in CHANNEL_REGIONS['right'])
        
        freqs, psd, signal = None, None, None
        if 'O1' in signals and len(signals['O1']) >= self.srate:
            signal = signals['O1']
            try:
                freqs, psd = scipy_signal.welch(signal, fs=self.srate, 
                                                nperseg=min(len(signal), self.srate))
            except:
                pass
        
        return {
            **avg, **avg_sub,
            'occ_alpha': occ_alpha,
            'temp_theta': temp_theta,
            'Alpha_left': safe_mean([powers.get(ch, {}).get('Alpha', 0) for ch in CHANNEL_REGIONS['left']]),
            'Alpha_right': safe_mean([powers.get(ch, {}).get('Alpha', 0) for ch in CHANNEL_REGIONS['right']]),
            'power_left': left_power,
            'power_right': right_power,
            'freqs': freqs,
            'psd': psd,
            'signal': signal,
        }
    
    def calculate(self, powers, sub_powers, signals):
        data = self.prepare(powers, sub_powers, signals)
        results = {}
        
        for key, cfg in METRICS.items():
            try:
                val = cfg.formula(data)
                val = safe_val(val, 0 if cfg.min_val == 0 else cfg.min_val)
                
                # Сглаживание для основных метрик
                if cfg.reliability != Reliability.HIGH or key not in ['spectral_entropy', 'alpha_peak']:
                    val = self.smooth(key, val)
                
                results[key] = val
            except:
                results[key] = 0
        
        return results


# ============================================================================
# АНАЛИЗАТОР
# ============================================================================
class EEGAnalyzer:
    def __init__(self, srate=250):
        self.srate = srate
        self.calc = MetricCalculator(srate)
        self.blink = BlinkDetector(srate)
        self.artifact = ArtifactDetector(srate)
        
        try:
            self.filt_b, self.filt_a = scipy_signal.butter(4, [0.5, 45], btype='band', fs=srate)
            self.filt_ok = True
        except:
            self.filt_ok = False
    
    def preprocess(self, sig):
        sig = np.asarray(sig, dtype=float)
        sig = sig - np.mean(sig)
        
        if self.filt_ok and len(sig) > 30:
            try:
                return scipy_signal.filtfilt(self.filt_b, self.filt_a, sig)
            except:
                pass
        return sig
    
    def compute_powers(self, data):
        powers = {ch: {} for ch in data}
        sub_powers = {ch: {} for ch in data}
        
        for ch, raw in data.items():
            if len(raw) < self.srate:
                continue
            
            sig = self.preprocess(raw)
            
            if len(sig) < self.srate // 4:
                continue
            
            try:
                freqs, psd = scipy_signal.welch(sig, fs=self.srate, 
                                                nperseg=min(len(sig), self.srate * 2))
            except:
                continue
            
            mask = (freqs >= 0.5) & (freqs <= 45)
            if not np.any(mask):
                continue
            
            total = np_trapz(psd[mask], freqs[mask])
            if total <= 0 or np.isnan(total):
                continue
            
            for bk, bc in BANDS.items():
                lo, hi = bc.range
                m = (freqs >= lo) & (freqs <= hi)
                if np.any(m):
                    bp = np_trapz(psd[m], freqs[m])
                    if not np.isnan(bp):
                        powers[ch][bk] = (bp / total) * 100
            
            for bk, bc in SUB_BANDS.items():
                lo, hi = bc.range
                m = (freqs >= lo) & (freqs <= hi)
                if np.any(m):
                    bp = np_trapz(psd[m], freqs[m])
                    if not np.isnan(bp):
                        sub_powers[ch][bk] = (bp / total) * 100
        
        return powers, sub_powers
    
    def analyze(self, data):
        buf_len = min(len(d) for d in data.values()) if data else 0
        
        artifacts = self.artifact.detect(data, self.blink)
        powers, sub_powers = self.compute_powers(data)
        
        signals = {ch: self.preprocess(np.array(d)) for ch, d in data.items() 
                   if len(d) >= self.srate}
        
        metrics = self.calc.calculate(powers, sub_powers, signals)
        
        avg_powers = {b: safe_mean([powers[ch].get(b, 0) for ch in powers if powers[ch]])
                      for b in BANDS}
        
        clean = 1.0 - np.mean(self.blink.get_artifact_mask(buf_len)) if buf_len > 0 else 1.0
        
        return {
            'powers': powers,
            'sub_powers': sub_powers,
            'avg_powers': avg_powers,
            'metrics': metrics,
            'blink_rate': self.blink.get_rate(),
            'blink_count': self.blink.blink_count,
            'artifacts': artifacts,
            'clean_ratio': clean,
        }


# ============================================================================
# ИСТОРИЯ
# ============================================================================
class History:
    def __init__(self, max_sec=120, rate=2):
        self.max_pts = max_sec * rate
        self.ts = deque(maxlen=self.max_pts)
        self.data = {k: deque(maxlen=self.max_pts) for k in list(BANDS.keys()) + list(METRICS.keys())}
        self.start = time.time()
    
    def add(self, powers, metrics):
        self.ts.append(time.time() - self.start)
        for b in BANDS:
            self.data[b].append(safe_val(powers.get(b, 0)))
        for m in METRICS:
            self.data[m].append(safe_val(metrics.get(m, 0)))
    
    def get(self, key, last_sec=60):
        if key not in self.data or not self.ts:
            return np.array([]), np.array([])
        
        t = np.array(self.ts)
        d = np.array(self.data[key])
        
        if len(t) == 0:
            return np.array([]), np.array([])
        
        rel = t - t[-1]
        mask = rel >= -last_sec
        
        t_m = rel[mask]
        d_m = d[-len(t_m):] if len(d) >= len(t_m) else d
        
        n = min(len(t_m), len(d_m))
        return t_m[:n], d_m[:n]


# ============================================================================
# ОКНО ИСТОРИИ
# ============================================================================
class HistoryDialog(QDialog):
    def __init__(self, history, keys, parent=None):
        super().__init__(parent)
        self.history = history
        self.keys = keys
        self.time_range = 60
        
        self.setWindowTitle("📈 История")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("background:#1a1a2e; color:#eee;")
        
        self.setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(500)
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        btns = QHBoxLayout()
        for sec, lbl in [(30, "30с"), (60, "1м"), (120, "2м")]:
            btn = QPushButton(lbl)
            btn.setStyleSheet("background:#333; border:1px solid #555; padding:5px 15px; border-radius:4px;")
            btn.clicked.connect(lambda _, s=sec: setattr(self, 'time_range', s))
            btns.addWidget(btn)
        btns.addStretch()
        layout.addLayout(btns)
        
        self.pw = pg.GraphicsLayoutWidget()
        self.pw.setBackground('#1a1a2e')
        layout.addWidget(self.pw)
        
        self.curves = {}
        for i, key in enumerate(self.keys):
            cfg = BANDS.get(key) or METRICS.get(key)
            if not cfg:
                continue
            
            p = self.pw.addPlot(row=i, col=0, title=cfg.name)
            p.showGrid(y=True, alpha=0.2)
            self.curves[key] = p.plot(pen=pg.mkPen(cfg.color, width=2))
        
        self.setLayout(layout)
    
    def update_plots(self):
        for k, c in self.curves.items():
            t, d = self.history.get(k, self.time_range)
            if len(t) > 0:
                c.setData(t, d)
    
    def closeEvent(self, e):
        self.timer.stop()
        super().closeEvent(e)


# ============================================================================
# UI
# ============================================================================
class ClickableGroup(QGroupBox):
    clicked = pyqtSignal()
    
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


def make_group(title, clickable=False):
    g = ClickableGroup(title) if clickable else QGroupBox(title)
    style = "QGroupBox{border:1px solid #333;border-radius:6px;margin-top:8px;padding-top:8px;}"
    if clickable:
        style = """QGroupBox{border:1px solid #444;border-radius:6px;margin-top:8px;
                   padding-top:8px;background:#252540;}
                   QGroupBox:hover{border-color:#666;background:#2a2a50;}
                   QGroupBox::title{color:#8af;}"""
    g.setStyleSheet(style)
    g.setFont(QFont('Arial', 10, QFont.Bold))
    return g


def make_bar(color):
    b = QProgressBar()
    b.setMaximum(100)
    b.setStyleSheet(f"""QProgressBar{{border:1px solid #333;border-radius:3px;
                       text-align:center;height:16px;background:#252540;}}
                       QProgressBar::chunk{{background:{color};border-radius:2px;}}""")
    return b


# ============================================================================
# ГЛАВНОЕ ОКНО
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainBit Viewer v4.3")
        self.setGeometry(50, 50, 1500, 1000)
        self.setStyleSheet("background:#1a1a2e; color:#eee;")
        
        print("🔍 Поиск BrainBitBlack...")
        streams = resolve_streams()
        target = next((s for s in streams if s.name() == 'BrainBitBlack'), None)
        
        if not target:
            print("❌ Не найден!")
            sys.exit(1)
        
        self.inlet = StreamInlet(target)
        print("✅ Подключено")
        
        self.buffers = {ch: deque(maxlen=SRATE * BUFFER_SEC) for ch in CHANNELS}
        self.sample_count = 0
        
        self.analyzer = EEGAnalyzer(SRATE)
        self.history = History(HISTORY_SEC, 2)
        
        self.setup_ui()
        
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_data)
        self.data_timer.start(20)
        
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.update_analysis)
        self.analysis_timer.start(500)
        
        self._hist_dlg = None
    
    def setup_ui(self):
        central = QWidget()
        main = QHBoxLayout()
        
        # LEFT
        left = QVBoxLayout()
        
        status = QHBoxLayout()
        self.status_lbl = QLabel("🟢 Ожидание...")
        status.addWidget(self.status_lbl)
        status.addStretch()
        self.artifact_lbl = QLabel("")
        self.artifact_lbl.setStyleSheet("color:#F97316;")
        status.addWidget(self.artifact_lbl)
        self.clean_lbl = QLabel("📊 100%")
        self.clean_lbl.setStyleSheet("color:#10B981;")
        status.addWidget(self.clean_lbl)
        self.blink_lbl = QLabel("👁️ --/м")
        self.blink_lbl.setStyleSheet("color:#06B6D4;")
        status.addWidget(self.blink_lbl)
        left.addLayout(status)
        
        self.pw = pg.GraphicsLayoutWidget()
        self.pw.setBackground('#1a1a2e')
        left.addWidget(self.pw, stretch=2)
        
        self.curves = {}
        self.blink_markers = {}
        
        for i, ch in enumerate(CHANNELS):
            p = self.pw.addPlot(row=i, col=0, title=ch)
            p.setYRange(-150, 150)
            p.showGrid(y=True, alpha=0.2)
            self.curves[ch] = p.plot(pen=pg.mkPen('#60a5fa', width=1.2))
            
            sc = pg.ScatterPlotItem(size=10, brush='#EF4444')
            p.addItem(sc)
            self.blink_markers[ch] = sc
        
        self.spec_plot = self.pw.addPlot(row=len(CHANNELS), col=0, title="Спектр")
        self.spec_plot.setXRange(0, 50)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen('#10b981', width=2))
        
        for bk, bc in BANDS.items():
            lo, hi = bc.range
            r = pg.LinearRegionItem([lo, hi], movable=False, brush=pg.mkBrush(bc.color + '30'))
            r.setZValue(-10)
            self.spec_plot.addItem(r)
        
        main.addLayout(left, stretch=2)
        
        # RIGHT
        right = QVBoxLayout()
        
        # Состояние
        sg = make_group("🧠 Состояние")
        sl = QVBoxLayout()
        self.state_lbl = QLabel("Анализ...")
        self.state_lbl.setFont(QFont('Arial', 14, QFont.Bold))
        self.state_lbl.setWordWrap(True)
        sl.addWidget(self.state_lbl)
        self.state_desc = QLabel("")
        self.state_desc.setStyleSheet("color:#aaa;")
        self.state_desc.setWordWrap(True)
        sl.addWidget(self.state_desc)
        self.conf_bar = make_bar('#10b981')
        sl.addWidget(self.conf_bar)
        sg.setLayout(sl)
        right.addWidget(sg)
        
        # Ритмы
        bg = make_group("📊 Ритмы ➜", clickable=True)
        bg.clicked.connect(lambda: self.show_hist(list(BANDS.keys())))
        bl = QGridLayout()
        self.band_bars = {}
        for i, (k, c) in enumerate(BANDS.items()):
            lbl = QLabel(c.name)
            lbl.setStyleSheet(f"color:{c.color};font-weight:bold;")
            bl.addWidget(lbl, i, 0)
            bar = make_bar(c.color)
            bl.addWidget(bar, i, 1)
            self.band_bars[k] = bar
        bg.setLayout(bl)
        right.addWidget(bg)
        
        # ✅ Достоверные
        hg = make_group("✅ Достоверные ➜", clickable=True)
        hg.clicked.connect(lambda: self.show_hist(METRICS_HIGH))
        hl = QGridLayout()
        self.high_bars = {}
        for i, k in enumerate(METRICS_HIGH):
            c = METRICS[k]
            lbl = QLabel(c.name)
            lbl.setToolTip(f"{c.desc}\n{c.formula_str}\n{c.source}")
            hl.addWidget(lbl, i, 0)
            bar = make_bar(c.color)
            hl.addWidget(bar, i, 1)
            self.high_bars[k] = bar
        hg.setLayout(hl)
        right.addWidget(hg)
        
        # ⚠️ Упрощённые
        mg = make_group("⚠️ Упрощённые ➜", clickable=True)
        mg.clicked.connect(lambda: self.show_hist(METRICS_MEDIUM))
        ml = QGridLayout()
        self.med_bars = {}
        for i, k in enumerate(METRICS_MEDIUM):
            c = METRICS[k]
            lbl = QLabel(c.name)
            lbl.setToolTip(f"{c.desc}\n{c.formula_str}")
            ml.addWidget(lbl, i, 0)
            bar = make_bar(c.color)
            ml.addWidget(bar, i, 1)
            self.med_bars[k] = bar
        mg.setLayout(ml)
        right.addWidget(mg)
        
        # ❓ Приблизительные
        lg = make_group("❓ Приблизительные")
        ll = QGridLayout()
        self.low_bars = {}
        for i, k in enumerate(METRICS_LOW):
            c = METRICS[k]
            lbl = QLabel(c.name)
            lbl.setToolTip(f"⚠️ {c.desc}")
            ll.addWidget(lbl, i, 0)
            bar = make_bar(c.color)
            ll.addWidget(bar, i, 1)
            self.low_bars[k] = bar
        lg.setLayout(ll)
        right.addWidget(lg)
        
        # Моргания
        blg = make_group("👁️ Моргания")
        bll = QVBoxLayout()
        self.blink_info = QLabel("--")
        bll.addWidget(self.blink_info)
        self.blink_bar = make_bar('#06B6D4')
        bll.addWidget(self.blink_bar)
        blg.setLayout(bll)
        right.addWidget(blg)
        
        # Доп
        eg = make_group("🔬 Доп.")
        el = QVBoxLayout()
        self.extra_lbl = QLabel("")
        self.extra_lbl.setStyleSheet("font-family:monospace;font-size:9pt;")
        self.extra_lbl.setWordWrap(True)
        el.addWidget(self.extra_lbl)
        eg.setLayout(el)
        right.addWidget(eg)
        
        right.addStretch()
        main.addLayout(right, stretch=1)
        
        central.setLayout(main)
        self.setCentralWidget(central)
    
    def show_hist(self, keys):
        if self._hist_dlg and self._hist_dlg.isVisible():
            self._hist_dlg.close()
        self._hist_dlg = HistoryDialog(self.history, keys, self)
        self._hist_dlg.show()
    
    def update_data(self):
        samples = []
        while True:
            s, _ = self.inlet.pull_sample(timeout=0.0)
            if s is None:
                break
            self.sample_count += 1
            for i, ch in enumerate(CHANNELS):
                self.buffers[ch].append(s[i])
            samples.append(s)
        
        if samples:
            blinks = self.analyzer.blink.process(samples)
            for b in blinks:
                print(f"👁️ Моргание: {b['amplitude']:.0f}µV, {b['duration_ms']:.0f}мс")
        
        for ch in CHANNELS:
            if self.buffers[ch]:
                data = np.array(self.buffers[ch])
                self.curves[ch].setData(data)
                
                pos = self.analyzer.blink.get_blink_positions(len(data))
                if pos:
                    self.blink_markers[ch].setData([p[0] for p in pos], [120]*len(pos))
                else:
                    self.blink_markers[ch].setData([], [])
        
        self.status_lbl.setText(f"🟢 {self.sample_count}")
    
    def update_analysis(self):
        if any(len(self.buffers[ch]) < SRATE * 2 for ch in CHANNELS):
            return
        
        data = {ch: np.array(self.buffers[ch]) for ch in CHANNELS}
        result = self.analyzer.analyze(data)
        
        self.history.add(result['avg_powers'], result['metrics'])
        self.update_ui(result)
    
    def update_ui(self, r):
        m = r['metrics']
        p = r['avg_powers']
        br = safe_val(r['blink_rate'])
        bc = r['blink_count']
        art = r['artifacts']
        clean = safe_val(r['clean_ratio'], 1)
        
        # Ритмы
        for k, bar in self.band_bars.items():
            v = safe_val(p.get(k, 0))
            bar.setValue(int(min(v, 100)))
            bar.setFormat(f"{v:.1f}%")
        
        # Достоверные
        for k, bar in self.high_bars.items():
            self._set_metric_bar(bar, k, m)
        
        # Упрощённые
        for k, bar in self.med_bars.items():
            self._set_metric_bar(bar, k, m)
        
        # Приблизительные
        for k, bar in self.low_bars.items():
            self._set_metric_bar(bar, k, m)
        
        # Состояние
        name, desc, conf, color = self.determine_state(m, p, art)
        self.state_lbl.setText(name)
        self.state_lbl.setStyleSheet(f"color:{color};")
        self.state_desc.setText(desc)
        self.conf_bar.setValue(int(safe_val(conf, 50)))
        
        # Моргания
        self.blink_lbl.setText(f"👁️ {br:.0f}/м")
        status = "⚠️ Высокая!" if br > 25 else "✓ Норма" if br > 0 else "..."
        self.blink_info.setText(f"{br:.1f}/мин | {bc} всего | {status}")
        self.blink_bar.setValue(int(np.clip(br / 30 * 100, 0, 100)))
        
        # Артефакты
        if art.has_artifact:
            self.artifact_lbl.setText(art.description)
        else:
            self.artifact_lbl.setText("")
        
        # Clean
        cp = clean * 100
        self.clean_lbl.setText(f"📊 {cp:.0f}%")
        self.clean_lbl.setStyleSheet(f"color:{'#10B981' if cp >= 90 else '#F59E0B' if cp >= 70 else '#EF4444'};")
        
        # Extra
        se = safe_val(m.get('spectral_entropy', 0))
        iaf = safe_val(m.get('alpha_peak', 10))
        sym = safe_val(m.get('symmetry', 100))
        val = safe_val(m.get('valence', 50))
        
        theta = safe_val(p.get('Theta', 0))
        beta = safe_val(p.get('Beta', 1))
        tbr = theta / (beta + 0.1)
        
        self.extra_lbl.setText(
            f"θ/β: {tbr:.2f}  |  IAF: {iaf:.1f}Hz\n"
            f"Энтропия: {se:.3f}  |  Симметрия: {sym:.0f}%\n"
            f"Валентность: {val:.0f}% {'🙂' if val > 60 else '😐' if val > 40 else '😔'}\n"
            f"Чистый сигнал: {cp:.0f}%"
        )
        
        # Спектр
        self.update_spectrum()
    
    def _set_metric_bar(self, bar, key, metrics):
        cfg = METRICS.get(key)
        v = safe_val(metrics.get(key, 0))
        
        if cfg and cfg.max_val != 100:
            pct = (v - cfg.min_val) / (cfg.max_val - cfg.min_val + 0.001) * 100
            bar.setValue(int(np.clip(pct, 0, 100)))
            bar.setFormat(f"{v:.2f}{cfg.unit}")
        else:
            bar.setValue(int(np.clip(v, 0, 100)))
            bar.setFormat(f"{v:.0f}%")
    
    def determine_state(self, m, p, art):
        if art.has_artifact:
            if ArtifactType.ELECTRODE in art.types:
                return ('⚠️ Проблема электрода', 'Проверьте контакт', 95, '#EF4444')
            if ArtifactType.MUSCLE in art.types:
                return ('💪 Мышцы', 'Расслабьте лицо', 80, '#F97316')
        
        states = []
        
        eng = safe_val(m.get('engagement', 0))
        foc = safe_val(m.get('focus', 0))
        rel = safe_val(m.get('relaxation', 0))
        med = safe_val(m.get('meditation', 0))
        dro = safe_val(m.get('drowsiness', 0))
        fat = safe_val(m.get('fatigue', 0))
        flo = safe_val(m.get('flow', 0))
        aro = safe_val(m.get('arousal', 0))
        str_ = safe_val(m.get('stress', 0))
        
        if eng > 70 and foc > 60:
            states.append(('🎯 Глубокий фокус', 'Высокая вовлечённость', (eng+foc)/2, '#F59E0B'))
        
        if flo > 65:
            states.append(('🌊 Поток', 'Оптимальная продуктивность', flo, '#06B6D4'))
        
        if rel > 60 and str_ < 30:
            if med > 50:
                states.append(('🧘 Медитация', 'α+θ высокие', (rel+med)/2, '#14B8A6'))
            else:
                states.append(('😌 Расслабление', 'Высокий Alpha', rel, '#10B981'))
        
        if dro > 60 and aro < 40:
            states.append(('😴 Сонливость', 'Высокий Theta', dro, '#8B5CF6'))
        
        if fat > 70:
            states.append(('🔋 Усталость', 'Нужен отдых', fat, '#F97316'))
        
        if str_ > 60:
            states.append(('😰 Стресс*', 'Высокий Beta (приблизительно)', str_, '#EF4444'))
        
        if not states:
            return ('😐 Нейтрально', 'Сбалансированная активность', 50, '#6B7280')
        
        states.sort(key=lambda x: -x[2])
        return states[0]
    
    def update_spectrum(self):
        if 'O1' not in self.buffers or len(self.buffers['O1']) < SRATE:
            return
        
        sig = self.analyzer.preprocess(np.array(self.buffers['O1']))
        try:
            freqs, psd = scipy_signal.welch(sig, fs=SRATE, nperseg=min(len(sig), SRATE * 2))
            mask = freqs <= 50
            self.spec_curve.setData(freqs[mask], np.log10(psd[mask] + 1e-10))
        except:
            pass


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
