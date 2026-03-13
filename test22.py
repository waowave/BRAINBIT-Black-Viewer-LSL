#!/usr/bin/env python3
"""
BrainBit Black — EEG Recorder v4.1

Использование:
  python brainbit.py              # Проверка контакта → запись
  python brainbit.py --check      # Только проверка контакта
  python brainbit.py --no-check   # Сразу запись
"""

import asyncio
import argparse
import struct
import time
import csv
import os
import sys
import numpy as np
from collections import deque
from bleak import BleakClient

# ============================================================================
# НАСТРОЙКИ (значения по умолчанию)
# ============================================================================
DEFAULT_MAC = "FD:1A:05:B6:20:71"

STATUS_UUID = "7e400002-b534-f393-68a9-e50e24dcca99"
CMD_UUID    = "7e400003-b534-f393-68a9-e50e24dcca99"
DATA_UUID   = "7e400004-b534-f393-68a9-e50e24dcca99"
AUX_UUID    = "7e400005-b534-f393-68a9-e50e24dcca99"

CMD_START_EEG = bytes([0x03])
CMD_START_IMPEDANCE = bytes([0x04])
CMD_STOP = bytes([0x02])

UV_SCALE = 0.02384
SRATE = 250
CHANNELS = ['O1', 'O2', 'T3', 'T4']

# Пороги импеданса
IMP_EXCELLENT = 50_000
IMP_GOOD = 150_000
IMP_ACCEPTABLE = 300_000
IMP_BAD = 500_000

# Пороги качества EEG
NOISE_THRESHOLD = 150.0
FLATLINE_THRESHOLD = 3.0

# ============================================================================
# КОНФИГ (заполняется из аргументов)
# ============================================================================
class Config:
    mac_address = DEFAULT_MAC
    enable_lsl = True
    enable_csv = True
    enable_print = True

config = Config()

# ============================================================================
# DC-ФИЛЬТР
# ============================================================================
class DCRemover:
    def __init__(self, alpha=0.996):
        self.alpha = alpha
        self.prev_raw = None
        self.prev_out = 0.0
        self.warmup = 50
        self.count = 0
    
    def process(self, raw_value):
        self.count += 1
        if self.prev_raw is None:
            self.prev_raw = raw_value
            return 0.0
        self.prev_out = self.alpha * (self.prev_out + raw_value - self.prev_raw)
        self.prev_raw = raw_value
        return 0.0 if self.count < self.warmup else self.prev_out
    
    @property
    def is_ready(self):
        return self.count >= self.warmup
    
    def reset(self):
        self.prev_raw = None
        self.prev_out = 0.0
        self.count = 0

# ============================================================================
# СОСТОЯНИЕ
# ============================================================================
class State:
    def __init__(self):
        self.battery = 0
        self.start_time = None
        self.mode = 'idle'
        
        # Импеданс
        self.impedance = {ch: None for ch in CHANNELS}
        self.impedance_status = {ch: '⏳' for ch in CHANNELS}
        self.impedance_ok = False
        
        # EEG
        self.dc_filters = {ch: DCRemover() for ch in CHANNELS}
        self.filters_ready = False
        self.sample_count = 0
        self.stats_buf = {ch: deque(maxlen=SRATE) for ch in CHANNELS}
        self.spectrum_buf = {ch: deque(maxlen=SRATE * 4) for ch in CHANNELS}
        self.eeg_status = {ch: '⏳' for ch in CHANNELS}
        
        # Файлы
        self.csv_file = None
        self.csv_writer = None
        self.csv_filename = None
        self.lsl_outlet = None

state = State()

# ============================================================================
# ИМПЕДАНС
# ============================================================================
def impedance_quality(value):
    if value is None:
        return '⏳', 'ожидание', False
    if value < IMP_EXCELLENT:
        return '✅', 'отлично', True
    if value < IMP_GOOD:
        return '🟢', 'хорошо', True
    if value < IMP_ACCEPTABLE:
        return '🟡', 'норма', True
    if value < IMP_BAD:
        return '🟠', 'слабый', False
    return '❌', 'плохой', False

def parse_impedance(data: bytes):
    if len(data) != 20:
        return None
    counter = struct.unpack_from('<I', data, 0)[0]
    values = [struct.unpack_from('<I', data, 4 + i*4)[0] for i in range(4)]
    return counter, values

def on_aux(sender, data: bytes):
    result = parse_impedance(data)
    if result is None:
        return
    
    counter, values = result
    
    # Порядок в пакете: O2, T3, T4, O1
    mapping = {'O2': 0, 'T3': 1, 'T4': 2, 'O1': 3}
    
    all_ok = True
    for ch in CHANNELS:
        val = values[mapping[ch]]
        state.impedance[ch] = val
        icon, text, ok = impedance_quality(val)
        state.impedance_status[ch] = icon
        if not ok:
            all_ok = False
    
    state.impedance_ok = all_ok
    
    if config.enable_print and state.mode == 'impedance':
        print_impedance_line()

def print_impedance_line():
    parts = []
    for ch in CHANNELS:
        val = state.impedance[ch]
        icon = state.impedance_status[ch]
        if val is not None:
            kohm = val / 1000
            parts.append(f"{ch}:{icon}{kohm:5.0f}k")
        else:
            parts.append(f"{ch}:{icon}    —")
    
    status = "✅ ВСЕ ОК" if state.impedance_ok else "⏳ Поправьте"
    print(f"\r  {' | '.join(parts)}  [{status}]   ", end='', flush=True)

# ============================================================================
# EEG
# ============================================================================
def parse_24bit(b0, b1, b2):
    val = b0 | (b1 << 8) | (b2 << 16)
    return val - 0x1000000 if val >= 0x800000 else val

def parse_eeg_packet(data: bytes):
    if len(data) != 108:
        return None, []
    counter = struct.unpack_from('<I', data, 0)[0]
    samples = []
    for i in range(8):
        off = 4 + i * 13 + 1
        o2 = parse_24bit(data[off],   data[off+1],  data[off+2])
        t3 = parse_24bit(data[off+3], data[off+4],  data[off+5])
        t4 = parse_24bit(data[off+6], data[off+7],  data[off+8])
        o1 = parse_24bit(data[off+9], data[off+10], data[off+11])
        samples.append({'O1': o1, 'O2': o2, 'T3': t3, 'T4': t4})
    return counter, samples

def eeg_quality(std):
    if std > NOISE_THRESHOLD:
        return '⚠️', 'шум'
    if std < FLATLINE_THRESHOLD:
        return '❌', 'нет'
    return '✅', 'ок'

last_quality_check = 0

def check_eeg_quality():
    global last_quality_check
    now = time.time()
    if now - last_quality_check < 2.0:
        return
    last_quality_check = now
    
    if not state.filters_ready:
        return
    
    issues = []
    for ch in CHANNELS:
        if len(state.stats_buf[ch]) < SRATE // 2:
            continue
        arr = np.array(state.stats_buf[ch])
        std = np.std(arr)
        icon, text = eeg_quality(std)
        old_status = state.eeg_status[ch]
        state.eeg_status[ch] = icon
        
        # Уведомление при ухудшении
        if old_status == '✅' and icon != '✅':
            issues.append(f"⚠️  {ch}: {text} ({std:.0f} µV)")
    
    if issues and config.enable_print:
        print("\n" + "\n".join(issues) + "\n")

def on_data(sender, data: bytes):
    if state.start_time is None:
        state.start_time = time.time()
    
    counter, samples = parse_eeg_packet(data)
    if not samples:
        return
    
    now = time.time()
    
    for raw in samples:
        state.sample_count += 1
        
        uv = {}
        for ch in CHANNELS:
            raw_uv = raw[ch] * UV_SCALE
            uv[ch] = state.dc_filters[ch].process(raw_uv)
            state.stats_buf[ch].append(uv[ch])
            state.spectrum_buf[ch].append(uv[ch])
        
        if not state.filters_ready:
            if all(f.is_ready for f in state.dc_filters.values()):
                state.filters_ready = True
                if config.enable_print:
                    print("\n✅ Запись началась\n")
            elif state.sample_count % 100 == 0 and config.enable_print:
                print(f"\r   Прогрев... {state.sample_count}/50", end='')
            continue
        
        # LSL
        if state.lsl_outlet:
            state.lsl_outlet.push_sample([uv[ch] for ch in CHANNELS])
        
        # CSV
        if state.csv_writer:
            ts = now - state.start_time
            state.csv_writer.writerow([
                f"{ts:.4f}", state.sample_count,
                f"{uv['O1']:.2f}", f"{uv['O2']:.2f}",
                f"{uv['T3']:.2f}", f"{uv['T4']:.2f}"
            ])
        
        # Печать
        if config.enable_print and state.sample_count % 50 == 0:
            elapsed = now - state.start_time
            vals = "  ".join(f"{ch}:{uv[ch]:+7.1f}" for ch in CHANNELS)
            eeg_st = "".join(state.eeg_status[ch] for ch in CHANNELS)
            imp_st = "".join(state.impedance_status[ch] for ch in CHANNELS)
            print(f"[{elapsed:5.1f}s] {vals} µV  EEG:[{eeg_st}] IMP:[{imp_st}]")
        
        check_eeg_quality()

def on_status(sender, data: bytes):
    if len(data) >= 1 and data[0] != state.battery:
        state.battery = data[0]
        if config.enable_print and state.mode == 'eeg':
            print(f"🔋 {state.battery}%")

# ============================================================================
# LSL
# ============================================================================
def setup_lsl():
    if not config.enable_lsl:
        return None
    try:
        from pylsl import StreamInfo, StreamOutlet
        info = StreamInfo('BrainBitBlack', 'EEG', 4, SRATE, 'float32',
                          f'BBB-{config.mac_address.replace(":", "")}')
        chns = info.desc().append_child("channels")
        for label in CHANNELS:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
        info.desc().append_child_value("manufacturer", "BrainBit")
        return StreamOutlet(info)
    except ImportError:
        print("⚠️  LSL недоступен (pip install pylsl)")
        return None

# ============================================================================
# CSV
# ============================================================================
def setup_csv():
    if not config.enable_csv:
        return None, None, None
    filename = f"brainbit_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(filename, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'sample', 'O1_uV', 'O2_uV', 'T3_uV', 'T4_uV'])
    return f, writer, filename

# ============================================================================
# СПЕКТР
# ============================================================================
def print_spectrum():
    bands = {
        'Delta (1-4)': (1, 4),
        'Theta (4-8)': (4, 8),
        'Alpha (8-13)': (8, 13),
        'Beta (13-30)': (13, 30),
        'Gamma (30-45)': (30, 45)
    }
    
    print(f"\n{'='*65}")
    print("📊 СПЕКТР")
    print(f"{'='*65}")
    
    for ch in CHANNELS:
        if len(state.spectrum_buf[ch]) < SRATE * 2:
            continue
        
        data = np.array(state.spectrum_buf[ch])
        window = np.hanning(len(data))
        fft = np.fft.rfft(data * window)
        psd = np.abs(fft) ** 2 / len(data)
        freqs = np.fft.rfftfreq(len(data), 1.0 / SRATE)
        
        print(f"\n  {ch} {state.eeg_status[ch]}:")
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            power = np.mean(psd[mask]) if np.any(mask) else 0
            log_p = np.log10(power + 1) if power > 0 else 0
            bar = '█' * min(int(log_p * 8), 35)
            print(f"    {name:<12}: {power:8.1f}  {bar}")

# ============================================================================
# РЕЖИМ ПРОВЕРКИ
# ============================================================================
async def run_impedance_check(client, auto_continue=False, timeout=30):
    state.mode = 'impedance'
    
    print("\n" + "="*60)
    print("  📡 ПРОВЕРКА КОНТАКТА ЭЛЕКТРОДОВ")
    print("  Наденьте устройство и поправляйте электроды")
    if auto_continue:
        print(f"  Автопереход к записи когда всё ОК (таймаут {timeout}с)")
    else:
        print("  Ctrl+C для выхода")
    print("="*60 + "\n")
    
    await client.start_notify(AUX_UUID, on_aux)
    await client.write_gatt_char(CMD_UUID, CMD_START_IMPEDANCE)
    
    start = time.time()
    ok_since = None
    
    try:
        while True:
            await asyncio.sleep(0.3)
            
            if state.impedance_ok:
                if ok_since is None:
                    ok_since = time.time()
                    print("\n\n🎉 Все электроды в норме!")
                
                if auto_continue and (time.time() - ok_since) >= 1.5:
                    print("   Переход к записи...\n")
                    await asyncio.sleep(0.5)
                    break
            else:
                ok_since = None
            
            if auto_continue and (time.time() - start) > timeout:
                print(f"\n\n⚠️  Таймаут {timeout}с. Продолжаем...")
                break
    
    except KeyboardInterrupt:
        if auto_continue:
            print("\n\n⏹ Пропуск проверки...")
        else:
            raise
    
    finally:
        await client.write_gatt_char(CMD_UUID, CMD_STOP)
        await asyncio.sleep(0.2)
        await client.stop_notify(AUX_UUID)
        state.mode = 'idle'
    
    # Вывод итогов проверки
    print(f"\n{'─'*40}")
    print("Результат проверки:")
    for ch in CHANNELS:
        val = state.impedance.get(ch)
        if val:
            icon, text, _ = impedance_quality(val)
            print(f"  {ch}: {val/1000:6.0f} кОм  {icon} {text}")
    print(f"{'─'*40}\n")

# ============================================================================
# РЕЖИМ ЗАПИСИ
# ============================================================================
async def run_eeg_recording(client):
    state.mode = 'eeg'
    state.start_time = time.time()
    
    # Сбросить фильтры (на случай если был режим проверки)
    for f in state.dc_filters.values():
        f.reset()
    state.filters_ready = False
    state.sample_count = 0
    
    # Настройка
    state.lsl_outlet = setup_lsl()
    if state.lsl_outlet:
        print("📡 LSL поток 'BrainBitBlack' активен")
    
    state.csv_file, state.csv_writer, state.csv_filename = setup_csv()
    if state.csv_filename:
        print(f"💾 CSV: {state.csv_filename}")
    
    print("\n" + "="*60)
    print("  🧠 ЗАПИСЬ EEG")
    print("  Ctrl+C для остановки")
    print("="*60 + "\n")
    
    await client.start_notify(DATA_UUID, on_data)
    await client.start_notify(STATUS_UUID, on_status)
    await client.write_gatt_char(CMD_UUID, CMD_START_EEG)
    
    try:
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⏹ Остановка...")
    
    finally:
        await client.write_gatt_char(CMD_UUID, CMD_STOP)
        await asyncio.sleep(0.2)
        await client.stop_notify(DATA_UUID)
        await client.stop_notify(STATUS_UUID)
        state.mode = 'idle'
        
        # Итоги
        elapsed = time.time() - state.start_time if state.start_time else 0
        
        print(f"\n{'='*65}")
        print("📊 ИТОГИ")
        print(f"{'='*65}")
        
        if elapsed > 0 and state.sample_count > 0:
            print(f"Сэмплов: {state.sample_count} | "
                  f"Время: {elapsed:.1f}с | "
                  f"Частота: {state.sample_count/elapsed:.1f} Гц")
        
        print(f"\n{'Канал':<6} {'EEG':<6} {'СКО':>10} {'Размах':>10} {'Импеданс':>12}")
        print("-" * 54)
        for ch in CHANNELS:
            arr = np.array(state.stats_buf[ch])
            imp = state.impedance.get(ch)
            imp_str = f"{imp/1000:.0f}k" if imp else "—"
            
            if len(arr) > 0:
                std = np.std(arr)
                pp = np.ptp(arr)
                print(f"{ch:<6} {state.eeg_status[ch]:<6} {std:>9.1f}µV {pp:>9.1f}µV {imp_str:>12}")
        
        print_spectrum()
        
        if state.csv_file:
            state.csv_file.close()
            sz = os.path.getsize(state.csv_filename) / 1024
            print(f"\n💾 Сохранено: {state.csv_filename} ({sz:.1f} KB)")
        
        print("\n✅ Готово!")

# ============================================================================
# MAIN
# ============================================================================
async def main():
    parser = argparse.ArgumentParser(
        description='BrainBit Black EEG Recorder v4.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python brainbit.py              # Проверка → запись
  python brainbit.py --check      # Только проверка
  python brainbit.py --no-check   # Сразу запись
  python brainbit.py --no-lsl     # Без LSL
        """
    )
    parser.add_argument('--check', action='store_true',
                        help='Только проверка контакта')
    parser.add_argument('--no-check', action='store_true',
                        help='Пропустить проверку')
    parser.add_argument('--no-lsl', action='store_true',
                        help='Отключить LSL')
    parser.add_argument('--no-csv', action='store_true',
                        help='Отключить CSV')
    parser.add_argument('--mac', type=str, default=DEFAULT_MAC,
                        help=f'MAC адрес (по умолчанию: {DEFAULT_MAC})')
    
    args = parser.parse_args()
    
    # Применить конфиг
    config.mac_address = args.mac
    config.enable_lsl = not args.no_lsl
    config.enable_csv = not args.no_csv
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║  BrainBit Black — EEG Recorder v4.1                               ║
║                                                                   ║
║  MAC: {config.mac_address}                               ║
║  LSL: {'✅' if config.enable_lsl else '❌'}   CSV: {'✅' if config.enable_csv else '❌'}                                            ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        async with BleakClient(config.mac_address, timeout=20) as client:
            print("✅ Подключено")
            
            status = await client.read_gatt_char(STATUS_UUID)
            state.battery = status[0]
            print(f"🔋 Батарея: {state.battery}%")
            
            if args.check:
                await run_impedance_check(client, auto_continue=False)
            
            elif args.no_check:
                await run_eeg_recording(client)
            
            else:
                await run_impedance_check(client, auto_continue=True, timeout=30)
                await run_eeg_recording(client)
    
    except KeyboardInterrupt:
        print("\n⏹ Выход")
    
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
