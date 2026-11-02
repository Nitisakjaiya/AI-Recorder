import cv2
import numpy as np
import pygame
import math
import os
import time
import wave

from ai_musician import AIMusician

import mediapipe.python.solutions.holistic as mp_holistic
import mediapipe.python.solutions.drawing_utils as mp_draw

# --- 1. ตั้งค่าระบบเสียงขลุ่ยรีคอร์ดเดอร์ 9 โน้ต ---
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(14)

# กำหนดข้อมูลโน้ตทั้ง 9 ระดับ
NOTES = [
    {"cnt": 1, "name": "NOTE 1", "freq": 493.88, "file": "recorder_note1.wav", "holes": [1,1,1,1,1,1,1]},
    {"cnt": 2, "name": "NOTE 2", "freq": 523.25, "file": "recorder_note2.wav", "holes": [1,1,1,1,1,1,0]},
    {"cnt": 3, "name": "NOTE 3", "freq": 587.33, "file": "recorder_note3.wav", "holes": [1,1,1,1,1,0,0]},
    {"cnt": 4, "name": "NOTE 4", "freq": 659.25, "file": "recorder_note4.wav", "holes": [1,1,1,1,0,0,0]},
    {"cnt": 5, "name": "NOTE 5", "freq": 698.46, "file": "recorder_note5.wav", "holes": [1,1,1,0,0,0,0]},
    {"cnt": 6, "name": "NOTE 6", "freq": 783.99, "file": "recorder_note6.wav", "holes": [1,1,0,0,0,0,0]},
    {"cnt": 7, "name": "NOTE 7", "freq": 880.00, "file": "recorder_note7.wav", "holes": [1,0,0,0,0,0,0]},
    {"cnt": 8, "name": "NOTE 8", "freq": 987.77, "file": "recorder_note8.wav", "holes": [0,1,0,0,0,0,0]},
    {"cnt": 9, "name": "NOTE 9", "freq": 1046.50,"file": "recorder_note9.wav", "holes": [0,0,0,0,0,0,0]},
]

def load_or_synth_sound(n_info, duration=4.0):
    sr = 44100
    n_samples = int(sr * duration)
    wav_path = n_info["file"]
    freq = n_info["freq"]

    # ถ้ามีไฟล์เสียงจริงตามชื่อที่กำหนด จะโหลดมาใช้ทันที
    if os.path.exists(wav_path):
        try:
            with wave.open(wav_path, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                if wf.getnchannels() > 1:
                    data = data[::wf.getnchannels()]
            data /= (np.max(np.abs(data)) + 1e-6)
            # เผื่อความยาวไฟล์ไม่พอ ให้ Loop เสียงต่อกัน
            repeats = int(np.ceil(n_samples / len(data)))
            data_extended = np.tile(data, repeats)[:n_samples]
            buf = np.zeros((n_samples, 2), dtype=np.int16)
            buf[:, 0] = (data_extended * 26000).astype(np.int16)
            buf[:, 1] = buf[:, 0]
            print(f"[Loaded Real Sound] {wav_path}")
            return pygame.sndarray.make_sound(buf)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}, using synth fallback.")

    # ระบบสังเคราะห์เสียงสำรอง (กรณีไฟล์ใดยังไม่เสร็จ)
    t = np.linspace(0, duration, n_samples, False)
    vib = 1.0 + 0.002 * np.sin(2 * np.pi * 5.2 * t)
    phase = 2 * np.pi * freq * t * vib
    w = (
        1.00 * np.sin(phase) +
        0.28 * np.sin(2 * phase) +
        0.45 * np.sin(3 * phase) +
        0.15 * np.sin(4 * phase) +
        0.08 * np.sin(6 * phase)
    )
    air = np.random.normal(0, 0.022, n_samples)
    w = (w + air) * 0.35
    f_len = int(sr * 0.06)
    w[:f_len] *= np.linspace(0, 1, f_len)
    w[-f_len:] *= np.linspace(1, 0, f_len)

    buf = np.zeros((n_samples, 2), dtype=np.int16)
    buf[:, 0] = (w * 32767).astype(np.int16)
    buf[:, 1] = buf[:, 0]
    return pygame.sndarray.make_sound(buf)

print("กำลังเตรียมเสียงขลุ่ยรีคอร์ดเดอร์ 9 โน้ต...")
sounds = {}
channels = {}
curr_vols = {}

for idx, n in enumerate(NOTES):
    c = n["cnt"]
    sounds[c] = load_or_synth_sound(n)
    channels[c] = pygame.mixer.Channel(idx)
    channels[c].play(sounds[c], loops=-1)
    channels[c].set_volume(0.0)
    curr_vols[c] = 0.0

def count_fingers_9(hand_landmarks):
    lm = hand_landmarks.landmark
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    cnt = 0
    for tip, pip in zip(tips, pips):
        if lm[tip].y < lm[pip].y:
            cnt += 1
    # นิ้วโป้งกางออกนับเป็น 1 นิ้วเต็ม (มือละ 0-5 นิ้ว สองมือรวมได้ถึง 9 โน้ต)
    # เดิมบวก 0.5 แล้วใช้ round() ซึ่ง Python ปัดครึ่งเข้าหาเลขคู่ (round(4.5) = 4)
    # ทำให้นิ้วโป้งนับบ้างไม่นับบ้าง และรวมสองมือได้สูงสุดแค่ 8 จึงเล่น NOTE 9 ไม่ได้
    thumb_ext = math.hypot(lm[4].x - lm[17].x, lm[4].y - lm[17].y)
    if thumb_ext > 0.15:
        cnt += 1
    return cnt

holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

history_cnt = []
SMOOTH_FRAMES = 4

# --- AI นักดนตรี : ฟังเราเล่น แล้วแต่งท่อนตอบกลับเอง ---
ai = AIMusician()
ai_enabled = True
AI_VOLUME = 0.7
PLAY_VOLUME = 0.8        # ความดังเมื่อเล่นแบบชูนิ้วอย่างเดียว
mouth_mode = True        # True = ต้องอ้าปาก (เป่า) ด้วยถึงมีเสียง ความดังตามการอ้าปาก, False = ชูนิ้วอย่างเดียว
status_msg = ""
status_until = 0.0


def draw_ai_brain(frame, ai, x, y, cell=13):
    """วาดตาราง P(โน้ตถัดไป | โน้ตปัจจุบัน) ที่ AI เรียนรู้ ยิ่งสว่างยิ่งมีโอกาสสูง"""
    k = cell / 13                     # อัตราขยายตัวอักษรตามขนาดช่อง
    t = max(1, round(k))
    mat = ai.matrix()
    cv2.putText(frame, "AI brain P(next|now)", (x, y - int(8 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * k, (0, 255, 180), t)
    for r, row in enumerate(mat):
        peak = max(row)
        for c, p in enumerate(row):
            v = int(255 * p / peak) if peak > 0 else 0
            x0, y0 = x + c * cell, y + r * cell
            cv2.rectangle(frame, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (v // 3, v, v // 2), -1)
    for i in range(9):
        cv2.putText(frame, str(i + 1), (x + i * cell + int(3 * k), y + 9 * cell + int(11 * k)), cv2.FONT_HERSHEY_PLAIN, 0.8 * k, (200, 200, 200), t)
        cv2.putText(frame, str(i + 1), (x - int(11 * k), y + i * cell + int(10 * k)), cv2.FONT_HERSHEY_PLAIN, 0.8 * k, (200, 200, 200), t)


def toggle_ai(now):
    """เปิด/ปิด AI (ใช้ทั้งปุ่มบนจอและปุ่ม A)"""
    global ai_enabled, status_msg, status_until
    ai_enabled = not ai_enabled
    ai.stop()
    status_msg, status_until = ("AI ON" if ai_enabled else "AI OFF"), now + 1.5


def toggle_mouth(now):
    """สลับโหมดอ้าปาก (ใช้ทั้งปุ่มบนจอและปุ่ม M)"""
    global mouth_mode, status_msg, status_until
    mouth_mode = not mouth_mode
    status_msg = "MOUTH MODE: open mouth to blow" if mouth_mode else "FINGER MODE: fingers only"
    status_until = now + 1.5


def draw_button(frame, box, label, on, hint):
    """ปุ่มสี่เหลี่ยม : เขียว = เปิด, แดง = ปิด ย่อตัวหนังสือให้พอดีปุ่มอัตโนมัติ"""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 0) if on else (60, 60, 160), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), P(1))
    scale = 0.7 * S
    while cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, P(2))[0][0] > (x2 - x1) - P(10):
        scale *= 0.9
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, P(2))
    cv2.putText(frame, label, ((x1 + x2 - tw) // 2, (y1 + y2 + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), P(2))
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38 * S, P(1))
    cv2.putText(frame, hint, ((x1 + x2 - hw) // 2, P(72)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * S, (170, 170, 170), P(1))


# --- หน้าต่างแสดงผลขนาดใหญ่ + ปุ่มเปิด/ปิด AI และโหมดอ้าปากที่คลิกได้ ---
WIN = "AI Soprano Recorder 9 Notes"
DISPLAY_W = 1280          # ขยายภาพกล้อง (640) เป็น 2 เท่าก่อนวาด UI ตัวหนังสือจึงคมชัด
S = 1.0                   # อัตราขยาย UI เทียบกับแบบที่ออกแบบไว้ที่ 640x480 (คำนวณทุกเฟรม)
btn_box = (0, 0, 0, 0)    # ตำแหน่งปุ่ม AI ON/OFF บนภาพ
mouth_box = (0, 0, 0, 0)  # ตำแหน่งปุ่มโหมดอ้าปาก
btn_clicked = False
mouth_clicked = False
fullscreen = False


def P(v):
    """แปลงพิกัดจากแบบ 640x480 ให้เป็นพิกัดบนภาพที่ขยายแล้ว"""
    return int(v * S)


def on_mouse(event, x, y, flags, param):
    global btn_clicked, mouth_clicked
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x1, y1, x2, y2 = btn_box
    if x1 <= x <= x2 and y1 <= y <= y2:
        btn_clicked = True
    x1, y1, x2, y2 = mouth_box
    if x1 <= x <= x2 and y1 <= y <= y2:
        mouth_clicked = True


try:                       # ปรับขนาดหน้าต่างให้พอดีจอ (กันจอโน้ตบุ๊กเล็กกว่า 1280x960)
    import ctypes
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)
except Exception:
    screen_h = 1080
win_h = min(960, screen_h - 120)
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, win_h * 4 // 3, win_h)
cv2.setMouseCallback(WIN, on_mouse)


print("=== AI RECORDER 9 NOTES READY! ===")
print("ปุ่ม : คลิกปุ่ม AI บนจอ หรือ [A] เปิด/ปิด AI  [M] สลับโหมดอ้าปาก/ชูนิ้วอย่างเดียว")
print("      [G] ชอบท่อนที่ AI เล่น  [B] ไม่ชอบ")
print("      [ / ] ลด/เพิ่มความแปลกใหม่  [F] เต็มจอ  [Q] ออก")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)

    # ตรวจการอ้าปาก (เป่า) จากระยะริมฝีปากบน-ล่าง วัดเป็นพิกเซลบนภาพกล้อง 640x480
    is_blowing = False
    breath_vol = 0.0
    ldist = 0.0
    if res.face_landmarks:
        lt = res.face_landmarks.landmark[13]
        lb = res.face_landmarks.landmark[14]
        ldist = math.hypot((lt.x - lb.x) * w, (lt.y - lb.y) * h)
        if ldist > 5.5:
            is_blowing = True
            breath_vol = min(1.0, max(0.4, (ldist - 4.0) / 14.0))

    # ให้ MediaPipe ประมวลผลภาพเล็ก (เร็ว) แล้วค่อยขยายภาพไว้แสดงผล
    frame = cv2.resize(frame, (DISPLAY_W, h * DISPLAY_W // w))
    h, w, _ = frame.shape
    S = w / 640

    if mouth_mode and is_blowing:
        cv2.circle(frame, (int(lt.x * w), int(lt.y * h)), int(ldist * 1.4 * S), (0, 255, 255), P(2))

    # โหมดอ้าปาก : ต้องชูนิ้ว + อ้าปาก ถึงมีเสียง / โหมดชูนิ้ว : ชูนิ้วเมื่อไรมีเสียงทันที

    raw_cnt = 0
    if res.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        raw_cnt += count_fingers_9(res.right_hand_landmarks)

    if res.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        raw_cnt += count_fingers_9(res.left_hand_landmarks)

    # จำกัดช่วงให้อยู่ระหว่าง 1 ถึง 9 นิ้ว
    raw_cnt = max(0, min(9, raw_cnt))

    history_cnt.append(raw_cnt)
    if len(history_cnt) > SMOOTH_FRAMES:
        history_cnt.pop(0)
    active_cnt = max(set(history_cnt), key=history_cnt.count)

    # ตาใครเล่น : ถ้า AI กำลังเล่นท่อนตอบ ให้ปิดเสียงผู้เล่นไว้ก่อน
    now = time.time()
    if btn_clicked:
        btn_clicked = False
        toggle_ai(now)
    if mouth_clicked:
        mouth_clicked = False
        toggle_mouth(now)
    can_play = is_blowing or not mouth_mode
    ai_turn = ai_enabled and ai.is_playing(now)
    ai_note_cnt = ai.current_note(now) if ai_turn else None

    active_note = None
    for n in NOTES:
        c = n["cnt"]
        if ai_turn:
            t_vol = AI_VOLUME if c == ai_note_cnt else 0.0
            if c == ai_note_cnt:
                active_note = n
        elif can_play and active_cnt == c:
            t_vol = breath_vol if mouth_mode else PLAY_VOLUME
            active_note = n
        else:
            t_vol = 0.0

        curr_vols[c] += (t_vol - curr_vols[c]) * 0.3
        if curr_vols[c] < 0.01:
            curr_vols[c] = 0.0
        channels[c].set_volume(curr_vols[c])

    # ตาผู้เล่น : AI ฟังและเรียนรู้ พอผู้เล่นกำมือ (เงียบ) นานพอ AI จะแต่งท่อนตอบ
    if ai_enabled and not ai_turn:
        ai.hear(active_note["cnt"] if active_note else None, now)
        if ai.phrase_ready(now):
            reply = ai.compose(now)
            print("AI ตอบ :", " -> ".join(str(n) for n in reply))

    # วาด UI หน้าจอ (ตำแหน่งทั้งหมดออกแบบที่ 640x480 แล้วขยายด้วย P())
    t1, t2 = P(1), P(2)
    cv2.rectangle(frame, (0, 0), (w, P(80)), (30, 30, 30), -1)
    note_txt = active_note["name"] if active_note else "REST"
    playing = active_note is not None and not ai_turn
    if playing:
        status_txt = "BLOWING" if mouth_mode else "PLAYING"
    else:
        status_txt = "SILENT - open mouth" if mouth_mode and not is_blowing else "SILENT - raise fingers"
    if ai_turn:
        turn_txt, col_txt = "AI TURN", (60, 180, 255)
    else:
        turn_txt = "YOUR TURN" if ai_enabled else "SOLO"
        col_txt = (0, 255, 180) if active_note else (180, 180, 180)

    cv2.putText(frame, f"{turn_txt} | Note: {note_txt}", (P(20), P(35)), cv2.FONT_HERSHEY_DUPLEX, 0.8 * S, col_txt, t2)
    cv2.putText(frame, f"Fingers: {active_cnt}/9 | {status_txt}", (P(20), P(65)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (0, 255, 255) if playing else (200, 200, 200), t2)

    # ปุ่มมุมขวาบน (คลิกเมาส์ได้) : โหมดอ้าปาก และ เปิด/ปิด AI
    mouth_box = (w - P(250), P(12), w - P(135), P(56))
    btn_box = (w - P(125), P(12), w - P(15), P(56))
    draw_button(frame, mouth_box, "MOUTH : ON" if mouth_mode else "MOUTH : OFF", mouth_mode, "click / [M]")
    draw_button(frame, btn_box, "AI : ON" if ai_enabled else "AI : OFF", ai_enabled, "click / [A]")

    # แผงข้อมูล AI มุมล่างซ้าย
    if ai_enabled:
        panel_y = h - P(175)
        cv2.rectangle(frame, (0, panel_y - P(30)), (P(365), h), (30, 30, 30), -1)
        draw_ai_brain(frame, ai, P(25), panel_y, cell=P(13))
        info_x = P(160)
        info = [(f"Learned: {ai.learned}", (220, 220, 220)),
                (f"Creativity T: {ai.temperature:.1f}", (220, 220, 220)),
                (f"Heard: {len(ai.phrase)} notes", (220, 220, 220))]
        if ai.last_reply_notes:
            info.append(("AI: " + "-".join(str(n) for n in ai.last_reply_notes[1:]), (60, 180, 255)))
        if not ai_turn and not ai.feedback_given:
            info.append(("Rate it: [G] good [B] bad", (0, 255, 255)))
        for i, (txt, col) in enumerate(info):
            cv2.putText(frame, txt, (info_x, panel_y + P(5 + 22 * i)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, col, t1)
    if now < status_until:
        cv2.putText(frame, status_msg, (P(20), P(105)), cv2.FONT_HERSHEY_DUPLEX, 0.8 * S, (0, 255, 180), t2)

    # วาดรูปขลุ่ยรีคอร์ดเดอร์
    rx = w - P(75)
    ry_top = P(105)
    ry_bot = h - P(30)
    cv2.rectangle(frame, (rx - P(18), ry_top), (rx + P(18), ry_bot), (210, 235, 245), -1)
    cv2.rectangle(frame, (rx - P(18), ry_top), (rx + P(18), ry_bot), (80, 100, 120), t2)
    cv2.rectangle(frame, (rx - P(10), ry_top + P(15)), (rx + P(10), ry_top + P(28)), (50, 50, 50), -1)

    holes_state = active_note["holes"] if active_note else [0,0,0,0,0,0,0]
    for i in range(7):
        hy = ry_top + P(65 + i * 38)
        is_closed = (holes_state[i] == 1)
        if is_closed:
            cv2.circle(frame, (rx, hy), P(10), (20, 20, 20), -1)
        else:
            cv2.circle(frame, (rx, hy), P(10), (255, 255, 255), -1)
            cv2.circle(frame, (rx, hy), P(10), (40, 40, 40), t2)

    cv2.imshow(WIN, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break                                    # กด Q หรือกดกากบาทปิดหน้าต่าง
    elif key == ord('a'):
        toggle_ai(now)
    elif key == ord('m'):
        toggle_mouth(now)
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    elif key in (ord('g'), ord('b')):
        if ai.feedback(key == ord('g')):
            status_msg = "Thanks! AI will do more of this" if key == ord('g') else "OK, AI will avoid this"
            status_until = now + 2.0
    elif key == ord(']'):
        ai.temperature = min(2.0, ai.temperature + 0.1)
    elif key == ord('['):
        ai.temperature = max(0.2, ai.temperature - 0.1)

cap.release()
cv2.destroyAllWindows()
pygame.quit()