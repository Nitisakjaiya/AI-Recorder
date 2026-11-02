"""
ai_musician.py -- AI นักดนตรีที่ฟังเราเล่น แล้วแต่งท่อนตอบกลับเอง (Call & Response)
=====================================================================================

แนวคิด
------
1) ฟังและเรียนรู้ (Learning)
   ทุกครั้งที่ผู้เล่นเปลี่ยนจากโน้ต s ไปโน้ต s'  AI จะนับไว้ในตาราง counts[s][s']
   แล้วแปลงเป็นความน่าจะเป็นของการเปลี่ยนโน้ต  P(s' | s)
   (เหมือน Transition Probability ใน MDP แต่ AI ประมาณค่าเองจากข้อมูลที่ได้ยิน)

2) คิดเอง (Decision)
   เมื่อผู้เล่นหยุดเป่านานพอ AI จะแต่งท่อนตอบกลับ
     - เริ่มต่อจากโน้ตสุดท้ายที่ผู้เล่นเล่น
     - สุ่มโน้ตถัดไปตาม P(s' | s) ที่เรียนมา
     - จบท่อนด้วยโน้ตแรกของผู้เล่น ให้ฟังเหมือน "กลับบ้าน"
     - เล่นด้วยจังหวะเดียวกับที่ผู้เล่นเพิ่งเล่น

3) รับคำติชม (Reward)
   ผู้ฟังกด ชอบ / ไม่ชอบ ท่อนที่ AI เล่น
     ชอบ    -> เพิ่มน้ำหนักการเปลี่ยนโน้ตที่ใช้ในท่อนนั้น
     ไม่ชอบ -> ลดน้ำหนักลง (ครึ่งหนึ่ง แล้วหักเพิ่มอีก 0.5)
   AI จึงแต่งเพลงเปลี่ยนไปตามรสนิยมของผู้ฟัง (reward-based learning)

ช่วงแรกที่ยังไม่ได้ยินอะไร AI จะใช้ "ความรู้ดนตรีพื้นฐาน" (prior)
ว่าโน้ตที่อยู่ใกล้กันมักเล่นต่อกัน เพื่อไม่ให้ท่อนแรก ๆ กระโดดไปมามั่ว ๆ
"""

import math
import random

N_NOTES = 9


class AIMusician:
    def __init__(self, n_notes=N_NOTES, temperature=0.8, prior_strength=1.0,
                 min_hold=0.12, phrase_gap=1.2, seed=None):
        self.n = n_notes
        self.temperature = temperature          # ค่าความกล้าเสี่ยง : ต่ำ = เล่นตามที่เรียนมา, สูง = แปลกใหม่
        self.prior_strength = prior_strength    # น้ำหนักของความรู้ดนตรีพื้นฐาน
        self.min_hold = min_hold                # ต้องค้างโน้ตนานเท่านี้ (วินาที) จึงนับว่าเล่นจริง
        self.phrase_gap = phrase_gap            # เงียบนานเท่านี้ (วินาที) ถือว่าจบท่อน
        self.rng = random.Random(seed)

        # ตารางนับการเปลี่ยนโน้ต counts[a][b] (ใช้ index 0..n-1 แทนโน้ต 1..n)
        self.counts = [[0.0] * self.n for _ in range(self.n)]
        self.learned = 0                        # จำนวนการเปลี่ยนโน้ตที่ได้ยินมาแล้วทั้งหมด

        # สถานะการฟัง
        self.phrase = []                        # [(note, เวลาเริ่ม)] ของท่อนปัจจุบัน
        self._candidate = None                  # โน้ตที่กำลังได้ยิน แต่ยังค้างไม่นานพอ
        self._candidate_t = 0.0
        self._silence_t = None                  # เวลาที่เริ่มเงียบ
        self.last_user_phrase = []

        # สถานะการเล่นตอบ
        self.reply = []                         # [(note, ระยะเวลา)]
        self.reply_start = None
        self.last_reply_notes = []
        self.feedback_given = True

    # ------------------------------------------------------------------
    # ความน่าจะเป็นของการเปลี่ยนโน้ต
    # ------------------------------------------------------------------
    @staticmethod
    def _prior(a, b):
        """ความรู้ดนตรีพื้นฐาน : โน้ตใกล้กันมีโอกาสสูง, โน้ตเดิมซ้ำมีโอกาสรองลงมา"""
        step = abs(a - b)
        if step == 0:
            return 0.5
        return math.exp(-(step - 1) / 1.5)

    def probs(self, cur):
        """P(next | cur) สำหรับโน้ต 1..n  (คืนค่า list ยาว n)"""
        a = cur - 1
        weights = []
        for b in range(self.n):
            w = self.counts[a][b] + self.prior_strength * self._prior(a, b)
            weights.append(max(w, 1e-6) ** (1.0 / self.temperature))
        total = sum(weights)
        return [w / total for w in weights]

    def _sample_next(self, cur):
        p = self.probs(cur)
        return self.rng.choices(range(1, self.n + 1), weights=p, k=1)[0]

    # ------------------------------------------------------------------
    # ฟังผู้เล่น
    # ------------------------------------------------------------------
    def hear(self, note, t):
        """
        เรียกทุกเฟรมขณะเป็นตาของผู้เล่น
        note = โน้ตที่กำลังเป่า (1..n) หรือ None ถ้าไม่ได้เป่า
        """
        if note is None:
            self._candidate = None
            if self._silence_t is None:
                self._silence_t = t
            return

        if note != self._candidate:
            self._candidate = note
            self._candidate_t = t
            return

        if t - self._candidate_t < self.min_hold:
            return                                  # ยังค้างไม่นานพอ อาจเป็นมือสั่น

        after_silence = self._silence_t is not None
        self._silence_t = None
        last = self.phrase[-1][0] if self.phrase else None

        if last != note or after_silence:
            if last is not None:
                self.counts[last - 1][note - 1] += 1.0   # เรียนรู้การเปลี่ยนโน้ต
                self.learned += 1
            self.phrase.append((note, self._candidate_t))

    def phrase_ready(self, t, min_notes=3):
        """ผู้เล่นเงียบนานพอและเล่นมาอย่างน้อย min_notes โน้ต -> ถึงตา AI"""
        return (self._silence_t is not None
                and t - self._silence_t >= self.phrase_gap
                and len(self.phrase) >= min_notes)

    def _user_tempo(self):
        """ความยาวโน้ตเฉลี่ยของผู้เล่น (วินาที) ใช้เป็นจังหวะของท่อนตอบ"""
        times = [st for _, st in self.phrase]
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        if not gaps:
            return 0.45
        gaps.sort()
        return min(0.9, max(0.2, gaps[len(gaps) // 2]))    # ใช้ค่ามัธยฐาน กันค่าโดด

    # ------------------------------------------------------------------
    # คิดและเล่นท่อนตอบ
    # ------------------------------------------------------------------
    def compose(self, t):
        """แต่งท่อนตอบจากท่อนที่ผู้เล่นเพิ่งเล่น แล้วเริ่มเล่นที่เวลา t"""
        notes_in = [n for n, _ in self.phrase]
        length = min(8, max(3, len(notes_in)))
        tempo = self._user_tempo()

        cur = notes_in[-1]
        out = []
        for _ in range(length - 1):
            cur = self._sample_next(cur)
            out.append(cur)
        out.append(notes_in[0])                     # จบที่โน้ตแรกของผู้เล่น ("กลับบ้าน")

        self.reply = [(n, tempo) for n in out]
        self.reply[-1] = (out[-1], tempo * 2)       # โน้ตสุดท้ายค้างยาวเป็นการปิดท่อน
        self.reply_start = t
        self.last_user_phrase = notes_in
        self.last_reply_notes = [notes_in[-1]] + out
        self.feedback_given = False

        self.phrase = []
        self._candidate = None
        self._silence_t = None
        return out

    def is_playing(self, t):
        if self.reply_start is None:
            return False
        return t - self.reply_start < sum(d for _, d in self.reply)

    def current_note(self, t):
        """โน้ตที่ AI ต้องเล่น ณ เวลา t (None = เล่นจบแล้ว หรือกำลังเว้นช่วงสั้น ๆ ระหว่างโน้ต)"""
        if not self.is_playing(t):
            return None
        elapsed = t - self.reply_start
        for note, dur in self.reply:
            if elapsed < dur:
                return note if elapsed < dur * 0.85 else None   # เว้นช่วงท้ายโน้ตให้ได้ยินการตัดโน้ต
            elapsed -= dur
        return None

    def stop(self):
        self.reply_start = None

    # ------------------------------------------------------------------
    # รับคำติชม
    # ------------------------------------------------------------------
    def feedback(self, good):
        """ให้รางวัลกับท่อนล่าสุด : good=True เพิ่มน้ำหนัก, False ลดน้ำหนัก"""
        if self.feedback_given or len(self.last_reply_notes) < 2:
            return False
        for a, b in zip(self.last_reply_notes, self.last_reply_notes[1:]):
            if good:
                self.counts[a - 1][b - 1] += 2.0
            else:
                # ลดครึ่งแล้วหักเพิ่ม เพื่อให้ลดได้แม้เป็นการเปลี่ยนโน้ตที่ AI ยังไม่เคยได้ยิน (นับเป็น 0)
                self.counts[a - 1][b - 1] = self.counts[a - 1][b - 1] * 0.5 - 0.5
        self.feedback_given = True
        return True

    def matrix(self):
        """ตาราง P(s'|s) ทั้งหมด ใช้วาด heatmap บนหน้าจอ"""
        return [self.probs(s) for s in range(1, self.n + 1)]
