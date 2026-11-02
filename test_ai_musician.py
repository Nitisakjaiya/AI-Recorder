import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_musician import AIMusician

FPS = 30
DT = 1 / FPS


def play(ai, t, notes, hold=0.4, jitter_frames=0):
    """จำลองการเป่าโน้ตตามลำดับ แต่ละโน้ตค้าง hold วินาที แล้วคืนเวลาปัจจุบัน"""
    for n in notes:
        # มือสั่น : โน้ตอื่นแวบมา jitter_frames เฟรม ก่อนเข้าโน้ตจริง
        for _ in range(jitter_frames):
            ai.hear(9 if n != 9 else 1, t); t += DT
        for _ in range(int(hold * FPS)):
            ai.hear(n, t); t += DT
    return t


def silence(ai, t, seconds):
    for _ in range(int(seconds * FPS)):
        ai.hear(None, t); t += DT
    return t


ai = AIMusician(seed=1)
t = 0.0

# 1) ผู้เล่นเล่นสเกลขึ้นลง โดยมีมือสั่นแวบ 2 เฟรมก่อนทุกโน้ต
t = play(ai, t, [1, 2, 3, 4, 3, 2, 1], hold=0.4, jitter_frames=2)
assert [n for n, _ in ai.phrase] == [1, 2, 3, 4, 3, 2, 1], ai.phrase
print("ท่อนที่ได้ยิน :", [n for n, _ in ai.phrase], "(กรองมือสั่นออกแล้ว)")
print("เรียนการเปลี่ยนโน้ตไป :", ai.learned, "ครั้ง")
assert ai.learned == 6

# 2) ยังเงียบไม่นานพอ ยังไม่ถึงตา AI
t2 = silence(ai, t, 0.5)
assert not ai.phrase_ready(t2)
t = silence(ai, t2, 1.0)
assert ai.phrase_ready(t), "ควรถึงตา AI แล้ว"

# 3) AI แต่งท่อนตอบ
reply = ai.compose(t)
print("AI ตอบ :", reply, "จังหวะ", ai.reply[0][1], "วินาที/โน้ต")
assert 3 <= len(reply) <= 8
assert all(1 <= n <= 9 for n in reply)
assert reply[-1] == 1, "ต้องจบที่โน้ตแรกของผู้เล่น"
assert abs(ai.reply[0][1] - 0.4667) < 0.05  # ใช้จังหวะผู้เล่น (0.4 + มือสั่น 2 เฟรม)

# 4) เล่นท่อนตอบทีละเฟรม ต้องได้โน้ตตามลำดับ
heard, last = [], None
while ai.is_playing(t):
    n = ai.current_note(t)
    if n is not None and n != last:
        heard.append(n)
    last = n
    t += DT
assert heard == reply, (heard, reply)
print("เสียงที่ AI เล่นจริง :", heard)

# 5) ความน่าจะเป็นที่เรียนมา : หลังโน้ต 2 ควรไป 1 หรือ 3 เป็นหลัก
p2 = ai.probs(2)
print("P(ต่อจากโน้ต 2) :", [round(x, 2) for x in p2])
assert p2[0] + p2[2] > 0.6

# 6) รางวัล : ชอบ -> เพิ่ม, ไม่ชอบ -> ลด, กดซ้ำไม่ได้
pairs = list(zip(ai.last_reply_notes, ai.last_reply_notes[1:]))
a, b = pairs[0]
before = ai.counts[a - 1][b - 1]
assert ai.feedback(True)
assert ai.counts[a - 1][b - 1] > before
assert not ai.feedback(False), "ให้คะแนนท่อนเดียวซ้ำไม่ได้"

# 7) ไม่ชอบซ้ำหลายรอบ การเปลี่ยนโน้ตนั้นต้องมีโอกาสลดลงจริง
ai2 = AIMusician(seed=2)
ai2.last_reply_notes = [5, 8]; ai2.feedback_given = False
p_before = ai2.probs(5)[7]
for _ in range(3):
    ai2.feedback_given = False
    ai2.feedback(False)
p_after = ai2.probs(5)[7]
print(f"P(5->8) ก่อน {p_before:.3f} หลังไม่ชอบ 3 ครั้ง {p_after:.6f}")
assert p_after < p_before

# 8) ทุกแถวของตาราง P รวมได้ 1
for row in ai.matrix():
    assert abs(sum(row) - 1) < 1e-9

print("\nผ่านทุกการทดสอบ")
