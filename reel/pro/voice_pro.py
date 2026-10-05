"""Narration for the professional cut. Reuses the TTS + timing code in ../voice.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import voice as V  # noqa: E402

# (scene, spoken, caption (*word* = accent), pause after, sfx cue)
SCRIPT = [
    ("hook", "Two sensors at the ears caught more seizures than an eighteen-electrode hospital cap.",
     "Two sensors at the ears caught *more* *seizures* than an *18-electrode* hospital cap.", 0.25, "impact"),
    ("hook", "Here's how I built it.", None, 0.35, "whoosh"),
    ("problem", "Fifty million people live with epilepsy.", "*50* *million* people live with epilepsy.", 0.15, "whoosh"),
    ("problem", "Seizures strike without warning.", "Seizures strike *without* *warning.*", 0.15, "pulse"),
    ("problem", "And proper monitoring still means a hospital cap full of wires.", None, 0.35, "tick"),
    ("spark", "Then Zomato's founder started wearing a brain sensor on his temple.",
     "Then Zomato's founder started wearing a brain sensor on his *temple.*", 0.15, "whoosh"),
    ("spark", "So I asked: could something that small catch a seizure?", "So I asked: could something that small *catch* *a* *seizure?*", 0.35, "riser"),
    ("twist", "It can't. It reads blood flow, not brain activity.", "It *can't.* It reads *blood* *flow,* not brain activity.", 0.4, "glitch"),
    ("build", "So I built my own lab.", "So I built my *own* *lab.*", 0.1, "whoosh"),
    ("build", "A physics-based seizure simulator. Ions, neurons, blood flow.", None, 0.1, "tick"),
    ("build", "Cross-checked against The Virtual Brain.", "Cross-checked against *The* *Virtual* *Brain.*", 0.35, "tick"),
    ("data", "Then I tested it on real hospital EEG.", "Then I tested it on *real* *hospital* *EEG.*", 0.1, "whoosh"),
    ("data", "Over a hundred and sixty patients. Eleven hundred seizures.", "*160+* patients. *1,100+* seizures.", 0.35, "count"),
    ("design", "The design that won: a question-mark ear hook.", "The design that won: a *question-mark* *ear* *hook.*", 0.1, "whoosh"),
    ("design", "Just two channels.", "Just *two* *channels.*", 0.35, "tick"),
    ("result", "On adults it had never seen, it caught ninety-six percent of seizures.",
     "On adults it had never seen, it caught *96%* of seizures.", 0.15, "riser"),
    ("result", "The full hospital cap: ninety-one.", "The full hospital cap: *91%.*", 0.4, "impact"),
    ("honest", "The hard part is false alarms.", "The hard part is *false* *alarms.*", 0.1, "whoosh"),
    ("honest", "Adding heart signals cut them by up to forty-eight percent.", "Adding *heart* *signals* cut them by up to *48%.*", 0.35, "pulse"),
    ("next", "Now I'm building prototype one.", "Now I'm building *Prototype* *1.*", 0.1, "whoosh"),
    ("next", "Follow the build.", "*Follow* *the* *build.*", 1.4, "impact"),
]

if __name__ == "__main__":
    V.SCRIPT = SCRIPT
    V.BUILD = os.path.join(HERE, "build")
    V.SPEED = float(os.environ.get("REEL_SPEED", "1.05"))
    V.main()
