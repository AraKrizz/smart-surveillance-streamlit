import os
import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM

# If it is too slow on your laptop, use "HuggingFaceTB/SmolVLM-256M-Instruct"
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Per-event confirmation thresholds (raise to be stricter, lower to catch more)
# Falls: detector already checks "was upright, then down", so VLM does not veto.
THRESHOLDS = {"Person Down": 0.0, "Altercation/Proximity": 0.50}

OPTIONS = {
    "Person Down": ("a person lying or collapsed on the floor",
                    "people standing, sitting, walking or working normally"),
    "Altercation/Proximity": ("people fighting, punching, kicking or violently struggling",
                              "people standing, sitting, talking, walking or working normally"),
}
CATEGORY = {"Person Down": "Person Down", "Altercation/Proximity": "Fighting / Altercation"}
DEBUG_DIR = os.path.join("data", "keyframes")


class EventExplainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = None, None
        try:
            print(f"[+] Loading VLM {MODEL_ID} on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, do_image_splitting=False)
            self.model = AutoVLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(self.device).eval()
            tok = self.processor.tokenizer
            self.ids = {L: [tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)] for L in "AB"}
            print("[+] VLM ready.")
        except Exception as e:
            print(f"[-] VLM FAILED to load, events will be UNVERIFIED: {e}")

    def _crop(self, img, box, pad, min_side=512):
        W, H = img.size
        if box is None:
            return img
        x1, y1, x2, y2 = box
        half = max(max(x2 - x1, y2 - y1) * (1 + pad) / 2, 60)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop = img.crop((int(max(0, cx - half)), int(max(0, cy - half)),
                         int(min(W, cx + half)), int(min(H, cy + half))))
        scale = max(1.0, min_side / min(crop.size))
        if scale > 1:
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        return crop

    def _run(self, image, text, max_new_tokens, scores=False):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      output_scores=scores, return_dict_in_generate=True)
        return inputs, out

    def _prob_A(self, image, a, b):
        q = (f"Look at this security camera image. Which statement best describes it?\n"
             f"A) {a}\nB) {b}\nAnswer with only the letter A or B.")
        _, out = self._run(image, q, 1, scores=True)
        logits = out.scores[0][0].float()
        la = torch.logsumexp(logits[self.ids["A"]], 0)
        lb = torch.logsumexp(logits[self.ids["B"]], 0)
        return float(torch.softmax(torch.stack([la, lb]), 0)[0])

    def _describe(self, image):
        inputs, out = self._run(image, "Describe what is happening in this security image in one short sentence.", 40)
        new = out.sequences[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()

    def analyze_event(self, image_path, trigger_type, box=None):
        normal = {"category": "Normal Activity", "confidence": 0.5, "summary": "Trigger rejected by VLM."}
        if trigger_type not in OPTIONS or not os.path.exists(image_path):
            return normal
        category = CATEGORY[trigger_type]
        if self.model is None:
            return {"category": category, "confidence": 0.5, "summary": "(Unverified) " + trigger_type}

        img = Image.open(image_path).convert("RGB")
        crops = [self._crop(img, box, 0.4), self._crop(img, box, 1.5)]
        probs = []
        for i, c in enumerate(crops):
            try:
                c.save(os.path.join(DEBUG_DIR, f"debug_crop_{i}.jpg"))
            except Exception:
                pass
            probs.append(self._prob_A(c, *OPTIONS[trigger_type]))
        p = max(probs)
        best = crops[probs.index(p)]
        thr = THRESHOLDS[trigger_type]
        ok = p >= thr
        print(f"[VLM] {trigger_type}: tight={probs[0]:.2f} wide={probs[1]:.2f} "
              f"-> p={p:.2f} (need {thr}) {'CONFIRMED' if ok else 'rejected'}")

        if not ok:
            normal["confidence"] = round(1 - p, 2)
            return normal

        summary = self._describe(best)
        if trigger_type == "Person Down":
            conf = 0.85
            if len(summary) < 10:
                summary = "A person fell and is lying on the ground."
        else:
            conf = round(max(p, 0.5), 2)
        return {"category": category, "confidence": conf, "summary": summary}