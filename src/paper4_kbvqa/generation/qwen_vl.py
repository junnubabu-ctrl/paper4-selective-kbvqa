from __future__ import annotations
import json
import math
import re
from paper4_kbvqa.generation.base import AnswerGenerator, build_prompt_payload
from paper4_kbvqa.types import Evidence

_JSON_RE = re.compile(r"\{.*\}", re.S)

class Qwen25VLGenerator(AnswerGenerator):
    """Lazy-loading Qwen2.5-VL adapter for free-GPU execution.

    The returned ``raw_confidence`` is the geometric mean probability of generated
    tokens. It is an uncalibrated model-likelihood signal, not a correctness
    probability. Paper-level confidence is produced only after validation-set
    calibration.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", revision="main", load_in_4bit=True, max_pixels=None):
        self.model_name=model_name
        self.revision=revision
        self.load_in_4bit=load_in_4bit
        self.max_pixels=max_pixels
        self._loaded=False

    def _load(self):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError("Install VLM extras in a CUDA environment") from e
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen VLM benchmark execution requires a CUDA GPU")
        quant = None
        if self.load_in_4bit:
            try:
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
            except Exception as e:
                raise RuntimeError("4-bit mode requested but bitsandbytes-compatible quantization is unavailable") from e
        proc_kwargs = {}
        if self.max_pixels is not None:
            proc_kwargs["max_pixels"] = int(self.max_pixels)
        self.processor=AutoProcessor.from_pretrained(self.model_name,revision=self.revision,**proc_kwargs)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            revision=self.revision,
            device_map="auto",
            quantization_config=quant,
            torch_dtype="auto",
        )
        self.model.eval()
        self._loaded=True

    @staticmethod
    def _parse_output(text: str, fallback_ids: list[str]) -> tuple[str, list[str]]:
        m = _JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                ans = str(obj.get("answer", "")).strip()
                ids = [str(x) for x in obj.get("evidence_ids", [])]
                if ans:
                    return ans, ids
            except Exception:
                pass
        # Parse failure must not manufacture citations for every supplied item.
        return text.strip(), []

    def extract_visual_entities(self, image_path: str, question: str, max_entities: int = 8) -> list[str]:
        if not self._loaded:
            self._load()
        import torch
        from PIL import Image
        image=Image.open(image_path).convert("RGB")
        prompt=(
            "Identify only visible entities in the image that are relevant to answering the question. "
            "Return strict JSON: {\"entities\":[...]} with at most " + str(int(max_entities)) + " short noun phrases. "
            "Do not answer the question.\nQuestion: " + question
        )
        messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=[image],return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=64,do_sample=False)
        gen=out[0][inputs.input_ids.shape[1]:]
        decoded=self.processor.decode(gen,skip_special_tokens=True).strip()
        m=_JSON_RE.search(decoded)
        if not m:
            return []
        try:
            obj=json.loads(m.group(0)); vals=obj.get("entities",[])
            if not isinstance(vals,list): return []
            clean=[]
            for x in vals:
                x=str(x).strip()
                if x and x.lower() not in {y.lower() for y in clean}: clean.append(x)
            return clean[:max_entities]
        except Exception:
            return []

    def generate(self,image_path: str,question: str,evidence: list[Evidence]) -> dict:
        if not self._loaded:
            self._load()
        import torch
        from PIL import Image
        payload=build_prompt_payload(question,evidence)
        evidence_text="\n".join(f"[{x['id']}] {x['text']}" for x in payload['evidence']) or "[NONE] No external evidence supplied."
        prompt=(
            "Answer the visual question using the image and only relevant supplied evidence. "
            "Do not invent evidence. Return strict JSON with keys answer and evidence_ids. "
            "answer must be concise. evidence_ids must contain only supplied IDs that directly support the answer.\n"
            f"Question: {question}\nEvidence:\n{evidence_text}"
        )
        image = Image.open(image_path).convert('RGB')
        messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=[image],return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out=self.model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        seq = out.sequences[0]
        prompt_len = int(inputs.input_ids.shape[1])
        gen_ids = seq[prompt_len:]
        decoded=self.processor.decode(gen_ids,skip_special_tokens=True).strip()
        token_logps=[]
        for token_id, logits in zip(gen_ids, out.scores):
            lp=torch.log_softmax(logits[0].float(),dim=-1)[int(token_id)].item()
            token_logps.append(lp)
        raw_conf=float(math.exp(sum(token_logps)/len(token_logps))) if token_logps else float('nan')
        fallback_ids=[e.evidence_id for e in evidence]
        answer, support_ids=self._parse_output(decoded, fallback_ids)
        allowed=set(fallback_ids)
        support_ids=[x for x in support_ids if x in allowed]
        return {"answer":answer,"raw_confidence":raw_conf,"supporting_evidence_ids":support_ids,"raw_text":decoded}
