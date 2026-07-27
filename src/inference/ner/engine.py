"""NerEngine: load model 1 lần, expose predict_text() / predict_file().

Class NerAssertionModel giữ nguyên kiến trúc từ train_ner_colab_crf —
KHÔNG được sửa layer nào ở đây mà không sửa đồng bộ bên train, nếu không
load_state_dict sẽ báo lỗi shape mismatch (hoặc tệ hơn: load "thành công"
nhưng assertion head học sai vùng pooling).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchcrf import CRF  # pip install pytorch-crf
from transformers import AutoModel, AutoTokenizer

from .. import config as cfg
from . import offset_mapper as om
from . import postprocessor as pp
from . import repair_gate
from . import sectioner
from ..schemas import NerEntity, SectionResult


class NerAssertionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_ner_tags: int,
        num_assert_labels: int = 3,
        dropout: float = 0.1,
        assertion_dropout: float = 0.3,
        context_window: int = 10,
        num_frozen_layers: int = 0,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.ner_dropout = nn.Dropout(dropout)
        self.assertion_dropout = nn.Dropout(assertion_dropout)

        self.ner_head = nn.Linear(hidden_size, num_ner_tags)
        self.crf = CRF(num_ner_tags, batch_first=True)

        self.assertion_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, num_assert_labels),
        )

        self.num_assert_labels = num_assert_labels
        self.context_window = context_window

        if num_frozen_layers > 0 and freeze_embeddings:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        if num_frozen_layers > 0:
            for layer in self.encoder.encoder.layer[:num_frozen_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def _pool_spans(self, hidden_states: torch.Tensor, assertion_spans_batch: list,
                     attention_mask: torch.Tensor):
        seq_len = hidden_states.size(1)
        span_vectors = []
        span_owner = []
        w = self.context_window

        for b, spans in enumerate(assertion_spans_batch):
            real_len = int(attention_mask[b].sum().item())
            for span in spans:
                s, e = span['token_start'], span['token_end']
                s = max(0, min(s, seq_len))
                e = max(s, min(e, seq_len))
                if e <= s:
                    continue

                local_repr = hidden_states[b, s:e, :].mean(dim=0)

                ctx_left_s = max(0, s - w)
                ctx_left_e = s
                ctx_right_s = e
                ctx_right_e = min(real_len, e + w)

                ctx_slices = []
                if ctx_left_e > ctx_left_s:
                    ctx_slices.append(hidden_states[b, ctx_left_s:ctx_left_e, :])
                if ctx_right_e > ctx_right_s:
                    ctx_slices.append(hidden_states[b, ctx_right_s:ctx_right_e, :])

                if ctx_slices:
                    context_repr = torch.cat(ctx_slices, dim=0).mean(dim=0)
                else:
                    context_repr = local_repr

                combined = torch.cat([local_repr, context_repr], dim=-1)
                span_vectors.append(combined)
                span_owner.append((b, span))

        if len(span_vectors) == 0:
            span_vectors = hidden_states.new_zeros((0, hidden_states.size(-1) * 2))
        else:
            span_vectors = torch.stack(span_vectors, dim=0)

        return span_vectors, span_owner

    def forward(self, input_ids, attention_mask, assertion_spans_batch, ner_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden = outputs.last_hidden_state

        ner_hidden = self.ner_dropout(encoder_hidden)
        assertion_hidden = self.assertion_dropout(encoder_hidden)

        ner_emissions = self.ner_head(ner_hidden)

        span_vectors, span_owner = self._pool_spans(assertion_hidden, assertion_spans_batch, attention_mask)
        assertion_logits = self.assertion_head(span_vectors)

        mask = attention_mask.bool()
        result = {
            "ner_emissions": ner_emissions,
            "assertion_logits": assertion_logits,
            "span_owner": span_owner,
            "ner_pred_tags": self.crf.decode(ner_emissions, mask=mask),
        }
        return result


class NerEngine:
    """Wrap model + tokenizer + VnCoreNLP, expose predict API cấp cao."""

    def __init__(self, model: NerAssertionModel, tokenizer, rdr, id2nerlabel: dict,
                 id2assertlabel: dict, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.rdr = rdr
        self.id2nerlabel = id2nerlabel
        self.id2assertlabel = id2assertlabel
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path = cfg.DEFAULT_CHECKPOINT_PATH,
        label_dicts_path: str | Path = cfg.DEFAULT_LABEL_DICTS_PATH,
        backbone: str = cfg.DEFAULT_BACKBONE,
        vncorenlp_jar: str | Path = cfg.DEFAULT_VNCORENLP_JAR,
        device: str = cfg.DEFAULT_DEVICE,
        context_window: int = cfg.CONTEXT_WINDOW,
    ) -> "NerEngine":
        from vncorenlp import VnCoreNLP

        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")

        with open(label_dicts_path, "r", encoding="utf-8") as f:
            label_dicts = json.load(f)
        nerlabel2id = label_dicts["nerlabel2id"]
        assertlabel2id = label_dicts["assertlabel2id"]
        id2nerlabel = {v: k for k, v in nerlabel2id.items()}
        id2assertlabel = {v: k for k, v in assertlabel2id.items()}

        tokenizer = AutoTokenizer.from_pretrained(backbone)

        model = NerAssertionModel(
            model_name=backbone,
            num_ner_tags=len(nerlabel2id),
            num_assert_labels=len(assertlabel2id),
            context_window=context_window,
        )
        state_dict = torch.load(checkpoint_path, map_location=resolved_device)
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        model.eval()

        rdr = VnCoreNLP(str(vncorenlp_jar), annotators="wseg", max_heap_size="-Xmx2g")

        return cls(model, tokenizer, rdr, id2nerlabel, id2assertlabel, resolved_device)

    @torch.no_grad()
    def predict_text(
        self,
        text: str,
        *,
        max_len: int = cfg.MAX_LEN,
        overlap_words: int = cfg.OVERLAP_WORDS,
        assertion_threshold: float = cfg.ASSERTION_THRESHOLD,
        single_assertion: bool = cfg.SINGLE_ASSERTION,
        apply_repair_gate: bool = cfg.ENABLE_REPAIR_GATE,
    ) -> list[NerEntity]:
        """Predict trên 1 đoạn text ĐÃ ĐƯỢC LÀM SẠCH (không tự clean ở đây
        — gọi postprocessor.clean_text_for_inference() trước nếu cần, để
        tách bạch rõ input nào đã qua bước nào)."""
        model = self.model
        tokens, offsets, line_ids = om.segment_with_offsets(text, self.rdr)
        chunks = om.make_word_chunks(tokens, self.tokenizer, max_len=max_len, overlap_words=overlap_words)

        all_results: list[dict] = []

        for chunk_start, chunk_end in chunks:
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_line_ids = line_ids[chunk_start:chunk_end]

            input_ids, attention_mask, word_to_subword_start, word_num_subtokens = (
                om.encode_words_for_inference(self.tokenizer, chunk_tokens, max_len=max_len)
            )

            valid_word_count = len(word_to_subword_start)
            if valid_word_count == 0:
                continue

            input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=self.device)

            encoder_out = model.encoder(input_ids=input_ids_t, attention_mask=attention_mask_t)
            hidden = model.ner_dropout(encoder_out.last_hidden_state)
            ner_emissions = model.ner_head(hidden)

            mask = attention_mask_t.bool()
            pred_tag_ids = model.crf.decode(ner_emissions, mask=mask)[0]

            word_tags = []
            word_confs = []
            for sw_start in word_to_subword_start:
                pred_id = pred_tag_ids[sw_start]
                word_tags.append(pp.get_label(self.id2nerlabel, pred_id))
                word_confs.append(1.0)  # CRF không có prob per-token -> điểm cố định

            local_entities = pp.extract_entities_from_word_tags(
                word_tags, line_ids=chunk_line_ids[:valid_word_count],
            )

            spans_for_pool = []
            for ent in local_entities:
                lws, lwe = ent["word_start"], ent["word_end"]
                if lws >= valid_word_count:
                    continue
                if chunk_start > 0 and lws == 0:
                    continue

                global_ws = chunk_start + lws
                global_we = chunk_start + lwe
                if not om.is_valid_char_span(offsets, global_ws, global_we):
                    continue

                sw_s = word_to_subword_start[lws]
                last_word = min(lwe - 1, valid_word_count - 1)
                sw_e = min(word_to_subword_start[last_word] + word_num_subtokens[last_word], max_len)
                if sw_e <= sw_s:
                    continue

                score = sum(word_confs[lws:lwe]) / max(1, lwe - lws)
                spans_for_pool.append({
                    "token_start": sw_s, "token_end": sw_e,
                    "word_start": lws, "word_end": lwe,
                    "global_word_start": global_ws, "global_word_end": global_we,
                    "type": ent["type"], "score": score,
                })

            if not spans_for_pool:
                continue

            span_vectors, span_owner = model._pool_spans(hidden, [spans_for_pool], attention_mask_t)
            assertion_logits = model.assertion_head(span_vectors)
            assertion_probs = torch.sigmoid(assertion_logits).detach().cpu().tolist()

            for (_, span), probs in zip(span_owner, assertion_probs):
                gws, gwe = span["global_word_start"], span["global_word_end"]
                char_start = offsets[gws][0]
                char_end = offsets[gwe - 1][1]
                entity_text = text[char_start:char_end]

                active_assertions = []
                for i, p in enumerate(probs):
                    label = pp.get_label(self.id2assertlabel, i)
                    if label == "NONE":
                        continue
                    if p >= assertion_threshold:
                        active_assertions.append(label)
                if single_assertion:
                    active_assertions = pp.collapse_assertions(active_assertions)

                all_results.append({
                    "text": entity_text, "type": span["type"],
                    "assertions": active_assertions,
                    "char_start": char_start, "char_end": char_end,
                    "score": span["score"],
                })

        merged = pp.merge_chunk_results(all_results)

        final_dicts = [
            {"text": r["text"], "type": r["type"], "assertions": r["assertions"],
             "position": [r["char_start"], r["char_end"]]}
            for r in merged
        ]

        if apply_repair_gate:
            final_dicts, _dropped = repair_gate.filter_entities(final_dicts)

        return [
            NerEntity(text=d["text"], type=d["type"], assertions=d["assertions"],
                      position=(d["position"][0], d["position"][1]))
            for d in final_dicts
        ]

    def predict_file(self, filepath: str | Path, **predict_kwargs) -> dict[int, SectionResult]:
        """Đọc 1 file .txt, tách section (EMR + QA), làm sạch + predict
        RIÊNG từng section — đúng kiến trúc bạn đang dùng."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        sections = sectioner.split_sections_by_header(raw_text)
        results: dict[int, SectionResult] = {}

        for sec_no in sorted(sections.keys()):
            title = sections[sec_no]["title"]
            body = sections[sec_no]["body"]

            if not body.strip():
                results[sec_no] = SectionResult(sec_no, title, [])
                continue

            cleaned = pp.clean_text_for_inference(body)
            entities = self.predict_text(cleaned, **predict_kwargs)
            results[sec_no] = SectionResult(sec_no, title, entities)

        return results