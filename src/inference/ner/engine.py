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
        assertion_threshold: float | dict[str, float] = cfg.ASSERTION_THRESHOLD,
        single_assertion: bool = cfg.SINGLE_ASSERTION,
        apply_repair_gate: bool = cfg.ENABLE_REPAIR_GATE,
    ) -> list[NerEntity]:
        """Predict trên 1 đoạn text ĐÃ ĐƯỢC LÀM SẠCH (không tự clean ở đây
        — gọi postprocessor.clean_text_for_inference() trước nếu cần, để
        tách bạch rõ input nào đã qua bước nào)."""
        model = self.model
        model.eval()
        tokens, offsets, line_ids = om.segment_with_offsets(text, self.rdr)
        if not tokens:
            return []
        chunks = om.make_word_chunks(
            tokens, self.tokenizer, max_len=max_len, overlap_words=overlap_words,
        )

        # Notebook V11: choose one BIO tag per global word from the chunk where
        # that word is furthest from an edge. Entity extraction happens only
        # once after the complete global BIO sequence has been reconciled.
        global_tag_choices: dict[int, tuple[int, str, float]] = {}
        chunk_cache: list[dict[str, Any]] = []

        for chunk_start, requested_chunk_end in chunks:
            chunk_tokens = tokens[chunk_start:requested_chunk_end]
            input_ids, attention_mask, word_starts, word_lengths = (
                om.encode_words_for_inference(
                    self.tokenizer, chunk_tokens, max_len=max_len,
                )
            )
            valid_word_count = len(word_starts)
            if valid_word_count == 0:
                continue
            effective_chunk_end = chunk_start + valid_word_count
            input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            attention_mask_t = torch.tensor(
                [attention_mask], dtype=torch.long, device=self.device,
            )
            encoder_hidden = model.encoder(
                input_ids=input_ids_t, attention_mask=attention_mask_t,
            ).last_hidden_state
            ner_hidden = model.ner_dropout(encoder_hidden)
            emissions = model.ner_head(ner_hidden)
            emission_probs = torch.softmax(emissions, dim=-1)
            pred_ids = model.crf.decode(emissions, mask=attention_mask_t.bool())[0]

            for local_index, subword_start in enumerate(word_starts):
                if subword_start >= len(pred_ids):
                    continue
                global_index = chunk_start + local_index
                pred_id = pred_ids[subword_start]
                tag = pp.get_label(self.id2nerlabel, pred_id)
                confidence = float(emission_probs[0, subword_start, pred_id].item())
                edge_distance = min(
                    local_index, valid_word_count - 1 - local_index,
                )
                previous = global_tag_choices.get(global_index)
                if previous is None or edge_distance > previous[0]:
                    global_tag_choices[global_index] = (
                        edge_distance, tag, confidence,
                    )

            # Keep the encoder result on CPU so assertion aggregation does not
            # retain all long-document chunks in VRAM.
            chunk_cache.append({
                "chunk_start": chunk_start,
                "chunk_end": effective_chunk_end,
                "word_starts": word_starts,
                "word_lengths": word_lengths,
                "attention_mask": attention_mask_t.detach().cpu(),
                "encoder_hidden": encoder_hidden.detach().cpu(),
            })

        if not chunk_cache:
            return []

        global_tags = [
            global_tag_choices.get(index, (-1, "O", 0.0))[1]
            for index in range(len(tokens))
        ]
        global_entities = pp.extract_entities_from_word_tags(
            pp.repair_bio_tags(global_tags), line_ids=line_ids,
        )
        global_entities = [
            entity for entity in global_entities
            if om.is_valid_char_span(
                offsets, entity["word_start"], entity["word_end"],
            )
        ]
        if not global_entities:
            return []

        # Aggregate assertion probabilities from every chunk containing the
        # complete entity, weighted by the same centrality rule as validation.
        assertion_choices: dict[tuple[int, int, str], dict[str, Any]] = {}
        for cached in chunk_cache:
            chunk_start, chunk_end = cached["chunk_start"], cached["chunk_end"]
            word_starts, word_lengths = cached["word_starts"], cached["word_lengths"]
            spans = []
            for entity in global_entities:
                global_start, global_end = entity["word_start"], entity["word_end"]
                if global_start < chunk_start or global_end > chunk_end:
                    continue
                local_start, local_end = global_start - chunk_start, global_end - chunk_start
                if local_start < 0 or local_end <= local_start or local_end > len(word_starts):
                    continue
                subword_start = word_starts[local_start]
                last_word = local_end - 1
                subword_end = min(
                    word_starts[last_word] + word_lengths[last_word], max_len - 1,
                )
                if subword_end <= subword_start:
                    continue
                centrality = min(global_start - chunk_start, chunk_end - global_end)
                spans.append({
                    "token_start": subword_start,
                    "token_end": subword_end,
                    "global_word_start": global_start,
                    "global_word_end": global_end,
                    "type": entity["type"],
                    "weight": float(max(1, centrality + 1)),
                })
            if not spans:
                continue

            encoder_hidden = cached["encoder_hidden"].to(self.device)
            attention_mask_t = cached["attention_mask"].to(self.device)
            assertion_hidden = model.assertion_dropout(encoder_hidden)
            span_vectors, owners = model._pool_spans(
                assertion_hidden, [spans], attention_mask_t,
            )
            if owners:
                probabilities = torch.sigmoid(
                    model.assertion_head(span_vectors)
                ).detach().cpu()
                for row, (_, span) in enumerate(owners):
                    key = (
                        span["global_word_start"], span["global_word_end"], span["type"],
                    )
                    weight = span["weight"]
                    weighted = probabilities[row] * weight
                    if key not in assertion_choices:
                        assertion_choices[key] = {
                            "weighted_prob": weighted.clone(), "weight": weight,
                        }
                    else:
                        assertion_choices[key]["weighted_prob"] += weighted
                        assertion_choices[key]["weight"] += weight

        final_dicts = []
        for entity in global_entities:
            global_start, global_end = entity["word_start"], entity["word_end"]
            entity_type = entity["type"]
            char_start, char_end = offsets[global_start][0], offsets[global_end - 1][1]
            key = (global_start, global_end, entity_type)
            active_assertions = []
            accumulator = assertion_choices.get(key)
            if accumulator is not None and accumulator["weight"] > 0:
                averaged = (
                    accumulator["weighted_prob"] / accumulator["weight"]
                ).tolist()
                for label_index, probability in enumerate(averaged):
                    label = pp.get_label(self.id2assertlabel, label_index)
                    if label != "NONE" and probability >= pp.get_assertion_threshold(
                        assertion_threshold, label,
                    ):
                        active_assertions.append(label)
            if single_assertion:
                active_assertions = pp.collapse_assertions(active_assertions)
            confidences = [
                global_tag_choices.get(index, (-1, "O", 0.0))[2]
                for index in range(global_start, global_end)
            ]
            final_dicts.append({
                "text": text[char_start:char_end],
                "type": entity_type,
                "assertions": active_assertions,
                "position": [char_start, char_end],
                "score": sum(confidences) / max(1, len(confidences)),
            })
        final_dicts.sort(key=lambda item: (
            item["position"][0], item["position"][1], item["type"],
        ))

        if apply_repair_gate:
            final_dicts, _dropped = repair_gate.filter_entities(final_dicts)

        return [
            NerEntity(text=d["text"], type=d["type"], assertions=d["assertions"],
                      position=(d["position"][0], d["position"][1]),
                      score=float(d.get("score", 1.0)), flag=d.get("flag"))
            for d in final_dicts
        ]

    def predict_file(self, filepath: str | Path, **predict_kwargs) -> dict[int, SectionResult]:
        """Đọc 1 file .txt, tách section (EMR + QA), làm sạch + predict
        RIÊNG từng section — đúng kiến trúc bạn đang dùng."""
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            raw_text = f.read()

        sections = sectioner.split_sections_by_header(raw_text)
        results: dict[int, SectionResult] = {}

        for block_id in sorted(sections.keys()):
            block = sections[block_id]
            section_no = block["section_no"]
            title = block["title"]
            body = block["body"]

            if not body.strip():
                results[block_id] = SectionResult(section_no, title, [])
                continue

            cleaned = pp.clean_text_for_inference(body)
            local_entities = self.predict_text(cleaned, **predict_kwargs)
            block_start = int(block["start"])
            entities = []
            for entity in local_entities:
                local_start, local_end = entity.position
                global_start, global_end = block_start + local_start, block_start + local_end
                if raw_text[global_start:global_end] != entity.text:
                    raise ValueError(
                        f"invalid global entity offset for {entity.text!r}: "
                        f"{[global_start, global_end]}"
                    )
                entities.append(NerEntity(
                    entity.text, entity.type, list(entity.assertions),
                    (global_start, global_end), entity.score, entity.flag,
                    list(entity.review_hints),
                ))
            results[block_id] = SectionResult(section_no, title, entities)

        return results
