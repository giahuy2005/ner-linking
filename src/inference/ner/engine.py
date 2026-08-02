"""NerEngine: load model 1 lần, expose predict_text() / predict_file().

Class NerAssertionModel giữ nguyên kiến trúc từ train_ner_colab_crf —
KHÔNG được sửa layer nào ở đây mà không sửa đồng bộ bên train, nếu không
load_state_dict sẽ báo lỗi shape mismatch (hoặc tệ hơn: load "thành công"
nhưng assertion head học sai vùng pooling).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF  # pip install pytorch-crf
from transformers import AutoModel, AutoTokenizer

from .. import config as cfg
from . import offset_mapper as om
from . import postprocessor as pp
from . import repair_gate
from . import sectioner
from ..schemas import NerEntity, SectionResult
from .evidence import CrfMarginalEvidence, NerDetailedResult, SpanCandidateEvidence, WordEvidence


def crf_token_marginals(crf, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return exact token marginals for a linear-chain CRF."""
    mask = mask.bool()
    batch_size, sequence_length, num_tags = emissions.shape
    marginals = emissions.new_zeros(batch_size, sequence_length, num_tags)
    for batch_index in range(batch_size):
        length = int(mask[batch_index].long().sum().item())
        if length <= 0:
            continue
        current = emissions[batch_index, :length]
        alpha = emissions.new_empty(length, num_tags)
        alpha[0] = crf.start_transitions + current[0]
        for token_index in range(1, length):
            alpha[token_index] = torch.logsumexp(
                alpha[token_index - 1].unsqueeze(1)
                + crf.transitions
                + current[token_index].unsqueeze(0),
                dim=0,
            )
        log_partition = torch.logsumexp(alpha[-1] + crf.end_transitions, dim=0)
        beta = emissions.new_empty(length, num_tags)
        beta[-1] = crf.end_transitions
        for token_index in range(length - 2, -1, -1):
            beta[token_index] = torch.logsumexp(
                crf.transitions
                + current[token_index + 1].unsqueeze(0)
                + beta[token_index + 1].unsqueeze(0),
                dim=1,
            )
        marginals[batch_index, :length] = torch.softmax(alpha + beta - log_partition, dim=-1)
    return marginals


def _split_bio_tag(tag: str) -> tuple[str, str | None]:
    """Split a BIO tag without changing unknown tags.

    Returns ``(prefix, entity_type)``. ``O`` becomes ``("O", None)``.
    Unknown/malformed tags get prefix ``"INVALID"`` so they lose BIO
    tie-breaks but are still handled later by ``repair_bio_tags``.
    """
    if tag == "O":
        return "O", None
    if isinstance(tag, str) and len(tag) > 2 and tag[1] == "-" \
            and tag[0] in {"B", "I"}:
        return tag[0], tag[2:]
    return "INVALID", None


def _bio_transition_quality(previous_tag: str, candidate_tag: str) -> int:
    """Rank BIO compatibility for a candidate at the current word.

    This score is used only after edge distance and model confidence are tied.
    It therefore cannot make a low-confidence edge prediction beat a stronger
    central prediction; it only prevents an arbitrary ``I-X`` from winning an
    otherwise exact tie after ``O`` or another entity type.
    """
    prefix, entity_type = _split_bio_tag(candidate_tag)
    if prefix in {"O", "B"}:
        return 2
    if prefix != "I" or entity_type is None:
        return 0

    previous_prefix, previous_type = _split_bio_tag(previous_tag)
    if previous_prefix in {"B", "I"} and previous_type == entity_type:
        return 2
    return 0


def _should_replace_global_tag_choice(
    previous_choice: tuple[int, str, float] | None,
    candidate_choice: tuple[int, str, float],
    *,
    previous_global_tag: str = "O",
    confidence_epsilon: float = 1e-12,
) -> bool:
    """Choose between overlapping-chunk predictions deterministically.

    Priority is intentionally strict:

    1. prediction farther from the chunk edge;
    2. higher emission confidence for the CRF-decoded label;
    3. BIO-compatible transition from the already reconciled previous word;
    4. exact ties keep the existing choice, making output deterministic.

    Tuple layout remains ``(edge_distance, tag, confidence)`` so the rest of
    ``predict_text`` and downstream score aggregation stay API-compatible.
    """
    if previous_choice is None:
        return True

    previous_edge, previous_tag, previous_confidence = previous_choice
    candidate_edge, candidate_tag, candidate_confidence = candidate_choice

    if candidate_edge != previous_edge:
        return candidate_edge > previous_edge

    confidence_delta = candidate_confidence - previous_confidence
    if abs(confidence_delta) > confidence_epsilon:
        return confidence_delta > 0

    previous_bio_quality = _bio_transition_quality(
        previous_global_tag, previous_tag,
    )
    candidate_bio_quality = _bio_transition_quality(
        previous_global_tag, candidate_tag,
    )
    if candidate_bio_quality != previous_bio_quality:
        return candidate_bio_quality > previous_bio_quality

    return False


class NerAssertionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_ner_tags: int,
        num_span_labels: int | None = None,
        num_assert_labels: int = 3,
        dropout: float = 0.1,
        assertion_dropout: float = 0.3,
        span_dropout: float = cfg.SPAN_DROPOUT,
        span_width_embedding_dim: int = cfg.SPAN_WIDTH_EMBEDDING_DIM,
        max_span_width_words: int = cfg.SPAN_MAX_WIDTH_WORDS,
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

        self.num_span_labels = int(num_span_labels or 0)
        self.max_span_width_words = int(max_span_width_words)
        if self.num_span_labels:
            self.span_dropout = nn.Dropout(span_dropout)
            self.span_width_embedding = nn.Embedding(
                self.max_span_width_words + 1, span_width_embedding_dim,
            )
            span_feature_size = hidden_size * 3 + span_width_embedding_dim
            self.span_head = nn.Sequential(
                nn.LayerNorm(span_feature_size),
                nn.Linear(span_feature_size, hidden_size),
                nn.GELU(),
                nn.Dropout(span_dropout),
                nn.Linear(hidden_size, self.num_span_labels),
            )
        else:
            self.span_dropout = None
            self.span_width_embedding = None
            self.span_head = None

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

    def _pool_candidate_spans(self, hidden_states: torch.Tensor, span_candidates_batch: list):
        if self.span_width_embedding is None:
            return hidden_states.new_zeros((0, 0)), []
        batch_indices, starts, ends, widths, owners = [], [], [], [], []
        for batch_index, spans in enumerate(span_candidates_batch):
            for span in spans:
                start, end = int(span["token_start"]), int(span["token_end"])
                if end <= start:
                    continue
                batch_indices.append(batch_index)
                starts.append(start)
                ends.append(end)
                widths.append(min(int(span.get("width_words", end - start)), self.max_span_width_words))
                owners.append((batch_index, span))
        if not owners:
            size = hidden_states.size(-1) * 3 + self.span_width_embedding.embedding_dim
            return hidden_states.new_zeros((0, size)), owners
        device = hidden_states.device
        batches = torch.tensor(batch_indices, dtype=torch.long, device=device)
        starts_t = torch.tensor(starts, dtype=torch.long, device=device)
        ends_t = torch.tensor(ends, dtype=torch.long, device=device)
        widths_t = torch.tensor(widths, dtype=torch.long, device=device)
        prefix = F.pad(hidden_states.cumsum(dim=1), (0, 0, 1, 0))
        mean_repr = (prefix[batches, ends_t] - prefix[batches, starts_t]) / (
            ends_t - starts_t
        ).clamp_min(1).unsqueeze(-1)
        return torch.cat([
            hidden_states[batches, starts_t],
            hidden_states[batches, ends_t - 1],
            mean_repr,
            self.span_width_embedding(widths_t),
        ], dim=-1), owners

    def forward(self, input_ids, attention_mask, assertion_spans_batch, ner_labels=None,
                span_candidates_batch=None, decode_ner: bool = True):
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
            "ner_pred_tags": self.crf.decode(ner_emissions, mask=mask) if decode_ner else [],
        }
        if self.span_head is not None:
            span_candidates_batch = span_candidates_batch or [[] for _ in range(input_ids.size(0))]
            features, owners = self._pool_candidate_spans(
                self.span_dropout(encoder_hidden), span_candidates_batch,
            )
            result["span_logits"] = self.span_head(features)
            result["span_candidate_owner"] = owners
        else:
            result["span_logits"] = encoder_hidden.new_zeros((0, 0))
            result["span_candidate_owner"] = []
        return result


class NerEngine:
    """Wrap model + tokenizer + VnCoreNLP, expose predict API cấp cao."""

    def __init__(self, model: NerAssertionModel, tokenizer, rdr, id2nerlabel: dict,
                 id2assertlabel: dict, device: torch.device, id2spanlabel: dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.rdr = rdr
        self.id2nerlabel = id2nerlabel
        self.id2assertlabel = id2assertlabel
        self.id2spanlabel = id2spanlabel or {}
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
        spanlabel2id = label_dicts.get("spanlabel2id") or label_dicts.get("span_label2id")
        id2nerlabel = {v: k for k, v in nerlabel2id.items()}
        id2assertlabel = {v: k for k, v in assertlabel2id.items()}
        id2spanlabel = {v: k for k, v in (spanlabel2id or {}).items()}

        tokenizer = AutoTokenizer.from_pretrained(backbone)

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        has_span_weights = any(
            key.startswith("span_head.") or key.startswith("span_width_embedding.")
            for key in state_dict
        )
        if has_span_weights and spanlabel2id:
            num_span_labels = len(spanlabel2id)
        elif has_span_weights:
            output_keys = [
                key for key in state_dict
                if key.startswith("span_head.") and key.endswith(".weight")
            ]
            if not output_keys:
                raise ValueError("span checkpoint has no span_head output weight")
            output_key = sorted(output_keys, key=lambda key: int(key.split(".")[1]))[-1]
            num_span_labels = int(state_dict[output_key].shape[0])
        else:
            num_span_labels = 0

        model = NerAssertionModel(
            model_name=backbone,
            num_ner_tags=len(nerlabel2id),
            num_span_labels=num_span_labels,
            num_assert_labels=len(assertlabel2id),
            context_window=context_window,
        )
        model.load_state_dict(state_dict, strict=True)
        if not has_span_weights:
            warnings.warn(
                "CRF-only checkpoint: span-head inference is disabled",
                RuntimeWarning,
            )
        model.to(resolved_device)
        model.eval()

        rdr = VnCoreNLP(str(vncorenlp_jar), annotators="wseg", max_heap_size="-Xmx2g")

        return cls(
            model, tokenizer, rdr, id2nerlabel, id2assertlabel,
            resolved_device, id2spanlabel,
        )

    def move_to(self, device: str | torch.device) -> None:
        """Move model and update tensor-creation device atomically."""
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            resolved = torch.device("cpu")
        self.model.to(resolved)
        self.device = resolved
        self.model.eval()

    def offload_to_cpu(self) -> None:
        self.move_to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                previous_global_choice = global_tag_choices.get(global_index - 1)
                previous_global_tag = (
                    previous_global_choice[1]
                    if previous_global_choice is not None
                    else "O"
                )
                candidate_choice = (edge_distance, tag, confidence)
                if _should_replace_global_tag_choice(
                    previous,
                    candidate_choice,
                    previous_global_tag=previous_global_tag,
                ):
                    global_tag_choices[global_index] = candidate_choice

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

    @torch.no_grad()
    def predict_text_detailed(
        self,
        text: str,
        *,
        max_len: int = cfg.MAX_LEN,
        overlap_words: int = cfg.OVERLAP_WORDS,
        span_add_threshold: float = cfg.SPAN_ADD_THRESHOLD,
        span_audit_threshold: float = cfg.SPAN_AUDIT_THRESHOLD,
        **predict_kwargs,
    ) -> NerDetailedResult:
        """Return CRF output plus exact marginals and optional span-head evidence.

        Span candidates below the final lattice threshold are retained down to
        ``span_audit_threshold`` for proposal-oracle audit.  A CRF-only
        checkpoint produces the same final entities and an explicit disabled log.
        """
        self.model.eval()
        crf_entities = self.predict_text(
            text, max_len=max_len, overlap_words=overlap_words, **predict_kwargs,
        )
        tokens, offsets, line_ids = om.segment_with_offsets(text, self.rdr)
        result = NerDetailedResult(
            raw_text_length=len(text),
            clean_text_length=len(text),
            crf_entities=list(crf_entities),
            lattice_entities=list(crf_entities),
            final_entities=list(crf_entities),
            thresholds={
                "span_add": float(span_add_threshold),
                "span_audit": float(span_audit_threshold),
                "span_repair": float(cfg.SPAN_REPAIR_THRESHOLD),
                "span_retype": float(cfg.SPAN_RETYPE_THRESHOLD),
                "o_token_entity_mass": float(cfg.O_TOKEN_ENTITY_MASS_THRESHOLD),
                "local_verification": float(cfg.LOCAL_VERIFICATION_THRESHOLD),
            },
            span_head_enabled=self.model.span_head is not None,
        )
        if not tokens:
            return result

        global_choices: dict[int, dict[str, Any]] = {}
        span_by_key: dict[tuple[int, int, str], SpanCandidateEvidence] = {}
        for chunk_start, requested_end in om.make_word_chunks(
            tokens, self.tokenizer, max_len=max_len, overlap_words=overlap_words,
        ):
            chunk_tokens = tokens[chunk_start:requested_end]
            input_ids, attention_mask, word_starts, word_lengths = om.encode_words_for_inference(
                self.tokenizer, chunk_tokens, max_len=max_len,
            )
            valid_count = len(word_starts)
            if not valid_count:
                continue
            ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
            hidden = self.model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            emissions = self.model.ner_head(self.model.ner_dropout(hidden))
            decoded = self.model.crf.decode(emissions, mask=mask.bool())[0]
            marginals = crf_token_marginals(self.model.crf, emissions, mask.bool())[0]
            for local_index, subword_start in enumerate(word_starts):
                if subword_start >= len(decoded):
                    continue
                global_index = chunk_start + local_index
                edge = min(local_index, valid_count - 1 - local_index)
                probabilities = marginals[subword_start].detach().cpu().tolist()
                choice = {
                    "edge": edge,
                    "decoded": pp.get_label(self.id2nerlabel, decoded[subword_start]),
                    "probabilities": probabilities,
                }
                previous = global_choices.get(global_index)
                if previous is None or edge > previous["edge"]:
                    global_choices[global_index] = choice

            if self.model.span_head is None:
                continue
            candidates = []
            for local_start in range(valid_count):
                for local_end in range(local_start + 1, min(valid_count, local_start + self.model.max_span_width_words) + 1):
                    if line_ids[chunk_start + local_start] != line_ids[chunk_start + local_end - 1]:
                        break
                    last = local_end - 1
                    candidates.append({
                        "token_start": word_starts[local_start],
                        "token_end": word_starts[last] + word_lengths[last],
                        "global_word_start": chunk_start + local_start,
                        "global_word_end": chunk_start + local_end,
                        "width_words": local_end - local_start,
                    })
            features, owners = self.model._pool_candidate_spans(
                self.model.span_dropout(hidden), [candidates],
            )
            if not owners:
                continue
            probabilities = torch.softmax(self.model.span_head(features), dim=-1)
            for row, (_, owner) in enumerate(owners):
                score, label_id = probabilities[row].max(dim=-1)
                score_value = float(score.item())
                label = self.id2spanlabel.get(int(label_id.item()))
                if label in {None, "NONE", "O"} or score_value < span_audit_threshold:
                    continue
                word_start, word_end = owner["global_word_start"], owner["global_word_end"]
                char_start, char_end = offsets[word_start][0], offsets[word_end - 1][1]
                if not (0 <= char_start < char_end <= len(text)):
                    continue
                evidence = SpanCandidateEvidence(
                    char_start, char_end, label, score_value, word_start, word_end,
                )
                key = (char_start, char_end, label)
                if key not in span_by_key or score_value > span_by_key[key].score:
                    span_by_key[key] = evidence

        result.span_candidates = sorted(
            span_by_key.values(), key=lambda item: (item.start, item.end, item.type),
        )
        repaired_tags = pp.repair_bio_tags([
            global_choices.get(index, {}).get("decoded", "O") for index in range(len(tokens))
        ])
        id_to_label = {int(key): value for key, value in self.id2nerlabel.items()}
        o_id = next((key for key, value in id_to_label.items() if value == "O"), None)
        for index, token in enumerate(tokens):
            choice = global_choices.get(index)
            probabilities = choice["probabilities"] if choice else []
            prob_map = {id_to_label[i]: float(value) for i, value in enumerate(probabilities) if i in id_to_label}
            non_o = [(label, probability) for label, probability in prob_map.items() if label != "O"]
            top_label, top_probability = max(non_o, key=lambda item: item[1], default=(None, 0.0))
            o_probability = probabilities[o_id] if o_id is not None and o_id < len(probabilities) else 1.0
            result.words.append(WordEvidence(
                index=index,
                text=token,
                start=offsets[index][0],
                end=offsets[index][1],
                line_id=line_ids[index],
                crf=CrfMarginalEvidence(
                    decoded_tag=choice["decoded"] if choice else "O",
                    repaired_tag=repaired_tags[index],
                    probabilities=prob_map,
                    entity_mass=max(0.0, min(1.0, 1.0 - float(o_probability))),
                    top_non_o_label=top_label,
                    top_non_o_probability=float(top_probability),
                ),
            ))

        # Conservative lattice rescue: only non-overlapping, high-confidence
        # span-only candidates are added. Boundary/retype conflicts remain audit
        # evidence for the locked editor instead of silently mutating offsets.
        lattice = list(crf_entities)
        for span in sorted(result.span_candidates, key=lambda item: -item.score):
            if span.score < span_add_threshold:
                continue
            if any(span.start < entity.position[1] and span.end > entity.position[0] for entity in lattice):
                continue
            lattice.append(NerEntity(text[span.start:span.end], span.type, [], (span.start, span.end), span.score))
        lattice.sort(key=lambda item: (*item.position, item.type))
        result.lattice_entities = lattice
        result.final_entities = lattice
        if self.model.span_head is None:
            result.logs.append({"level": "warning", "event": "span_head_disabled", "reason": "checkpoint_is_crf_only"})
        result.validate_offsets(text)
        return result

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
                ))
            results[block_id] = SectionResult(section_no, title, entities)

        return results
