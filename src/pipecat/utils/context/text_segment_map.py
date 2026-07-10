#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Segment-level mapping between original and TTS-transformed text."""

import difflib
import re
from dataclasses import dataclass

from pipecat.utils.text.transforms._alnum_utils import advance_by_alnums, normalize


def strip_markup(text: str) -> str:
    """Remove XML/SSML-like markup from text without depending on tag names.

    This is intentionally syntax-based, not tag-name based. It treats anything
    between '<' and '>' as markup and preserves text outside markup.
    """
    result = []
    in_tag = False

    for ch in text:
        if in_tag:
            if ch == ">":
                in_tag = False
            continue

        if ch == "<":
            in_tag = True
            continue

        result.append(ch)

    return "".join(result)


def _raw_len_for_clean_chars(text: str, n: int) -> int:
    """Return the raw offset into *text* after producing *n* markup-stripped chars.

    Stateless counterpart to :func:`strip_markup`: walks *text* the same way
    (tracking ``<...>`` spans) but stops as soon as *n* non-markup characters
    have been produced, instead of stripping the whole string. Used to convert
    a match found in markup-stripped space back into a raw-text offset.
    """
    if n <= 0:
        return 0

    produced = 0
    in_tag = False
    for pos, ch in enumerate(text):
        if in_tag:
            if ch == ">":
                in_tag = False
            continue
        if ch == "<":
            in_tag = True
            continue
        produced += 1
        if produced == n:
            return pos + 1

    return len(text)


@dataclass(frozen=True)
class TextSegment:
    """Immutable aligned chunk between original and TTS text.

    Parameters:
        original: Chunk of the user-facing / LLM text.
        tts: Corresponding chunk in the TTS-transformed text.
        original_start: Byte offset in original_text where this chunk begins.
        original_end: Byte offset in original_text where this chunk ends.
    """

    original: str
    tts: str
    original_start: int
    original_end: int

    @property
    def is_transformed(self) -> bool:
        """True when this segment cannot be tracked by proportional char advancement.

        This holds when:

        - alphanumeric content differs between original and TTS sides;
        - a replacement changed word count / tokenization;
        - the TTS side contains markup, even if the spoken alphanumeric content is
          the same as the original.

        The markup check is syntax-based and tag-name independent. For example,
        ``<phoneme ...>Siobhan</phoneme>`` is transformed because the TTS segment
        has raw markup around the original word, so the raw segment cursor can move
        while the original/LLM cursors must remain held.
        """
        if self.tts != strip_markup(self.tts):
            return True
        if normalize(self.original) != normalize(self.tts):
            return True
        return len(self.original.split()) != len(self.tts.split())

    @property
    def tts_alnum_count(self) -> int:
        """Number of alphanumeric characters in the spoken TTS content."""
        return len(normalize(self.tts))

    @property
    def original_alnum_count(self) -> int:
        """Number of alphanumeric characters in the original side of this segment."""
        return len(normalize(self.original))


class TextSegmentMap:
    """Maps cursor positions between transformed TTS text and original text.

    Tracks a single raw-text cursor (``_seg_raw_pos``) into the current
    segment's ``tts`` text. Each incoming word-timestamp token is matched
    against the segment's remaining raw text -- literally, or (as a stateless
    fallback) with markup stripped from both sides -- so the same mechanism
    drives segment completion and cursor advancement without needing to parse
    tag structure out of the token stream.

    For unchanged segments, ``user_facing_pos``/``llm_pos`` advance
    proportionally to the alphanumeric content of each consumed raw span. For
    transformed segments (e.g. a phoneme-wrapped word, or ``"$42.50"`` ->
    ``"forty two dollars and fifty cents"``), those cursors are held until the
    segment's entire raw text has been matched, then jump to the end of the
    corresponding original span in one step.

    Callers drive the map word-by-word: :meth:`word_belongs_current_segment`
    asks whether a raw word-timestamp token plausibly continues the remaining
    TTS text, and :meth:`advance_word` consumes it.
    """

    def __init__(
        self,
        tts_text: str,
        original_text: str,
        llm_text: str | None = None,
    ):
        """Initialize the segment map.

        Args:
            tts_text: Post-transform text sent to TTS.
            original_text: User-facing pre-transform text.
            llm_text: LLM-produced text, which may have surrounding tags. Defaults
                to ``original_text`` when not provided.
        """
        self._tts_text = tts_text
        self._original_text = original_text
        self._llm_text = llm_text if llm_text is not None else original_text
        self._segments: list[TextSegment] = self._build(tts_text, original_text)
        self._reset_state()

    @staticmethod
    def _build(tts_text: str, original_text: str) -> list[TextSegment]:
        """Build aligned TextSegments from a word-level SequenceMatcher diff.

        Each diff opcode (equal, replace, insert, delete) becomes a segment.
        Segments whose normalized alphanumeric content differs are later treated
        as transformed/atomic units during cursor advancement.
        """

        def tokenize(text: str) -> list[str]:
            return re.split(r"(\s+)", text)

        orig_tokens = tokenize(original_text)
        tts_tokens = tokenize(tts_text)

        # SequenceMatcher produces a word-level alignment between the original
        # and TTS texts. Each opcode becomes a TextSegment whose boundaries are
        # tracked in the original text.
        #
        # Example:
        #
        #     original_text = "Your balance is $42.50"
        #     tts_text      = "Your balance is forty two dollars and fifty cents"
        #
        # Tokenization preserves whitespace, so SequenceMatcher sees:
        #
        #     equal:
        #         "Your balance is "
        #
        #     replace:
        #         "$42.50"
        #         ->
        #         "forty two dollars and fifty cents"
        #
        # This produces two segments:
        #
        #     TextSegment(
        #         original="Your balance is ",
        #         tts="Your balance is ",
        #         original_start=0,
        #         original_end=16,
        #     )
        #
        #     TextSegment(
        #         original="$42.50",
        #         tts="forty two dollars and fifty cents",
        #         original_start=16,
        #         original_end=22,
        #     )
        #
        # During playback, unchanged segments advance cursors
        # proportionally. Transformed segments are treated as atomic:
        # the cursors are held while the expanded TTS text is being
        # consumed and jump to original_end only when the entire
        # transformed segment completes.
        matcher = difflib.SequenceMatcher(None, orig_tokens, tts_tokens, autojunk=False)

        segments: list[TextSegment] = []
        orig_pos = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            orig_chunk = "".join(orig_tokens[i1:i2])
            tts_chunk = "".join(tts_tokens[j1:j2])
            orig_end = orig_pos + len(orig_chunk)
            segments.append(
                TextSegment(
                    original=orig_chunk,
                    tts=tts_chunk,
                    original_start=orig_pos,
                    original_end=orig_end,
                )
            )
            orig_pos = orig_end

        return segments

    def _reset_state(self) -> None:
        self._seg_idx: int = 0
        self._seg_raw_pos: int = 0
        self._user_facing_pos: int = 0
        self._llm_pos: int = 0
        self._last_completed: TextSegment | None = None
        self._last_overflow: str | None = None
        self._touched_current_segment: bool = False

    @staticmethod
    def _classify_hop(
        segment_remaining: str, remaining_word: str, seg: TextSegment
    ) -> tuple[str, int, int]:
        """Classify how *remaining_word* relates to *segment_remaining*.

        Purely positional/textual -- no tag-name parsing or cross-call state.
        Returns a ``(kind, skip, value)`` tuple:

        - ``("found", skip, raw_len)``: the whole word is matched inside this
          segment, after skipping *skip* leading raw chars; the match spans
          *raw_len* raw chars from there. Tried first against the segment's
          remaining text as-is (*skip* 0 -- e.g. a TTS provider whose word
          tokens carry their own leading/trailing whitespace, like Inworld's
          ``" world"``), then, if that fails, against it with leading
          whitespace stripped (*skip* > 0 -- the more common case where the
          word omits the separating space); and finally by comparing both
          sides with markup stripped too (needed when a TTS provider wraps
          the word-timestamp token in tags that never appear in ``tts_text``,
          or vice versa; recomputed fresh each call, no persisted tag state).
        - ``("consume", 0, trim)``: the segment's remaining raw text (as-is,
          or with leading whitespace stripped) is a prefix of the word --
          drain the segment and trim *trim* chars off the front of the word
          before continuing into the next segment.
        - ``("consume", 0, 0)``: nothing above matched, and nothing more can be
          spoken in this segment anyway -- its *remaining* raw text has zero
          alphanumeric content, whether because the segment itself carries
          none at all (e.g. a self-closing ``<break/>`` tag) or because only
          trailing whitespace/punctuation is left after everything
          alphanumeric has already been consumed. Should be drained so the
          word gets a chance to match the next segment instead. Checked only
          after the match attempts above, so a word that *does* literally
          match trailing zero-alnum content (e.g. an emoji) is still found
          there rather than skipped over.
        - ``("fallback", 0, 0)``: nothing above matched (e.g. a TTS provider
          symbol substitution).
        """
        if segment_remaining.startswith(remaining_word):
            return "found", 0, len(remaining_word)
        if remaining_word.startswith(segment_remaining):
            return "consume", 0, len(segment_remaining)

        stripped = segment_remaining.lstrip()
        skip = len(segment_remaining) - len(stripped)

        if skip:
            if stripped.startswith(remaining_word):
                return "found", skip, len(remaining_word)
            if stripped and remaining_word.startswith(stripped):
                return "consume", 0, len(stripped)

        clean_word = strip_markup(remaining_word)
        if clean_word and strip_markup(stripped).startswith(clean_word):
            return "found", skip, _raw_len_for_clean_chars(stripped, len(clean_word))

        if not normalize(segment_remaining):
            return "consume", 0, 0

        return "fallback", 0, 0

    def _commit_raw_span(self, seg: TextSegment, new_pos: int) -> None:
        """Apply raw progress on *seg* up to *new_pos*, advancing cursors.

        For an unchanged segment, ``user_facing_pos``/``llm_pos`` advance
        proportionally to the alphanumeric content of the newly-consumed span.
        Once *new_pos* reaches the end of the segment's raw text, the segment
        completes: a transformed segment's cursors jump to the end of its
        original span; an unchanged segment's cursors are already there from
        the proportional advance above (never snapped to ``original_end``, to
        avoid overshooting when a segment ends in trailing whitespace).
        """
        consumed_span = seg.tts[self._seg_raw_pos : new_pos]
        self._seg_raw_pos = new_pos

        if not seg.is_transformed:
            n_alnum = len(normalize(consumed_span))
            self._user_facing_pos = advance_by_alnums(
                self._original_text, self._user_facing_pos, n_alnum
            )
            self._llm_pos = advance_by_alnums(self._llm_text, self._llm_pos, n_alnum)
        elif not normalize(seg.tts[new_pos:]):
            # Transformed segment: a trailing markup-only remainder (e.g. a
            # closing tag) will never arrive as its own word -- TTS providers
            # don't emit a separate word-timestamp event for it. Fold it into
            # this call so the segment still completes. (Unchanged segments
            # don't get this treatment: a trailing symbol/emoji there is a real
            # output position and IS expected to arrive as its own word.)
            new_pos = len(seg.tts)
            self._seg_raw_pos = new_pos

        if new_pos >= len(seg.tts):
            if seg.is_transformed:
                self._user_facing_pos = seg.original_end
                self._llm_pos = advance_by_alnums(
                    self._llm_text, self._llm_pos, seg.original_alnum_count
                )
            self._last_completed = seg
            self._seg_idx += 1
            self._seg_raw_pos = 0

    def _advance_raw(self, word: str) -> None:
        """Match *word* against the remaining raw TTS text, advancing cursors.

        Hops across segments as needed for a word that straddles a segment
        boundary. If the word runs past the end of ``tts_text`` (no segments
        left to carry the remainder into), the unconsumed raw suffix is stored
        in ``last_overflow``.
        """
        remaining_word = word

        while remaining_word and self._seg_idx < len(self._segments):
            seg = self._segments[self._seg_idx]
            old_pos = self._seg_raw_pos
            segment_remaining = seg.tts[old_pos:]
            kind, skip, value = self._classify_hop(segment_remaining, remaining_word, seg)

            if kind == "found":
                self._commit_raw_span(seg, old_pos + skip + value)
                return

            if kind == "consume":
                self._commit_raw_span(seg, len(seg.tts))
                if value:
                    remaining_word = remaining_word[value:]
                continue

            # Fallback: nudge past this segment's leading run of non-alnum raw
            # chars only -- never past real (alnum) content -- so a provider
            # symbol substitution (e.g. "->" reported as "-") is absorbed
            # without risking eating an unspoken word.
            skip_len = 0
            while skip_len < len(segment_remaining) and not segment_remaining[skip_len].isalnum():
                skip_len += 1
            self._seg_raw_pos = old_pos + skip_len
            return

        if remaining_word:
            self._last_overflow = remaining_word

    def advance_word(self, word: str) -> None:
        """Match *word* against the remaining TTS text and advance cursors.

        Args:
            word: Raw TTS word-timestamp token. May be a fragment of a tag, a
                spoken word, or a mix -- the matching is purely textual, no
                tag parsing is required from callers.
        """
        self._last_completed = None
        self._last_overflow = None
        seg_idx_before = self._seg_idx

        if word:
            self._advance_raw(word)

        self._touched_current_segment = self._seg_idx == seg_idx_before

    def word_belongs_current_segment(self, word: str) -> bool:
        """Return True if *word* plausibly continues the remaining TTS text.

        A non-mutating dry run of the same matching :meth:`advance_word` uses.
        Used to detect when a TTS provider silently dropped a word-timestamp
        event: if the incoming word does not match, the caller should
        force-complete this slot and route the word to the next.
        """
        if not word:
            return True
        if self._word_matches_remaining(word):
            return True
        if not normalize(word):
            return self._symbol_word_belongs(word)
        return False

    def _word_matches_remaining(self, word: str) -> bool:
        """Dry run of :meth:`_advance_raw`'s matching loop; does not mutate state.

        Returns True once a "found" hop occurs (word fully matches, whether
        entirely within the current segment or a legitimate straddle across
        further segments that get fully drained), or once such a straddle
        exhausts every remaining segment. Returns False only when the map was
        already exhausted before this call, or a hop can't be classified as
        anything but a fallback (no recognizable match at all).
        """
        if self._seg_idx >= len(self._segments):
            return False

        seg_idx = self._seg_idx
        raw_pos = self._seg_raw_pos
        remaining_word = word

        while remaining_word and seg_idx < len(self._segments):
            seg = self._segments[seg_idx]
            segment_remaining = seg.tts[raw_pos:]
            kind, _skip, value = self._classify_hop(segment_remaining, remaining_word, seg)

            if kind == "found":
                return True
            if kind == "consume":
                if value:
                    remaining_word = remaining_word[value:]
                seg_idx += 1
                raw_pos = 0
                continue
            return False

        return True

    def _symbol_word_belongs(self, word: str) -> bool:
        """Return True if a non-alnum word (emoji, punctuation, symbol) belongs here.

        Two checks are applied in order:

        1. **Literal substring**: search for the raw word in the remaining TTS
           text. The search window is backed up over any already-consumed
           trailing punctuation, since that may have been swept past already.

        2. **Symbol substitution fallback**: some TTS providers substitute
           Unicode symbols with ASCII punctuation in word-timestamp events (e.g.
           ElevenLabs reports "->" as "-"), so check 1 always fails even though
           the word belongs here. If alnum content still remains unconsumed and
           the next non-space character in the TTS text is itself a non-alnum
           symbol, accept the word as a substitution.
        """
        pos = self.raw_pos
        search_start = pos
        while search_start > 0:
            ch = self._tts_text[search_start - 1]
            if ch.isalnum() or ch.isspace() or ch == ">":
                break
            search_start -= 1
        if word in self._tts_text[search_start:]:
            return True

        if self._seg_idx >= len(self._segments):
            return False

        while pos < len(self._tts_text) and self._tts_text[pos].isspace():
            pos += 1
        return pos < len(self._tts_text) and not self._tts_text[pos].isalnum()

    @property
    def user_facing_pos(self) -> int:
        """Current byte offset in the original user-facing text."""
        return self._user_facing_pos

    @property
    def llm_pos(self) -> int:
        """Current byte offset in the LLM text."""
        return self._llm_pos

    @property
    def raw_pos(self) -> int:
        """Current global byte offset into ``tts_text``."""
        pos = sum(len(s.tts) for s in self._segments[: self._seg_idx])
        if self._seg_idx < len(self._segments):
            pos += self._seg_raw_pos
        return pos

    @property
    def last_overflow(self) -> str | None:
        """Raw suffix of the last :meth:`advance_word` call that overflowed.

        ``None`` unless that call's word ran past the end of ``tts_text`` (no
        segments left to carry the remainder into). Always a suffix of the
        word passed to that call -- the consumed prefix is
        ``word[: len(word) - len(last_overflow)]``.
        """
        return self._last_overflow

    @property
    def is_complete(self) -> bool:
        """True once every segment's alphanumeric content has been accounted for.

        Not simply "cursor past the last segment": a frame whose remaining
        content is entirely punctuation/markup (zero alphanumeric chars) is
        already complete even if its raw text hasn't been walked yet.
        """
        if self._seg_idx >= len(self._segments):
            return True
        seg = self._segments[self._seg_idx]
        if normalize(seg.tts[self._seg_raw_pos :]):
            return False
        return all(not normalize(s.tts) for s in self._segments[self._seg_idx + 1 :])

    @property
    def in_transformed_segment(self) -> bool:
        """True when the cursor is on a transformed segment that is not complete."""
        if self._seg_idx >= len(self._segments):
            return False

        seg = self._segments[self._seg_idx]
        return seg.is_transformed and (self._seg_raw_pos > 0 or self._touched_current_segment)

    @property
    def last_completed_segment(self) -> TextSegment | None:
        """The segment completed by the last :meth:`advance_word` call."""
        return self._last_completed

    def reset(self) -> None:
        """Reset all cursor and consumption state."""
        self._reset_state()
