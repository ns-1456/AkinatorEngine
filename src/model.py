"""
GPU Bayesian inference engine for Akinator.
Expert-system style: resolve categories on Yes (never ask again), cap questions per
category, rotate categories. Session state (current_probs, asked set, resolved_categories,
asked_count_per_category) is managed by the server.
"""
import os
import json
import torch

DATA_DIR = "data"
# Max questions per category per game (stops country/gender/occupation spam)
MAX_QUESTIONS_PER_CATEGORY = 4
# Answer value >= this means "Yes" -> resolve that category (never ask from it again)
YES_THRESHOLD = 0.7
# Don't ask if P(Yes) is outside this range (question would be uninformative)
MIN_P_YES = 0.03
MAX_P_YES = 0.97
# Mistake-tolerant: soft likelihood so one wrong answer does not zero out the true character
SOFT_MIN_LIKELIHOOD = 0.1
SOFT_MATCH_LIKELIHOOD = 0.9
# Prefer questions that split the current top candidates ("untie the top guesses")
TOP_K_CANDIDATES = 20
TOP_K_DISCRIMINATION_BETA = 0.3
# When we've narrowed to this many effective candidates, skip P(Yes) mask so we still have questions (e.g. occupation)
NARROWED_EFFECTIVE_N = 25
# Min probability to count as "effective" candidate for narrowed check
EFFECTIVE_PROB_THRESHOLD = 0.001


class AkinatorEngine:
    """
    Bayesian engine: pick next question by max entropy (information gain),
    update belief with likelihood, return top guess when confident.
    """

    def __init__(self, data_dir=None):
        data_dir = data_dir or DATA_DIR
        # Device: CUDA if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tensors and move to device
        kb_path = os.path.join(data_dir, "knowledge_base.pt")
        pop_path = os.path.join(data_dir, "popularity.pt")
        self.KB = torch.load(kb_path, map_location="cpu").to(self.device)
        prior = torch.load(pop_path, map_location="cpu").to(self.device)
        self.prior_probs = prior / prior.sum()

        with open(os.path.join(data_dir, "candidates.json"), encoding="utf-8") as f:
            self.candidates = json.load(f)
        with open(os.path.join(data_dir, "features.json"), encoding="utf-8") as f:
            self.features = json.load(f)

        self.N, self.M = self.KB.shape

    def _feature_category(self, feature_text):
        """Return category prefix for question diversity: 'gender', 'country', 'occupation', or None."""
        t = (feature_text or "").strip().lower()
        if t.startswith("is gender "):
            return "gender"
        if t.startswith("is country "):
            return "country"
        if t.startswith("is occupation "):
            return "occupation"
        return None

    def _feature_value(self, feature_text):
        """Return the value part of the question, e.g. 'Is gender male?' -> 'male'. Used to skip 'Unknown'."""
        t = (feature_text or "").strip()
        for prefix in ("Is gender ", "Is country ", "Is occupation "):
            if t.startswith(prefix):
                value = t[len(prefix) :].rstrip("?")
                return value.strip()
        return ""

    def _is_unknown_value(self, feature_text):
        """True if the question is about 'Unknown' (not useful to ask)."""
        return self._feature_value(feature_text).lower() == "unknown"

    def get_category_of_question(self, question_idx):
        """Return category string for question index, or None."""
        if 0 <= question_idx < self.M:
            return self._feature_category(self.features[question_idx])
        return None

    def _entropy_topk(self, current_probs):
        """Entropy of P(Yes) restricted to top-K candidates by probability. Used to prefer questions that split the front-runners."""
        k = min(TOP_K_CANDIDATES, self.N)
        if k <= 0:
            return torch.zeros(self.M, device=self.device)
        top_probs, top_idx = current_probs.topk(k, dim=0)
        top_sum = top_probs.sum().clamp(1e-12)
        # p_yes_topk[j] = sum over top-K of (probs[n] * KB[n,j]) / top_sum
        kb_top = self.KB[top_idx, :]  # (k, M)
        p_yes_topk = (top_probs.unsqueeze(1) * kb_top).sum(0) / top_sum
        p = p_yes_topk.clamp(1e-7, 1.0 - 1e-7)
        return -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)

    def _compute_entropy_mask(
        self,
        current_probs,
        asked_mask,
        last_asked_idx=None,
        resolved_categories=None,
        asked_count_per_category=None,
        skip_same_category=True,
        apply_cap=True,
        min_p_yes=None,
        max_p_yes=None,
    ):
        """Compute entropy and apply masks. Returns (entropy tensor, p_yes).
        min_p_yes/max_p_yes: when None use module constants; pass 0.01/0.99 or skip (see get_next_question) to relax.
        """
        if isinstance(asked_mask, list):
            asked_mask = torch.tensor(asked_mask, dtype=torch.bool, device=self.device)
        resolved_categories = resolved_categories or set()
        asked_count_per_category = asked_count_per_category or {}
        use_p_yes_bounds = min_p_yes is not None and max_p_yes is not None

        p_yes = torch.mv(self.KB.T, current_probs)
        p = p_yes.clamp(1e-7, 1.0 - 1e-7)
        entropy = -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)

        # 1) Already asked
        entropy = entropy.masked_fill(asked_mask, -float("inf"))

        # 2) Never ask "Unknown" (Is gender Unknown? etc.) – not useful
        for j in range(self.M):
            if self._is_unknown_value(self.features[j]):
                entropy[j] = -float("inf")

        # 3) Skip uninformative questions (optional; relaxed in fallback passes)
        if use_p_yes_bounds:
            entropy = entropy.masked_fill(p_yes < min_p_yes, -float("inf"))
            entropy = entropy.masked_fill(p_yes > max_p_yes, -float("inf"))

        # 4) Resolved categories
        for j in range(self.M):
            cat = self._feature_category(self.features[j])
            if cat is not None and cat in resolved_categories:
                entropy[j] = -float("inf")

        # 5) Cap per category (optional, so we can relax when we run out)
        if apply_cap:
            for j in range(self.M):
                cat = self._feature_category(self.features[j])
                if cat is not None:
                    count = asked_count_per_category.get(cat, 0)
                    if count >= MAX_QUESTIONS_PER_CATEGORY:
                        entropy[j] = -float("inf")

        # 6) Don't ask same category twice in a row (optional)
        if skip_same_category and last_asked_idx is not None and 0 <= last_asked_idx < self.M:
            last_cat = self._feature_category(self.features[last_asked_idx])
            if last_cat is not None:
                for j in range(self.M):
                    if self._feature_category(self.features[j]) == last_cat:
                        entropy[j] = -float("inf")

        return entropy, p_yes

    def _pick_best_question(self, entropy, entropy_topk):
        """Combine global entropy with top-K discrimination; return (idx, text) or (None, None)."""
        score = entropy + TOP_K_DISCRIMINATION_BETA * entropy_topk
        if score.max().item() <= -float("inf"):
            return None, None
        idx = score.argmax().item()
        return idx, self.features[idx]

    def get_next_question(
        self,
        current_probs,
        asked_mask,
        last_asked_idx=None,
        resolved_categories=None,
        asked_count_per_category=None,
    ):
        """
        Expert-system style: pick next question by entropy + top-K discrimination, subject to rules.
        Never asks "Unknown", skips uninformative P(Yes), then applies resolved/cap/same-category.
        If no question found, relaxes constraints (wider P(Yes), same-category, then cap, then no P(Yes) mask).
        Returns (question_idx, question_text) or (None, None) if really none.
        """
        resolved_categories = resolved_categories or set()
        asked_count_per_category = asked_count_per_category or {}
        entropy_topk = self._entropy_topk(current_probs)

        # When we've narrowed down (few effective candidates), skip P(Yes) mask so we still have questions
        # (e.g. after gender + country, occupation questions often have p_yes 0 or 1 and get masked)
        effective_n = (current_probs > EFFECTIVE_PROB_THRESHOLD).sum().item()
        use_p_yes_in_pass1 = effective_n > NARROWED_EFFECTIVE_N
        min_p1, max_p1 = (MIN_P_YES, MAX_P_YES) if use_p_yes_in_pass1 else (None, None)

        # Pass 1: full constraints (optionally skip P(Yes) when narrowed)
        entropy, _ = self._compute_entropy_mask(
            current_probs,
            asked_mask,
            last_asked_idx=last_asked_idx,
            resolved_categories=resolved_categories,
            asked_count_per_category=asked_count_per_category,
            skip_same_category=True,
            apply_cap=True,
            min_p_yes=min_p1,
            max_p_yes=max_p1,
        )
        idx, text = self._pick_best_question(entropy, entropy_topk)
        if idx is not None:
            return idx, text

        # Pass 2: wider P(Yes) (0.01–0.99), KEEP same-category (so we rotate to occupation, not country spam)
        entropy, _ = self._compute_entropy_mask(
            current_probs,
            asked_mask,
            last_asked_idx=last_asked_idx,
            resolved_categories=resolved_categories,
            asked_count_per_category=asked_count_per_category,
            skip_same_category=True,
            apply_cap=True,
            min_p_yes=0.01,
            max_p_yes=0.99,
        )
        idx, text = self._pick_best_question(entropy, entropy_topk)
        if idx is not None:
            return idx, text

        # Pass 3: ignore cap, no P(Yes) mask, KEEP same-category
        entropy, _ = self._compute_entropy_mask(
            current_probs,
            asked_mask,
            last_asked_idx=last_asked_idx,
            resolved_categories=resolved_categories,
            asked_count_per_category={},
            skip_same_category=True,
            apply_cap=False,
            min_p_yes=None,
            max_p_yes=None,
        )
        idx, text = self._pick_best_question(entropy, entropy_topk)
        if idx is not None:
            return idx, text

        # Pass 4: only then allow same category (so we have something to ask)
        entropy, _ = self._compute_entropy_mask(
            current_probs,
            asked_mask,
            last_asked_idx=None,
            resolved_categories=resolved_categories,
            asked_count_per_category={},
            skip_same_category=False,
            apply_cap=False,
            min_p_yes=None,
            max_p_yes=None,
        )
        idx, text = self._pick_best_question(entropy, entropy_topk)
        if idx is not None:
            return idx, text

        return None, None

    def update_belief(self, current_probs, question_idx, answer_val):
        """
        Bayesian update: likelihood * current_probs, then normalize.
        Mistake-tolerant: use soft likelihood (SOFT_MIN_LIKELIHOOD / SOFT_MATCH_LIKELIHOOD)
        so one wrong answer does not zero out the true character; disagreeing candidates
        stay in the game with low probability.
        answer_val: 1.0 (Yes), 0.0 (No), 0.5 (Unknown), 0.75 (Probably), 0.25 (Probably not).
        Returns new probs tensor (same device).
        """
        col = self.KB[:, question_idx]
        has_feature = (col >= 0.5).float()
        if answer_val >= YES_THRESHOLD:
            # Yes / Probably: high likelihood if has feature, soft min if not
            likelihood = has_feature * SOFT_MATCH_LIKELIHOOD + (1 - has_feature) * SOFT_MIN_LIKELIHOOD
        elif answer_val <= (1.0 - YES_THRESHOLD):
            # No / Probably not: high likelihood if does not have feature, soft min if has
            likelihood = (1 - has_feature) * SOFT_MATCH_LIKELIHOOD + has_feature * SOFT_MIN_LIKELIHOOD
        else:
            # Unknown: soft update
            likelihood = 1.0 - torch.abs(col - answer_val)
            likelihood = likelihood.clamp(min=1e-6)
        new_probs = current_probs * likelihood
        total = new_probs.sum().item()
        if total < 1e-12:
            total = 1e-12  # avoid collapse
        new_probs = new_probs / total
        return new_probs

    def get_top_guess(self, current_probs, threshold=0.8):
        """
        Return (candidate_name, probability) if max prob >= threshold, else None.
        """
        max_prob = current_probs.max().item()
        if max_prob < threshold:
            return None
        idx = current_probs.argmax().item()
        return self.candidates[idx], max_prob
