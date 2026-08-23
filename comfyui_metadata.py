"""
Extract ComfyUI generation metadata (LoRAs used, prompt text) embedded in
PNG images saved by ComfyUI's default Save Image node.

ComfyUI embeds the "API format" node graph that actually ran as a JSON
string in the PNG's "prompt" text chunk (a dict of node_id -> {class_type,
inputs, ...}).

This does NOT assume a plain/default workflow. Real workflows - especially
ones built with custom node packs (rgthree, pysssss, easy-use, WAS,
ComfyUIChat's own nodes, etc.) - very often don't put literal text/lora
names directly on the CLIPTextEncode/LoraLoader node itself. Instead:
  - A CLIPTextEncode's "text" input is a [node_id, slot] reference to some
    other node (a wildcard/dynamic-prompt node, an LLM-augmentation node, a
    concatenation node, ...), sometimes several hops away.
  - Some of those nodes compute their output dynamically at runtime (an LLM
    call, a random pick) - their own *input* fields only show what was fed
    into them (e.g. the brief given to the LLM), not what they produced.
    The saved JSON has no field for "what this node's output actually was"
    in general - EXCEPT when the workflow also routes that same output into
    a "ShowText"-style display node (pysssss's ShowText|pysssss and
    similar), which caches the literal value that flowed through it. When
    present, that's the authoritative source of "what the sampler actually
    saw" and takes priority over guessing among an upstream node's config.
  - LoRA "stack" nodes (e.g. rgthree's) hold multiple LoRAs in numbered
    fields (lora_01, lora_02, ...) instead of a single lora_name.
  - Which CLIPTextEncode is actually "positive" vs "negative" is determined
    by wiring into the sampler/guider node, not by node titles (titles can
    be stale/misleading after a workflow gets rewired).

So this traces the graph rather than pattern-matching a handful of known
field names: it finds whichever node actually feeds a sampler's
positive/negative inputs (regardless of that node's class_type), and
resolves references by first checking for a sibling ShowText-style cache of
that exact output, then falling back to the most substantial literal found
on/near the referenced node. Some values genuinely can't be recovered this
way - e.g. a LoRA strength driven by a "Random Number" node with no
ShowText tap and no filename_prefix tap (see below) has no runtime output
cached anywhere in this JSON, only its min/max/seed config - those are
reported as None rather than guessed.

CAVEAT - ShowText caches can be stale in batch/queued generation: a
ShowText-style widget's cached value is whatever the ComfyUI UI last
rendered, and when several prompts are queued back-to-back that can lag
behind by one or more generations - i.e. the "prompt" JSON embedded in
image N can carry a ShowText cache that actually reflects image N-1's (or
an even earlier) run, not image N's own. This was caught and confirmed by
comparing a cached LoRA-strength ShowText value against the ground-truth
value baked into that image's own filename (ComfyUI evaluates
filename_prefix fresh, server-side, at the moment each specific file is
written - it can't be stale for that file). Where a value's reference chain
also feeds a SaveImage node's filename_prefix, the actual saved filename is
used in preference to any ShowText cache, since it's guaranteed accurate.
Prompt text has no comparable ground truth to cross-check against, so it
still relies on ShowText caches when a literal isn't available on the
source node directly - correct in the common (non-batched, or last-in-batch)
case, but be aware the same staleness risk applies to prompt text in
rapid-batch-queued generations, without our being able to detect or correct
it the way we can for a filename-encoded numeric value.
"""

import json
import math
import os
import re

LORA_CLASS_HINT = "lora"
MAX_RESOLVE_DEPTH = 6
_CACHED_TEXT_KEY_RE = re.compile(r"^text_\d+$", re.IGNORECASE)


def _has_nan(value):
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, list):
        return any(_has_nan(v) for v in value)
    return False


def _is_dynamic_node(node):
    """ComfyUI's IS_CHANGED convention: a node reporting NaN there means
    "recompute unconditionally every run" - the standard way custom nodes
    mark non-deterministic/runtime-computed output (Random Number,
    DPRandomGenerator, "Text Load Line From File" cycling through a list,
    LLM-augmentation nodes, ...). For these, the node's own literal fields
    describe its config/source, not the value it actually produced - trusting
    them as "the output" gives a plausible-looking but wrong answer (e.g.
    grabbing a line-from-file node's file_path as if it were the line that
    was actually read that run)."""
    return _has_nan((node or {}).get("is_changed"))


def extract_comfyui_info(image_path):
    """
    Read ComfyUI's embedded workflow metadata from an image, if present.

    Args:
        image_path: Path to the image file

    Returns:
        dict with keys "loras" (list of {"name", "strength_model",
        "strength_clip"} - strengths are None if not resolvable),
        "model_name" (str or None - the checkpoint/UNET the sampler
        actually used), "positive_prompt" (str or None), "negative_prompt"
        (str or None), "other_prompts" (list of str, used when no
        positive/negative sampler wiring could be found at all). Returns
        None if no ComfyUI metadata is present, or the image can't be read.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            info = dict(img.info or {})
    except Exception:
        return None

    prompt_json = info.get("prompt")
    if not prompt_json:
        return None

    try:
        nodes = json.loads(prompt_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(nodes, dict):
        return None

    ref_consumers = _build_ref_consumers(nodes)
    filename_prefix_hint = _derive_filename_prefix(image_path)
    sampler_node = _find_sampler_node(nodes)

    loras = _extract_loras(nodes, ref_consumers, filename_prefix_hint)
    positive, negative, others = _extract_prompts(nodes, ref_consumers, sampler_node)
    model_name = _extract_model(nodes, ref_consumers, sampler_node)

    if not loras and not positive and not negative and not others and not model_name:
        return None

    return {
        "loras": loras,
        "model_name": model_name,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "other_prompts": others,
    }


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _leaves_for_concat_chain(nodes, value, visited=None):
    """Given an input value that feeds a "concatenate strings" style node,
    return the ordered list of (node_id, slot) leaf references that are
    "format a single number as text" nodes (e.g. JWFloatToString) - i.e.
    the numeric values actually baked into the final string, in
    left-to-right order. Plain literal text segments contribute nothing
    (they're already correctly embedded, nothing to recover)."""
    if visited is None:
        visited = set()
    if not isinstance(value, list) or len(value) < 2:
        return []
    node_id = str(value[0])
    if node_id in visited:
        return []
    visited = visited | {node_id}
    node = nodes.get(node_id)
    if not isinstance(node, dict):
        return []
    inputs = node.get("inputs", {}) or {}
    ref_items = [
        (k, v) for k, v in inputs.items() if isinstance(v, list) and len(v) >= 2
    ]

    if len(ref_items) == 1:
        # Single-reference "format as string" node - its one upstream
        # reference is a numeric leaf
        return [tuple(ref_items[0][1][:2])]

    if len(ref_items) > 1:
        # Multi-reference "concatenate" node - recurse into each reference
        # input, in declaration order
        leaves = []
        for _, v in ref_items:
            leaves.extend(_leaves_for_concat_chain(nodes, v, visited))
        return leaves

    return []


def _derive_filename_prefix(image_path):
    """Best-effort: recover the filename_prefix value ComfyUI actually used
    to save this specific file, by stripping its standard auto-appended
    counter suffix (e.g. "..._00001_") and extension.

    Unlike a cached ShowText widget (which reflects whatever the UI last
    happened to render - and can lag behind by one or more queued
    generations when several prompts are submitted in a batch, since the
    widget's cached text gets bundled into every subsequent submission
    until the UI re-renders it), the saved *filename* is evaluated by
    ComfyUI's backend at the moment that specific file was written, so it's
    guaranteed to correspond to this exact image.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    match = re.match(r"^(.*)_\d{4,}_$", base)
    return match.group(1) if match else None


def _resolve_via_cached_concat(
    nodes, ref_consumers, target_node_id, target_slot, filename_prefix_hint
):
    """Last-resort numeric resolution for a value with no cached output of
    its own (e.g. a "Random Number" node - the JSON only has its
    min/max/seed config, not what it actually sampled).

    Forward-searches for either (a) this exact reference feeding a
    SaveImage node's filename_prefix - in which case the actual saved
    filename is used, since it's guaranteed accurate for this specific
    image - or (b), failing that, a ShowText-style cache fed by it,
    possibly through a chain of "format number as string" and "concatenate
    strings" nodes. Either way, the reference's position among all the
    numeric-formatted leaves feeding that same chain is used to pick the
    matching number back out of the resulting string. Returns None (rather
    than a guess) if the position can't be determined unambiguously.
    """
    seen = set()

    def walk(node_id, slot, depth=0):
        if depth > MAX_RESOLVE_DEPTH:
            return None
        for consumer_id in ref_consumers.get((node_id, slot), []):
            if consumer_id in seen:
                continue
            consumer = nodes.get(consumer_id)
            if not isinstance(consumer, dict):
                continue
            consumer_inputs = consumer.get("inputs", {}) or {}

            def _ref_key_matching(v):
                return (
                    isinstance(v, list)
                    and len(v) >= 2
                    and str(v[0]) == node_id
                    and v[1] == slot
                )

            if (
                filename_prefix_hint is not None
                and str(consumer.get("class_type", "")).lower() == "saveimage"
            ):
                fp_ref = consumer_inputs.get("filename_prefix")
                if _ref_key_matching(fp_ref):
                    seen.add(consumer_id)
                    return filename_prefix_hint, _leaves_for_concat_chain(nodes, fp_ref)

            cached = _cached_display_value(consumer)
            if cached is not None:
                seen.add(consumer_id)
                for v in consumer_inputs.values():
                    if _ref_key_matching(v):
                        return cached, _leaves_for_concat_chain(nodes, v)
                continue

            result = walk(consumer_id, 0, depth + 1)
            if result is not None:
                return result
        return None

    result = walk(target_node_id, target_slot)
    if result is None:
        return None
    cached_string, leaves = result

    target_key = (target_node_id, target_slot)
    if target_key not in leaves:
        return None

    numbers = _NUMBER_RE.findall(cached_string)
    if len(numbers) != len(leaves):
        # Can't align matches to leaves unambiguously (e.g. a literal text
        # segment elsewhere in the string also looked number-shaped)
        return None

    try:
        return float(numbers[leaves.index(target_key)])
    except ValueError:
        return None


def _build_ref_consumers(nodes):
    """Map (source_node_id, source_slot) -> [consumer_node_id, ...], from
    every list-reference found in any node's inputs. Used to find sibling
    nodes that consume the same output as the one we're trying to resolve."""
    consumers = {}
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {}) or {}
        for v in inputs.values():
            if isinstance(v, list) and len(v) >= 2:
                key = (str(v[0]), v[1])
                consumers.setdefault(key, []).append(node_id)
    return consumers


def _cached_display_value(node):
    """If this node looks like a "ShowText"-style cache/display node (has a
    literal in a text_0-style field), return it."""
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs", {}) or {}
    for key, val in inputs.items():
        if _CACHED_TEXT_KEY_RE.match(key) and isinstance(val, str) and val.strip():
            return val
    return None


def _resolve_ref(nodes, ref_consumers, value, visited=None, depth=0):
    """Resolve a value that may be a [node_id, output_slot] reference.

    Priority order:
      1. A sibling node that consumes this exact (node_id, slot) output and
         caches it as a literal (a ShowText-style display node) - the most
         reliable source when the upstream node computes its output
         dynamically (LLM call, random pick, ...).
      2. A literal directly on the referenced node itself (preferring the
         most substantial one - custom-node packs use many different field
         names for "the actual text/number", so this doesn't hardcode
         names).
      3. Recursing into whichever of that node's own inputs are themselves
         references.

    Returns None if nothing resolvable is found within the depth limit.
    """
    if visited is None:
        visited = set()

    if isinstance(value, (str, int, float)):
        return value
    if not isinstance(value, list) or not value:
        return None
    if depth > MAX_RESOLVE_DEPTH:
        return None

    node_id = str(value[0])
    slot = value[1] if len(value) > 1 else 0
    if node_id in visited or node_id not in nodes:
        return None
    node = nodes.get(node_id)
    if not isinstance(node, dict):
        return None

    # ConditioningZeroOut is a standard core ComfyUI node whose entire
    # purpose is to null out its input conditioning (a common way to get an
    # effectively-empty negative prompt on models that don't need a real
    # one). Following the reference through it would surface the *positive*
    # text it zeroed as if it were the negative prompt - report it as
    # genuinely empty instead.
    if str(node.get("class_type", "")).lower() == "conditioningzeroout":
        return None

    visited = visited | {node_id}

    for consumer_id in ref_consumers.get((node_id, slot), []):
        if consumer_id in visited:
            continue
        cached = _cached_display_value(nodes.get(consumer_id))
        if cached:
            return cached

    inputs = node.get("inputs", {}) or {}

    if not _is_dynamic_node(node):
        string_literals = [
            v for v in inputs.values() if isinstance(v, str) and v.strip()
        ]
        if string_literals:
            return max(string_literals, key=len)

        numeric_literals = [v for v in inputs.values() if isinstance(v, (int, float))]
        if numeric_literals:
            return numeric_literals[0]

    resolved = []
    for v in inputs.values():
        if isinstance(v, list):
            r = _resolve_ref(nodes, ref_consumers, v, visited, depth + 1)
            if r is not None:
                resolved.append(r)

    strings = [r for r in resolved if isinstance(r, str) and r.strip()]
    if strings:
        return max(strings, key=len)
    numbers = [r for r in resolved if isinstance(r, (int, float))]
    if numbers:
        return numbers[0]

    return None


def _resolve_text(nodes, ref_consumers, value):
    if value is None:
        return None
    resolved = _resolve_ref(nodes, ref_consumers, value)
    return resolved if isinstance(resolved, str) and resolved.strip() else None


def _resolve_numeric(nodes, ref_consumers, value, filename_prefix_hint=None):
    if value is None:
        return None
    resolved = _resolve_ref(nodes, ref_consumers, value)
    if isinstance(resolved, (int, float)):
        return resolved

    if not (isinstance(value, list) and value):
        return None

    # Last resort: the source node (e.g. a "Random Number" node) has no
    # cached output of its own, but its value may still be recoverable if
    # it was also piped into a filename_prefix or a formatted/concatenated
    # ShowText elsewhere
    node_id = str(value[0])
    slot = value[1] if len(value) > 1 else 0
    via_cache = _resolve_via_cached_concat(
        nodes, ref_consumers, node_id, slot, filename_prefix_hint
    )
    if via_cache is None:
        return None

    # Sanity-check against the source node's own configured range, if it
    # has one (e.g. a "Random Number" node's minimum/maximum) - guards
    # against a subtly wrong position match returning a plausible-looking
    # but incorrect number
    source_inputs = (nodes.get(node_id, {}) or {}).get("inputs", {}) or {}
    lo, hi = source_inputs.get("minimum"), source_inputs.get("maximum")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        if not (lo - 1e-6 <= via_cache <= hi + 1e-6):
            return None

    return via_cache


_LORA_SLOT_RE = re.compile(r"^lora[_ ]?0*(\d+)$", re.IGNORECASE)


def _extract_loras(nodes, ref_consumers, filename_prefix_hint=None):
    """Collect LoRA name + strength from any node whose class_type mentions
    lora, handling both single-lora nodes (lora_name) and numbered-slot
    "stack" nodes (lora_01, lora_02, ...)."""
    loras = []
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", "")).lower()
        if LORA_CLASS_HINT not in class_type:
            continue

        inputs = node.get("inputs", {}) or {}
        inputs_lower = {k.lower(): v for k, v in inputs.items()}

        # Simple single-lora nodes (e.g. LoraLoader)
        simple_name = (
            inputs.get("lora_name") or inputs.get("lora") or inputs.get("name")
        )
        if isinstance(simple_name, str) and simple_name.strip().lower() not in (
            "",
            "none",
        ):
            loras.append(
                {
                    "name": simple_name,
                    "strength_model": _resolve_numeric(
                        nodes,
                        ref_consumers,
                        inputs.get("strength_model"),
                        filename_prefix_hint,
                    ),
                    "strength_clip": _resolve_numeric(
                        nodes,
                        ref_consumers,
                        inputs.get("strength_clip"),
                        filename_prefix_hint,
                    ),
                }
            )

        # Numbered lora-stack nodes (e.g. rgthree's "Lora Loader Stack")
        for key, val in inputs.items():
            match = _LORA_SLOT_RE.match(key)
            if not match or not isinstance(val, str):
                continue
            name = val.strip()
            if not name or name.lower() == "none":
                continue

            strength_key = key.lower().replace("lora", "strength", 1)
            strength = _resolve_numeric(
                nodes,
                ref_consumers,
                inputs_lower.get(strength_key),
                filename_prefix_hint,
            )
            loras.append(
                {"name": name, "strength_model": strength, "strength_clip": strength}
            )

    return loras


def _find_sampler_node(nodes):
    """Find the first node whose inputs include both "positive" and
    "negative" as list-references - the sampler/guider that determines
    what the model actually saw, regardless of its class_type (KSampler,
    KSamplerAdvanced, CFGGuider, custom variants, ...) or node title
    (titles can be stale/misleading after a workflow gets rewired)."""
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {}) or {}
        pos_ref = inputs.get("positive")
        neg_ref = inputs.get("negative")
        if (
            isinstance(pos_ref, list)
            and pos_ref
            and isinstance(neg_ref, list)
            and neg_ref
        ):
            return node
    return None


def _extract_prompts(nodes, ref_consumers, sampler_node):
    """Find positive/negative prompt text from the sampler/guider node's
    positive+negative inputs.

    Falls back to resolving any CLIPTextEncode-ish node's text as
    "other_prompts" if no sampler node (or no positive/negative wiring) was
    found at all.
    """
    inputs = (sampler_node or {}).get("inputs", {}) or {}
    positive_ref = inputs.get("positive")
    negative_ref = inputs.get("negative")

    positive = (
        _resolve_text(nodes, ref_consumers, positive_ref) if positive_ref else None
    )
    negative = (
        _resolve_text(nodes, ref_consumers, negative_ref) if negative_ref else None
    )

    others = []
    if positive is None and negative is None:
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", "")).lower()
            if "textencode" not in class_type:
                continue
            text = _resolve_text(
                nodes, ref_consumers, (node.get("inputs", {}) or {}).get("text")
            )
            if text:
                others.append(text)

    return positive, negative, others


def _resolve_model_name(nodes, ref_consumers, value, visited=None, depth=0):
    """Walk a "model" reference backward through any LoRA loader nodes to
    find the underlying checkpoint/UNET loader and its model filename.

    LoRA loader nodes pass the model through unchanged (just modified), so
    naively grabbing "the most substantial literal string" off one of them
    would return a LoRA name instead of the actual base model/checkpoint -
    this specifically keeps following the "model" chain through them.
    """
    if visited is None:
        visited = set()
    if not isinstance(value, list) or not value:
        return None
    if depth > MAX_RESOLVE_DEPTH:
        return None

    node_id = str(value[0])
    if node_id in visited or node_id not in nodes:
        return None
    node = nodes.get(node_id)
    if not isinstance(node, dict):
        return None
    visited = visited | {node_id}

    class_type = str(node.get("class_type", "")).lower()
    inputs = node.get("inputs", {}) or {}

    if LORA_CLASS_HINT in class_type:
        model_ref = inputs.get("model")
        if isinstance(model_ref, list):
            return _resolve_model_name(
                nodes, ref_consumers, model_ref, visited, depth + 1
            )
        return None

    # Loader-style node - prefer a well-known checkpoint/model filename
    # field. Some workflows wire the name itself through a picker/reroute
    # node (e.g. a "type bypass" node) rather than setting it directly, so
    # a reference here needs resolving too, not just a literal. If a named
    # field exists but can't be resolved (e.g. it traces to a dynamic/
    # runtime-computed picker with nothing cached), the model name is
    # genuinely unknown - don't fall through to grabbing an unrelated
    # literal off the same node (e.g. a "weight_dtype" config value).
    named_field_present = False
    for key in ("ckpt_name", "unet_name", "model_name", "checkpoint"):
        val = inputs.get(key)
        if val is None:
            continue
        named_field_present = True
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list):
            resolved = _resolve_text(nodes, ref_consumers, val)
            if resolved:
                return resolved

    if named_field_present:
        return None

    # Generic fallback: the most substantial literal string on this node -
    # unless the node is flagged as non-deterministic/runtime-computed (e.g.
    # a "Text Load Line From File" cycling picker), in which case none of
    # its own literal fields represent the value it actually produced.
    if not _is_dynamic_node(node):
        string_literals = [
            v for v in inputs.values() if isinstance(v, str) and v.strip()
        ]
        if string_literals:
            return max(string_literals, key=len)

    # Otherwise this is likely a passthrough/patch node (e.g.
    # ModelSamplingXXX) - keep following whichever of its own inputs are
    # references, in case the model chain continues through it
    for v in inputs.values():
        if isinstance(v, list):
            result = _resolve_model_name(nodes, ref_consumers, v, visited, depth + 1)
            if result is not None:
                return result

    return None


def _extract_model(nodes, ref_consumers, sampler_node):
    """Resolve the checkpoint/UNET model name feeding the sampler node."""
    inputs = (sampler_node or {}).get("inputs", {}) or {}
    model_ref = inputs.get("model")
    if not isinstance(model_ref, list):
        return None
    return _resolve_model_name(nodes, ref_consumers, model_ref)
