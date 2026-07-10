import difflib
from dataclasses import dataclass
from typing import List


@dataclass
class Group:
    target: str
    members: List[str]
    confidence: float


# Domain hints map normalized terms to target group concepts.
DOMAIN_HINTS = {
    "helmet": "helmet",
    "hardhat": "helmet",
    "vest": "vest",
    "safetyvest": "vest",
    "person": "person",
    "worker": "person",
    "pedestrian": "person",
    "glove": "gloves",
    "gloves": "gloves",
    "goggle": "goggles",
    "goggles": "goggles",
    "boot": "boots",
    "boots": "boots",
    "mask": "mask",
}


def normalize_class_name(name: str) -> str:
    """Normalizes class names (lowercase, strip spaces/dashes/underscores, naive singularization)."""
    n = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    # Naive singularization: remove trailing 's' if length is > 3.
    # Note: domain hints explicitly map "gloves" and "boots" to avoid issues,
    # but we can do a naive check for un-mapped words.
    if len(n) > 3 and n.endswith("s") and n not in ["boots", "gloves", "goggles", "glass", "glasses"]:
        n = n[:-1]
    return n


def suggest_groups(class_names: List[str]) -> List[Group]:
    groups = []
    processed = set()

    # 1. Handle Domain Hints
    hint_groups = {}
    for name in class_names:
        if name in processed:
            continue
        norm_name = normalize_class_name(name)
        if norm_name in DOMAIN_HINTS:
            target_concept = DOMAIN_HINTS[norm_name]
            if target_concept not in hint_groups:
                hint_groups[target_concept] = []
            hint_groups[target_concept].append(name)
            processed.add(name)

    for target_concept, members in hint_groups.items():
        groups.append(Group(target=target_concept, members=members, confidence=0.95))

    # 2. Handle string similarity and substring matching
    unprocessed = [n for n in class_names if n not in processed]

    while unprocessed:
        base_name = unprocessed.pop(0)
        norm_base = normalize_class_name(base_name)
        current_group_members = [base_name]

        # Iterate over a copy of unprocessed
        i = 0
        while i < len(unprocessed):
            compare_name = unprocessed[i]
            norm_compare = normalize_class_name(compare_name)

            # Match conditions
            similarity = difflib.SequenceMatcher(None, norm_base, norm_compare).ratio()

            if similarity > 0.85 or norm_base in norm_compare or norm_compare in norm_base:
                current_group_members.append(compare_name)
                unprocessed.pop(i)
            else:
                i += 1

        # Determine target name for this group: shortest original name (since shortest normalized string)
        if current_group_members:
            # We want to use the shortest string as the target name
            target_name = min(current_group_members, key=len)

            if len(current_group_members) > 1:
                # Group based on similarity/substring
                groups.append(Group(target=target_name, members=current_group_members, confidence=0.80))
            else:
                # Single item group
                groups.append(Group(target=target_name, members=current_group_members, confidence=1.0))

    return groups
