import difflib
from dataclasses import dataclass
from typing import List, Dict

PPE_HINTS = {
    "helmet": ["helmet", "hard_hat", "hardhat"],
    "vest": ["vest", "safety_vest"],
    "person": ["person", "worker", "pedestrian"],
    "gloves": ["gloves"],
    "goggles": ["goggles"],
    "boots": ["boots"],
    "mask": ["mask"]
}

@dataclass
class Group:
    target: str
    members: List[str]
    confidence: float

def normalize_name(name: str) -> str:
    n = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    # Naive singularization
    if n.endswith("s") and len(n) > 1 and not n.endswith("ss"):
        n = n[:-1]
    return n

def get_similarity(n1: str, n2: str) -> float:
    seq_match = difflib.SequenceMatcher(None, n1, n2).ratio()
    if len(n1) >= 3 and len(n2) >= 3:
        if n1 in n2 or n2 in n1:
            seq_match = max(seq_match, 0.8)
    return seq_match

def suggest_groups(names: List[str]) -> List[Group]:
    # Flatten domain hints for quick lookup
    hint_map = {}
    for target, synonyms in PPE_HINTS.items():
        for syn in synonyms:
            hint_map[normalize_name(syn)] = target

    # Also add targets themselves
    for target in PPE_HINTS.keys():
        hint_map[normalize_name(target)] = target

    groups: List[Group] = []
    unassigned = set(names)

    # Pass 1: Combine names by similarity or hint maps
    clusters = []

    names_list = list(unassigned)

    def get_cluster(name: str) -> int:
        for i, cluster in enumerate(clusters):
            if name in cluster:
                return i
        return -1

    for name in names_list:
        clusters.append([name])

    def merge_clusters(i, j):
        if i == j: return
        clusters[i].extend(clusters[j])
        clusters[j] = []

    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            n1 = names_list[i]
            n2 = names_list[j]
            norm1 = normalize_name(n1)
            norm2 = normalize_name(n2)

            c1 = get_cluster(n1)
            c2 = get_cluster(n2)
            if c1 == c2: continue

            # Check domain hints
            if norm1 in hint_map and norm2 in hint_map and hint_map[norm1] == hint_map[norm2]:
                merge_clusters(c1, c2)
                continue

            # Check similarity
            sim = get_similarity(norm1, norm2)
            if sim >= 0.75:
                merge_clusters(c1, c2)

    clusters = [c for c in clusters if c]

    for cluster in clusters:
        # Determine target
        target = min(cluster, key=len) # fallback
        for name in cluster:
            norm = normalize_name(name)
            if norm in hint_map:
                target_hint = hint_map[norm]
                # Try to find a member that exactly matches the hint if possible
                exact_matches = [m for m in cluster if normalize_name(m) == normalize_name(target_hint)]
                if exact_matches:
                    target = min(exact_matches, key=len)
                else:
                    target = target_hint
                break

        # Calculate confidence
        if len(cluster) == 1:
            confidence = 1.0
        else:
            # Check if all members are in the same hint group
            all_in_hint = True
            hint_val = None
            for name in cluster:
                norm = normalize_name(name)
                if norm not in hint_map:
                    all_in_hint = False
                    break
                if hint_val is None:
                    hint_val = hint_map[norm]
                elif hint_map[norm] != hint_val:
                    all_in_hint = False
                    break

            if all_in_hint:
                confidence = 0.95
            else:
                # Based on pairwise similarities
                sims = []
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        sims.append(get_similarity(normalize_name(cluster[i]), normalize_name(cluster[j])))
                if sims:
                    avg_sim = sum(sims) / len(sims)
                    # if avg_sim < 0.82 it will have a review comment
                    confidence = round(avg_sim, 2)
                else:
                    confidence = 1.0

        groups.append(Group(target=target, members=cluster, confidence=confidence))

    return groups
