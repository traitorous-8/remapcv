import difflib
from dataclasses import dataclass
from typing import List, Dict, Set

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
    # Naive singularization: remove trailing 's' unless it ends with 'ss'
    if n.endswith("s") and len(n) > 1 and not n.endswith("ss"):
        n = n[:-1]
    return n

def get_similarity(n1: str, n2: str) -> float:
    seq_match = difflib.SequenceMatcher(None, n1, n2).ratio()
    # Boost similarity if one is a substring of another and they are relatively long
    if len(n1) >= 4 and len(n2) >= 4:
        if n1 in n2 or n2 in n1:
            seq_match = max(seq_match, 0.85)
    return seq_match

def suggest_groups(names: List[str]) -> List[Group]:
    # Reverse mapping for quick lookup: normalized synonym -> target hint
    hint_map = {}
    for target, synonyms in PPE_HINTS.items():
        for syn in synonyms:
            hint_map[normalize_name(syn)] = target
        # Also add the target itself
        hint_map[normalize_name(target)] = target

    groups: List[Group] = []

    # We will cluster names
    clusters: List[List[str]] = [[name] for name in set(names)]

    def merge_clusters(i: int, j: int):
        if i == j: return
        clusters[i].extend(clusters[j])
        clusters[j] = []

    def get_cluster(name: str) -> int:
        for i, cluster in enumerate(clusters):
            if name in cluster:
                return i
        return -1

    names_list = list(set(names))
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            n1 = names_list[i]
            n2 = names_list[j]
            norm1 = normalize_name(n1)
            norm2 = normalize_name(n2)

            c1 = get_cluster(n1)
            c2 = get_cluster(n2)
            if c1 == c2 or c1 == -1 or c2 == -1:
                continue

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
        cluster.sort() # for deterministic output

        target = min(cluster, key=len) # default to shortest name

        # Determine if any member has a domain hint
        for name in cluster:
            norm = normalize_name(name)
            if norm in hint_map:
                target_hint = hint_map[norm]
                # Try to use a member that perfectly matches the hint
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
                sims = []
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        sims.append(get_similarity(normalize_name(cluster[i]), normalize_name(cluster[j])))
                if sims:
                    avg_sim = sum(sims) / len(sims)
                    confidence = round(avg_sim, 2)
                else:
                    confidence = 1.0

        groups.append(Group(target=target, members=cluster, confidence=confidence))

    groups.sort(key=lambda g: g.target)
    return groups
