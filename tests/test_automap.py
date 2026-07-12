import pytest
from remapcv.automap import suggest_groups, Group, get_similarity, normalize_name

def test_suggest_groups_hints():
    names = ["helmet", "hard_hat", "person", "pedestrian"]
    groups = suggest_groups(names)
    assert len(groups) == 2

    # Order of groups is not strictly defined, so check by target
    targets = {g.target: g for g in groups}
    assert "helmet" in targets
    assert "person" in targets

    helmet_group = targets["helmet"]
    assert set(helmet_group.members) == {"helmet", "hard_hat"}
    assert helmet_group.confidence == 0.95

    person_group = targets["person"]
    assert set(person_group.members) == {"person", "pedestrian"}
    assert person_group.confidence == 0.95

def test_suggest_groups_similarity():
    names = ["worker", "workers"]
    groups = suggest_groups(names)
    assert len(groups) == 1

    g = groups[0]
    assert g.target == "person"
    assert set(g.members) == {"worker", "workers"}
    # One is in hint map, one is not (after normalization, worker is in hint map).
    # Wait, 'workers' becomes 'worker' after normalize_name!
    # normalize_name: n = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    # if n.endswith("s") and len(n) > 1 and not n.endswith("ss"): n = n[:-1]
    # So 'workers' -> 'worker'.
    # Since 'worker' maps to 'person', both map to 'person'!
    # Which means they are both in the hint map.
    assert g.confidence == 0.95

def test_suggest_groups_similarity_no_hints():
    # Find two words that are similar but don't map to the same hint or normalize to same string
    # Let's use "vehicle" and "vehicles". Wait, "vehicles" -> "vehicle"
    # Let's use "automobile" and "automobiles". Wait, "automobiles" -> "automobile"
    # We need to test the similarity path.
    # What about "truck" and "truc"
    names = ["truck", "truc"]
    groups = suggest_groups(names)
    assert len(groups) == 1

    g = groups[0]
    assert g.target == "truc" # min length
    assert set(g.members) == {"truck", "truc"}
    assert g.confidence == round(get_similarity("truck", "truc"), 2)

def test_suggest_groups_disjoint():
    names = ["apple", "orange", "banana"]
    groups = suggest_groups(names)
    assert len(groups) == 3

    targets = {g.target: g for g in groups}
    assert "apple" in targets
    assert "orange" in targets
    assert "banana" in targets

    for g in groups:
        assert len(g.members) == 1
        assert g.confidence == 1.0

def test_suggest_groups_empty():
    groups = suggest_groups([])
    assert groups == []

def test_suggest_groups_exact_hint_target():
    names = ["hardhat", "helmet"]
    groups = suggest_groups(names)
    assert len(groups) == 1

    g = groups[0]
    # Because both are mapped to "helmet", and "helmet" is in the cluster,
    # the target should be "helmet" exactly.
    assert g.target == "helmet"
    assert set(g.members) == {"hardhat", "helmet"}
    assert g.confidence == 0.95

def test_suggest_groups_partial_hints():
    # Mix of hinted and unhinted, but unhinted is very similar
    names = ["worker", "workr"]
    groups = suggest_groups(names)
    assert len(groups) == 1
    g = groups[0]
    assert g.target == "person" # 'worker' maps to 'person' in hints
    assert set(g.members) == {"worker", "workr"}
    # Because not all are in hint map, confidence should be based on similarity
    assert g.confidence == round(get_similarity("worker", "workr"), 2)

def test_get_similarity():
    # Length >= 3, and one in the other, min similarity should be 0.8
    sim = get_similarity("car", "cars")
    assert sim >= 0.8

    sim2 = get_similarity("apple", "orange")
    assert sim2 < 0.8

def test_normalize_name():
    assert normalize_name("Hard-Hat") == "hardhat"
    assert normalize_name("Workers") == "worker"
    assert normalize_name("Class") == "class" # ends with 'ss'
    assert normalize_name("Dogs") == "dog"
