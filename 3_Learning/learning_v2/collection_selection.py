"""Deterministic proportional selection from an already frozen split."""

from __future__ import annotations


def select_split_sites(split: dict, max_sites: int | None = None) -> list[tuple[str, str]]:
    partitions = [name for name in ("train", "val", "test") if split.get(name)]
    total = sum(len(split[name]) for name in partitions)
    target = total if max_sites is None else min(max(1, int(max_sites)), total)
    if target == total:
        return [(name, site) for name in partitions for site in split[name]]
    exact = {name: target * len(split[name]) / total for name in partitions}
    quotas = {name: min(len(split[name]), int(exact[name])) for name in partitions}
    if target >= len(partitions):
        for name in partitions:
            quotas[name] = max(1, quotas[name])
    while sum(quotas.values()) < target:
        eligible = [name for name in partitions if quotas[name] < len(split[name])]
        name = max(eligible, key=lambda item: (exact[item] - quotas[item], -partitions.index(item)))
        quotas[name] += 1
    while sum(quotas.values()) > target:
        eligible = [name for name in reversed(partitions) if quotas[name] > (1 if target >= len(partitions) else 0)]
        name = max(eligible, key=lambda item: (quotas[item] - exact[item], partitions.index(item)))
        quotas[name] -= 1
    return [(name, site) for name in partitions for site in split[name][:quotas[name]]]


def select_stratified_sites(
    split: dict, positive_sites: set[str], max_sites: int | None = None,
    positive_fraction: float = 0.6,
) -> list[tuple[str, str]]:
    """Preserve partition quotas while enriching a bounded pilot for support."""
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must be between zero and one")
    proportional = select_split_sites(split, max_sites)
    quota = {name: sum(partition == name for partition, _ in proportional) for name in ("train", "val", "test")}
    selected = []
    for partition in ("train", "val", "test"):
        sites = list(split.get(partition, [])); target = quota[partition]
        positives = [site for site in sites if site in positive_sites]
        negatives = [site for site in sites if site not in positive_sites]
        positive_target = min(len(positives), max(1 if target else 0, round(target * positive_fraction)))
        chosen = positives[:positive_target]
        chosen.extend(negatives[:target - len(chosen)])
        if len(chosen) < target:
            remaining = [site for site in positives if site not in chosen]
            chosen.extend(remaining[:target - len(chosen)])
        selected.extend((partition, site) for site in chosen)
    return selected


def select_multilabel_stratified_sites(
    split: dict,
    positive_sites_by_rule: dict[str, set[str]],
    max_sites: int | None = None,
    positive_fraction: float = 0.6,
    minimum_per_rule: int = 1,
) -> list[tuple[str, str]]:
    """Select a bounded cohort while covering every predeclared rule.

    Coverage is enforced independently inside train, validation, and test. A
    deterministic greedy set-cover pass runs before the ordinary positive /
    negative fill. If the frozen partition or quota cannot satisfy a rule, the
    caller can expose that fact by comparing available and selected support.
    """
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must be between zero and one")
    if minimum_per_rule < 1:
        raise ValueError("minimum_per_rule must be at least one")
    proportional = select_split_sites(split, max_sites)
    quotas = {
        name: sum(partition == name for partition, _ in proportional)
        for name in ("train", "val", "test")
    }
    rules = sorted(positive_sites_by_rule)
    selected: list[tuple[str, str]] = []
    for partition in ("train", "val", "test"):
        sites = list(split.get(partition, [])); target = quotas[partition]
        position = {site: index for index, site in enumerate(sites)}
        needs = {
            rule: min(minimum_per_rule, sum(site in positive_sites_by_rule[rule] for site in sites))
            for rule in rules
        }
        chosen: list[str] = []
        while len(chosen) < target and any(value > 0 for value in needs.values()):
            candidates = []
            for site in sites:
                if site in chosen:
                    continue
                covered = [rule for rule in rules if needs[rule] > 0 and site in positive_sites_by_rule[rule]]
                if covered:
                    candidates.append((len(covered), -position[site], site, covered))
            if not candidates:
                break
            _, _, site, covered = max(candidates)
            chosen.append(site)
            for rule in covered:
                needs[rule] -= 1
        union_positive = set().union(*(positive_sites_by_rule.values())) if rules else set()
        positive_target = min(
            sum(site in union_positive for site in sites),
            max(len(chosen), 1 if target else 0, round(target * positive_fraction)),
        )
        chosen.extend(
            site for site in sites
            if site in union_positive and site not in chosen and len(chosen) < positive_target
        )
        chosen.extend(
            site for site in sites
            if site not in union_positive and site not in chosen and len(chosen) < target
        )
        chosen.extend(site for site in sites if site not in chosen and len(chosen) < target)
        selected.extend((partition, site) for site in chosen)
    return selected
