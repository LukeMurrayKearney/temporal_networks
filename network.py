"""Synthetic population: households + layered community contacts (top-up model)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import networkx as nx

N_AGE = 9

# Right edges of the 9 age bins [0-4, 5-11, 12-17, 18-29, 30-39, 40-49, 50-59, 60-69, 70+].
_AGE_BIN_EDGES = np.array([5, 12, 18, 30, 40, 50, 60, 70])

# Cumulative proportions of the UK population per age group (0..8 inclusive).
AGE_CUMULATIVE = np.array([0.058, 0.145, 0.212, 0.364, 0.497, 0.623, 0.759, 0.866, 1.0])


def _bin_age(age: float) -> int:
    """Map a raw age to one of the N_AGE age-group indices (0–8)."""
    return int(np.searchsorted(_AGE_BIN_EDGES, float(age), side="left"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    node_id: int
    age: int           # age-group index in [0, N_AGE)
    household_id: int


@dataclass
class Household:
    hh_id: int
    members: list[Node] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    def edges(self) -> list[tuple[int, int]]:
        """Fully connected clique over all household members."""
        ids = [m.node_id for m in self.members]
        return [
            (ids[i], ids[j])
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
        ]


@dataclass
class CommunityLayer:
    """Non-household contact edges between individuals in different households."""
    name: str
    edges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

@dataclass
class Population:
    households: list[Household]
    nodes: dict[int, Node]           # node_id → Node
    layers: dict[str, CommunityLayer] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        n_nodes: int,
        hh_ages: list[list[int]],
        contact_model,
        layer_names: list[str] | None = None,
        layer_weights: list[float] | None = None,
        rng=None,
    ) -> "Population":
        """Build a synthetic population with households and community top-up.

        Phase 1 — Households: create ``n_nodes`` individuals following the UK
        age-structure (``AGE_CUMULATIVE``), then assign them to households by
        sampling age compositions from the observed ego data in ``hh_ages``.
        Any household slot that cannot be filled from the existing unhoused pool
        is satisfied by creating a top-up node, so every household is complete.

        Phase 2 — Stubs: for each node sample a total contact vector from
        ``contact_model`` and subtract the within-household contacts already
        satisfied, yielding the community stub vector.

        Phase 3 — Matching: pair stubs with bipartite age-group matching,
        excluding self-loops, multi-edges, and intra-household pairs.

        Phase 4 — Layers: distribute community edges across named layers by
        ``layer_weights`` (default: uniform).

        Parameters
        ----------
        n_nodes :
            Target number of seed individuals; actual population may be
            slightly larger due to top-up nodes added during household filling.
        hh_ages :
            Observed household ego data; each inner list is the raw ages of
            all members of one household (e.g. from uk_reconnect_hh_ages.json).
            Ages are binned to ``[0, N_AGE)`` using ``AGE_CUMULATIVE`` breakpoints.
        contact_model :
            Fitted contact model with interface
            ``sample_contacts(age_group: int, rng) → np.ndarray`` of shape
            ``(N_AGE,)``.  Works with any popnet ContactModel, GMMContactModel,
            DPLNMarginalModel, etc.
        layer_names :
            Names for community contact layers.  If ``None``, a single
            ``"community"`` layer is produced.
        layer_weights :
            Relative weight per layer (same length as ``layer_names``).
            Edges are assigned proportionally.  Defaults to uniform.
        rng :
            Seed or ``numpy.random.Generator``.
        """
        rng = np.random.default_rng(rng)

        if layer_names is None:
            layer_names = ["community"]
        n_layers = len(layer_names)
        if layer_weights is None:
            layer_weights = [1.0] * n_layers
        w = np.asarray(layer_weights, dtype=float)
        w = w / w.sum()

        # ---- Phase 1: households from ego data --------------------------
        # Create n_nodes with the UK age distribution.
        boundaries = np.round(AGE_CUMULATIVE * n_nodes).astype(int)
        counts = np.diff(np.concatenate(([0], boundaries)))
        age_arr = np.repeat(np.arange(N_AGE), counts)
        rng.shuffle(age_arr)

        nodes: dict[int, Node] = {}
        for i, ag in enumerate(age_arr):
            nodes[i] = Node(node_id=i, age=int(ag), household_id=-1)

        # Unhoused pool by age group; sets give O(1) discard/pop.
        pool: dict[int, set[int]] = {a: set() for a in range(N_AGE)}
        for nid, node in nodes.items():
            pool[node.age].add(nid)

        # Convert template ages to age-group indices; index templates by the
        # age groups they contain so we can quickly find valid templates for a
        # given ego age.
        templates: list[list[int]] = [[_bin_age(a) for a in hh] for hh in hh_ages]
        by_age: dict[int, list[int]] = {a: [] for a in range(N_AGE)}
        for idx, t in enumerate(templates):
            for ag in set(t):
                by_age[ag].append(idx)

        households: list[Household] = []
        node_order = list(range(n_nodes))
        rng.shuffle(node_order)
        next_id = n_nodes  # counter for top-up nodes

        for seed_id in node_order:
            if nodes[seed_id].household_id != -1:
                continue

            seed_age = nodes[seed_id].age
            hh_id = len(households)

            pool[seed_age].discard(seed_id)
            nodes[seed_id].household_id = hh_id
            members: list[Node] = [nodes[seed_id]]

            tmpl_candidates = by_age.get(seed_age, [])
            if tmpl_candidates:
                chosen = int(rng.integers(0, len(tmpl_candidates)))
                composition = list(templates[tmpl_candidates[chosen]])

                # Consume one slot for the seed's own age group.
                try:
                    composition.remove(seed_age)
                except ValueError:
                    if composition:
                        composition.pop()

                for target_age in composition:
                    if pool[target_age]:
                        cand_id = pool[target_age].pop()
                        nodes[cand_id].household_id = hh_id
                        members.append(nodes[cand_id])
                    else:
                        new_node = Node(node_id=next_id, age=target_age, household_id=hh_id)
                        nodes[next_id] = new_node
                        members.append(new_node)
                        next_id += 1

            households.append(Household(hh_id=hh_id, members=members))

        # ---- Phase 2: community stubs (top-up after household contacts) --
        # stubs[a][b] = node_ids of age-group-a nodes wanting community contacts
        #               with age-group-b nodes
        stubs: list[list[list[int]]] = [[[] for _ in range(N_AGE)] for _ in range(N_AGE)]

        hh_member_sets: dict[int, set[int]] = {
            h.hh_id: {m.node_id for m in h.members} for h in households
        }
        hh_by_node: dict[int, int] = {
            m.node_id: h.hh_id for h in households for m in h.members
        }

        for node in nodes.values():
            a = node.age
            total_vec = contact_model.sample_contacts(a, rng=rng).astype(int)

            # Subtract contacts already wired within the household
            home_by_age = np.zeros(N_AGE, dtype=int)
            for other_id in hh_member_sets[node.household_id]:
                if other_id != node.node_id:
                    home_by_age[nodes[other_id].age] += 1

            community_vec = np.maximum(0, total_vec - home_by_age)
            for b, count in enumerate(community_vec):
                if count > 0:
                    stubs[a][b].extend([node.node_id] * int(count))

        # ---- Phase 3: stub matching (no self-loops / multi-edges / intra-hh) --
        existing: set[tuple[int, int]] = set()
        for hh in households:
            for u, v in hh.edges():
                existing.add((min(u, v), max(u, v)))

        community_edges: list[tuple[int, int]] = []
        for a in range(N_AGE):
            for b in range(a, N_AGE):
                if a == b:
                    new = _match_within(stubs[a][a], rng, existing, hh_by_node)
                else:
                    new = _match_between(stubs[a][b], stubs[b][a], rng, existing, hh_by_node)
                community_edges.extend(new)

        # ---- Phase 4: distribute edges across named layers ---------------
        if n_layers == 1:
            layer_edge_lists: list[list[tuple[int, int]]] = [community_edges]
        else:
            rng.shuffle(community_edges)
            layer_edge_lists = [[] for _ in range(n_layers)]
            assignments = rng.choice(n_layers, size=len(community_edges), p=w)
            for edge, lyr in zip(community_edges, assignments):
                layer_edge_lists[lyr].append(edge)

        layers = {
            name: CommunityLayer(name=name, edges=edges)
            for name, edges in zip(layer_names, layer_edge_lists)
        }

        return cls(households=households, nodes=nodes, layers=layers)

    # ------------------------------------------------------------------
    # Post-build layer addition
    # ------------------------------------------------------------------

    def add_layer(
        self,
        name: str,
        stub_counts: dict[int, int],
        rng=None,
    ) -> CommunityLayer:
        """Wire up an additional community layer from per-node stub counts.

        Useful for injecting a structurally distinct layer (e.g. a workplace or
        school contact network) after the initial population build.  Avoids
        multi-edges with any previously created household or community edges.

        Parameters
        ----------
        name :
            Layer name, e.g. ``"workplace"``.
        stub_counts :
            ``{node_id: n_stubs}`` — how many contact stubs each node
            contributes to this layer.
        rng :
            Seed or ``numpy.random.Generator``.
        """
        rng = np.random.default_rng(rng)

        stub_list: list[int] = []
        for nid, count in stub_counts.items():
            stub_list.extend([nid] * int(count))

        existing: set[tuple[int, int]] = set()
        for hh in self.households:
            for u, v in hh.edges():
                existing.add((min(u, v), max(u, v)))
        for layer in self.layers.values():
            for u, v in layer.edges:
                existing.add((min(u, v), max(u, v)))

        hh_by_node = {
            m.node_id: h.hh_id for h in self.households for m in h.members
        }
        edges = _match_within(stub_list, rng, existing, hh_by_node)
        layer = CommunityLayer(name=name, edges=edges)
        self.layers[name] = layer
        return layer

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def to_graph(
        self,
        include_layers: list[str] | None = None,
    ) -> nx.Graph:
        """Return a ``networkx.Graph`` with household cliques + community layers.

        Parameters
        ----------
        include_layers :
            Names of community layers to include.  Defaults to all layers.
        """
        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(
                node.node_id,
                age=node.age,
                household_id=node.household_id,
            )
        for hh in self.households:
            for u, v in hh.edges():
                G.add_edge(u, v, layer="household")

        layer_keys = include_layers if include_layers is not None else list(self.layers)
        for lname in layer_keys:
            layer = self.layers.get(lname)
            if layer is None:
                continue
            for u, v in layer.edges:
                G.add_edge(u, v, layer=lname)

        return G

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_households(self) -> int:
        return len(self.households)

    def layer_edge_counts(self) -> dict[str, int]:
        return {name: layer.n_edges for name, layer in self.layers.items()}

    def household_size_distribution(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for hh in self.households:
            s = hh.size
            counts[s] = counts.get(s, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Internal stub-matching helpers
# ---------------------------------------------------------------------------

def _match_between(
    stubs_a: list[int],
    stubs_b: list[int],
    rng: np.random.Generator,
    existing: set[tuple[int, int]],
    hh_by_node: dict[int, int],
) -> list[tuple[int, int]]:
    """Pair stubs_a (age-group a) against stubs_b (age-group b).

    Skips self-loops, duplicate edges, and intra-household pairs.
    Mirrors the scan-and-swap approach from popnet.graph so that stubs are
    not wasted when the first candidate is invalid.
    """
    arr_a = np.array(stubs_a, dtype=np.int64)
    arr_b = np.array(stubs_b, dtype=np.int64)
    rng.shuffle(arr_a)
    rng.shuffle(arr_b)
    n = min(len(arr_a), len(arr_b))
    arr_a = arr_a[:n]
    arr_b = arr_b[:n].copy()

    edges: list[tuple[int, int]] = []
    for i in range(n):
        u = int(arr_a[i])
        for j in range(i, n):
            v = int(arr_b[j])
            if u == v:
                continue
            if hh_by_node.get(u) == hh_by_node.get(v):
                continue
            edge = (min(u, v), max(u, v))
            if edge in existing:
                continue
            arr_b[i], arr_b[j] = arr_b[j], arr_b[i]
            existing.add(edge)
            edges.append(edge)
            break
    return edges


def _match_within(
    stub_list: list[int] | np.ndarray,
    rng: np.random.Generator,
    existing: set[tuple[int, int]],
    hh_by_node: dict[int, int],
) -> list[tuple[int, int]]:
    """Pair stubs within a single pool.

    Used for within-age-group community contacts and for ``add_layer``.
    Skips self-loops, duplicate edges, and intra-household pairs.
    """
    arr = np.array(stub_list, dtype=np.int64)
    rng.shuffle(arr)
    n = len(arr)
    used = np.zeros(n, dtype=bool)

    edges: list[tuple[int, int]] = []
    for i in range(n):
        if used[i]:
            continue
        u = int(arr[i])
        for j in range(i + 1, n):
            if used[j]:
                continue
            v = int(arr[j])
            if u == v:
                continue
            if hh_by_node.get(u) == hh_by_node.get(v):
                continue
            edge = (min(u, v), max(u, v))
            if edge in existing:
                continue
            used[i] = used[j] = True
            existing.add(edge)
            edges.append(edge)
            break
    return edges
