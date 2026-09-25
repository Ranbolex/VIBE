import torch as th

from vibe_core import CandidateBuilder, EdgeStudent, GraphMessage
from vibe_core import PairwiseBeliefEncoder, VIBETeacher
from vibe_core.graph_message import sparse_topk_weights


def test_candidate_builder_respects_action_availability():
    q_values = th.tensor([[[[3.0, 1.0], [2.0, 0.0], [1.0, 4.0]]]])
    available = th.tensor([[[[1, 1], [1, 0], [0, 1]]]])
    candidates = CandidateBuilder("local_topk", top_k=2, top_m=8).build(
        q_values, available
    )
    assert candidates.size.item() == 2
    valid_actions = candidates.actions[candidates.valid]
    assert valid_actions[:, 1].eq(0).all()
    assert valid_actions[:, 2].eq(1).all()


def test_pow_qr_callback_ranks_bounded_candidates():
    q_values = th.zeros(1, 1, 4, 2)
    available = th.ones_like(q_values, dtype=th.long)
    observed = {}

    def score_fn(actions, candidate_qs):
        del candidate_qs
        observed["pool_size"] = actions.size(-2)
        weights = th.tensor([8, 4, 2, 1], device=actions.device)
        return (actions * weights).sum(dim=-1).float()

    candidates = CandidateBuilder(
        "pow_qr", top_k=2, top_m=2, pool_size=6
    ).build(q_values, available, score_fn=score_fn)
    assert observed["pool_size"] == 6
    assert candidates.size.item() == 2
    assert candidates.scores[0, 0, 0] >= candidates.scores[0, 0, 1]


def test_teacher_selects_pivotal_edge_in_analytic_diagnostic():
    teacher = VIBETeacher(n_agents=3, edge_budget=1)
    delta = teacher.compute_cct_delta(
        2, 1, th.device("cpu"), th.float32, 8.0, 24.0, 0.05
    )
    labels = teacher.select_edges(delta)
    assert th.allclose(delta[..., 0, 2], th.full((2, 1), 2.8))
    assert labels[..., 0, 2].eq(1).all()
    assert labels.sum().item() == 2


def test_teacher_scores_candidate_regret_with_uncertainty():
    teacher = VIBETeacher(n_agents=3, edge_budget=1)
    values = th.tensor([[[10.0, 9.0]]])
    candidate_qs = th.tensor([[[[0.0, 8.0, 1.0], [0.0, 0.0, 9.0]]]])
    candidate_actions = th.tensor([[[[0, 0, 0], [1, 0, 1]]]])
    uncertainty = th.ones(1, 1, 3, 3)
    uncertainty[..., 0, 1] = 0.25
    delta = teacher.compute_from_candidates(
        values, candidate_qs, candidate_actions,
        th.ones(1, 1, 2, dtype=th.bool), uncertainty,
    )
    assert delta[0, 0, 0, 1] > delta[0, 0, 0, 2]
    assert th.allclose(delta[0, 0, 0, 1], th.tensor(0.25))


def test_student_and_belief_are_receiver_local():
    th.manual_seed(1)
    student = EdgeStudent(hidden_dim=8, n_agents=3)
    belief = PairwiseBeliefEncoder(hidden_dim=8, n_agents=3, n_actions=4)
    hidden = th.randn(2, 3, 8)
    changed = hidden.clone()
    changed[:, 1:] = th.randn_like(changed[:, 1:])
    assert th.allclose(student(hidden)[:, 0], student(changed)[:, 0])
    assert th.allclose(belief(hidden)[:, 0], belief(changed)[:, 0])
    weights = sparse_topk_weights(
        student(hidden), top_k=1, straight_through=False,
        min_probability=0.0,
    )
    assert weights.diagonal(dim1=-2, dim2=-1).eq(0).all()
    assert weights.sum(dim=-1).le(1).all()


def test_graph_message_uses_receiver_local_payload():
    graph = GraphMessage(hidden_dim=5, message_dim=2, local_dim=2)
    with th.no_grad():
        graph.local_encoder[0].weight.copy_(th.eye(2))
        graph.local_encoder[0].bias.zero_()
    edges = th.zeros(1, 3, 3)
    edges[0, 0, 1] = 1.0
    payload = th.zeros(1, 3, 3, 2)
    payload[0, 0, 1] = th.tensor([1.0, 2.0])
    result = graph(edges, payload, payload_mode="receiver_local_ally")
    assert th.equal(result[0, 0], th.tensor([1.0, 2.0]))
    assert result[0, 1:].eq(0).all()
