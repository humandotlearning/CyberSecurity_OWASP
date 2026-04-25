from .helpers import make_env


def test_same_seed_reproducible_visible_facts():
    a = make_env(22)
    b = make_env(22)
    assert a.state.task_id == b.state.task_id
    assert a.state.hidden_facts["owner_invoice_id"] == b.state.hidden_facts["owner_invoice_id"]
    assert a.state.hidden_facts["other_invoice_id"] == b.state.hidden_facts["other_invoice_id"]
    assert a.state.visible_facts == b.state.visible_facts
