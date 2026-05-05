from src.inference.alarm import AlarmConfig, AlarmStateMachine


def test_alarm_state_machine_confirms_with_persistence():
    sm = AlarmStateMachine(AlarmConfig(high_threshold=0.7, low_threshold=0.4, suspect_threshold=0.55, confirm_frames=3))
    states = []
    for p in [0.6, 0.75, 0.8, 0.9]:
        st, ev, _, _ = sm.update(decision_prob=p, top10_intensity=0.8, largest_component_area=0.02)
        states.append((st, ev))
    assert states[-1][0] == "confirmed"
    assert states[-1][1] == 1


def test_alarm_state_machine_enters_cooldown():
    sm = AlarmStateMachine(AlarmConfig(high_threshold=0.7, low_threshold=0.4, suspect_threshold=0.55, confirm_frames=2, cooldown_frames=2))
    sm.update(0.8)
    sm.update(0.9)
    st, ev, _, _ = sm.update(0.2)
    assert st == "cooldown"
    assert ev == 0
