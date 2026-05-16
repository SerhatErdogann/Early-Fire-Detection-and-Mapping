from src.inference.frame_sampling import AdaptiveFrameSampler


def test_adaptive_sampler_increases_step_on_low_motion_low_risk():
    s = AdaptiveFrameSampler(base_step=5, min_step=1, max_step=12, low_motion_threshold=0.03, high_risk_threshold=0.65)
    step = s.update(motion_mae=0.01, decision_prob=0.1, alarm_state="idle")
    assert step >= 6
    assert s.skip_count() == step - 1


def test_adaptive_sampler_min_step_on_high_risk():
    s = AdaptiveFrameSampler(base_step=5, min_step=1, max_step=12, low_motion_threshold=0.03, high_risk_threshold=0.65)
    step = s.update(motion_mae=0.02, decision_prob=0.9, alarm_state="suspected")
    assert step == 1
