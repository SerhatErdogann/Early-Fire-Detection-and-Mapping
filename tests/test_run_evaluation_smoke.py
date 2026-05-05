import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.eval import run_evaluation


def _set_fake_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    project_root = tmp_path / "proj"
    fake_script = project_root / "src" / "eval" / "run_evaluation.py"
    fake_script.parent.mkdir(parents=True, exist_ok=True)
    fake_script.write_text("# fake", encoding="utf-8")
    monkeypatch.setattr(run_evaluation, "__file__", str(fake_script))
    return project_root


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["run_evaluation.py", *argv])
    run_evaluation.main()


def test_collect_videos_discovers_supported_and_ignores_nonvideo(tmp_path: Path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "a.mp4").write_text("", encoding="utf-8")
    (videos_dir / "b.avi").write_text("", encoding="utf-8")
    (videos_dir / "note.txt").write_text("x", encoding="utf-8")
    nested = videos_dir / "nested"
    nested.mkdir()
    (nested / "c.mkv").write_text("", encoding="utf-8")

    vids = run_evaluation._collect_videos(videos_dir, recursive=True)
    assert [p.name for p in vids] == ["a.mp4", "b.avi", "c.mkv"]


def test_profile_balanced_runs_single_profile_and_creates_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_root = _set_fake_project_root(monkeypatch, tmp_path)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "video1.mp4").write_text("", encoding="utf-8")
    (videos_dir / "ignore.csv").write_text("", encoding="utf-8")

    calls = []

    def fake_run_cmd(cmd, cwd):
        calls.append((cmd, cwd))

    monkeypatch.setattr(run_evaluation, "_run_cmd", fake_run_cmd)
    monkeypatch.setattr(
        run_evaluation,
        "compute_event_metrics",
        lambda events_csv, duration_sec: {
            "event_count": 1,
            "avg_event_duration": 3.0,
            "max_event_duration": 3.0,
            "min_event_duration": 3.0,
            "false_alarms_per_hour": 30.0,
            "events_per_minute": 0.5,
            "confirmed_frames_total": 4,
            "confirmed_coverage_ratio": 0.03,
        },
    )
    monkeypatch.setattr(
        run_evaluation,
        "_compute_event_metrics",
        lambda scored_csv, events_csv: {
            "frames_total": 10,
            "confirmed_frames": 2,
            "fire_event_frames": 1,
            "num_events": 1,
            "total_event_duration": 3,
            "max_event_duration": 3,
            "mean_event_duration": 3.0,
            "max_event_prob": 0.9,
            "mean_event_prob": 0.8,
        },
    )

    out_csv = tmp_path / "eval_summary.csv"
    _run_main(
        monkeypatch,
        [
            "--videos_dir",
            str(videos_dir),
            "--profile",
            "balanced",
            "--output",
            str(out_csv),
        ],
    )

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.loc[0, "profile"] == "balanced"
    assert df.loc[0, "status"] == "ok"
    assert df.loc[0, "event_metrics_source"] == "computed"
    assert "balanced" in str(df.loc[0, "pred_csv"])
    metrics_json = Path(df.loc[0, "event_metrics_json"])
    assert metrics_json.exists()
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert payload["event_count"] == 1
    assert len(calls) == 3  # infer + risk + events
    assert all(str(cwd) == str(project_root) for _, cwd in calls)


def test_profile_all_expands_to_three_profiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_fake_project_root(monkeypatch, tmp_path)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "video1.mp4").write_text("", encoding="utf-8")

    monkeypatch.setattr(run_evaluation, "_run_cmd", lambda cmd, cwd: None)
    monkeypatch.setattr(
        run_evaluation,
        "compute_event_metrics",
        lambda events_csv, duration_sec: {
            "event_count": 0,
            "avg_event_duration": 0.0,
            "max_event_duration": 0.0,
            "min_event_duration": 0.0,
            "false_alarms_per_hour": 0.0,
            "events_per_minute": 0.0,
            "confirmed_frames_total": 0,
            "confirmed_coverage_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        run_evaluation,
        "_compute_event_metrics",
        lambda scored_csv, events_csv: {
            "frames_total": 1,
            "confirmed_frames": 0,
            "fire_event_frames": 0,
            "num_events": 0,
            "total_event_duration": 0,
            "max_event_duration": 0,
            "mean_event_duration": 0.0,
            "max_event_prob": 0.0,
            "mean_event_prob": 0.0,
        },
    )

    out_csv = tmp_path / "all_profiles_summary.csv"
    _run_main(
        monkeypatch,
        [
            "--videos_dir",
            str(videos_dir),
            "--profile",
            "all",
            "--output",
            str(out_csv),
        ],
    )

    df = pd.read_csv(out_csv)
    assert len(df) == 3
    assert set(df["profile"].tolist()) == {"fast", "balanced", "safe"}


def test_max_videos_1_limits_processing_and_writes_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_fake_project_root(monkeypatch, tmp_path)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "a.mp4").write_text("", encoding="utf-8")
    (videos_dir / "b.mp4").write_text("", encoding="utf-8")

    monkeypatch.setattr(run_evaluation, "_run_cmd", lambda cmd, cwd: None)
    monkeypatch.setattr(
        run_evaluation,
        "compute_event_metrics",
        lambda events_csv, duration_sec: {
            "event_count": 0,
            "avg_event_duration": 0.0,
            "max_event_duration": 0.0,
            "min_event_duration": 0.0,
            "false_alarms_per_hour": 0.0,
            "events_per_minute": 0.0,
            "confirmed_frames_total": 0,
            "confirmed_coverage_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        run_evaluation,
        "_compute_event_metrics",
        lambda scored_csv, events_csv: {
            "frames_total": 1,
            "confirmed_frames": 0,
            "fire_event_frames": 0,
            "num_events": 0,
            "total_event_duration": 0,
            "max_event_duration": 0,
            "mean_event_duration": 0.0,
            "max_event_prob": 0.0,
            "mean_event_prob": 0.0,
        },
    )

    out_csv = tmp_path / "limited_summary.csv"
    _run_main(
        monkeypatch,
        [
            "--videos_dir",
            str(videos_dir),
            "--profile",
            "balanced",
            "--max_videos",
            "1",
            "--output",
            str(out_csv),
        ],
    )

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.loc[0, "video_name"] == "a.mp4"


def test_skip_existing_skips_per_video_profile_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_root = _set_fake_project_root(monkeypatch, tmp_path)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video = videos_dir / "video1.mp4"
    video.write_text("", encoding="utf-8")

    vid = run_evaluation._safe_id(video, videos_dir)
    profile_dir = project_root / "outputs" / "eval" / "balanced"
    profile_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = profile_dir / f"{vid}__pred.csv"
    scored_csv = profile_dir / f"{vid}__scored.csv"
    events_csv = profile_dir / f"{vid}__events.csv"
    bench_json = profile_dir / f"{vid}__bench.json"
    event_metrics_json = profile_dir / f"{vid}__event_metrics.json"
    pred_csv.write_text("frame_idx,alarm_state,decision_prob\n0,idle,0.1\n", encoding="utf-8")
    scored_csv.write_text("frame_idx,alarm_state,decision_prob,fire_event\n0,idle,0.1,0\n", encoding="utf-8")
    events_csv.write_text("event_id,start_frame,end_frame,duration,max_prob,avg_prob\n", encoding="utf-8")
    bench_json.write_text("{}", encoding="utf-8")

    calls = []
    monkeypatch.setattr(run_evaluation, "_run_cmd", lambda cmd, cwd: calls.append((cmd, cwd)))
    monkeypatch.setattr(
        run_evaluation,
        "compute_event_metrics",
        lambda events_csv, duration_sec: {
            "event_count": 0,
            "avg_event_duration": 0.0,
            "max_event_duration": 0.0,
            "min_event_duration": 0.0,
            "false_alarms_per_hour": 0.0,
            "events_per_minute": 0.0,
            "confirmed_frames_total": 0,
            "confirmed_coverage_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        run_evaluation,
        "_compute_event_metrics",
        lambda scored_csv, events_csv: {
            "frames_total": 1,
            "confirmed_frames": 0,
            "fire_event_frames": 0,
            "num_events": 0,
            "total_event_duration": 0,
            "max_event_duration": 0,
            "mean_event_duration": 0.0,
            "max_event_prob": 0.0,
            "mean_event_prob": 0.0,
        },
    )

    out_csv = tmp_path / "skip_summary.csv"
    _run_main(
        monkeypatch,
        [
            "--videos_dir",
            str(videos_dir),
            "--profile",
            "balanced",
            "--skip_existing",
            "--output",
            str(out_csv),
        ],
    )

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped"
    assert df.loc[0, "event_metrics_source"] == "computed"
    assert event_metrics_json.exists()
    assert len(calls) == 0


def test_skip_existing_reuses_existing_event_metrics_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_root = _set_fake_project_root(monkeypatch, tmp_path)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video = videos_dir / "video1.mp4"
    video.write_text("", encoding="utf-8")

    vid = run_evaluation._safe_id(video, videos_dir)
    profile_dir = project_root / "outputs" / "eval" / "balanced"
    profile_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = profile_dir / f"{vid}__pred.csv"
    scored_csv = profile_dir / f"{vid}__scored.csv"
    events_csv = profile_dir / f"{vid}__events.csv"
    bench_json = profile_dir / f"{vid}__bench.json"
    event_metrics_json = profile_dir / f"{vid}__event_metrics.json"
    pred_csv.write_text("frame_idx,alarm_state,decision_prob\n0,idle,0.1\n", encoding="utf-8")
    scored_csv.write_text("frame_idx,alarm_state,decision_prob,fire_event\n0,idle,0.1,0\n", encoding="utf-8")
    events_csv.write_text("event_id,start_frame,end_frame,duration,max_prob,avg_prob\n", encoding="utf-8")
    bench_json.write_text("{}", encoding="utf-8")
    original_payload = {
        "event_count": 123,
        "avg_event_duration": 1.0,
        "max_event_duration": 1.0,
        "min_event_duration": 1.0,
        "false_alarms_per_hour": 0.0,
        "events_per_minute": 0.0,
        "confirmed_frames_total": 0,
        "confirmed_coverage_ratio": 0.0,
    }
    event_metrics_json.write_text(json.dumps(original_payload), encoding="utf-8")

    monkeypatch.setattr(run_evaluation, "_run_cmd", lambda cmd, cwd: None)

    def _must_not_be_called(events_csv, duration_sec):
        raise AssertionError("compute_event_metrics should not be called when metrics json exists")

    monkeypatch.setattr(run_evaluation, "compute_event_metrics", _must_not_be_called)
    monkeypatch.setattr(
        run_evaluation,
        "_compute_event_metrics",
        lambda scored_csv, events_csv: {
            "frames_total": 1,
            "confirmed_frames": 0,
            "fire_event_frames": 0,
            "num_events": 0,
            "total_event_duration": 0,
            "max_event_duration": 0,
            "mean_event_duration": 0.0,
            "max_event_prob": 0.0,
            "mean_event_prob": 0.0,
        },
    )

    out_csv = tmp_path / "skip_reuse_summary.csv"
    _run_main(
        monkeypatch,
        [
            "--videos_dir",
            str(videos_dir),
            "--profile",
            "balanced",
            "--skip_existing",
            "--output",
            str(out_csv),
        ],
    )

    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert int(df.loc[0, "event_count"]) == 123
    assert df.loc[0, "event_metrics_source"] == "reused"


def test_failed_subprocess_recorded_with_error_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_fake_project_root(monkeypatch, tmp_path)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "video1.mp4").write_text("", encoding="utf-8")

    def fail_once(cmd, cwd):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(run_evaluation, "_run_cmd", fail_once)
    out_csv = tmp_path / "failed_summary.csv"

    _run_main(
        monkeypatch,
        [
            "--videos_dir",
            str(videos_dir),
            "--profile",
            "balanced",
            "--output",
            str(out_csv),
        ],
    )

    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.loc[0, "status"] == "failed"
    assert "error_message" in df.columns
    assert str(df.loc[0, "error_message"]).strip() != ""


def test_empty_folder_handled_cleanly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_fake_project_root(monkeypatch, tmp_path)
    empty_dir = tmp_path / "empty_videos"
    empty_dir.mkdir()

    with pytest.raises(SystemExit, match="No video files found"):
        _run_main(
            monkeypatch,
            [
                "--videos_dir",
                str(empty_dir),
                "--profile",
                "balanced",
                "--output",
                str(tmp_path / "summary.csv"),
            ],
        )
