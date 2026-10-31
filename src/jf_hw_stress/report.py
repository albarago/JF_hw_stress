"""Results as JSON (schema 2, the source of truth) and as a self-contained HTML page built from it."""
from __future__ import annotations

import datetime
import html
import platform as host_platform
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .engine import MODE_LABELS, Engine

SCHEMA_VERSION = 2


def _iso(ts: Optional[float]) -> Optional[str]:
    if not ts:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)


def build(engine: Engine, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    cfg, src, tools, t = engine.cfg, engine.source, engine.tools, engine.cfg.thresholds
    controller = engine.controller
    streams = []
    for s in engine.streams:
        info = s.info
        streams.append({
            "id": info.id, "scenario": info.scenario_id, "scenario_label": info.scenario_label,
            "platform": info.platform_id, "output": info.output,
            "encode": {"hw": info.encode_hw, "encoder": info.encoder},
            "decode": {"hw": info.decode_hw}, "tonemap": info.tonemap or None,
            "status": s.status, "started_at": _iso(s.started_at), "ended_at": _iso(s.ended_at),
            "frames": s.frames, "fps": s.fps_summary.to_dict(),
            "output_mbps": _round(engine.output_mbps(info.id), 2),
            "error": s.error or None,
        })
    kept_up = None
    if controller is None:  # fixed mode: did each stream's median keep the real-time threshold?
        threshold = src.fps * t.fps_ratio
        medians = [s.fps_summary.percentile(0.5) for s in engine.streams]
        kept_up = sum(1 for m in medians if m is not None and m >= threshold)
    ended = engine.ended_at or engine._wall()
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "jf-hw-stress", "version": __version__},
        "started_at": _iso(engine.started_at), "ended_at": _iso(ended),
        "duration_secs": round(ended - engine.started_at, 1) if engine.started_at else 0.0,
        "host": {"os": host_platform.system(), "os_version": host_platform.release(),
                 "machine": host_platform.machine(), "python": sys.version.split()[0]},
        "ffmpeg": {"path": tools.ffmpeg, "ffprobe": tools.ffprobe, "version": tools.version, "origin": tools.origin},
        "platforms": [{"id": p.id, "name": p.name, "device": p.device, "tested": p.tested} for p in cfg.platforms],
        "source": src.to_dict(),
        "config": {
            "mode": cfg.mode,
            "scenario": None if cfg.mode == "mixed" else cfg.scenario.id,
            "scenario_pool": [s.id for s in cfg.pool] if cfg.mode == "mixed" else None,
            "fixed_streams": cfg.streams if cfg.mode == "fixed" else None,
            "hw_decode": cfg.hw_decode, "audio": cfg.options.audio, "loop": cfg.options.loop,
            "realtime": cfg.options.realtime,
            "hls": {"segment_secs": cfg.options.hls_segment_secs, "list_size": cfg.options.hls_list_size},
            "thresholds": {"fps_ratio": t.fps_ratio, "threshold_fps": round(src.fps * t.fps_ratio, 2),
                           "ramp_interval_secs": t.ramp_interval, "fail_secs": t.fail_secs,
                           "warmup_secs": t.warmup_secs},
            "duration_limit_secs": cfg.duration, "max_streams": cfg.max_streams, "seed": cfg.seed,
        },
        "result": {
            "stop_reason": engine.stop_reason or None,
            "limit_reached": engine.stop_reason in ("limit-reached", "max-streams"),
            "max_stable_streams": engine.max_stable(),
            "hw_max_stable_streams": engine.hw_max_stable,
            "streams_keeping_up": kept_up,
            "peak_concurrent_streams": engine.peak_concurrent,
            "peak_combined_fps": round(engine.peak_combined_fps, 1),
            "avg_output_mbps": _round(engine.avg_output_mbps(), 2),
            "failure": engine.failure,
        },
        "io": {"segments_written_mb": round(engine.io.total_bytes / 1e6, 1),
               "peak_write_mb_s": round(engine.io.peak_rate / 1e6, 2), "work_dir": engine.work_dir},
        "streams": streams,
        "timeline": engine.timeline,
        "events": [{"t": e.t, "message": e.message} for e in engine.events],
        "warnings": list(warnings or []) + engine.warnings,
    }
    return report


# --- HTML ------------------------------------------------------------------------------------

CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;
--good:#3fb950;--warn:#d29922;--bad:#f85149;--hdr:#bc8cff}
*{box-sizing:border-box}body{margin:0;padding:32px 24px;background:var(--bg);color:var(--text);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
h1{margin:0;font-size:22px;color:var(--accent)}h2{margin:0 0 12px;font-size:12px;letter-spacing:.08em;
text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:8px}
.meta{color:var(--muted);font-size:12px;margin:6px 0 28px}.grid{display:grid;gap:20px;
grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}.panel{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:18px 22px;margin-bottom:20px}.kv{display:grid;grid-template-columns:max-content 1fr;
gap:4px 16px}.kv dt{color:var(--muted)}.kv dd{margin:0;word-break:break-word}code{font:12px "SF Mono",Menlo,Consolas,monospace;
color:var(--accent)}.big{font-size:44px;font-weight:700;color:var(--good);line-height:1}.big small{font-size:14px;
color:var(--muted);font-weight:400;margin-left:8px}table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:6px 8px;border-bottom:1px solid var(--line)}.badge{display:inline-block;padding:1px 7px;border-radius:4px;
font-size:11px;font-weight:600}.hw{background:#3fb95026;color:var(--good)}.sw{background:#d2992226;color:var(--warn)}
.err{background:#f8514926;color:var(--bad)}.hdr{background:#bc8cff26;color:var(--hdr)}.note{color:var(--warn)}
svg{width:100%;height:180px}
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _kv(rows: List[tuple]) -> str:
    return "<dl class='kv'>" + "".join(f"<dt>{_e(k)}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def _chart(timeline: List[Dict], threshold: float) -> str:
    """Combined fps (line) and stream count (steps) over time, as inline SVG."""
    points = [p for p in timeline if p.get("t") is not None]
    if len(points) < 2:
        return "<p class='meta'>Not enough samples for a chart.</p>"
    w, h, pad = 800.0, 180.0, 24.0
    t_max = max(p["t"] for p in points) or 1.0
    f_max = max(max(p["combined_fps"] or 0 for p in points), 1.0)
    n_max = max(max(p["streams"] for p in points), 1)
    x = lambda t: pad + (w - 2 * pad) * t / t_max
    yf = lambda f: h - pad - (h - 2 * pad) * f / f_max
    yn = lambda n: h - pad - (h - 2 * pad) * n / n_max
    fps_line = " ".join(f"{x(p['t']):.1f},{yf(p['combined_fps'] or 0):.1f}" for p in points)
    steps = " ".join(f"{x(p['t']):.1f},{yn(p['streams']):.1f}" for p in points)
    min_line = " ".join(f"{x(p['t']):.1f},{yf(p['min_fps']):.1f}" for p in points if p.get("min_fps") is not None)
    thr = yf(threshold) if threshold <= f_max else None
    return (f"<svg viewBox='0 0 {w:.0f} {h:.0f}' preserveAspectRatio='none' role='img' "
            f"aria-label='Combined frame rate and stream count over time'>"
            f"<polyline points='{steps}' fill='none' stroke='#8b949e' stroke-width='1.5' stroke-dasharray='4 3'/>"
            f"<polyline points='{fps_line}' fill='none' stroke='#58a6ff' stroke-width='2'/>"
            + (f"<polyline points='{min_line}' fill='none' stroke='#d29922' stroke-width='1'/>" if min_line else "")
            + (f"<line x1='{pad}' x2='{w - pad}' y1='{thr:.1f}' y2='{thr:.1f}' stroke='#f85149' stroke-width='1' "
               f"stroke-dasharray='2 4'/>" if thr is not None else "")
            + f"<text x='{pad}' y='14' fill='#8b949e' font-size='11'>peak {f_max:.0f} fps combined · up to {n_max} streams"
              f" · {t_max:.0f} s</text></svg>"
            "<p class='meta'>Blue: combined fps. Orange: slowest stream. Grey dashes: streams running. "
            "Red: per-stream real-time threshold.</p>")


def _mbps(value: Optional[float], cfg: Dict[str, Any]) -> str:
    if value is None:
        return "per stream (mixed targets)" if cfg["mode"] == "mixed" else "—"
    return _e(f"{value:.1f} Mbps per stream (video + audio, as delivered)")


def render_html(report: Dict[str, Any]) -> str:
    src, cfg, res, ff = report["source"], report["config"], report["result"], report["ffmpeg"]
    thr = cfg["thresholds"]
    platforms = ", ".join(f"{p['name']} ({p['device']})" + ("" if p["tested"] else " — untested backend")
                          for p in report["platforms"])
    if res["max_stable_streams"] is not None:
        hw_max = res.get("hw_max_stable_streams")
        headline = (f"<div class='big'>{_e(res['max_stable_streams'])}<small>stable concurrent streams"
                    + (f" · {_e(hw_max)} on hardware" if hw_max is not None else "")
                    + "</small></div>")
    else:
        headline = (f"<div class='big'>{_e(res['streams_keeping_up'])} / {_e(cfg['fixed_streams'])}"
                    "<small>streams kept up with real time (median fps)</small></div>")
    failure = res.get("failure")
    summary = headline + (f"<p class='note'>{_e(failure['reason'])}</p>" if failure else "") + _kv([
        ("Stopped because", _e(res["stop_reason"])),
        ("Peak streams", _e(res["peak_concurrent_streams"])),
        ("Peak combined fps", _e(res["peak_combined_fps"])),
        ("Average output bitrate", _mbps(res.get("avg_output_mbps"), cfg)),
        ("Real-time threshold", f"{_e(thr['threshold_fps'])} fps per stream ({_e(round(thr['fps_ratio'] * 100))}% of source)"),
    ])
    hdr = (f"<span class='badge hdr'>{_e(src['hdr_type'])}"
           + (f" profile {_e(src['dovi_profile'])}" if src.get("dovi_profile") else "") + "</span>") if src["is_hdr"] else "SDR"
    video = (f"{src['width']}×{src['height']} {src['codec'].upper()} {src['profile']}, "
             f"{src['bit_depth']}-bit ({src['pix_fmt']})")
    source = _kv([
        ("File", f"<code>{_e(src['path'])}</code>"),
        ("Video", _e(video)),
        ("Frame rate", _e(f"{src['fps']} fps")), ("Bitrate", _e(f"{src['bitrate_mbps']} Mbps")),
        ("Dynamic range", hdr), ("Transfer / primaries", _e(f"{src['color_transfer']} / {src['color_primaries']}")),
        ("Audio", _e(src.get("audio_codec") or "none")),
    ])
    setup = _kv([
        ("Mode", _e(MODE_LABELS.get(cfg["mode"], cfg["mode"]))),
        ("Scenario", _e(cfg["scenario"] or ", ".join(cfg["scenario_pool"] or []))),
        ("Platforms", _e(platforms)),
        ("ffmpeg", f"<code>{_e(ff['version'])}</code> {_e(ff['origin'])}"),
        ("Decode", "hardware when supported" if cfg["hw_decode"] else "software"),
        ("Audio", _e(cfg["audio"])), ("Source read", "real-time (-re)" if cfg["realtime"] else "as fast as possible"),
        ("HLS segment", _e(f"{cfg['hls']['segment_secs']} s")),
        ("Ramp / fail / warm-up", _e(f"{thr['ramp_interval_secs']} s / {thr['fail_secs']} s / {thr['warmup_secs']} s")),
    ])
    rows = []
    for s in report["streams"]:
        f = s["fps"]
        enc = f"<span class='badge {'hw' if s['encode']['hw'] else 'sw'}'>{'HW' if s['encode']['hw'] else 'SW'}</span>"
        dec = f"<span class='badge {'hw' if s['decode']['hw'] else 'sw'}'>{'HW' if s['decode']['hw'] else 'SW'}</span>"
        status = f"<span class='badge err'>{_e(s['status'])}</span>" if s["status"] == "error" else _e(s["status"])
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in (
            _e(s["id"]), _e(s["scenario_label"]), _e(s["platform"]), _e(s["encode"]["encoder"]), enc, dec,
            _e(s["tonemap"] or "—"), status, _e(f["p50"]), _e(f["p5"]), _e(f["min"]), _e(f"{s['frames']:,}"),
            _e(s.get("output") or ""), _e("—" if s.get("output_mbps") is None else s["output_mbps"]))) + "</tr>")
        if s.get("error"):
            rows.append(f"<tr><td></td><td colspan='13' class='note'>{_e(s['error'])}</td></tr>")
    table = ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in (
        "#", "Scenario", "Platform", "Encoder", "Enc", "Dec", "Tone map", "Status", "Median fps", "5th pct fps",
        "Min fps", "Frames", "Target", "Output Mbps")) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    events = "".join(f"<tr><td>{_e(e['t'])} s</td><td>{_e(e['message'])}</td></tr>" for e in report["events"])
    warnings = "".join(f"<p class='note'>{_e(w)}</p>" for w in report["warnings"])
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>jf-hw-stress report — {_e(report['started_at'])}</title><style>{CSS}</style></head><body>"
            f"<h1>Jellyfin transcode stress test</h1><div class='meta'>{_e(report['started_at'])} · "
            f"{_e(report['duration_secs'])} s · jf-hw-stress {_e(report['tool']['version'])} · "
            f"{_e(report['host']['os'])} {_e(report['host']['machine'])}</div>{warnings}"
            f"<div class='grid'><div class='panel'><h2>Result</h2>{summary}</div>"
            f"<div class='panel'><h2>Source</h2>{source}</div><div class='panel'><h2>Setup</h2>{setup}</div></div>"
            f"<div class='panel'><h2>Over time</h2>{_chart(report['timeline'], thr['threshold_fps'])}</div>"
            f"<div class='panel'><h2>Streams</h2>{table}</div>"
            f"<div class='panel'><h2>Events</h2><table><tbody>{events}</tbody></table></div>"
            f"</body></html>")
