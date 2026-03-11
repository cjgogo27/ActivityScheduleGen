

import re
import json
import warnings
import numpy as np
from scipy.spatial import distance as sp_dist
from sklearn.metrics import f1_score

ACTIVITY_CODE = {
    "home": 1, "work": 2, "education": 3, "shopping": 4, "service": 5,
    "medical": 6, "dine_out": 7, "socialize": 8, "exercise": 9, "dropoff_pickup": 10,
}
N_ACTIVITIES = 11      
SLOT_MINUTES = 15      
N_SLOTS      = 96    

_JSON_RE   = re.compile(r"\[JSON\](.*?)\[/JSON\]",     re.DOTALL)
_THOUGHT_RE = re.compile(r"\[THOUGHT\].*?\[/THOUGHT\]", re.DOTALL)



def _parse_completion(completion) -> str:

    if isinstance(completion, list):
        return completion[0]["content"]
    return str(completion)


def _time_to_minutes(t: str) -> int:
    try:
        s = str(t).strip()
        if s in ("24:00", "23:59"):
            return 1440
        sep = ":" if ":" in s else "."
        h, m = s.split(sep, 1)
        return min(int(float(h)) * 60 + int(float(m)), 1440)
    except Exception:
        return 0


def _schedule_to_slots(schedule: list) -> np.ndarray:
    slots = np.zeros(N_SLOTS, dtype=int)
    for i in range(N_SLOTS):
        slot_s, slot_e = i * SLOT_MINUTES, (i + 1) * SLOT_MINUTES
        durations: dict = {}
        for seg in schedule:
            seg_s = _time_to_minutes(seg.get("start_time", "00:00"))
            seg_e = _time_to_minutes(seg.get("end_time",   "24:00"))
            overlap = min(slot_e, seg_e) - max(slot_s, seg_s)
            if overlap > 0:
                act = seg.get("activity", "home")
                durations[act] = durations.get(act, 0) + overlap
        if durations:
            slots[i] = ACTIVITY_CODE.get(max(durations, key=durations.get), 0)
    return slots


def _extract_schedule(text: str):
    """Parse the [JSON]...[/JSON] block; return list or None on failure."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass
    return None


def _segment_lengths(slots: np.ndarray) -> np.ndarray:
    mat = np.zeros((N_ACTIVITIES, N_SLOTS), dtype=float)
    cur_act, cur_len = int(slots[0]), 1
    for a in slots[1:]:
        a = int(a)
        if a == cur_act:
            cur_len += 1
        else:
            mat[cur_act, cur_len - 1] += 1
            cur_act, cur_len = a, 1
    mat[cur_act, cur_len - 1] += 1
    return mat


def _similarity(gen: list, gt: list) -> float:
    gen_s = _schedule_to_slots(gen)
    gt_s  = _schedule_to_slots(gt)

    acc = float(np.sum(gen_s == gt_s)) / N_SLOTS

    gen_p = np.array([np.sum(gen_s == i) for i in range(N_ACTIVITIES)], dtype=float)
    gt_p  = np.array([np.sum(gt_s  == i) for i in range(N_ACTIVITIES)], dtype=float)
    gen_p /= gen_p.sum() + 1e-9
    gt_p  /= gt_p.sum()  + 1e-9
    act_sim = 1.0 - min(float(sp_dist.jensenshannon(gen_p, gt_p)), 1.0)

    gen_mat = _segment_lengths(gen_s)
    gt_mat  = _segment_lengths(gt_s)
    g_sum, t_sum = gen_mat.sum(), gt_mat.sum()
    if g_sum > 0 and t_sum > 0:
        int_sim = 1.0 - min(
            float(sp_dist.jensenshannon(gen_mat.sum(0) / g_sum,
                                        gt_mat.sum(0)  / t_sum)), 1.0)
    else:
        int_sim = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = float(f1_score(gt_s, gen_s, average="macro", zero_division=0))

    return 0.40 * acc + 0.25 * act_sim + 0.25 * int_sim + 0.10 * f1

def reward_format(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    scores = []
    for c in completions:
        text = _parse_completion(c)
        has_thought = bool(_THOUGHT_RE.search(text))
        has_json    = bool(_JSON_RE.search(text))
        if has_thought and has_json:
            scores.append(1.0)
        elif has_thought or has_json:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return scores


def reward_schedule_similarity(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    if ground_truth_schedule is None:
        return [0.0] * len(completions)

    scores = []
    for c, gt_raw in zip(completions, ground_truth_schedule):
        text = _parse_completion(c)
        gen  = _extract_schedule(text)
        if gen is None:
            scores.append(0.0)
            continue
        gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        try:
            scores.append(_similarity(gen, gt))
        except Exception:
            scores.append(0.0)
    return scores


def reward_constraints(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    scores = []
    for c in completions:
        text     = _parse_completion(c)
        schedule = _extract_schedule(text)
        if not schedule:
            scores.append(0.0)
            continue
        try:
            segs = sorted(
                [(_time_to_minutes(s.get("start_time", "00:00")),
                  _time_to_minutes(s.get("end_time",   "24:00")))
                 for s in schedule],
                key=lambda x: x[0],
            )
            full_day   = segs[0][0] == 0 and segs[-1][1] == 1440
            continuous = all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1))
            valid_dur  = all(5 <= e - s <= 720 for s, e in segs)
            scores.append((float(full_day) + float(continuous) + float(valid_dur)) / 3.0)
        except Exception:
            scores.append(0.0)
    return scores
