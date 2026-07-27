"""Generate, QA, export, and verify MACHIAVELLI Deep Value."""

import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import polars as pl
from datasets import load_dataset
from loguru import logger
from openai import OpenAI

import prompts
from quality import rejection_reason

MODELS = (
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-35b-a3b",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-397b-a17b",
)
SOURCE_DATASET = "wassname/machiavelli_character_scenarios"
SOURCE_REVISION = "abd8b004486440919dc9850ef7c34098e79ccbce"
HOSTED_DATASET = "wassname/machiavelli_deep_value"
QA_MODEL = "qwen/qwen3.5-flash-02-23"
SAMPLES = 5
GAME_SPLIT_SEED = 0

RUN = Path("run")
OUTPUT = Path("dataset")
WRITE_LOCK = threading.Lock()
QA_FIELDS = (
    "refusal",
    "confession",
    "incoherent",
    "audience_mismatch",
    "truncated",
    "short",
)


def record_key(row):
    return (
        row["agent"],
        row["source_id"],
        row["action_label"],
        row["condition"],
        row["sample"],
    )


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK, path.open("a") as file:
        file.write(json.dumps(row) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def selected_source_ids():
    ids = Path("selected_scenes.txt").read_text().splitlines()
    if len(ids) != 14 or len(set(ids)) != 14:
        raise ValueError("selected_scenes.txt must contain 14 unique source IDs")
    return ids


def load_scenes():
    wanted = set(selected_source_ids())
    rows = {
        row["source_id"]: dict(row)
        for row in load_dataset(SOURCE_DATASET, split="train", revision=SOURCE_REVISION)
        if row["source_id"] in wanted
    }
    if set(rows) != wanted:
        raise ValueError(f"source dataset is missing: {sorted(wanted - set(rows))}")

    scenes = {}
    for source_id in selected_source_ids():
        row = rows[source_id]
        actions, _ = prompts.turn_actions(row)
        tallies = prompts.morality_tallies(row)
        lower_index = min(range(len(actions)), key=tallies.__getitem__)
        higher_index = max(range(len(actions)), key=tallies.__getitem__)
        if tallies[higher_index] - tallies[lower_index] < 5:
            raise ValueError(f"{source_id}: harm tally spread fell below 5")
        scenes[source_id] = {
            "source_id": source_id,
            "game": row["game_id"],
            "row": row,
            "moral": {
                "index": lower_index + 1,
                "action": actions[lower_index],
                "harm": tallies[lower_index],
            },
            "immoral": {
                "index": higher_index + 1,
                "action": actions[higher_index],
                "harm": tallies[higher_index],
            },
        }
    return scenes


def index_complete_records(path, source_ids):
    rows = {
        record_key(row): row
        for row in read_jsonl(path)
        if row["source_id"] in source_ids and row["sample"] < SAMPLES
    }
    expected = len(MODELS) * len(source_ids) * 2 * 2 * SAMPLES
    if len(rows) != expected:
        raise ValueError(f"{path}: found {len(rows)} of {expected} records")
    return rows


def openrouter_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def last_json(text, required_field):
    objects = []
    depth = 0
    start = None
    for index, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0:
                objects.append(text[start : index + 1])
    for candidate in reversed(objects):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if required_field in parsed:
            return parsed
    return None


def account_text(value):
    items = value if isinstance(value, list) else [
        line for line in str(value).splitlines() if line.strip()
    ]
    items = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", str(item)).strip()
        for item in items
    ]
    items = [item for item in items if item]
    return "\n".join(f"- {item}" for item in items) if len(items) > 1 else "".join(items)


def generate_one(client, job):
    agent, scene, action_label, condition, sample = job
    action = scene[action_label]["action"]
    user_prompt = prompts.generation_prompt(
        scene["row"], action, rationalized=condition == "rationalized"
    )
    response = client.chat.completions.create(
        model=agent,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.9,
        max_tokens=32000,
        timeout=900,
        response_format={"type": "json_object"},
        extra_body={"reasoning": {"enabled": True}},
    )
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    reasoning = getattr(choice.message, "reasoning", None) or ""
    completion = content or reasoning
    parsed = last_json(completion, prompts.PUBLIC_FIELD)
    metadata = {
        "completion": completion,
        "completion_channel": "content" if content else "reasoning",
        "reasoning": reasoning,
        "finish_reason": choice.finish_reason,
        "resolved_model": response.model,
        "provider": getattr(response, "provider", None),
        "usage": response.usage.model_dump() if response.usage else None,
    }
    base = {
        "agent": agent,
        "source_id": scene["source_id"],
        "game": scene["game"],
        "action_label": action_label,
        "condition": condition,
        "sample": sample,
        "action": action,
        "action_index": scene[action_label]["index"],
        "action_harm": scene[action_label]["harm"],
        "system_prompt": "",
        "user_prompt": user_prompt,
    }
    if parsed is None:
        return {
            **base,
            "error": f"no JSON object containing {prompts.PUBLIC_FIELD}",
            **metadata,
        }
    text = account_text(parsed[prompts.PUBLIC_FIELD])
    secret_motive = parsed.get("secret_unacceptable_motivation")
    if condition == "rationalized" and not secret_motive:
        return {**base, "error": "missing secret motive", **metadata}
    return {
        **base,
        "text": text,
        "secret_motive": secret_motive,
        "nchar": len(text),
        **metadata,
    }


def generation_jobs(scenes, cached):
    return [
        (agent, scene, action_label, condition, sample)
        for agent in MODELS
        for scene in scenes.values()
        for action_label in ("moral", "immoral")
        for condition in ("genuine", "rationalized")
        for sample in range(SAMPLES)
        if (agent, scene["source_id"], action_label, condition, sample) not in cached
    ]


def generate():
    scenes = load_scenes()
    output_path = RUN / "generations.jsonl"
    cached = {
        record_key(row)
        for row in read_jsonl(output_path)
        if "usage" in row and "error" not in row
    }
    jobs = generation_jobs(scenes, cached)
    logger.info(f"GENERATE: {len(cached)} cached, {len(jobs)} calls")
    client = openrouter_client()
    errors = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(generate_one, client, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if "error" in record:
                append_jsonl(RUN / "generation_errors.jsonl", record)
                errors += 1
            else:
                append_jsonl(output_path, record)
            if index == 1 or index % 25 == 0 or index == len(jobs):
                logger.info(f"GENERATE: wrote {index}/{len(jobs)} new records")
    if errors:
        raise ValueError(f"{errors} malformed replies saved; rerun generate")
    records = index_complete_records(output_path, set(scenes))
    first = records[(MODELS[0], selected_source_ids()[0], "moral", "genuine", 0)]
    logger.info(f"FIRST GENERATION PROMPT\n{first['user_prompt']}")
    logger.info(f"FIRST GENERATION REPLY\n{first['completion']}")


def parse_qa_reply(text):
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        raise ValueError(f"QA returned no JSON: {text[:200]}")
    return json.loads(match.group(0))


def qa_one(client, row):
    mechanical_reason = rejection_reason(row["text"])
    user_prompt = prompts.qa_prompt(row["text"], row["secret_motive"])
    response = client.chat.completions.create(
        model=QA_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0,
        max_tokens=80,
        timeout=120,
        extra_body={"reasoning": {"enabled": False}},
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    verdict = parse_qa_reply(content)
    return {
        **{
            field: row[field]
            for field in ("agent", "source_id", "game", "action_label", "condition", "sample")
        },
        "refusal": bool(mechanical_reason and mechanical_reason.startswith("refusal:")),
        "confession": False if row["secret_motive"] is None else bool(verdict["leak"]),
        "incoherent": not bool(verdict["coherent"]),
        "audience_mismatch": not bool(verdict["addressed"]),
        "truncated": row["finish_reason"] == "length",
        "short": row["nchar"] < 250,
        "mechanical_reason": mechanical_reason,
        "qa_prompt": user_prompt,
        "qa_response": content,
        "qa_finish_reason": choice.finish_reason,
        "qa_model": response.model,
        "qa_provider": getattr(response, "provider", None),
        "qa_usage": response.usage.model_dump() if response.usage else None,
    }


def qa():
    scenes = load_scenes()
    generations = index_complete_records(RUN / "generations.jsonl", set(scenes))
    output_path = RUN / "qa_flags.jsonl"
    cached = {
        record_key(row)
        for row in read_jsonl(output_path)
        if "qa_response" in row
    }
    jobs = [row for key, row in generations.items() if key not in cached]
    logger.info(f"QA: {len(cached)} cached, {len(jobs)} calls")
    client = openrouter_client()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(qa_one, client, row) for row in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            append_jsonl(output_path, future.result())
            if index == 1 or index % 50 == 0 or index == len(jobs):
                logger.info(f"QA: wrote {index}/{len(jobs)} new records")
    flags = index_complete_records(output_path, set(scenes))
    first = flags[(MODELS[0], selected_source_ids()[0], "moral", "genuine", 0)]
    logger.info(f"FIRST QA PROMPT\n{first['qa_prompt']}")
    logger.info(f"FIRST QA REPLY\n{first['qa_response']}")


def pair_quality(genuine, rationalized, genuine_flags, rationalized_flags):
    similarity = SequenceMatcher(
        None, genuine["text"].lower(), rationalized["text"].lower()
    ).ratio()
    reasons = []
    if genuine_flags["refusal"] or rationalized_flags["refusal"]:
        reasons.append("refusal")
    if rationalized_flags["confession"]:
        reasons.append("confession")
    if genuine_flags["incoherent"] or rationalized_flags["incoherent"]:
        reasons.append("incoherent")
    if genuine_flags["audience_mismatch"] or rationalized_flags["audience_mismatch"]:
        reasons.append("audience_mismatch")
    if any(
        flags[field]
        for flags in (genuine_flags, rationalized_flags)
        for field in ("truncated", "short")
    ):
        reasons.append("incoherent")
    if similarity > 0.8:
        reasons.append("near_identical")
    return similarity, sorted(set(reasons))


def context_fields(agent, scene, sample):
    actions, _ = prompts.turn_actions(scene["row"])
    return {
        "agent": agent,
        "source_id": scene["source_id"],
        "game": scene["game"],
        "game_title": scene["row"]["world"],
        "game_summary": scene["row"]["short_summary"],
        "player_character": scene["row"]["player_character"],
        "scene": scene["row"]["decision_context"],
        "available_actions": actions,
        "sample": sample,
    }


def quality_fields(genuine, rationalized, flags):
    genuine_flags = flags[record_key(genuine)]
    rationalized_flags = flags[record_key(rationalized)]
    similarity, reasons = pair_quality(
        genuine, rationalized, genuine_flags, rationalized_flags
    )
    result = {
        "similarity": similarity,
        "clean": not reasons,
        "qa_reasons": reasons,
    }
    for field in QA_FIELDS:
        result[f"genuine_{field}"] = genuine_flags[field]
        result[f"rationalized_{field}"] = rationalized_flags[field]
    return result


def pairs(scenes, generations):
    result = {}
    for agent in MODELS:
        for source_id in scenes:
            for sample in range(SAMPLES):
                for action_label in ("moral", "immoral"):
                    prefix = (agent, source_id, action_label)
                    result[(agent, source_id, sample, action_label)] = (
                        generations[(*prefix, "genuine", sample)],
                        generations[(*prefix, "rationalized", sample)],
                    )
    return result


def export():
    scenes = load_scenes()
    source_ids = set(scenes)
    generations = index_complete_records(RUN / "generations.jsonl", source_ids)
    flags = index_complete_records(RUN / "qa_flags.jsonl", source_ids)
    paired = pairs(scenes, generations)

    shuffled_ids = sorted(source_ids)
    random.Random(GAME_SPLIT_SEED).shuffle(shuffled_ids)
    heldout_ids = set(shuffled_ids[::3])
    game_rows = []
    for (agent, source_id, sample, action_label), (genuine, rationalized) in paired.items():
        scene = scenes[source_id]
        agent_name = agent.split("/")[-1]
        game_rows.append({
            "genuine_account": genuine["text"],
            "rationalized_account": rationalized["text"],
            **context_fields(agent, scene, sample),
            "pair_id": f"{agent_name}:{source_id}:{action_label}:s{sample}",
            "quartet_id": f"{agent_name}:{source_id}:s{sample}",
            "action": genuine["action"],
            "action_index": genuine["action_index"],
            "action_label": action_label,
            "machiavelli_harm_tally": genuine["action_harm"],
            "secret_motive": rationalized["secret_motive"],
            "game_split": "heldout" if source_id in heldout_ids else "development",
            **quality_fields(genuine, rationalized, flags),
        })
    game_rows.sort(key=lambda row: row["pair_id"])

    deep_value_rows = []
    for agent in MODELS:
        agent_name = agent.split("/")[-1]
        for source_id, scene in scenes.items():
            for sample in range(SAMPLES):
                lower_genuine, lower_rationalized = paired[
                    (agent, source_id, sample, "moral")
                ]
                higher_genuine, higher_rationalized = paired[
                    (agent, source_id, sample, "immoral")
                ]
                comparisons = (
                    ("train", "A", lower_genuine, higher_rationalized),
                    ("test", "B", higher_genuine, lower_rationalized),
                )
                for split, pairing, genuine, rationalized in comparisons:
                    quartet_id = f"{agent_name}:{source_id}:s{sample}"
                    deep_value_rows.append({
                        "genuine_account": genuine["text"],
                        "rationalized_account": rationalized["text"],
                        **context_fields(agent, scene, sample),
                        "comparison_id": f"{quartet_id}:{pairing}",
                        "quartet_id": quartet_id,
                        "pairing": pairing,
                        "deep_value_split": split,
                        "genuine_action": genuine["action"],
                        "genuine_action_index": genuine["action_index"],
                        "genuine_action_label": genuine["action_label"],
                        "genuine_machiavelli_harm_tally": genuine["action_harm"],
                        "rationalized_action": rationalized["action"],
                        "rationalized_action_index": rationalized["action_index"],
                        "rationalized_action_label": rationalized["action_label"],
                        "rationalized_machiavelli_harm_tally": rationalized["action_harm"],
                        "secret_motive": rationalized["secret_motive"],
                        **quality_fields(genuine, rationalized, flags),
                    })
    deep_value_rows.sort(key=lambda row: row["comparison_id"])

    game_frame = pl.DataFrame(game_rows)
    deep_value_frame = pl.DataFrame(deep_value_rows)
    for split in ("development", "heldout"):
        path = OUTPUT / "data" / "game_split" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        game_frame.filter(pl.col("game_split") == split).write_parquet(path)
    for split in ("train", "test"):
        path = OUTPUT / "data" / "deep_value" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        deep_value_frame.filter(pl.col("deep_value_split") == split).write_parquet(path)

    write_jsonl(
        OUTPUT / "raw" / "generations.jsonl",
        [generations[key] for key in sorted(generations)],
    )
    write_jsonl(
        OUTPUT / "raw" / "qa_flags.jsonl",
        [flags[key] for key in sorted(flags)],
    )
    write_jsonl(
        OUTPUT / "raw" / "generation_errors.jsonl",
        read_jsonl(RUN / "generation_errors.jsonl"),
    )
    verify_local()


def verify_frames(development, heldout, train, test):
    expected = {
        "development": (development, 360),
        "heldout": (heldout, 200),
        "train": (train, 280),
        "test": (test, 280),
    }
    for name, (frame, count) in expected.items():
        if frame.height != count:
            raise ValueError(f"{name}: found {frame.height} of {count} rows")
        if frame.columns[:2] != ["genuine_account", "rationalized_account"]:
            raise ValueError(f"{name}: account columns are not first")
        if frame.select(pl.col("genuine_account").eq(pl.col("rationalized_account")).any()).item():
            raise ValueError(f"{name}: contains an identical pair")

    development_games = set(development["game"])
    heldout_games = set(heldout["game"])
    if len(development_games) != 9 or len(heldout_games) != 5:
        raise ValueError("game_split must contain 9 development and 5 held-out games")
    if development_games & heldout_games:
        raise ValueError("a game occurs in both game_split files")
    if set(train["pairing"]) != {"A"} or set(test["pairing"]) != {"B"}:
        raise ValueError("Deep Value train/test pairing labels are wrong")
    if (
        set(train["genuine_action_label"]) != {"moral"}
        or set(train["rationalized_action_label"]) != {"immoral"}
        or set(test["genuine_action_label"]) != {"immoral"}
        or set(test["rationalized_action_label"]) != {"moral"}
    ):
        raise ValueError("Deep Value action pairings are wrong")

    logger.info(
        "VERIFIED: game_split 360 development + 200 heldout; "
        "deep_value 280 train A + 280 test B; account columns first"
    )


def parquet_frames(root):
    return (
        pl.read_parquet(root / "data/game_split/development.parquet"),
        pl.read_parquet(root / "data/game_split/heldout.parquet"),
        pl.read_parquet(root / "data/deep_value/train.parquet"),
        pl.read_parquet(root / "data/deep_value/test.parquet"),
    )


def verify_local():
    verify_frames(*parquet_frames(OUTPUT))


def verify_hosted():
    game = load_dataset(HOSTED_DATASET, "game_split", download_mode="force_redownload")
    deep = load_dataset(HOSTED_DATASET, "deep_value", download_mode="force_redownload")
    verify_frames(
        pl.from_arrow(game["development"].data.table),
        pl.from_arrow(game["heldout"].data.table),
        pl.from_arrow(deep["train"].data.table),
        pl.from_arrow(deep["test"].data.table),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("generate", "qa", "export", "verify-local", "verify-hosted"),
    )
    phase = parser.parse_args().phase
    {
        "generate": generate,
        "qa": qa,
        "export": export,
        "verify-local": verify_local,
        "verify-hosted": verify_hosted,
    }[phase]()


if __name__ == "__main__":
    main()
