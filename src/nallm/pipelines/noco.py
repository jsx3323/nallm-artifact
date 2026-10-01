"""LLM-NoCo 跑批引擎(rule_first 链路)。

下沉自 scripts/baselines/noco.py。保留自己的 provider(AsyncOpenAI,
OPENCODE_GO 网关)与 asyncio.Semaphore 并发。
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from nallm.classification.candidates import candidate_sort_key_v3, format_candidates
from nallm.core.config import DATA_DIR, RESULTS_DIR
from nallm.disambiguation.parser import parse_output
from nallm.disambiguation.prompts import LLM_NOCO_SYSTEM_PROMPT
from nallm.utils.io import load_json


class LLMRunner:
    """OpenAI SDK 封装，与 llm_co.py 风格一致。"""

    def __init__(self, model="deepseek-v4-flash", api_key=None, base_url=None):
        self.model = model
        # 走 OPENCODE_GO 网关
        api_key = api_key or os.environ.get("OPENCODE_GO_API_KEY", "")
        base_url = base_url or os.environ.get("OPENCODE_GO_API_BASE", "")
        if not api_key or not base_url:
            raise ValueError("需要设置 OPENCODE_GO_API_KEY/BASE")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=10)

    async def call(self, system_prompt, human_text):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_text},
        ]
        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    timeout=httpx.Timeout(connect=30, read=600, write=30, pool=30),
                )
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                usage = resp.usage
                return content, {
                    "input_tokens": usage.prompt_tokens or 0,
                    "output_tokens": usage.completion_tokens or 0,
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    return str(e), {}
        return "", {}


async def run_one(runner, uid, candidates_text, aid_map, sem, gt_aid):
    async with sem:
        human = f"候选作者列表:\n{candidates_text}\n\n待消歧论文信息: （简略信息）"
        content, usage = await runner.call(LLM_NOCO_SYSTEM_PROMPT, human)

        pred = "UNFINISHED"
        status = "ok"
        if content and not content.startswith(("err:", "429:", "conn:", "api(")):
            parsed = parse_output(content)
            pred = getattr(parsed, "author_id", None) or "PARSE_FAIL"
            if pred == "PARSE_FAIL":
                status = "parse_fail"
            # 编号 → aid 映射
            if pred and pred != "none" and pred not in aid_map.values():
                if pred in aid_map:
                    pred = aid_map[pred]
        else:
            status = "unfinished"

        match = str(pred).lower() == str(gt_aid).lower() if gt_aid else False
        return {
            "case_id": uid,
            "gt": gt_aid or "none",
            "pred": pred,
            "match": match,
            "status": status,
            "raw": content[:200] if isinstance(content, str) else str(content)[:200],
            "in_tokens": usage.get("input_tokens", 0),
            "out_tokens": usage.get("output_tokens", 0),
        }


async def run_llm_noco(
    split: str = "valid",
    *,
    concurrency: int = 10,
    model: str | None = None,
    force: bool = False,
    out_path: Path | None = None,
) -> list[dict]:
    """跑全量 LLM-NoCo 案例,输出 {split}_noco_co_prompt.json。"""
    model = model or os.environ.get("OPENCODE_GO_MODEL", "deepseek-v4-flash")
    print(f"split={split} conc={concurrency} model={model}")

    out_path = out_path or (RESULTS_DIR / f"{split}_noco_co_prompt.json")
    gt_path = DATA_DIR / split / f"cna_{split}_unass_gt.json"
    cands_path = RESULTS_DIR / f"{split}_filtered_candidates.json"
    classified_path = RESULTS_DIR / f"{split}_unass_classified.csv"

    gt = load_json(gt_path) if gt_path.exists() else {}
    cands_data = load_json(cands_path)

    print(f"GT: {len(gt)} 条")
    print(f"分类: {classified_path}")

    # 确定 LLM-NoCo 案例
    noco_uids: set[str] = set()
    if classified_path.exists():
        with open(classified_path) as f:
            for row in csv.DictReader(f):
                if row["subtype"] == "LLM-NoCo":
                    noco_uids.add(row["unass_record"])
    print(f"LLM-NoCo 案例: {len(noco_uids)}")

    if not noco_uids:
        print("无 LLM-NoCo 案例，跳过")
        return []

    # 加载已有结果（断点续传）
    existing: dict[str, dict] = {}
    if out_path.exists() and not force:
        for r in load_json(out_path):
            existing[r["case_id"]] = r
        noco_uids -= set(existing.keys())
        print(f"已有结果: {len(existing)} 条，剩余: {len(noco_uids)} 条")

    if not noco_uids:
        print("全部已完成")
        return list(existing.values())

    # 构造任务
    tasks = []
    for uid in noco_uids:
        if uid not in cands_data:
            continue
        candidates = cands_data[uid].get("candidates", [])
        sorted_cands = sorted(candidates, key=candidate_sort_key_v3)
        candidates_text, aid_map = format_candidates(sorted_cands)
        gt_aid = gt.get(uid, "none")
        tasks.append((uid, candidates_text, aid_map, gt_aid))

    print(f"待运行: {len(tasks)} 条")

    # 并行运行
    runner = LLMRunner(model=model)
    sem = asyncio.Semaphore(concurrency)

    results = list(existing.values())
    done = 0
    for i in range(0, len(tasks), concurrency):
        batch = tasks[i: i + concurrency]
        batch_results = await asyncio.gather(*[
            run_one(runner, uid, ct, am, sem, gt_aid)
            for uid, ct, am, gt_aid in batch
        ])
        results.extend(batch_results)
        done += len(batch_results)

        # 增量保存
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"进度: {done}/{len(tasks)}")
        await asyncio.sleep(1)  # 节流

    # 统计
    total = len(results)
    match = sum(1 for r in results if r["match"])
    none_pred = sum(1 for r in results if r["pred"] == "none")
    print(f"\n=== LLM-NoCo {split} ===")
    print(f"总案例: {total}")
    print(f"匹配: {match}/{total} = {match / total * 100:.1f}% (仅 GT 存在时)")
    print(f"pred=none: {none_pred}/{total} = {none_pred / total * 100:.1f}%")
    print(f"保存: {out_path}")

    return results
