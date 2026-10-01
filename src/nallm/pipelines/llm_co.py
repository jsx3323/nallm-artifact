"""LLM-Co 跑批引擎(rule_first 链路)。

下沉自 scripts/run_llm_co.py。保留自己的 provider(AsyncOpenAI + OPENCODE_GO)
与 asyncio.Semaphore 并发(独立于 llm_first 的批处理引擎)。
删除 sys.path hack / ROOT 硬编码 / global EMB,路径走 config + split。
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import time
from pathlib import Path

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    tqdm_asyncio = None
    HAS_TQDM = False
else:
    HAS_TQDM = True

from openai import AsyncOpenAI

from nallm.core.config import DATA_DIR, EMBEDDING_DB_DIR, RESULTS_DIR
from nallm.disambiguation.parser import parse_output
from nallm.disambiguation.prompt_builder import build_llm_co_prompt
from nallm.disambiguation.sample import sample_papers_for_candidate
from nallm.embedding.database import EmbeddingDB

# LLM-Co 子类型集合(含旧分类兼容)
LLM_CO = {
    "LLM-Co",
    "P1-High", "P1-Strong-Comp", "P1-High-Comp", "P1-Medium",
    "P1-Medium-Comp", "P1-Weak", "P1-Weak-Comp",
    "P2", "P2-Comp", "P3",
}

SYSTEM_PROMPT = "你是学术作者消歧助手。基于候选信息和代表论文，判断待消歧论文的真实作者。"


def _load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class CombinedEmbeddingDB:
    """合并查询：unass 论文（split/title.lmdb）+ profiles 论文（profiles/title.lmdb）"""

    def __init__(self, split_db, profile_db):
        self.split_db = split_db
        self.profile_db = profile_db

    def get(self, pid):
        v = self.split_db.get(pid)
        if v is not None:
            return v
        return self.profile_db.get(pid)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def candidate_sort_key(c):
    co = c.get("coauthor_count", 0)
    org = c.get("coauthor_org_match_count", 0)
    org_sim = c.get("org_sim", c.get("org_sim_max", c.get("main_org_sim", 0)))
    venue_sim = c.get("venue_sim", c.get("venue_sim_max", 0))
    if co > 0 and org > 0:
        bucket = 0
    elif co > 0:
        bucket = 1
    else:
        bucket = 2
    return (
        bucket, -org, -org_sim, -venue_sim,
        -c.get("title_sim_max", c.get("title_sim", 0)),
    )


class LLMRunner:
    def __init__(self, model: str):
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OEPNCODE_GO_API_KEY") or os.environ.get("OPENCODE_GO_API_KEY", ""),
            base_url=os.environ["OPENCODE_GO_API_BASE"],
            timeout=600.0,
            max_retries=10,  # SDK 自带：自动重试 408/409/429/500/502/503/504
        )
        self.model = model

    async def call(self, system: str, user: str) -> tuple[str, dict]:
        """返回 (content, usage_dict)。SDK 处理 5xx/429 重试 10 次，empty content 也重试。"""
        last_err = ""
        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                )
                content = (resp.choices[0].message.content or "") if resp.choices else ""
                usage = {}
                if resp.usage:
                    usage = {
                        "input_tokens": resp.usage.prompt_tokens or 0,
                        "output_tokens": resp.usage.completion_tokens or 0,
                    }
                if not content.strip():
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                return content, usage
            except Exception as e:
                last_err = f"err: {e}"
                # SDK 已重试 10 次仍失败
                return last_err, {}
        return last_err, {}


def build_samples(sorted_cands, new_pub, profiles, profiles_pub_ext, gt_aid, db, all_real=False, new_pub_paper_id=None):
    """构造每候选的 sample papers

    all_real=True: 所有候选都从 profiles 取真实论文（test 集用，公平判断）
    gt_aid=None (无 GT): 所有候选给空（让 LLM 基于数值判 none）
    new_pub_paper_id: 8 字符 paper id（用于 embedding 查询）
    """
    samples_by_aid = {}

    def _aid(c):
        return c.get("aid") or c.get("candidate_aid")

    if gt_aid is None and not all_real:
        # 无 GT（如 test 集不指定）→ 给空（基于数值判）
        for c in sorted_cands[:5]:
            samples_by_aid[_aid(c)] = []
        return samples_by_aid
    for c in sorted_cands[:5]:
        aid = _aid(c)
        # test all_real 模式 / 正常 GT=author 模式：都从 profiles 取真实论文
        if all_real or aid == gt_aid:
            pubs_ids = profiles[aid].get("pubs", []) if aid in profiles else []
        else:
            pubs_ids = ["FAKE_PUB_FOR_NONGT"]
        pubs = [profiles_pub_ext[pid] for pid in pubs_ids if pid in profiles_pub_ext]
        # 传 8 字符 paper_id 给 sampling（embedding db key 是 8 字符）
        samples_by_aid[aid] = sample_papers_for_candidate(
            pubs, new_pub, top_k=3, title_db=db,
            new_pub_paper_id=new_pub_paper_id,
        )
    return samples_by_aid


async def run_one(runner: LLMRunner, uid: str, gt_aid: str,
                  sem: asyncio.Semaphore, new_pub: dict, sorted_cands: list,
                  samples_by_aid: dict, author_order: int) -> dict:
    async with sem:
        prompt_text = build_llm_co_prompt(
            new_pub=new_pub, samples_by_aid=samples_by_aid,
            candidates=sorted_cands[:5], author_order=author_order,
        )
        content, usage = await runner.call(SYSTEM_PROMPT, prompt_text)

        pred = "UNFINISHED"
        status = "ok"
        if content and not content.startswith(("err:", "429:", "conn:", "api(")):
            parsed = parse_output(content)
            pred = getattr(parsed, "author_id", None) or "PARSE_FAIL"
            if pred == "PARSE_FAIL":
                status = "parse_fail"
        else:
            status = "unfinished"

        match = str(pred).lower() == str(gt_aid).lower()

        return {
            "case_id": uid, "gt": gt_aid, "pred": pred,
            "match": match, "status": status,
            "raw": content[:200] if isinstance(content, str) else str(content)[:200],
            "in_tokens": usage.get("input_tokens", 0),
            "out_tokens": usage.get("output_tokens", 0),
        }


async def run_llm_co(
    split: str = "valid",
    *,
    out_path: Path | None = None,
    concurrency: int = 10,
    model: str | None = None,
    n: int = 0,
    seed: int = 42,
    subtypes: list[str] | None = None,
    exclude_gt_none: bool = False,
    force_restart: bool = False,
) -> list[dict]:
    """跑全量/抽样 LLM-Co 案例。

    Args:
        split: 数据集 (valid/test)
        out_path: 输出 JSON 路径(默认 RESULTS_DIR/{split}_llm_co_full.json)
        concurrency: 并发数(默认 10)
        model: 模型名(默认 env OPENCODE_GO_MODEL 或 deepseek-v4-flash)
        n: 随机抽样数(0=全量)
        seed: 抽样随机种子
        subtypes: 子类型过滤列表
        exclude_gt_none: 排除 GT=none 案例
        force_restart: 忽略已有结果重跑

    Returns:
        结果列表(含本轮 + resume 的历史),同时写 out_path。
    """
    model = model or os.environ.get("OPENCODE_GO_MODEL", "deepseek-v4-flash")
    print(f"split={split} n={'全量' if n == 0 else n} conc={concurrency} model={model}")

    # 路径(config + split;profiles 档案库固定用 valid)
    gt_path = DATA_DIR / split / f"cna_{split}_unass_gt.json"
    profiles_path = DATA_DIR / "valid" / "whole_author_profiles.json"
    profiles_pub_path = DATA_DIR / "valid" / "whole_author_profiles_pub.json"
    unass_pub_path = DATA_DIR / split / f"cna_{split}_unass_pub.json"
    cands_path = RESULTS_DIR / f"{split}_filtered_candidates.json"
    classified_csv = RESULTS_DIR / f"{split}_unass_classified.csv"
    split_db_path = EMBEDDING_DB_DIR / split / "title.lmdb"
    profile_db_path = EMBEDDING_DB_DIR / "profiles" / "title.lmdb"
    out_path = out_path or (RESULTS_DIR / f"{split}_llm_co_full.json")

    unass_gt = _load_json(gt_path) if gt_path.exists() else {}
    profiles = _load_json(profiles_path)
    profiles_pub = _load_json(profiles_pub_path)
    unass_pub = _load_json(unass_pub_path)
    cands_data = _load_json(cands_path)

    profiles_pub_ext = dict(profiles_pub)
    profiles_pub_ext["FAKE_PUB_FOR_NONGT"] = {
        "id": "FAKE_PUB_FOR_NONGT",
        "title": "(No detailed paper available for this candidate)",
        "authors": [], "year": "?", "venue": "",
    }

    # 子类型映射
    uid2sub: dict[str, str] = {}
    if classified_csv.exists():
        with open(classified_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid2sub[row["unass_record"]] = row["subtype"]

    # case 来源：test 无 GT 用 cands_data；valid 用 unass_gt（可对照）
    source_uids = list(unass_gt.keys()) if unass_gt else list(cands_data.keys())
    cases = []
    for uid in source_uids:
        gt = unass_gt.get(uid, "none")  # test 默认 gt=none
        if exclude_gt_none and gt == "none":
            continue
        if uid not in cands_data:
            continue
        sub = uid2sub.get(uid, "")
        if subtypes:
            if sub not in set(subtypes):
                continue
        elif sub not in LLM_CO and sub != "none":
            continue
        new_pub_id = uid.split("-")[0]
        if new_pub_id not in unass_pub:
            continue
        cases.append((uid, gt, sub))

    cases.sort()
    if n > 0:
        random.seed(seed)
        cases = random.sample(cases, min(n, len(cases)))
    print(f"案例: {len(cases)}")

    runner = LLMRunner(model=model)
    sem = asyncio.Semaphore(concurrency)

    # Resume: 读已有 JSON，跳过已完成的 case_id
    existing_map: dict = {}
    if out_path.exists():
        existing_list = _load_json(out_path)
        existing_map = {r["case_id"]: r for r in existing_list}
        if not force_restart:
            pending = [(uid, gt, sub) for uid, gt, sub in cases
                       if existing_map.get(uid, {}).get("status") not in ("ok",)]
            finished_n = len(cases) - len(pending)
            print(f"Resume: 已有 {finished_n} 完成，跳过；待跑 {len(pending)}")
            cases = pending
        else:
            print(f"--force-restart: 重跑全部 {len(cases)} 案例")

    with EmbeddingDB(str(split_db_path), readonly=True) as split_db, \
         EmbeddingDB(str(profile_db_path), readonly=True) as prof_db:
        db = CombinedEmbeddingDB(split_db, prof_db)
        print(f"打开 embedding: {split_db_path} + {profile_db_path}")

        async def task(uid, gt_aid):
            new_pub_id = uid.split("-")[0]
            new_pub = unass_pub[new_pub_id]
            candidates = cands_data[uid].get("candidates", [])
            sorted_cands = sorted(candidates, key=candidate_sort_key)
            samples_by_aid = build_samples(
                sorted_cands, new_pub, profiles, profiles_pub_ext,
                gt_aid if gt_aid != "none" else None, db,
                all_real=(split == "test"),
                new_pub_paper_id=new_pub_id,
            )
            return await run_one(
                runner, uid, gt_aid, sem, new_pub,
                sorted_cands, samples_by_aid, int(uid.split("-")[1]),
            )

        t0 = time.time()
        coros = [task(uid, gt) for uid, gt, _ in cases]
        if HAS_TQDM and tqdm_asyncio is not None:
            new_results = await tqdm_asyncio.gather(
                *coros, total=len(coros),
                desc=f"LLM-Co ({concurrency} 并发)", ncols=80,
            )
        else:
            new_results = await asyncio.gather(*coros)
        elapsed = time.time() - t0

    # Merge: 新结果覆盖旧记录
    for r in new_results:
        existing_map[r["case_id"]] = r
    results = list(existing_map.values())
    results.sort(key=lambda r: r["case_id"])

    n_done = len(results)
    correct = sum(1 for r in results if r["match"])
    unfinished = sum(1 for r in results if r["status"] == "unfinished")
    parse_fail = sum(1 for r in results if r["status"] == "parse_fail")
    sum_in = sum(r["in_tokens"] for r in results)
    sum_out = sum(r["out_tokens"] for r in results)

    print(f"\n=== 结果 (n={n_done}, 本轮 {elapsed:.1f}s) ===")
    print(f"正确 (有结果): {correct}/{n_done} = {correct / n_done:.1%}")
    print(f"未完成 (UNFINISHED): {unfinished}/{n_done} = {unfinished / n_done:.1%}")
    print(f"PARSE_FAIL: {parse_fail}/{n_done} = {parse_fail / n_done:.1%}")
    print(f"Token: in={sum_in:,} out={sum_out:,} total={sum_in + sum_out:,}")
    print(f"平均: in={sum_in // max(n_done, 1)} out={sum_out // max(n_done, 1)}")
    if unfinished:
        uids = [r["case_id"] for r in results if r["status"] == "unfinished"]
        print(f"\n未完成案例 ({len(uids)}): {uids[:10]}{'...' if len(uids) > 10 else ''}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")

    return results
