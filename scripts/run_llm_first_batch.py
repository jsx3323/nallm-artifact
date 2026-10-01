"""用 OpenRouter batch 端点跑 llm_first ③(半价、24h 完成窗口)。

batch 走 `/api/beta/batches`,与同步 chat completions 不是同一套接口,
故本脚本把同一条 llm_first 链路跑两趟,只替换 LLM 调用点:

  趟 A(collect):patch chat_completion 为「记录 messages 并返回占位」,不发请求
  → 提交 batch(custom_id = 记录 ID)→ 轮询到终态
  趟 B(apply) :patch chat_completion 为「按记录 ID 查表返回 batch content」

这样 parse_output / 置信度校准 / 结果 schema / ground_truth 全部沿用同步链路,无重复实现。

关联键必须是记录 ID,不能用 messages 哈希:prompt_builder 把 set 交集直接
list() 截断(`matched_keywords[:10]`),而 str 的 hash 每进程随机,故约 13%
的记录跨进程 prompt 不逐位一致(实测 1000 条中 129 条),用哈希关联会大面积查表失败。
记录 ID 经 contextvar 从 disambiguate_record 传到 chat_completion,与 prompt 是否稳定解耦。

用法:
    # 提交并轮询到完成,写结果 CSV
    uv run python scripts/run_llm_first_batch.py \
        --target-csv results/disambiguation/valid1000_target.csv \
        --out results/disambiguation/valid1000_or_luna_batch.csv

    # 只提交,拿 batch id 后退出(--state 记录 id,便于稍后取回)
    uv run python scripts/run_llm_first_batch.py --target-csv ... --out ... --submit-only

    # 用已有 batch id 取回并落盘(跳过提交)
    uv run python scripts/run_llm_first_batch.py --target-csv ... --out ... --batch-id batch_xxx
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import logging
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

import nallm.disambiguation.batch.engine as engine
from nallm.disambiguation.batch import ERROR_SENTINEL
from nallm.pipelines.llm_first import LlmFirstConfig, RecordFilter, run_llm_first_from_path
from nallm.utils.io import safe_write_csv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

BATCH_API = "https://openrouter.ai/api/beta/batches"
# 占位 content:趟 A 用,能被 parse_output 正常解析,避免走 error 哨兵路径
COLLECT_PLACEHOLDER = '{"author_id": "none", "confidence": "medium", "reasoning": "collect"}'
TERMINAL = {"completed", "failed", "expired", "cancelled"}


# 当前正在处理的记录 ID:disambiguate_record 包装层设置,patch 后的 chat_completion 读取。
# asyncio 每个 task 复制一份 context,故并发下互不串扰。
_current_record: contextvars.ContextVar[str] = contextvars.ContextVar("current_record")


def wrap_record_id(fn):
    """包装 disambiguate_record,把 unass_record 放进 contextvar"""

    async def wrapped(unass_record, *a, **kw):
        _current_record.set(unass_record)
        return await fn(unass_record, *a, **kw)

    return wrapped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用 OpenRouter batch 端点跑 llm_first ③")
    p.add_argument("--target-csv", required=True, type=Path, help="定义 ID 界的 CSV(需含 unass_record 列)")
    p.add_argument("--out", required=True, type=Path, help="结果 CSV 输出路径")
    p.add_argument("--dataset", "-d", choices=["valid", "test"], default="valid")
    p.add_argument("--max-candidates", type=int, default=10)
    p.add_argument("--model", default=None, help="batch 模型 id;缺省取 LLM_MODEL 并补 ':batch' 后缀")
    p.add_argument("--state", type=Path, default=None, help="状态文件(batch id 等);缺省为 <out>.state.json")
    p.add_argument("--batch-id", action="append", default=None, help="跳过提交,直接取回该 batch(可多次传,对应多片)")
    p.add_argument("--chunk-size", type=int, default=5000, help="每片条数上限,服务端硬限 5000")
    p.add_argument("--max-payload-mb", type=float, default=180.0, help="每片 payload 上限 MB,服务端硬限 200,默认留余量 180")
    p.add_argument("--dry-run", action="store_true", help="只打印分片计划(片数/条数/体积),不提交")
    p.add_argument("--submit-only", action="store_true", help="只提交并记录 batch id 后退出")
    p.add_argument("--poll-interval", type=int, default=60, help="轮询间隔秒,默认 60")
    p.add_argument("--no-temperature", action="store_true", help="body 不带 temperature(部分模型不支持该参数)")
    return p.parse_args()


def run_pipeline(dataset: str, ids: set[str], max_candidates: int) -> list[dict]:
    """跑一趟 llm_first(LLM 调用点已被调用方 patch)"""
    run = asyncio.run(
        run_llm_first_from_path(
            dataset,
            config=LlmFirstConfig(max_candidates=max_candidates, concurrency=8, is_test=dataset == "test"),
            filt=RecordFilter(record_ids=ids),
        )
    )
    return run.results


def collect_prompts(dataset: str, ids: set[str], max_candidates: int) -> dict[str, list[dict]]:
    """趟 A:收集 messages(记录 ID → messages),不发任何请求"""
    collected: dict[str, list[dict]] = {}

    async def fake_chat(messages: list[dict]) -> str:
        collected[_current_record.get()] = messages
        return COLLECT_PLACEHOLDER

    orig_chat, orig_rec = engine.chat_completion, engine.disambiguate_record
    engine.chat_completion = fake_chat
    engine.disambiguate_record = wrap_record_id(orig_rec)
    try:
        run_pipeline(dataset, ids, max_candidates)
    finally:
        engine.chat_completion, engine.disambiguate_record = orig_chat, orig_rec
    return collected


def apply_contents(
    dataset: str, ids: set[str], max_candidates: int, contents: dict[str, str]
) -> list[dict]:
    """趟 B:按记录 ID 查表返回 batch content,产出与同步链路同 schema 的结果"""

    async def lookup_chat(messages: list[dict]) -> str:
        record = _current_record.get()
        content = contents.get(record)
        if content is None:
            raise RuntimeError(f"batch 结果缺失该记录: {record}")
        return content

    orig_chat, orig_rec = engine.chat_completion, engine.disambiguate_record
    engine.chat_completion = lookup_chat
    engine.disambiguate_record = wrap_record_id(orig_rec)
    try:
        return run_pipeline(dataset, ids, max_candidates)
    finally:
        engine.chat_completion, engine.disambiguate_record = orig_chat, orig_rec


def build_requests(prompts: dict[str, list[dict]], extra: dict, with_temp: bool) -> list[dict]:
    """prompts → batch requests(custom_id = 记录 ID)"""
    requests = []
    for key, messages in prompts.items():
        body: dict = {"messages": messages}
        if with_temp:
            body["temperature"] = 0.0
        body.update(extra)
        requests.append({"custom_id": key, "body": body})
    return requests


# 服务端双重硬限(均来自 413 报错文案,文档未载):≤5000 条 且 ≤200MB payload。
# 真实记录每条约 80KB,故体积先撞线(200MB ≈ 2500 条),条数上限通常用不到。
ENVELOPE_BYTES = 120  # {"endpoint":..,"model":..,"requests":[]} 外壳与逗号的粗略开销


def chunk(items: list, size: int, max_bytes: int | None = None) -> list[list]:
    """按条数与 payload 体积双约束切片

    max_bytes 为 None 时只按条数切(用于测试与不含真实 prompt 的场景)。
    """
    if max_bytes is None:
        return [items[i : i + size] for i in range(0, len(items), size)]

    out: list[list] = []
    cur: list = []
    cur_bytes = ENVELOPE_BYTES
    for it in items:
        # 必须与上线序列化口径一致:httpx json= 用默认 ensure_ascii=True,
        # 中文转 \uXXXX 后体积约翻倍,用 ensure_ascii=False 估会低算一半
        b = len(json.dumps(it).encode()) + 1
        if cur and (len(cur) >= size or cur_bytes + b > max_bytes):
            out.append(cur)
            cur, cur_bytes = [], ENVELOPE_BYTES
        cur.append(it)
        cur_bytes += b
    if cur:
        out.append(cur)
    return out


def plan_chunks(requests: list[dict], size: int, max_bytes: int | None = None) -> list[tuple[int, int]]:
    """分片计划:[(条数, payload 字节数), ...]"""
    return [
        (len(c), len(json.dumps({"endpoint": "/v1/chat/completions", "model": "x", "requests": c}).encode()))
        for c in chunk(requests, size, max_bytes)
    ]


def submit_batch(model: str, requests: list[dict]) -> str:
    """提交单片 batch,返回 batch id"""
    payload = {"endpoint": "/v1/chat/completions", "model": model, "requests": requests}
    logger.info(f"提交 batch: {len(requests)} 条,模型 {model}")
    # 超时按体积放宽:实测 80MB 上传约 4 分钟,5000 条一片约 400MB 需 20 分钟量级
    r = httpx.post(
        BATCH_API,
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
        json=payload,
        timeout=3600,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"提交失败 {r.status_code}: {r.text[:1000]}")
    data = r.json()
    batch_id = data.get("id") or data.get("data", {}).get("id")
    if not batch_id:
        raise RuntimeError(f"响应无 batch id: {json.dumps(data)[:500]}")
    return batch_id


def poll_batch(batch_id: str, interval: int, not_found_grace: int = 10) -> dict:
    """轮询到终态,返回终态响应

    刚创建的 batch 有传播延迟,前 not_found_grace 次 404 视为「还没就绪」继续等,
    不当失败(实测 1000 条提交后首次轮询即 404,稍后同一 id 正常返回)。
    """
    headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
    last = None
    not_found = 0
    while True:
        r = httpx.get(f"{BATCH_API}/{batch_id}", headers=headers, timeout=600)
        if r.status_code == 404 and not_found < not_found_grace:
            not_found += 1
            logger.info(f"batch 未就绪(404 第 {not_found} 次),{interval}s 后重试")
            time.sleep(interval)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"轮询失败 {r.status_code}: {r.text[:500]}")
        data = r.json()
        data = data.get("data", data)
        status = data.get("status")
        counts = data.get("request_counts", {})
        if (status, tuple(sorted(counts.items()))) != last:
            logger.info(f"状态 {status} | {counts}")
            last = (status, tuple(sorted(counts.items())))
        if status in TERMINAL:
            return data
        time.sleep(interval)


def extract_contents(data: dict) -> tuple[dict[str, str], list[str]]:
    """从终态响应抽 custom_id → content;返回(内容表, 失败 id 列表)"""
    contents: dict[str, str] = {}
    failed: list[str] = []
    for item in data.get("results") or []:
        cid = item.get("custom_id")
        resp = item.get("response") or {}
        body = resp.get("body") or resp
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            failed.append(cid)
            continue
        if not (content or "").strip():
            failed.append(cid)
            continue
        contents[cid] = content
    return contents, failed


def merge_contents(datas: list[dict]) -> tuple[dict[str, str], list[str]]:
    """合并多片终态响应为一张 custom_id → content 表

    片间 custom_id 不重叠(切片自同一 prompts 字典的键),重复出现视为异常并记警告。
    """
    merged: dict[str, str] = {}
    failed: list[str] = []
    for data in datas:
        contents, bad = extract_contents(data)
        dup = merged.keys() & contents.keys()
        if dup:
            logger.warning(f"片间 custom_id 重复 {len(dup)} 个,保留先到的:{sorted(dup)[:3]}")
        for k, v in contents.items():
            merged.setdefault(k, v)
        failed.extend(bad)
    return merged, failed


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    state_path = args.state or args.out.with_suffix(".state.json")
    model = args.model or os.environ.get("LLM_MODEL", "")
    if not model:
        raise SystemExit("需要 --model 或 LLM_MODEL")
    if not model.endswith(":batch"):
        model += ":batch"

    ids = set(pd.read_csv(args.target_csv, usecols=["unass_record"])["unass_record"])
    logger.info(f"目标界 {len(ids)} 条 | 数据集 {args.dataset}")

    extra = json.loads(os.environ.get("LLM_EXTRA_BODY", "").strip() or "{}")

    logger.info("趟 A:收集 prompt(不发请求)")
    prompts = collect_prompts(args.dataset, ids, args.max_candidates)
    logger.info(f"需 LLM 的记录 {len(prompts)} 条(其余走 NONE/AUTO 规则,无 prompt)")

    requests = build_requests(prompts, extra, with_temp=not args.no_temperature)

    max_bytes = int(args.max_payload_mb * 1e6)

    if args.dry_run:
        logger.info(f"分片计划(≤{args.chunk_size} 条 且 ≤{args.max_payload_mb}MB):")
        total = 0
        for i, (n, size) in enumerate(plan_chunks(requests, args.chunk_size, max_bytes), 1):
            total += n
            logger.info(f"  片 {i}: {n} 条, payload {size / 1e6:.1f}MB")
        logger.info(f"合计 {total} 条(应等于 {len(requests)});服务端硬限 5000 条 / 200MB")
        return

    batch_ids = args.batch_id
    if not batch_ids:
        batch_ids = []
        chunks = chunk(requests, args.chunk_size, max_bytes)
        for i, c in enumerate(chunks, 1):
            logger.info(f"提交片 {i}/{len(chunks)}")
            batch_ids.append(submit_batch(model, c))
            # 每片提交后立刻落盘:后续片失败时,已提交的片不丢(其结果仍可用 --batch-id 取回)
            state_path.write_text(
                json.dumps({"batch_ids": batch_ids, "model": model, "n": len(prompts)}, indent=2, ensure_ascii=False)
            )
        logger.info(f"{len(batch_ids)} 片 → {state_path}")
        if args.submit_only:
            return

    datas = []
    for i, bid in enumerate(batch_ids, 1):
        logger.info(f"轮询片 {i}/{len(batch_ids)}: {bid}")
        d = poll_batch(bid, args.poll_interval)
        if d.get("status") != "completed":
            logger.error(f"片 {i} 终态非 completed: {d.get('status')}")
        if d.get("usage"):
            logger.info(f"片 {i} usage: {json.dumps(d['usage'], ensure_ascii=False)}")
        datas.append(d)

    contents, failed = merge_contents(datas)
    logger.info(f"取回 content {len(contents)} 条,失败 {len(failed)} 条")

    # 失败的 prompt 用 error 哨兵占位,保持与同步链路一致的错误语义
    for key in prompts:
        contents.setdefault(key, f'{{"author_id": "{ERROR_SENTINEL}", "confidence": "low", "reasoning": "batch failed"}}')

    logger.info("趟 B:回填 content 并产出结果")
    results = apply_contents(args.dataset, ids, args.max_candidates, contents)
    df = pd.DataFrame(results)
    safe_write_csv(df, args.out)
    logger.info(f"已写 {len(df)} 条 → {args.out}")

    if "ground_truth_author_id" in df.columns:
        norm = lambda s: s.where(s.notna(), "").astype(str)
        correct = (norm(df["predicted_author_id"]) == norm(df["ground_truth_author_id"])).sum()
        logger.info(f"准确率 {correct}/{len(df)} = {correct / len(df):.2%}")


if __name__ == "__main__":
    main()
