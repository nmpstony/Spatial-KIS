"""Crawler for DRES evaluation tasks, prompts, metadata, and answer videos.

Supports downloading:
- Task prompts & multi-stage textual hints
- Target answer videos (exact video segments)
- Hint query videos (for Visual KIS)
- Question & Answer text targets
- Structured summaries in JSON and CSV format
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


def make_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    return session


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


class DresCrawler:
    def __init__(self, base_url: str, username: str, password: str, out_dir: Path, skip_media: bool = False):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.out_dir = Path(out_dir)
        self.skip_media = skip_media
        self.session = make_session()
        self.session_id: str | None = None

    def login(self) -> bool:
        login_url = f"{self.base_url}/api/v2/login"
        print(f"[*] Logging in to {login_url} as '{self.username}'...")
        payload = {"username": self.username, "password": self.password}
        try:
            resp = self.session.post(login_url, json=payload, timeout=15)
            if resp.status_code != 200:
                print(f"[!] Login failed: HTTP {resp.status_code} - {resp.text}", file=sys.stderr)
                return False
            data = resp.json()
            self.session_id = data.get("sessionId")
            print(f"[+] Login successful! Session ID: {self.session_id}")
            return True
        except Exception as e:
            print(f"[!] Login error: {e}", file=sys.stderr)
            return False

    def list_evaluations(self) -> list[dict[str, Any]]:
        url = f"{self.base_url}/api/v2/client/evaluation/list"
        resp = self.session.get(url, params={"session": self.session_id}, timeout=15)
        if resp.status_code != 200:
            print(f"[!] Failed to list evaluations: HTTP {resp.status_code}", file=sys.stderr)
            return []
        return resp.json()

    def get_template(self, template_id: str) -> dict[str, Any] | None:
        url = f"{self.base_url}/api/v2/download/template/{template_id}"
        resp = self.session.get(url, params={"session": self.session_id}, timeout=30)
        if resp.status_code != 200:
            print(f"[!] Failed to download template {template_id}: HTTP {resp.status_code}", file=sys.stderr)
            return None
        return resp.json()

    def download_file(self, url: str, dest_path: Path, params: dict | None = None) -> bool:
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return True  # Already downloaded
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.session.get(url, params=params, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    return False
                tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(dest_path)
            return True
        except Exception as e:
            print(f"[!] Error downloading {url} -> {dest_path.name}: {e}", file=sys.stderr)
            return False

    def crawl_evaluation(self, eval_info: dict[str, Any]) -> dict[str, Any]:
        eval_id = eval_info["id"]
        eval_name = eval_info.get("name", eval_id)
        template_id = eval_info.get("templateId")
        slug = f"{sanitize_filename(eval_name)}_{eval_id[:8]}"
        eval_dir = self.out_dir / slug
        eval_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"[*] Crawling Evaluation: '{eval_name}' (ID: {eval_id})")
        print(f"[*] Output directory: {eval_dir}")
        print(f"=======================================================")

        # Save eval info
        with open(eval_dir / "evaluation_info.json", "w", encoding="utf-8") as f:
            json.dump(eval_info, f, ensure_ascii=False, indent=2)

        if not template_id:
            print(f"[!] No templateId found for evaluation {eval_id}")
            return {"evaluation": eval_info, "tasks": []}

        template_data = self.get_template(template_id)
        if not template_data:
            print(f"[!] Could not fetch template {template_id}")
            return {"evaluation": eval_info, "tasks": []}

        with open(eval_dir / "template_full.json", "w", encoding="utf-8") as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)

        raw_tasks = template_data.get("tasks", [])
        print(f"[+] Found {len(raw_tasks)} tasks in template '{template_data.get('name')}'")

        tasks_summary: list[dict[str, Any]] = []
        tasks_dir = eval_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)

        for idx, task in enumerate(raw_tasks, start=1):
            task_id = task.get("id")
            task_name = task.get("name", f"task_{idx}")
            task_group = task.get("taskGroup", "")
            task_type = task.get("taskType", "")
            duration = task.get("duration", 0)
            task_folder = tasks_dir / f"{idx:02d}_{sanitize_filename(task_name)}"
            task_folder.mkdir(parents=True, exist_ok=True)

            # Process hints (prompts / video hints)
            hints = task.get("hints", [])
            text_descriptions = []
            hint_video_saved = None

            for h_idx, hint in enumerate(hints):
                htype = hint.get("type", "")
                if htype == "TEXT":
                    desc = (hint.get("description") or "").strip()
                    start_s = hint.get("start", 0)
                    end_s = hint.get("end", "")
                    time_tag = f"[{start_s}s - {end_s}s]" if end_s else f"[{start_s}s]"
                    text_descriptions.append(f"{time_tag} {desc}")
                elif htype == "VIDEO" and not self.skip_media:
                    hint_vid_path = task_folder / f"hint_{h_idx}_query.mp4"
                    hint_url = f"{self.base_url}/api/v2/evaluation/{eval_id}/template/task/{task_id}/hint/{h_idx}/media"
                    success = self.download_file(hint_url, hint_vid_path, params={"session": self.session_id})
                    if not success:
                        h_item = hint.get("item", {})
                        h_media_id = hint.get("target") or h_item.get("mediaItemId")
                        h_rng = hint.get("range", {})
                        h_start = h_rng.get("start", {}).get("value", 0)
                        h_end = h_rng.get("end", {}).get("value", h_start)
                        if h_media_id:
                            preview_url = f"{self.base_url}/api/v2/preview/{h_media_id}/{h_start}/{h_end}"
                            success = self.download_file(preview_url, hint_vid_path, params={"session": self.session_id})
                    if success:
                        hint_video_saved = str(hint_vid_path.relative_to(self.out_dir))

            full_prompt = "\n".join(text_descriptions).strip()
            if full_prompt:
                with open(task_folder / "prompt.txt", "w", encoding="utf-8") as f:
                    f.write(full_prompt + "\n")

            # Process targets (answer videos or answer texts)
            targets = task.get("targets", [])
            target_video_id = ""
            target_start_ms = 0
            target_end_ms = 0
            target_start_sec = 0.0
            target_end_sec = 0.0
            target_text_answer = ""
            answer_video_saved = None

            for t_idx, target in enumerate(targets):
                ttype = target.get("type", "")
                if ttype in ("MEDIA_ITEM_TEMPORAL_RANGE", "MEDIA_ITEM"):
                    item = target.get("item", {})
                    target_video_id = item.get("location") or item.get("name", "")
                    rng = target.get("range", {})
                    try:
                        target_start_ms = int(rng.get("start", {}).get("value", 0))
                        target_end_ms = int(rng.get("end", {}).get("value", target_start_ms))
                    except (ValueError, TypeError):
                        target_start_ms = 0
                        target_end_ms = 0

                    target_start_sec = round(target_start_ms / 1000.0, 2)
                    target_end_sec = round(target_end_ms / 1000.0, 2)

                    if not self.skip_media:
                        ans_vid_path = task_folder / f"answer_{t_idx}.mp4"
                        ans_url = f"{self.base_url}/api/v2/evaluation/{eval_id}/template/task/{task_id}/target/{t_idx}/media"
                        success = self.download_file(ans_url, ans_vid_path, params={"session": self.session_id})
                        if not success:
                            media_id = target.get("target") or item.get("mediaItemId")
                            if media_id:
                                preview_url = f"{self.base_url}/api/v2/preview/{media_id}/{target_start_ms}/{target_end_ms}"
                                success = self.download_file(preview_url, ans_vid_path, params={"session": self.session_id})
                        if success:
                            answer_video_saved = str(ans_vid_path.relative_to(self.out_dir))

                elif ttype == "TEXT":
                    target_text_answer = target.get("target", "")
                    with open(task_folder / "answer_text.txt", "w", encoding="utf-8") as f:
                        f.write(target_text_answer + "\n")

            # Save task metadata
            task_meta = {
                "task_id": task_id,
                "task_name": task_name,
                "task_group": task_group,
                "task_type": task_type,
                "duration_seconds": duration,
                "prompt": full_prompt,
                "text_descriptions": text_descriptions,
                "target_video_id": target_video_id,
                "target_start_ms": target_start_ms,
                "target_end_ms": target_end_ms,
                "target_start_sec": target_start_sec,
                "target_end_sec": target_end_sec,
                "target_text_answer": target_text_answer,
                "answer_video_file": answer_video_saved,
                "hint_video_file": hint_video_saved,
                "raw_task": task,
            }
            with open(task_folder / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(task_meta, f, ensure_ascii=False, indent=2)

            # Add to summary row
            summary_row = {
                "idx": idx,
                "evaluation_name": eval_name,
                "evaluation_id": eval_id,
                "task_name": task_name,
                "task_group": task_group,
                "task_type": task_type,
                "duration": duration,
                "prompt": full_prompt.replace("\n", " || "),
                "target_video_id": target_video_id,
                "target_start_ms": target_start_ms,
                "target_end_ms": target_end_ms,
                "target_start_sec": target_start_sec,
                "target_end_sec": target_end_sec,
                "target_text_answer": target_text_answer,
                "answer_video_file": answer_video_saved or "",
                "hint_video_file": hint_video_saved or "",
            }
            tasks_summary.append(summary_row)
            print(f"  [{idx:02d}/{len(raw_tasks)}] {task_name} ({task_group}) -> Video: {target_video_id} ({target_start_sec}s - {target_end_sec}s) {'[MP4 saved]' if answer_video_saved else ''}")

        # Save CSV summary
        csv_path = eval_dir / "tasks_summary.csv"
        if tasks_summary:
            keys = list(tasks_summary[0].keys())
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(tasks_summary)

        # Save JSON summary
        with open(eval_dir / "tasks_summary.json", "w", encoding="utf-8") as f:
            json.dump(tasks_summary, f, ensure_ascii=False, indent=2)

        print(f"[+] Finished evaluation '{eval_name}'. Summary saved to {csv_path}")
        return {"evaluation": eval_info, "tasks": tasks_summary}

    def run(self, target_eval_id: str | None = None, crawl_all: bool = False):
        if not self.login():
            return

        all_evals = self.list_evaluations()
        if not all_evals:
            print("[!] No evaluations found.")
            return

        evals_to_crawl = []
        if crawl_all:
            evals_to_crawl = all_evals
        elif target_eval_id:
            evals_to_crawl = [e for e in all_evals if e["id"] == target_eval_id]
            if not evals_to_crawl:
                print(f"[!] Evaluation ID '{target_eval_id}' not found in accessible evaluations:")
                for e in all_evals:
                    print(f"    - {e['name']}: {e['id']}")
                return
        else:
            # Default to first active evaluation
            evals_to_crawl = [all_evals[0]]

        all_results = []
        for ev in evals_to_crawl:
            res = self.crawl_evaluation(ev)
            all_results.append(res)

        # Global combined summary
        combined_rows = []
        for res in all_results:
            combined_rows.extend(res.get("tasks", []))

        if combined_rows:
            global_csv = self.out_dir / "all_evaluations_summary.csv"
            with open(global_csv, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
                writer.writeheader()
                writer.writerows(combined_rows)
            print(f"\n[✓] ALL DONE! Total tasks crawled: {len(combined_rows)}")
            print(f"[✓] Global summary saved to: {global_csv}")


def main():
    parser = argparse.ArgumentParser(description="Crawler for DRES tasks, prompts, and answer videos")
    parser.add_argument("--url", default="https://aiclub.uit.edu.vn/dres", help="DRES base URL")
    parser.add_argument("--username", default="team02_t4", help="DRES username")
    parser.add_argument("--password", default="123456", help="DRES password")
    parser.add_argument(
        "--eval-id",
        default="bcb482c6-2b9e-4908-877a-e5ccd7c04640",
        help="Evaluation ID (default: 'Đề thi thử')",
    )
    parser.add_argument("--all", action="store_true", help="Crawl all available evaluations")
    parser.add_argument(
        "--out-dir",
        default="D:/Courses/HK7/CS336/dres_dataset",
        help="Output directory for crawled dataset",
    )
    parser.add_argument("--skip-media", action="store_true", help="Skip downloading answer/hint video files")

    args = parser.parse_args()

    crawler = DresCrawler(
        base_url=args.url,
        username=args.username,
        password=args.password,
        out_dir=Path(args.out_dir),
        skip_media=args.skip_media,
    )
    crawler.run(target_eval_id=args.eval_id, crawl_all=args.all)


if __name__ == "__main__":
    main()
