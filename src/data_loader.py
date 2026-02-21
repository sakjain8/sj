"""
data_loader.py — GitHub Issue Fetcher

Fetches real-world GitHub issues from public repositories using the GitHub
REST API. Handles rate limiting, pagination, and filtering. Caches results
to disk for reproducibility.

Author: Research Framework
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_REPOS = [
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "microsoft/vscode",
    "nodejs/node",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_headers() -> Dict[str, str]:
    """Build request headers, optionally including a personal access token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _handle_rate_limit(response: requests.Response) -> None:
    """Sleep until the rate-limit window resets if we are throttled."""
    if response.status_code == 403 and "rate limit" in response.text.lower():
        reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_ts - int(time.time()), 5)
        print(f"[data_loader] Rate-limited. Sleeping {wait}s until reset …")
        time.sleep(wait + 1)
    elif response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        print(f"[data_loader] 429 Too Many Requests. Sleeping {retry_after}s …")
        time.sleep(retry_after)


# ---------------------------------------------------------------------------
# Core fetcher
# ---------------------------------------------------------------------------

def fetch_issues_from_repo(
    repo: str,
    per_page: int = 30,
    min_body_length: int = 100,
    max_issues: int = 15,
) -> List[Dict]:
    """
    Fetch open issues (not PRs) from a single GitHub repository.

    Parameters
    ----------
    repo : str
        Full repo name, e.g. "pytorch/pytorch".
    per_page : int
        Number of issues to request per API page.
    min_body_length : int
        Minimum character length of the issue body to keep.
    max_issues : int
        Maximum number of qualifying issues to return.

    Returns
    -------
    list[dict]
        List of issue dicts with keys: repo, number, title, body, labels.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo}/issues"
    params = {
        "state": "open",
        "per_page": per_page,
        "sort": "created",
        "direction": "desc",
    }
    headers = _get_headers()
    collected: List[Dict] = []
    page = 1

    while len(collected) < max_issues and page <= 5:
        params["page"] = page
        resp = requests.get(url, headers=headers, params=params, timeout=30)

        # Handle rate limiting
        if resp.status_code in (403, 429):
            _handle_rate_limit(resp)
            continue  # Retry the same page

        if resp.status_code != 200:
            print(f"[data_loader] Warning: {repo} returned HTTP {resp.status_code}")
            break

        items = resp.json()
        if not items:
            break

        for item in items:
            # Skip pull requests (they also appear in /issues endpoint)
            if "pull_request" in item:
                continue

            body = (item.get("body") or "").strip()
            if len(body) < min_body_length:
                continue

            collected.append({
                "repo": repo,
                "number": item["number"],
                "title": item.get("title", ""),
                "body": body,
                "labels": [lbl["name"] for lbl in item.get("labels", [])],
            })

            if len(collected) >= max_issues:
                break

        page += 1
        time.sleep(1)  # Politeness delay

    print(f"[data_loader] Fetched {len(collected)} issues from {repo}")
    return collected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all_issues(
    repos: Optional[List[str]] = None,
    issues_per_repo: int = 10,
    min_body_length: int = 100,
    max_total: int = 50,
    data_dir: str = "data",
    force_refresh: bool = False,
) -> List[Dict]:
    """
    Fetch issues from multiple repos, cache to data/github_issues.json.

    Parameters
    ----------
    repos : list[str] | None
        Repository names. Defaults to DEFAULT_REPOS.
    issues_per_repo : int
        Max issues to fetch per repo.
    min_body_length : int
        Minimum body length filter.
    max_total : int
        Hard cap on total issues returned.
    data_dir : str
        Directory for cached data.
    force_refresh : bool
        If True, ignore cache and re-fetch.

    Returns
    -------
    list[dict]
        Combined list of issue dicts.
    """
    repos = repos or DEFAULT_REPOS
    cache_path = os.path.join(data_dir, "github_issues.json")

    # Return cached data if available and not forcing refresh
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            issues = json.load(f)
        print(f"[data_loader] Loaded {len(issues)} cached issues from {cache_path}")
        return issues

    # Fetch from GitHub
    all_issues: List[Dict] = []
    for repo in repos:
        print(f"[data_loader] Fetching issues from {repo} …")
        issues = fetch_issues_from_repo(
            repo,
            per_page=30,
            min_body_length=min_body_length,
            max_issues=issues_per_repo,
        )
        all_issues.extend(issues)
        if len(all_issues) >= max_total:
            all_issues = all_issues[:max_total]
            break

    # Persist to disk
    os.makedirs(data_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_issues, f, indent=2, ensure_ascii=False)

    print(f"[data_loader] Saved {len(all_issues)} issues to {cache_path}")
    return all_issues


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    issues = fetch_all_issues(force_refresh=True)
    for i, issue in enumerate(issues):
        print(f"  [{i+1}] {issue['repo']}#{issue['number']}: {issue['title'][:80]}")
