import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import requests

GRAPH_ROOT = 'https://graph.microsoft.com/v1.0'


@dataclass
class GraphClient:
    token: str
    session: requests.Session
    min_delay_s: float = 0.0

    @classmethod
    def create(cls, token: str, *, min_delay_s: float = 0.0) -> 'GraphClient':
        s = requests.Session()
        s.headers.update({
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
        })
        return cls(token=token, session=s, min_delay_s=float(min_delay_s or 0.0))

    def _sleep(self, seconds: float) -> None:
        try:
            time.sleep(max(0.0, float(seconds)))
        except Exception:
            pass

    def _request(self, method: str, url: str, *, params: Optional[Dict[str, Any]] = None, stream: bool = False) -> requests.Response:
        """Request with retry/backoff for throttling.

        Fix for Graph 429 (OneNote) code 20166: we now backoff more aggressively
        and add a small global pacing (min_delay_s) between requests.
        """
        if self.min_delay_s:
            self._sleep(self.min_delay_s)

        last = None
        for attempt in range(10):
            r = self.session.request(method, url, params=params, stream=stream, timeout=180)
            last = r
            if r.status_code not in (429, 503, 504, 500):
                return r

            retry_after = r.headers.get('Retry-After')
            if retry_after:
                try:
                    sleep_s = float(retry_after)
                except Exception:
                    sleep_s = 5.0
            else:
                # No Retry-After header: exponential backoff with jitter.
                # OneNote throttling (20166) is often sensitive to burstiness.
                sleep_s = min(5.0 * (2 ** attempt), 120.0)
                sleep_s += random.uniform(0.0, 1.0)

            self._sleep(sleep_s)

        return last

    def get_json(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = path if path.startswith('http') else f'{GRAPH_ROOT}{path}'
        r = self._request('GET', url, params=params)
        if r.status_code >= 400:
            raise RuntimeError(f'Graph error {r.status_code} url={url} body={r.text[:2000]}')
        return r.json()

    def iter_paged(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        url = path if path.startswith('http') else f'{GRAPH_ROOT}{path}'
        while url:
            r = self._request('GET', url, params=params)
            if r.status_code >= 400:
                raise RuntimeError(f'Graph error {r.status_code} url={url} body={r.text[:2000]}')
            data = r.json()
            for item in data.get('value', []):
                yield item
            url = data.get('@odata.nextLink')
            params = None  # already encoded in nextLink

    def download(self, url: str) -> bytes:
        r = self._request('GET', url, stream=True)
        if r.status_code >= 400:
            raise RuntimeError(f'Download error {r.status_code} url={url} body={r.text[:2000]}')
        return r.content


def list_notebooks(gc: GraphClient) -> List[Dict[str, Any]]:
    return list(gc.iter_paged('/me/onenote/notebooks?$select=id,displayName'))


def list_sections(gc: GraphClient, notebook_id: str) -> List[Dict[str, Any]]:
    q = f"/me/onenote/notebooks/{notebook_id}/sections?$select=id,displayName"
    return list(gc.iter_paged(q))


def list_section_groups(gc: GraphClient, notebook_id: str) -> List[Dict[str, Any]]:
    q = f"/me/onenote/notebooks/{notebook_id}/sectionGroups?$select=id,displayName"
    return list(gc.iter_paged(q))


def list_pages_in_section(gc: GraphClient, section_id: str) -> List[Dict[str, Any]]:
    q = (
        f"/me/onenote/sections/{section_id}/pages"
        "?$select=id,title,createdDateTime,lastModifiedDateTime,contentUrl,links"
    )
    return list(gc.iter_paged(q))


def get_page_content_html(gc: GraphClient, page_id: str) -> str:
    url = f"{GRAPH_ROOT}/me/onenote/pages/{page_id}/content"
    r = gc._request('GET', url, params={'includeIDs': 'true'})
    if r.status_code >= 400:
        raise RuntimeError(f'Graph page content error {r.status_code} page={page_id} body={r.text[:2000]}')
    return r.text


def get_section_group_children(gc: GraphClient, group_id: str) -> Dict[str, Any]:
    """Fetch a sectionGroup and expand its direct child groups + sections in ONE call.

    This reduces request volume (important for OneNote throttling / 429).
    """
    gid = (group_id or '').strip()
    if not gid:
        return {}
    q = (
        f"/me/onenote/sectionGroups/{gid}"
        "?$select=id,displayName"
        "&$expand=sectionGroups($select=id,displayName),sections($select=id,displayName)"
    )
    return gc.get_json(q)
