import os
import json
import re
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

from .auth import acquire_token_device_flow, load_env_scopes
from .graph import (
    GraphClient,
    list_notebooks,
    list_sections,
    list_section_groups,
    get_section_group_children,
    list_pages_in_section,
    get_page_content_html,
)
from .markdown import slugify, md5_hex, render_markdown, html_to_blocks


def _sanitize_filename(name: str) -> str:
    name = (name or '').strip()
    name = re.sub(r"[\\/:*?\"<>\n]", "-", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip() or 'page'


def _jwt_claims_noverify(token: str) -> Dict[str, Any]:
    try:
        parts = (token or '').split('.')
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        payload_b64 += '=' * (-len(payload_b64) % 4)
        raw = base64.urlsafe_b64decode(payload_b64.encode('utf-8'))
        return json.loads(raw.decode('utf-8', errors='replace'))
    except Exception:
        return {}


def log_token_scopes(token: str, *, label: str = 'Graph token') -> None:
    claims = _jwt_claims_noverify(token)
    scp = claims.get('scp')
    roles = claims.get('roles')
    aud = claims.get('aud')
    appid = claims.get('appid')
    tid = claims.get('tid')
    exp = claims.get('exp')
    print(f"[{label}] aud={aud} appid={appid} tid={tid} exp={exp}")
    if scp:
        print(f"[{label}] scp={scp}")
    if roles:
        print(f"[{label}] roles={roles}")
    if not scp and not roles:
        print(f"[{label}] scp/roles not present in token")


@dataclass
class ExportConfig:
    tenant_id: str
    client_id: str
    additional_scopes: str
    output_dir: Path
    token_cache: Path
    notebook_name: Optional[str] = None
    notebook_id: Optional[str] = None
    merge: bool = False
    formats: str = 'md'


def resolve_notebook(gc: GraphClient, notebook_name: Optional[str], notebook_id: Optional[str]) -> Dict[str, Any]:
    nbs = list_notebooks(gc)
    if notebook_id:
        for nb in nbs:
            if nb.get('id') == notebook_id:
                return nb
        raise RuntimeError(f"Notebook id not found: {notebook_id}")
    if not notebook_name:
        raise RuntimeError("Provide --notebook or --notebook-id")
    needle = notebook_name.strip().lower()
    for nb in nbs:
        if (nb.get('displayName') or '').strip().lower() == needle:
            return nb
    for nb in nbs:
        if needle in (nb.get('displayName') or '').strip().lower():
            return nb
    raise RuntimeError(f"Notebook not found (name match): {notebook_name}")


def iter_sections_recursive(gc: GraphClient, notebook_id: str, *, debug_token: Optional[str] = None) -> Iterator[Tuple[Dict[str, Any], List[str]]]:
    # Top-level sections
    for sec in list_sections(gc, notebook_id):
        yield sec, []

    if debug_token:
        log_token_scopes(debug_token, label='Before sectionGroups traversal')

    # Breadth-first traversal to reduce burstiness
    queue: List[Tuple[Dict[str, Any], List[str]]] = []
    for grp in list_section_groups(gc, notebook_id):
        queue.append((grp, []))

    seen_groups = set()
    while queue:
        group, path = queue.pop(0)
        gid = (group.get('id') or '').strip()
        if not gid or gid in seen_groups:
            continue
        seen_groups.add(gid)

        gname = (group.get('displayName') or '').strip()
        gpath = path + ([gname] if gname else [])

        # ONE call per group (expand child groups + sections)
        data = get_section_group_children(gc, gid)

        # sections
        for sec in (data.get('sections') or []):
            if isinstance(sec, dict):
                yield sec, gpath

        # child groups
        for child in (data.get('sectionGroups') or []):
            if isinstance(child, dict):
                queue.append((child, gpath))


def export_notebook(cfg: ExportConfig) -> Path:
    delegated = load_env_scopes(cfg.additional_scopes)
    delegated = [f"https://graph.microsoft.com/{s}" for s in delegated]
    token = acquire_token_device_flow(cfg.client_id, cfg.tenant_id, delegated, cfg.token_cache)
    log_token_scopes(token, label='After auth')

    # Pace requests a bit to reduce 429 likelihood
    min_delay = float(os.environ.get('GRAPH_MIN_DELAY', '0.15') or 0.15)
    gc = GraphClient.create(token, min_delay_s=min_delay)

    nb = resolve_notebook(gc, cfg.notebook_name, cfg.notebook_id)
    nb_name = nb.get('displayName') or 'notebook'
    notebook_slug = nb_name
    out_root = cfg.output_dir / notebook_slug
    pages_dir = out_root / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'notebook': notebook_slug,
        'notebook_id': nb.get('id'),
        'generated_at': None,
        'pages_written': [],
        'assets_written': [],
        'errors': [],
    }

    merged_lines: List[str] = []
    jsonl_lines: List[str] = []

    for sec, group_path in iter_sections_recursive(gc, nb['id'], debug_token=token):
        sec_name = (sec.get('displayName') or '').strip()
        sec_id = (sec.get('id') or '').strip()
        if not sec_id:
            continue

        pages = list_pages_in_section(gc, sec_id)
        for p in pages:
            try:
                page_id = p.get('id')
                title = p.get('title') or 'Page'
                created = p.get('createdDateTime') or ''
                modified = p.get('lastModifiedDateTime') or ''
                links = p.get('links') or {}
                web_url = (links.get('oneNoteWebUrl') or {}).get('href') or ''
                client_url = (links.get('oneNoteClientUrl') or {}).get('href') or ''

                html = get_page_content_html(gc, page_id)
                blocks = html_to_blocks(html)

                asset_pairs: List[Tuple[str, str]] = []
                page_folder = out_root / page_id
                page_folder.mkdir(parents=True, exist_ok=True)
                for kind, payload in blocks:
                    if kind not in ('image', 'audio'):
                        continue
                    src = payload.get('src', '')
                    if not src:
                        continue
                    h = md5_hex(src)
                    ext = Path(src.split('?', 1)[0]).suffix
                    if not ext:
                        ext = '.jpg' if kind == 'image' else '.m4a'
                    fname = f"res-{h}{ext}" if kind == 'image' else f"aud-{h}{ext}"
                    rel = f"{page_id}/{fname}"
                    fpath = page_folder / fname
                    if not fpath.exists():
                        data = gc.download(src)
                        fpath.write_bytes(data)
                        manifest['assets_written'].append(rel)
                    asset_pairs.append((src, rel))

                try:
                    short = page_id.split('!')[0].split('-', 1)[1][:6]
                except Exception:
                    short = md5_hex(page_id)[:6]
                md_name = f"{slugify(title)}-{short}.md"
                md_path = pages_dir / md_name
                if md_path.exists():
                    i = 2
                    while True:
                        alt = pages_dir / f"{slugify(title)}-{short}-{i}.md"
                        if not alt.exists():
                            md_path = alt
                            break
                        i += 1

                group_str = ' / '.join([x for x in group_path if x])
                section_path = ' / '.join([x for x in (group_path + ([sec_name] if sec_name else [])) if x])

                frontmatter = {
                    'notebook': notebook_slug,
                    'section': sec_name,
                    'section_group': group_str,
                    'section_path': section_path or sec_name,
                    'section_id': f"1-{sec_id.split('-')[0]}" if sec_id else '',
                    'title': _sanitize_filename(title),
                    'page_id': page_id,
                    'created': created,
                    'modified': modified,
                    'web_url': web_url,
                    'client_url': client_url,
                }

                md_text = render_markdown(frontmatter, _sanitize_filename(title), blocks, asset_pairs)
                md_path.write_text(md_text, encoding='utf-8')

                manifest['pages_written'].append({
                    'page_id': page_id,
                    'file': str(md_path.relative_to(out_root)),
                    'section': sec_name,
                    'section_group': group_str,
                    'section_path': section_path or sec_name,
                    'title': title,
                })

                if cfg.merge:
                    merged_lines.append(md_text)

                if 'jsonl' in [f.strip() for f in cfg.formats.split(',')]:
                    rec = {
                        'notebook': notebook_slug,
                        'section': sec_name,
                        'section_group': group_str,
                        'section_path': section_path or sec_name,
                        'section_id': sec_id,
                        'page_id': page_id,
                        'title': title,
                        'created': created,
                        'modified': modified,
                        'web_url': web_url,
                        'client_url': client_url,
                        'markdown_path': str(md_path.relative_to(out_root)),
                    }
                    jsonl_lines.append(json.dumps(rec, ensure_ascii=False))

            except Exception as e:
                manifest['errors'].append({'page_id': p.get('id'), 'error': str(e)})

    if cfg.merge:
        (out_root / 'merged.md').write_text('\n\n'.join(merged_lines).strip() + '\n', encoding='utf-8')
    if jsonl_lines:
        (out_root / 'merged.jsonl').write_text('\n'.join(jsonl_lines) + '\n', encoding='utf-8')

    (out_root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_root
