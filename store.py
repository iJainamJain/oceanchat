"""
Local persistence for OceanChat: conversation threads and long-term memories.

Everything lives in JSON under store/ so it survives restarts without needing a
database. Both files are written atomically (temp file + replace) so an interrupted
save cannot leave a truncated file behind.
"""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone

STORE_DIR = "store"
CHATS_PATH = os.path.join(STORE_DIR, "chats.json")
MEMORIES_PATH = os.path.join(STORE_DIR, "memories.json")

MEMORY_TYPES = ["user", "preference", "project", "reference"]

# Keep the injected block bounded so it cannot crowd out the agent's real prompt.
MAX_MEMORIES_IN_PROMPT = 25


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A corrupt store should degrade to empty, never crash the app.
        return default


def _write_json(path, payload, attempts=5):
    os.makedirs(STORE_DIR, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())

    # On Windows os.replace raises PermissionError when the destination is briefly
    # held open by another handle (search indexer, AV scanner, an overlapping
    # Streamlit rerun). Without the retry the update is silently lost: the temp
    # file keeps the new data and the live file keeps the old.
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))

    # Last resort: overwrite in place rather than drop the write entirely.
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    try:
        os.remove(tmp)
    except OSError:
        pass


def slugify(text, fallback="memory"):
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:48] or fallback


# ============================== CHATS ==============================
def load_chats():
    """Return the chat list, newest activity first."""
    chats = _read_json(CHATS_PATH, [])
    if not isinstance(chats, list):
        return []
    return sorted(chats, key=lambda c: c.get("updated_at", ""), reverse=True)


def save_chats(chats):
    _write_json(CHATS_PATH, chats)


def new_chat(chats, title="New conversation"):
    chat = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }
    chats.insert(0, chat)
    save_chats(chats)
    return chat


def get_chat(chats, chat_id):
    return next((c for c in chats if c["id"] == chat_id), None)


def delete_chat(chats, chat_id):
    remaining = [c for c in chats if c["id"] != chat_id]
    save_chats(remaining)
    return remaining


def rename_chat(chats, chat_id, title):
    chat = get_chat(chats, chat_id)
    if chat:
        chat["title"] = title.strip() or chat["title"]
        chat["updated_at"] = _now()
        save_chats(chats)
    return chats


def append_message(chats, chat_id, role, content):
    chat = get_chat(chats, chat_id)
    if not chat:
        return chats
    chat["messages"].append({"role": role, "content": content, "at": _now()})
    chat["updated_at"] = _now()

    # First user line becomes the thread title, the way most chat apps do it.
    if role == "user" and chat["title"] == "New conversation":
        words = content.strip().split()
        chat["title"] = " ".join(words[:7]) + ("..." if len(words) > 7 else "")

    save_chats(chats)
    return chats


def chat_to_markdown(chat):
    lines = [f"# {chat['title']}", "", f"_Started {chat['created_at']}_", ""]
    for msg in chat["messages"]:
        who = "You" if msg["role"] == "user" else "OceanChat"
        lines += [f"**{who}**", "", msg["content"], ""]
    return "\n".join(lines)


# ============================== MEMORIES ==============================
def load_memories():
    memories = _read_json(MEMORIES_PATH, [])
    return memories if isinstance(memories, list) else []


def save_memories(memories):
    _write_json(MEMORIES_PATH, memories)


def add_memory(memories, name, description, content, mem_type="project", save=True):
    """Add or update by slug. Returns (memories, created) so callers can report.

    Pass save=False when adding several in a loop, then call save_memories() once.
    """
    slug = slugify(name or description)
    existing = next((m for m in memories if m["name"] == slug), None)

    if existing:
        existing.update(
            {
                "description": description,
                "content": content,
                "type": mem_type,
                "updated_at": _now(),
            }
        )
        if save:
            save_memories(memories)
        return memories, False

    memories.append(
        {
            "id": uuid.uuid4().hex[:12],
            "name": slug,
            "description": description,
            "content": content,
            "type": mem_type if mem_type in MEMORY_TYPES else "project",
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    if save:
        save_memories(memories)
    return memories, True


def delete_memory(memories, memory_id):
    remaining = [m for m in memories if m["id"] != memory_id]
    save_memories(remaining)
    return remaining


def memories_block(memories):
    """Render memories as a prompt fragment, or '' when there are none."""
    if not memories:
        return ""

    ordered = sorted(memories, key=lambda m: m.get("updated_at", ""), reverse=True)
    ordered = ordered[:MAX_MEMORIES_IN_PROMPT]

    lines = [
        "",
        "Known context about this user and project (remember it, but only mention "
        "it when relevant):",
    ]
    for mem in ordered:
        lines.append(f"- [{mem['type']}] {mem['content']}")
    return "\n".join(lines)
