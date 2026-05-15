import json
import os
from typing import List, Dict
import redis.asyncio as aioredis

SESSION_TTL = int(os.getenv("SESSION_TTL", "3600"))
SESSION_PREFIX = "session:"

class SessionCache:
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)

    async def get_history(self, session_id: str) -> List[Dict]:
        raw = await self.redis.get(f"{SESSION_PREFIX}{session_id}")
        if raw:
            return json.loads(raw)
        return []

    async def append_message(self, session_id: str, role: str, content: str):
        history = await self.get_history(session_id)
        history.append({"role": role, "content": content})
        history = history[-20:]
        await self.redis.setex(
            f"{SESSION_PREFIX}{session_id}",
            SESSION_TTL,
            json.dumps(history),
        )

    async def clear(self, session_id: str):
        await self.redis.delete(f"{SESSION_PREFIX}{session_id}")
