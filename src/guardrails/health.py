"""
C3: Health check endpoints cho tất cả services.
/health/router    — kiểm tra Router model
/health/generator — kiểm tra LLM Generator (SGLang/vLLM)
/health/graph     — kiểm tra Neo4j connection
/health/cache     — kiểm tra Semantic Cache
/health           — overall aggregated health
"""

import asyncio
import logging
import time
from typing import Any, Dict, Literal, Optional

import httpx
from fastapi import APIRouter

logger = logging.getLogger(__name__)

HealthStatus = Literal["healthy", "degraded", "unhealthy"]

router = APIRouter(prefix="/health", tags=["health"])


# ---------------------------------------------------------------------------
# Individual checkers
# ---------------------------------------------------------------------------

async def check_router_model(timeout: float = 2.0) -> Dict[str, Any]:
    """Kiểm tra Router model đang load và có thể inference."""
    t0 = time.perf_counter()
    try:
        from src.router.classifier import RouterClassifier
        classifier = RouterClassifier.get_instance()
        # Quick classify test
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, classifier.classify, "test ping"
            ),
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 1),
            "model_loaded": True,
            "test_intent": result.get("intent", "?"),
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"[health/router] {e}")
        return {
            "status": "unhealthy",
            "latency_ms": round(latency_ms, 1),
            "error": str(e),
        }


async def check_generator(
    sglang_url: str = "http://localhost:30000",
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """Kiểm tra SGLang/vLLM Generator endpoint."""
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{sglang_url}/health")
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 200:
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 1),
                "url": sglang_url,
                "http_status": resp.status_code,
            }
        else:
            return {
                "status": "degraded",
                "latency_ms": round(latency_ms, 1),
                "url": sglang_url,
                "http_status": resp.status_code,
            }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"[health/generator] {e}")
        return {
            "status": "unhealthy",
            "latency_ms": round(latency_ms, 1),
            "url": sglang_url,
            "error": str(e),
        }


async def check_graph_db(
    neo4j_url: str = "bolt://localhost:7687",
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """Kiểm tra Neo4j connection."""
    t0 = time.perf_counter()
    try:
        from src.graph.client import get_neo4j_client
        client = get_neo4j_client()
        # Simple query
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, client.ping
            ),
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "healthy" if result else "degraded",
            "latency_ms": round(latency_ms, 1),
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"[health/graph] {e}")
        return {
            "status": "unhealthy",
            "latency_ms": round(latency_ms, 1),
            "error": str(e),
        }


async def check_cache() -> Dict[str, Any]:
    """Kiểm tra Semantic Cache stats."""
    t0 = time.perf_counter()
    try:
        from src.cache.stats import get_stats
        stats = get_stats().to_dict()
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 1),
            **stats,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "degraded",
            "latency_ms": round(latency_ms, 1),
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@router.get("/router")
async def health_router():
    result = await check_router_model()
    status_code = 200 if result["status"] == "healthy" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)


@router.get("/generator")
async def health_generator():
    import os
    url = os.environ.get("SGLANG_GENERATOR_URL", "http://localhost:30000")
    result = await check_generator(sglang_url=url)
    status_code = 200 if result["status"] == "healthy" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)


@router.get("/graph")
async def health_graph():
    import os
    url = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    result = await check_graph_db(neo4j_url=url)
    status_code = 200 if result["status"] == "healthy" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)


@router.get("/cache")
async def health_cache():
    result = await check_cache()
    status_code = 200 if result["status"] != "unhealthy" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)


@router.get("")
@router.get("/")
async def health_overall():
    """Aggregated health — chạy song song tất cả checks."""
    import os

    checks = await asyncio.gather(
        check_router_model(),
        check_generator(os.environ.get("SGLANG_GENERATOR_URL", "http://localhost:30000")),
        check_graph_db(os.environ.get("NEO4J_URL", "bolt://localhost:7687")),
        check_cache(),
        return_exceptions=True,
    )

    names = ["router", "generator", "graph", "cache"]
    results = {}
    for name, check in zip(names, checks):
        if isinstance(check, Exception):
            results[name] = {"status": "unhealthy", "error": str(check)}
        else:
            results[name] = check

    # Overall status
    statuses = [r.get("status", "unhealthy") for r in results.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
        http_code = 200
    elif any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
        http_code = 503
    else:
        overall = "degraded"
        http_code = 200   # 200 vì vẫn hoạt động được

    payload = {
        "status": overall,
        "timestamp": time.time(),
        "services": results,
    }

    from fastapi.responses import JSONResponse
    return JSONResponse(content=payload, status_code=http_code)