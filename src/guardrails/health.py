import asyncio
import logging
import time
from typing import Any, Dict, Literal, Optional

import httpx
from fastapi import APIRouter

logger = logging.getLogger(__name__)

HealthStatus = Literal["healthy", "degraded", "unhealthy"]

router = APIRouter(prefix="/health", tags=["health"])


async def check_router_model(timeout: float = 2.0) -> Dict[str, Any]:
    """Kiểm tra Router model đang load và có thể inference."""
    t0 = time.perf_counter()
    try:
        from src.router.model import RouterModel
        model = RouterModel()          # trả về singleton (không load lại)
        loaded = model.is_loaded()
        latency_ms = (time.perf_counter() - t0) * 1000

        if not loaded:
            return {
                "status": "unhealthy",
                "latency_ms": round(latency_ms, 1),
                "model_loaded": False,
                "error": "Model not loaded yet",
            }

        # Chạy 1 inference nhẹ để kiểm tra latency thực tế
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, model.generate, "test ping"
            ),
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 1),
            "model_loaded": True,
            "inference_ok": True,
        }
    except asyncio.TimeoutError:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "degraded",
            "latency_ms": round(latency_ms, 1),
            "model_loaded": True,
            "error": f"Inference timed out (>{timeout}s)",
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
    """Kiểm tra Neo4j connection bằng neo4j driver trực tiếp."""
    import os
    t0 = time.perf_counter()
    try:
        from neo4j import GraphDatabase, exceptions as neo4j_exc

        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "secret")

        def _ping():
            driver = GraphDatabase.driver(neo4j_url, auth=(user, password))
            try:
                driver.verify_connectivity()
                return True
            finally:
                driver.close()

        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _ping),
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 1),
            "url": neo4j_url,
        }
    except ImportError:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "degraded",
            "latency_ms": round(latency_ms, 1),
            "error": "neo4j package not installed (pip install neo4j)",
        }
    except asyncio.TimeoutError:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "unhealthy",
            "latency_ms": round(latency_ms, 1),
            "url": neo4j_url,
            "error": f"Connection timed out (>{timeout}s)",
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"[health/graph] {e}")
        return {
            "status": "unhealthy",
            "latency_ms": round(latency_ms, 1),
            "url": neo4j_url,
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