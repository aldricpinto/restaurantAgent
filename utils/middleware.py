import time
import json
import functools
import re
from utils.logger import agent_logger as logger

def log_node_execution(func):
    """
    Decorator/Middleware that wraps LangGraph nodes to automatically log 
    their metrics, latency, inputs, outputs, and any thrown exceptions.
    """
    @functools.wraps(func)
    async def wrapper(state, *args, **kwargs):
        start_time = time.time()
        node_name = func.__name__
        
        # 1. Resolve trace_id (booking_id if present, otherwise fallback)
        booking_id = state.get("booking_id")
        trace_id = booking_id if booking_id else "initiation_phase"

        # Try to find venue_id in state
        search_result = state.get("search_result", "")
        match = re.search(r"venue_id:\s*([a-zA-Z0-9_-]+)", search_result) if search_result else None
        venue_id = match.group(1) if match else "unknown"

        # Log node start event
        logger.info(json.dumps({
            "event": "node_started",
            "node": node_name,
            "trace_id": trace_id,
            "venue_id": venue_id,
            "timestamp": time.time()
        }))

        try:
            # Execute the actual node
            result = await func(state, *args, **kwargs)
            
            # If the node returns a state update, grab booking_id/status if they just got set
            if isinstance(result, dict):
                booking_id = result.get("booking_id") or booking_id
                trace_id = booking_id if booking_id else trace_id
                status = result.get("booking_status") or state.get("booking_status", "unknown")
            else:
                status = state.get("booking_status", "unknown")

            latency = time.time() - start_time
            
            # Log successful node completion
            logger.info(json.dumps({
                "event": "node_completed",
                "node": node_name,
                "trace_id": trace_id,
                "venue_id": venue_id,
                "latency_seconds": round(latency, 3),
                "status": status,
                "timestamp": time.time()
            }))
            return result

        except Exception as e:
            latency = time.time() - start_time
            # Log exceptions/failures with context
            logger.error(json.dumps({
                "event": "node_failed",
                "node": node_name,
                "trace_id": trace_id,
                "venue_id": venue_id,
                "latency_seconds": round(latency, 3),
                "error": str(e),
                "timestamp": time.time()
            }))
            raise e

    return wrapper
