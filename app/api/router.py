"""
app/api/router.py
==================
Aggregates all versioned API routers into one.

WHY THIS EXISTS:
  main.py includes exactly ONE thing: `api_router`.
  When v2 endpoints are added, only this file changes — main.py stays clean.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import analysis, clusters, comments, intent_summaries, jobs, recommendations, summaries, transcripts, videos

api_router = APIRouter(prefix="/api")

# v1 endpoints
api_router.include_router(jobs.router,            prefix="/v1")
api_router.include_router(comments.router,        prefix="/v1")
api_router.include_router(videos.router,          prefix="/v1")
api_router.include_router(transcripts.router,     prefix="/v1")
api_router.include_router(summaries.router,       prefix="/v1")
api_router.include_router(analysis.router,        prefix="/v1")
api_router.include_router(clusters.router,        prefix="/v1")
api_router.include_router(recommendations.router,   prefix="/v1")
api_router.include_router(intent_summaries.router, prefix="/v1")
