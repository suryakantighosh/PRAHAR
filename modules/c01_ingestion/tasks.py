"""
prahar/modules/c01_ingestion/tasks.py
Celery task definitions for C-01.
"""
import asyncio
from typing import Optional
from celery import Celery
from loguru import logger
import os

CELERY_BROKER_URL     = os.getenv("CELERY_BROKER_URL",
                                   "amqp://prahar:prahar_secret@localhost:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND",
                                   "redis://localhost:6379/0")

celery_app = Celery(
    "prahar",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    name="c01.ingest_domain",
)
def task_ingest_domain(self, domain: str, case_id: Optional[str] = None):
    from prahar.modules.c01_ingestion.engine import ingest_domain
    from uuid import UUID
    cid = UUID(case_id) if case_id else None
    return _run(ingest_domain(domain, cid))


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    name="c01.ingest_username",
)
def task_ingest_username(self, username: str, case_id: Optional[str] = None):
    from prahar.modules.c01_ingestion.engine import ingest_username
    from uuid import UUID
    cid = UUID(case_id) if case_id else None
    return _run(ingest_username(username, cid))
