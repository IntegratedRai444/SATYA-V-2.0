#!/usr/bin/env python3
"""
Gunicorn configuration for SatyaAI production deployment
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5002)}"
backlog = 2048

# Worker processes
workers = int(os.environ.get("WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2
max_requests = 1000
max_requests_jitter = 100

# Restart workers after this many requests, with up to jitter random requests
preload_app = True

# Logging
accesslog = os.environ.get("ACCESS_LOG", "-")  # stdout
errorlog = os.environ.get("ERROR_LOG", "-")  # stderr
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "satyaai-server"

# Server mechanics
daemon = False
pidfile = "/tmp/satyaai.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if certificates are provided)
keyfile = os.environ.get("SSL_KEYFILE")
certfile = os.environ.get("SSL_CERTFILE")

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190


def when_ready(server):
    """Called just after the server is started"""
    server.log.info("SatyaAI server is ready. Listening on: %s", server.address)


def worker_int(worker):
    """Called just after a worker has been killed by a signal"""
    worker.log.info("Worker received INT or QUIT signal")


def pre_fork(server, worker):
    """Called just before a worker is forked"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def post_fork(server, worker):
    """Called just after a worker has been forked"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal"""
    worker.log.info("Worker received SIGABRT signal")
