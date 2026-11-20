from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import FRONTEND_ORIGINS
from .db import pool
from .routers import audit_log, auth, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open()
    yield
    pool.close()


app = FastAPI(title="CCTV Incident Platform", lifespan=lifespan)

# Lets the dashboard (on another address, e.g. localhost:3000) call the API and send
# the login cookie. The dashboard must use fetch(..., {credentials: "include"}).
# Never use "*" here: browsers refuse cookies with a wildcard origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API = "/api/v1"
for r in (auth.router, users.router, audit_log.router):
    app.include_router(r, prefix=API)

# Yee Mon: add your routers here the same way, e.g.
#   from .routers import incidents
#   app.include_router(incidents.router, prefix=API)
# and protect them with `user: CurrentUser` / `user: AdminUser` from app.deps.


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}
