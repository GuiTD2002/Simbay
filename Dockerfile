ARG BASE_IMAGE=rayproject/ray:2.55.1-py312-cu129
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/serve_app:/serve_app/src \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libglew2.2 \
    libglfw3 \
    libgl1 \
    libosmesa6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /serve_app

RUN python -m pip install --no-cache-dir --upgrade pip poetry

COPY pyproject.toml README.md ./
RUN poetry install --only main --no-root

RUN mkdir -p saved_plots outputs && chown -R ray /serve_app

USER ray

CMD ["bash"]
