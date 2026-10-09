FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:689c34b3fdc12303418e90828359fcdb3c7053bc8aa8d7681823186dde1fe572
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=1 TENSORFOLD_NO_UPDATE_CHECK=1 TENSORFOLD_MEMORY_RESERVE_GIB=7
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev git build-essential ninja-build ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/runtime
ENV PATH=/opt/runtime/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
RUN python3 -m pip install --no-cache-dir torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
COPY docker/requirements.txt /opt/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /opt/requirements.txt
COPY patches/vision-prefix-cache.patch /opt/recipe/patches/vision-prefix-cache.patch
RUN printf '32e5e78565b4ac3a9b5f228e9828ab1ffc17d9a646bc037d26825e19a035b9c6  /opt/recipe/patches/vision-prefix-cache.patch\n' | sha256sum -c - && git init /opt/TensorFold && git -C /opt/TensorFold remote add origin https://github.com/ashhart/TensorFold.git && git -C /opt/TensorFold fetch --depth 1 origin 609ca419abecebdc5a059498a613680bd3aa847f && git -C /opt/TensorFold checkout --detach FETCH_HEAD && test "$(git -C /opt/TensorFold rev-parse HEAD)" = 609ca419abecebdc5a059498a613680bd3aa847f && git -C /opt/TensorFold apply --check /opt/recipe/patches/vision-prefix-cache.patch && git -C /opt/TensorFold apply /opt/recipe/patches/vision-prefix-cache.patch && python3 -m pip install --no-deps --no-build-isolation /opt/TensorFold
COPY scripts/serve.py /opt/recipe/serve.py
LABEL qwen.recipe=tensorfold-exl3-405 qwen.runtime.revision=609ca419abecebdc5a059498a613680bd3aa847f qwen.vision-prefix.patch.sha256=32e5e78565b4ac3a9b5f228e9828ab1ffc17d9a646bc037d26825e19a035b9c6
WORKDIR /opt/recipe
ENTRYPOINT ["python3", "/opt/recipe/serve.py"]
