FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg espeak-ng

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    transformers==4.39.2 diffusers==0.27.2 accelerate==0.28.0 omegaconf munch pydub phonemizer einops einops-exts git+https://github.com/resemble-ai/monotonic_align.git nltk==3.8 librosa matplotlib && \
	GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/yl4579/StyleTTS2 /content/tost-tts && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/TostAI/tost-tts/resolve/main/tost-tts-base-v1.3.pth -d /content/tost-tts/Models/LJSpeech -o epoch_2nd_00030.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/TostAI/tost-tts/raw/main/config_ft.yml -d /content/tost-tts/Models/LJSpeech -o config_ft.yml
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/bucilianus-1/resolve/main/epoch_2nd_00030.pth -d /content/tost-tts/Models/LJSpeech -o epoch_2nd_00030.pth && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/bucilianus-1/raw/main/config.yml -d /content/tost-tts/Models/LJSpeech -o config_ft.yml

COPY ./worker_runpod.py /content/tost-tts/worker_runpod.py
WORKDIR /content/tost-tts
CMD python worker_runpod.py