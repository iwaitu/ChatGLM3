FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

EXPOSE 8000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    apt-utils \
    vim \
    git \
    wget \
    gcc \
    g++ \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# RUN mkdir -p /root/.cache/torch/hub/checkpoints
# RUN wget -P /root/.cache/torch/hub/checkpoints https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_aa-076e3472.pth

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda init && conda config --set always_yes yes --set changeps1 no
RUN conda --version
RUN pip install --upgrade pip
RUN git clone https://github.com/THUDM/ChatGLM3 /root/ChatGLM3
WORKDIR /root/ChatGLM3
RUN pip install -r requirements.txt
RUN mkdir chatglm3-6b
COPY ./chatglm3-6b/ /root/ChatGLM3/chatglm3-6b/
COPY ./openai_api.py /root/ChatGLM3/openai_api.py
COPY ./utils.py /root/ChatGLM3/utils.py
CMD ["python", "openai_api.py"]
