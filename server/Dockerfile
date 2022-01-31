
FROM ubuntu:18.04

WORKDIR /app

# Install Python and its tools
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade

# Install all packages
RUN pip3 install \
    numpy pandas \
    torch torchvision \
    easydict \
    nltk \
    scikit-image \
    flask \
    pyyaml \
    pyopenssl

COPY . .

WORKDIR /app/code

EXPOSE 5000

ENTRYPOINT ["python3", "main.py", "--cfg", "cfg/coco.yml", "--port", "5000", "--ssl_cert", "/app/ssl/keydreams_media_mit_edu_cert.cer", "--ssl_key", "/app/ssl/keydreams.media.mit.edu.key"]
