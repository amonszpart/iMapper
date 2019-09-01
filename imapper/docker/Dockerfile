FROM tensorflow/tensorflow:1.13.2-gpu-py3

SHELL ["/bin/bash", "-c"]

ENV TZ=Europe/London
RUN apt update \
 && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    apt-utils \
 && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    git cmake build-essential wget ssh gdb ca-certificates rsync \
    python3 python3-numpy python3-wheel python3-setuptools python3-opencv \
    ffmpeg python-dev python-pip python-opencv python-setuptools python-tk \
 && rm -rf /var/lib/apt/lists/*

RUN pip2 install tensorflow-gpu==1.14 wheel repoze.lru future ujson

RUN git clone \
    https://github.com/amonszpart/Lifting-from-the-Deep-release.git \
    /opt/Lifting-from-the-Deep-release \
 && cd /opt/Lifting-from-the-Deep-release \
 && bash ./setup.sh \
 && sed -i 's/from stealth/from imapper/g' /opt/Lifting-from-the-Deep-release/demo_aron.py \
 && git clone https://github.com/amonszpart/iMapper.git /opt/iMapper \
 && echo -e "root\nroot\n" | passwd root \
 && sed -i 's/#PermitRootLogin .*/PermitRootLogin yes/g' /etc/ssh/sshd_config \
 && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/g' /etc/ssh/sshd_config \
 && sed -i -e "s/^export PS1/[ -z \"\$PS1\" ] \&\& return\n\nexport PS1/g" /etc/bash.bashrc \
 && ln -s /usr/bin/ffmpeg /usr/bin/avconv

WORKDIR "/opt/iMapper"
