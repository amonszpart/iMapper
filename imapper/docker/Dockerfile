# Build with
# ```shell
# cd iMapper/imapper
# docker build -t imapper docker
# ```
# Run with
# ```shell
# nvidia-docker run -it --name imapper -p 22 --cap-add SYS_PTRACE imapper bash
# ```

FROM tensorflow/tensorflow:1.13.2-gpu-py3

SHELL ["/bin/bash", "-c"]

ENV TZ=Europe/London
RUN apt update && \
 DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    apt-utils && \
 DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    git cmake build-essential wget ssh gdb ca-certificates rsync \
    python3 python3-numpy python3-wheel python3-setuptools python3-opencv \
    ffmpeg python-dev python-pip python-opencv python-setuptools python-tk \
    blender vim && \
 rm -rf /var/lib/apt/lists/* && \
 echo -e "root\nroot\n" | passwd root && \
 sed -i 's/#PermitRootLogin .*/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
 sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/g' /etc/ssh/sshd_config && \
 ln -s /usr/bin/ffmpeg /usr/bin/avconv

RUN pip2 install tensorflow-gpu==1.14 wheel repoze.lru future ujson && \
 pip3 install matplotlib ujson scipy shapely descartes cython

RUN git clone \
    https://github.com/amonszpart/Lifting-from-the-Deep-release.git \
    /opt/Lifting-from-the-Deep-release && \
 cd /opt/Lifting-from-the-Deep-release && \
 bash ./setup.sh && \
 sed -i 's/from stealth/from imapper/g' /opt/Lifting-from-the-Deep-release/demo_aron.py && \
 git clone https://github.com/amonszpart/iMapper.git /opt/iMapper && \
 cd /opt/iMapper && python3 imapper/scenelet_fit/setup.py build_ext --inplace && \
 cd /opt/iMapper/i3DB && \
 wget http://geometry.cs.ucl.ac.uk/projects/2019/imapper/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling.tar.gz && \
 tar -xvzf pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling.tar.gz && \
 rm -rf pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling.tar.gz

RUN \
 sed -i -E "s/^export PS1=(.*)/\
\# Added by Aron \n\
[ -z \"\$PS1\" ] \&\& return \n\
\n\
export PS1=\1 \n\
\n\
\# Added by Aron\n\
test \"\$\(service ssh status | grep 'not'\)\" \&\& service ssh start\n/" \
/etc/bash.bashrc

WORKDIR "/opt/iMapper"
ENV PYTHONPATH="/opt/iMapper"

