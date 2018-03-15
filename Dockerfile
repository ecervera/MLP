FROM ubuntu:xenial

# install pip
RUN apt-get update && apt-get install -y \
    python-pip wget\
    && rm -rf /var/lib/apt/lists/

RUN pip install --upgrade pip
RUN pip install jupyter matplotlib scipy scikit-learn pillow

ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}
