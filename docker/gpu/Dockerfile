FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

MAINTAINER Tetsuya Ishikawa <tiskw111@gmail.com>

# Set environment variables.
ENV DEBIAN_FRONTEND=noninteractive
ENV USERNAME=developer

# Copy and run the installer.
COPY installer.sh /installer.sh 
RUN sh installer.sh

# Copy a shell script for dynamic user creation.
COPY entrypoint.sh /entrypoint.sh

# Unlock permissions for the above "entrypoint.sh".
RUN chmod u+s /usr/sbin/useradd /usr/sbin/groupadd

# Set locales.
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

ENTRYPOINT ["sh", "/entrypoint.sh"]

RUN pip install matplotlib ipykernel

ARG GID=1000
ARG UID=1000
ARG USERNAME=user1
RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -s /bin/bash -u ${UID} -g ${GID} -G sudo ${USERNAME} \
    && echo ${USERNAME}:${USERNAME} | chpasswd \
    && echo "${USERNAME}   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USERNAME}

CMD ["/bin/bash"]
