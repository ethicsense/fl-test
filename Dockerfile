FROM python

WORKDIR /home
ADD ./model.tar.gz .

RUN mkdir -p flwr_logs
RUN mkdir -p runs

RUN apt-get update && apt-get install -y sudo
RUN chmod +w /etc/sudoers
RUN echo 'irteam ALL=(ALL) NOPASSWD:ALL' | tee -a /etc/sudoers
RUN chmod -w /etc/sudoers
RUN sudo apt-get install -y libgl1-mesa-glx
RUN sudo apt-get install -y python3-pip

# Install Packages
RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN pip install pandas
RUN pip install tensorboard
RUN pip install packaging
RUN pip install -r requirements.txt