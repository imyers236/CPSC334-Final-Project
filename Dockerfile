FROM ubuntu:latest
WORKDIR /CPSC334-Final-Project
COPY . .
RUN apt update
RUN apt install -y python3
RUN apt install -y python3-pip
RUN pip install numpy --break-system-packages
RUN pip install tabulate --break-system-packages
CMD ["python3", "trees.py"]