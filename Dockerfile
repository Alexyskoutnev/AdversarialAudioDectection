FROM ubuntu:22.04

# ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install
RUN apt-get install -y cmake git python3

RUN git clone git@github.com:Alexyskoutnev/AdversarialAudioDectection.git







# COPY . .