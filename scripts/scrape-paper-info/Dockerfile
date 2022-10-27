FROM python:3.11.0-slim

ENV wd=/c/Users/d6mao/Desktop/ml
WORKDIR ${wd}
COPY ${wd}/papers-cv/ .
COPY requirements.txt .
# COPY scrape_paper_info.py .
RUN python -m pip install -r requirements.txt
