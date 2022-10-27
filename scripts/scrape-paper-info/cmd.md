```bash
docker image rm $(docker image ls -q -f "dangling=true")
docker container rm $(docker container ls -aq -f "status=exited")
docker image build scripts/scrape-paper-info/
docker run --interactive --tty --rm \
    scrape-paper-info python "scrape_paper_info.py"
```
