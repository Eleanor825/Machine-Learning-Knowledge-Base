import glob
import re
from mapping import mapping, aliases


def update_single(filepath):
    with open(filepath, mode='r', encoding='utf8') as f:
        data = f.read()
        reading_list = re.findall(pattern=r"\* +\[\d+\] +.*", string=data)
        for paper in reading_list:
            match = re.findall(pattern="^\* +\[(\d+)\] +\[?([^\[\]]*)\]?\(?(https[^\(\)]*)?\)?$", string=paper)
            assert len(match) == 1 and len(match[0]) == 3, match
            idx, label, url = match[0]
            pattern = f" +\[{idx}\] +\[?{label}\]?.*"
            label = aliases.get(label, label)
            if mapping.get(label, None):
                repl = f" [{idx}] [{label}]({mapping[label]})"
            else:
                repl = f" [{idx}] {label}"
            data = re.sub(pattern=pattern, repl=repl, string=data)
    with open(filepath, mode='w', encoding='utf8') as f:
        f.write(data)


if __name__ == "__main__":
    base_dir = "notes-cv/architectures"
    file_list = glob.glob(pathname=base_dir+'/**/*.md', recursive=True)
    for filepath in file_list:
        print(f"[INFO] Updating links in {filepath}")
        update_single(filepath=filepath)
    # update_single(filepath="notes-cv/tasks/segmentation/RefineNet.md")
