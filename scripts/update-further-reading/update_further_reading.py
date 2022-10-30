import glob
import re
from mapping import mapping


def update_single(filepath):
    begin_pattern = r"\* +\[\d+\] +"
    end_pattern = r"\(https.+\)$"
    with open(filepath, mode='r', encoding='utf8') as f:
        data = f.read()
        reading_list = re.findall(pattern=begin_pattern+r".*", string=data)
        for paper in reading_list:
            assert len(re.findall(pattern=begin_pattern, string=paper)) == 1
            assert len(re.findall(pattern=end_pattern, string=paper)) <= 1
            label = re.split(pattern=begin_pattern+r"\[?|\]?"+end_pattern, string=paper)[1]
            if mapping.get(label, None):
                pattern = f" +\[(\d+)\] +\[?({label}).*"
                find = re.findall(pattern=pattern, string=paper)[0]
                replace = f" [{find[0]}] [{find[1]}]({mapping[label]})"
                data = re.sub(pattern=pattern, repl=replace, string=data)
    with open(filepath, mode='w', encoding='utf8') as f:
        f.write(data)


if __name__ == "__main__":
    base_dir = "notes-cv/tasks"
    file_list = glob.glob(pathname=base_dir+'/**/*.md', recursive=True)
    for filepath in file_list:
        print(f"[INFO] Updating links in {filepath}")
        update_single(filepath=filepath)
    # update_single(filepath="notes-cv/architectures/cnn/light-weight/ShuffleNetV1.md")
