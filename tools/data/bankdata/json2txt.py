import json
import os.path as osp
import os

def get_chinese_keys(anns: dict, key: set):
    for ann in anns.values():
        for values in ann.values():
            for value in values:
                key.add(value)



if __name__ == "__main__":
    modes = ["train"]
    kinds = ["amount", "date"]
    chinese_keys = set()
    for mode in modes:
        anns = dict()
        gt_txt = osp.join(mode, "gt.txt")
        for kind in kinds:
            sub_root = f"{mode}/{kind}"
            image_prefix = osp.join(sub_root, "images")
            gt_file = osp.join(sub_root, "gt.json")
            with open(gt_file, 'r', encoding='UTF-8') as f:
                ann = json.load(f)
                anns[kind] = ann
        with open(gt_txt, "w") as f:
            for kind, val in anns.items():
                for key, value in val.items():
                    f.writelines(f"{kind}/images/{key} {value}\n")

        get_chinese_keys(anns, chinese_keys)
        with open("dict_file.txt", "w", encoding='UTF-8') as f:
            for value in chinese_keys:
                f.writelines(value+"\n")



