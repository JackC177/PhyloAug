import json

data = []
chars = set("AGCTN")

with open("augmented_train.json", "r") as file:
    for line in file.readlines():
        temp = json.loads(line.strip())
        data.append({"seq":"".join(temp["seq"]).replace(" ","").upper().replace("U","T"), "label":"".join(temp["label"])})


# chars = set("AGCUN")
for row in data:
    if any(x not in chars for x in row["seq"]):
        print("".join([char for char in row["seq"] if char not in chars]))

with open("augmented_train.json", "w") as file:
    for line in data:
        line["seq"] = "".join([x for x in line["seq"] if x in chars])
        file.write(json.dumps(line)+"\n")
