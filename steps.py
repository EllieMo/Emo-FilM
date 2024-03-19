import json
import os
from copy import deepcopy as dc

import pandas as pd

f = open("jsonfile")
template = json.load(f)

scanlist = os.listdir("scans")

for fs in scanlist:
    scans = pd.read_csv(f"scans/{fs}", sep="\t")
    os.makedirs(os.path.join(fs[:7], fs[8:13], "func"), exist_ok=True)

    for t, d in scans.values:
        if "func" in t:
            print(t[:-7] + ".json")
            print(t[24:-12])
            print(d[11:])
            df = dc(template)
            df["TaskName"] = t[24:-12]
            df["AcquisitionTime"] = d[11:]

            filename = os.path.join(fs[:7], fs[8:13], f"{t[:-7]}.json")

            with open(filename, "w") as out:
                out.write(json.dumps(df, indent=4))
