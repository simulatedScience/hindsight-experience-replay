# print filenames containing 1e5 at the beginning of the filename with added prefix "plots/"

import os

for filename in os.listdir("final_plots"):
    if filename.startswith("1e5") and filename.endswith(".png"):
        print("plots/" + filename)

