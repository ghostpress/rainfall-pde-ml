import os
import subprocess

def dataDir(x):
    return os.getcwd() + "/data/" +  x

def outDir(x):
    return dataDir("cdo_out/" + x)


super_parent = dataDir("process")
year_paths = os.listdir(super_parent)

print("Processing data in: %s" % super_parent)

for y in year_paths:

    print("Descending into sub-directory: %s" % y)
    
    yf = super_parent + "/" + y
    month_paths = os.listdir(yf)

    for m in month_paths:

        mf = yf + "/" + m
        print("Descending into sub-directory: %s" % m)

        for f in os.listdir(mf):
            print("Processing: %s" % f)

            sel_name = "sst_geo_" + f[-26:-18] + ".nc"
            out = subprocess.run(["cdo", "-sellonlatbox,-18,-10,34,56", "-sellevel,0.494024992", "-selname,thetao", mf+"/"+f, outDir(sel_name)])
            print("Exit code: %d" % out.returncode)
