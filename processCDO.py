import os
import subprocess

def dataDir(x):
    return os.getcwd() + "/data/" +  x

def outDir(x):
    return dataDir("process/testCDO/" + x)


parent = dataDir("process/2022/04/")
for f in os.listdir(parent):

    print("Processing: %s" % f)

    sel_name = "sst_geo_" + f[-26:-18] + ".nc"

    # FIXME: error in selzaxisname
    # FIXME: thetao variable does not have the "surface" z axis, it's only recorded starting from 0.5m below surface
    # FIXME: find way to filter z axis by level=0.5 so not storing 5000+ m of data
    out = subprocess.run(["cdo", "-sellonlatbox,-18,-10,34,56", "-selzaxis,2", "-selname,thetao", parent+f, outDir(sel_name)])
    print("Exit code: %d" % out.returncode)
