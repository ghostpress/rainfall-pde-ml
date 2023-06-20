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
    out = subprocess.run(["cdo", "-sellonlatbox,-18,-10,34,56", "-selzaxisname,sfc", "-selname,thetao", parent+f, outDir(sel_name)])
    print("Exit code: %d" % out.returncode)
