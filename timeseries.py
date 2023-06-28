import os
import subprocess

def dataDir(x):
    return os.getcwd() + "/data/cdo_out/" +  x

def outDir(x):
    return dataDir("trend/" + x)

# Select lat/lon point -> store into temp file
# Run cdo -trend

pt = (-17.97, 34.03)

print("Creating time series for (lat,lon) point: (" + str(pt[1]) + ", " + str(pt[0]) + ").")

dataFile = outDir("trendData.nc")

# FIXME: average these?
out = subprocess.run(["cdo", "-sellonlatbox,-17.9699,-17.97,34.0299,34.03", dataDir("test.nc"), outDir("trendData.nc")])
print("Exit code: %d" % out.returncode)

print("Running trend analysis for chosen point.")

out = subprocess.run(["cdo", "-trend", dataFile, outDir("trendA.nc"), outDir("trendB.nc")])
print("Exit code: %d" % out.returncode)

