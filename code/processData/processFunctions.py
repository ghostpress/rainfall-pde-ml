import os
import subprocess

# TODO: use a config file instead of asking for user input
# TODO: don't assume number of subdirectories, try checking how many there are and/or using recursion to descend

def ask_for_data_dir():
    path = input("Enter top data directory: ")
    return path

def ask_for_dest_dir():
    path = input("Enter destination directory: ")
    return path

def process_data_in_dir(in_path, dest_path, vs, lon, lat, lev):

    print("Processing data in %s" % in_path)
    sub_path = os.listdir(in_path) # get all the sub-dirs of the top data dir; in this case, years

    for s in sub_path:

        print("Descending into %s" % s)

        sf = path + "/" + s
        ss_path = os.listdir(sf) # get all the sub-sub-dirs; in this case, months 

        for ss in ss_path:

            ssf = sf + "/" + ss
            print("Descending into %s" % ssf)

            for f in os.listdir(ssf): # now at the file level
                print("Processing: %s" % f)

                sel_name = "sst_geo_" + f[-26:-18] + ".nc"

                lonlat_command = "-sellonlatbox,"
                lonlat_command += str(lon[0]) + "," + str(lon[1]) + "," + str(lat[0]) + "," + str(lat[1])
                level_command = "-sellevel,"
                level_command += str(lev)

                vars_command = "-selname,"

                for i in range(len(vs)):
                    if i < (len(vs)-1):
                        vars_command += v + ","
                    else:
                        vars_command += v

                out = subprocess.run(["cdo", lonlat_command, level_command, vars_command, ssf+"/"+f, dest_path])
                print("Exit code: %d" % out.returncode)
            

            
