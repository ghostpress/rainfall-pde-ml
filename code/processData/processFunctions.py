import os
import subprocess

def read_config_file():

    print("Reading config settings...")

    with open("config.txt", "r") as f:
        lines = [line.rstrip() for line in f]

    return lines
        

def process_file(f, commands, out_name, dest_dir):
    assert os.path.isfile(f), "Can't run CDO commands on a directory."
        
    print("Processing %s" % f)

    fname = out_name + f[-26:-18] + ".nc"
    exe = commands.extend([f, dest_dir+"/"+fname])
    out = subprocess.run(commands)

    return out.returncode
