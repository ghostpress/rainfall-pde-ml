import os
import subprocess

# TODO: don't assume number of subdirectories, try checking how many there are and/or using recursion to descend

def process_files(wd, commands, out_name, dest_dir):

    print("Working directory: %s " % wd)
    
    for f in os.listdir(wd):

        assert os.path.isfile(f), "Can't run CDO commands on a directory, check that your config.txt is correct."
        
        print("Processing: %s" % f)

        fname = out_name + f[-26:-18] + ".nc"
        exe = commands.append(fname, wd+"/"+f, dest_dir+"/"+out_name)

        out = subprocess.run(commands)
        print("Exit code: %d" % out.returncode)

    return    
