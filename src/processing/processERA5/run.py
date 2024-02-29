import os
import datetime
import functions


def process_variables(source_parent, output_parent, vname, first, levels=None, mask=False, start=0):

    nc_files = functions_.get_files(source_parent)
    #print(nc_files)

    # Get dataset information
    functions_.print_info(nc_files)

    functions_.count_masked_variable(nc_files, vname)

    time = functions_.get_time(first, nc_files)
    #print(time)

    varr = functions_.extract_var_array(nc_files, vname, start, use_mask=mask)
    print(varr.shape)

    #varr_resized = functions_.resize(varr, vname, (levels is not None))

    functions_.save_to_daily_files(output_parent, time, varr, vname, resized=False, use_mask=mask, levels=levels)

    pass


if __name__ == "__main__":
    source = "/home/lucia/Downloads/download/"
    output = "/home/lucia/Downloads/test/daily/"
    firstDay = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)  # time units for ERA5 are hours since 1/1/1990

    levels = ['300', '500', '600', '700', '850', '925', '950']
    levels_variables = {"r": "relativeHumidity"}
    single_variables = {"t2m": "temperature2m"}

    #for var in levels_variables.keys():
    #    print("Processing variable", var, ":", levels_variables[var])
    #    process_variables(source + var + "/", output, var, firstDay, levels=levels, mask=False, start=0)

    for var in single_variables.keys():
        print("Processing variable", var, ":", single_variables[var])
        process_variables(source + var + "/", output, var, firstDay, mask=False, start=0)
