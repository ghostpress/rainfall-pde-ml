import os
import datetime
import functions

# TODO: redo with arrays only; no dicts


def process_variables(source_parent, output_parent, variable_names_dict, first, levels=False, mask=False):

    nc_files = functions.get_files(source_parent)

    # Get dataset information
    functions.print_info(nc_files)

    for vname in variable_names_dict.keys():
        functions.count_masked_variable(nc_files, vname)

    time = functions.get_time(first, nc_files)
    exit(0)
    # Load weather variables into a dictionary
    created_wind = False
    saved_wind = False

    for vname in variable_names_dict.keys():
        print("Processing variable", vname, ":", variable_names_dict[vname])
        # One variable at a time, to conserve memory
        variable_series = dict()

        if (vname == "u10" or vname == "v10") and not created_wind:
            functions.create_wind_series(nc_files, variable_series, use_mask=mask)
            created_wind = True

        elif (vname == "u10" or vname == "v10") and created_wind:
            pass
        else:
            functions.create_series(nc_files, vname, variable_series, use_mask=mask, levels=levels)

        # Rescale rainfall from m to mm
        if vname == "tp":
            functions.scale_variable(variable_series, "tp", 1000)

        #laplace = functions.compute_laplacian(precip_series[vname], vname, (10, 12))  # testing
        #print(laplace.shape)
        #print(laplace)
        #exit(0)

        # Resize variables
        resized_series = functions.resize_variables(variable_series, levels=levels)

        # Create new directory for each variable & save to daily files
        if (vname == "u10" or vname == "v10") and not saved_wind:  # wind only
            # Original size
            newdir = output_parent + "wind/"
            os.mkdir(newdir)
            functions.save_to_file(newdir, time, variable_series, "wind",
                                   one_series=False, resized=False, use_mask=mask, levels=levels)

            # Resized
            newdir = output_parent + "wind_resized/"
            os.mkdir(newdir)
            functions.save_to_file(newdir, time, resized_series, "wind",
                                   one_series=False, resized=False, use_mask=mask, levels=levels)

            saved_wind = True
        elif (vname == "u10" or vname == "v10") and saved_wind:
            pass
        else:
            # Original size
            newdir = output_parent + variable_names_dict[vname] + "/"
            os.mkdir(newdir)
            functions.save_to_file(newdir, time, variable_series, vname,
                                   one_series=False, resized=False, use_mask=mask, levels=levels)

            # Resized
            newdir = output_parent + variable_names_dict[vname] + "_resized/"
            os.mkdir(newdir)
            functions.save_to_file(newdir, time, resized_series, vname,
                                   one_series=False, resized=False, use_mask=mask, levels=levels)


if __name__ == "__main__":
    source = "/home/Lucia/Downloads/download/"
    output = "/home/lucia/Downloads/test/"
    firstDay = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)  # time units for ERA5 are hours since 1/1/1990

    levels_variables = {"r": "relativeHumidity", "t": "temperature"}
    single_variables = {"t2m": "temperature2m"}

    process_variables(source, output, levels_variables, firstDay, levels=True, mask=False)
