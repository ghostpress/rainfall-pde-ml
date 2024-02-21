import cdsapi

# ------------- SET UP -------------

years = [y for y in range(1990, 2021, 1)]  # 1990-2020
surface_vars = ['precipitation', '2m_dewpoint_temperature', '2m_temperature', 'convective_available_potential_energy',
                'convective_inhibition', 'k_index', 'surface_pressure', 'total_cloud_cover',
                'total_column_cloud_liquid_water', 'total_column_water_vapour',
                'vertically_integrated_moisture_divergence']  # streamfunction at 700 hPa
level_vars = ['relative_humidity', 'specific_humidity', 'temperature', 'u_component_of_wind']
levels = ['300', '500', '600', '700', '850', '925', '950']
geo = [11.6, -3.8, 4.3, 1.8]  # Ghana region bounding box
destination_path = '/home/lucia/projects/FORMES/rainfall-pde-ml/data/ERA5/new/'

# ----------------------------------

c = cdsapi.Client()

# Download surface variables:
for var in surface_vars:

    if var == 'precipitation':  # for precipitation, want daily accumulation -> all times
        times = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00',
                 '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00',
                 '22:00', '23:00']
    else:
        times = ['00:00', '06:00', '12:00', '18:00']  # all other variables, just 6h intervals

    for i in range(0, len(years), 10):  # 10-year intervals for download size limit
        subset = years[i:i + 10]
        subset = list(map(str, subset))  # convert to strings for API
        fsuff = subset[0] + '-' + subset[len(subset)-1]  # get first, last year for filename

        print(f"Requesting variable {var} for years {fsuff} on surface.")

        c.retrieve('reanalysis-era5-single-levels',
                   {'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'area': geo,
                    'variable': [var],
                    'year': subset,
                    'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                            '31'],
                    'time': times,
                    },
                   destination_path + var + '_download_' + fsuff + '.nc')

# Download variables at pressure levels:
for var in level_vars:

    for i in range(0, len(years), 10):  # 10-year intervals for download size limit
        subset = years[i:i + 10]
        subset = list(map(str, subset))  # convert to strings for API
        fsuff = subset[0] + '-' + subset[len(subset)-1]  # get first, last year for filename

        print(f"Requesting variable {var} for years {fsuff} on pressure levels {levels} (hPa).")

        c.retrieve('reanalysis-era5-pressure-levels',
                   {'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'area': geo,
                    'variable': [var],
                    'pressure_level': levels,
                    'year': subset,
                    'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                            '31'],
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    },
                   destination_path + var + '_download_' + fsuff + '.nc')
