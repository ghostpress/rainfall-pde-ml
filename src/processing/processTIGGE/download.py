from ecmwfapi import ECMWFDataServer

c = ECMWFDataServer()

dates_file = open('Rainfall_TestDate.txt', 'r')
read = dates_file.read()
dates_list = read.split('\n')

years = []
months = []
days = []

for i in range(len(dates_list)):
    year = dates_list[i][-8:-4]
    month = dates_list[i][-4:-2]
    day = dates_list[i][-2:]

    # TIGGE only available from Oct. 2006 onwards
    if (int(year) < 2006) or (int(year) == 2006 and int(month) < 10):
        continue
    else:
        years.append(year)
        months.append(month)
        days.append(day)

assert len(years) == len(months) == len(days)

def send_request(y, m, d):
    print(f"Requesting total precipitation in Ghana for date {y}-{m}-{d}.")

    c.retrieve({
        "area"    : "11.6/-3.8/4.3/1.8",
        "class"   : "ti",
        "dataset" : "tigge",
        "date"    : f"{y}-{m}-{d}",
        "expver"  : "prod",
        "grid"    : "0.25/0.25",
        "levtype" : "sfc",
        "origin"  : "ecmf",
        "param"   : "228228",
        "step"    : "0/18/42",
        "time"    : "12:00:00",
        "type"    : "cf",
        "format"  : "netcdf",
        "target"  : f"/home/lucia/projects/FORMES/rainfall-pde-ml/data/tigge_cf/tigge_precip_ensemble_{y}{m}{d}.nc"  # add directory path as needed
    })	#"number"  : "1/to/50",  # for perturbed ensemble forecasts only
 

if __name__ == "__main__":

    for i in range(len(years)):
        send_request(years[i], months[i], days[i])

    print("Done.")
