import pandas as pd
from geopy.geocoders import GoogleV3, Nominatim, Bing, DataBC

def format_zipcode(zipcode):
    zipcode = str(zipcode)
    if len(zipcode) == 0 or len(zipcode) == 5:
        return zipcode
    elif len(zipcode) == 4:
        return '0'+zipcode
    elif len(zipcode) == 8:
        return '0'+zipcode[:4]+'-'+zipcode[4:]


def get_full_address(npidata_file):
    df = pd.read_csv(npidata_file)
    mail_address = list(df['Provider First Line Business Mailing Address'].fillna('') + ', ' + \
                            df['Provider Second Line Business Mailing Address'].fillna('') + ', ' + \
                            df['Provider Business Mailing Address City Name'].fillna('') + ', ' + \
                            df['Provider Business Mailing Address State Name'].fillna('') + ', ' + \
                            df['Provider Business Mailing Address Postal Code'].fillna('').apply(str) + ', ' + \
                            df['Provider Business Mailing Address Country Code (If outside U.S.)'].fillna(''))
    addr_df = pd.DataFrame({'mailing_addr': mail_address}, index=df['NPI'])
    addr_dict = addr_df.to_dict()['mailing_addr']
    return addr_dict

def get_npi_coordinates(addr_dict):
    geolocator = Nominatim(user_agent="sumanta_mms")
    coord_dict = {}
    for n in addr_dict.keys():
        address_string = addr_dict[n]
        loc = geolocator.geocode(address_string)
        if loc is not None:
            coord_dict[n] = (loc.address, loc.latitude, loc.longitude)
            print("Got coordinates")
        else:
            print("Could not get coordinates for "+address_string)
            coord_dict[n] = ()
    return coord_dict

def get_npi_coordinates_from_file(npis, exact_file, zip_file):
    exact_loc_dict = {}
    zip_loc_dict = {}
    with open(exact_file, 'r') as ef:
        for l in ef:
            npi = int(l.split(',')[0])
            lat = float(l.split(',')[1])
            lng = float(l.split(',')[2])
            exact_loc_dict[npi] = (lat, lng)
    with open(zip_file, 'r') as zf:
        for l in zf:
            npi = int(l.split(',')[0])
            lat = float(l.split(',')[1])
            lng = float(l.split(',')[2])
            zip_loc_dict[npi] = (lat, lng)
    loc_dict = {}
    for n in npis:
        if n in exact_loc_dict.keys():
            loc_dict[n] = exact_loc_dict[n]
        elif n in zip_loc_dict.keys():
            loc_dict[n] = zip_loc_dict[n]
        else:
            loc_dict[n] = (0.0, 0.0)
    return loc_dict