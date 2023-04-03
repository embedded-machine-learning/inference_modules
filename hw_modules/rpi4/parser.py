# The compiled benchmark returns a profile csv with layer execution times for one model
# Example usage: python3 hw_modules/rpi4/parser.py --inreport tmp/resnet_v2_101_299_1thr.csv --outfold tmp/extr_reps

__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

import os, logging
import pandas, json
from pathlib import Path

def add_measured_to_input(time_df, input_df, measured_df):
    return True

def read_report(report="", outfolder="./tmp/"):
    """Reads file in a pandas dataframe and writes layer data into a json file

    Args:
        outfolder: folder where the json data will be stored
        report: filename of the report where the data will be extracted

    Returns: False if File does not exist, otherwise extracted data frame
    """

    # check if parsed report file exists 
    if not os.path.exists(Path(report)):
        return False

    # blank lines are needed to find the correct indices
    data = pandas.read_csv(Path(report), sep=",", header=2, skip_blank_lines=False)
    print(data)
    data.columns = ["node type","first [ms]","avg [ms]","%","cdf%","mem KB", "times called", "name"]

    delegate_index = None # there should be only one in the file
    operator_wise_index = None # there should be only one in the file
    whitespace_indices = []
    next_whitespace = None

    for i, d in enumerate(data["node type"]):
        # Use data after Delegate internal if it exists in file 
        if "Delegate internal:" in str(d):
            delegate_index = i # save the delegate index
        # Use Operator-wise Profiling Info for Regular Benchmark Runs: if it exists in file
        if "Operator-wise Profiling Info for Regular Benchmark Runs:" in str(d):
            operator_wise_index = i
        if str(d) == "nan":
            whitespace_indices.append(i)

    print("Delegate Internal index: {}, Operator-wise* index: {}".format(delegate_index, operator_wise_index))
    print("Whitespace indices:", whitespace_indices)

    # get layer information starting from one of the indices + 2 (Err:510, node type) untill an empty row
    # find the next largest whitespace index compared to either delegate* or operation wise*
    if delegate_index:
        # delegate* is mightier than operation-wise*
        start_index = delegate_index
        next_whitespace = [val for val in whitespace_indices if val > delegate_index][0] 
    elif operator_wise_index and not delegate_index:
        # only operation-wise* is valid
        start_index = operator_wise_index
        next_whitespace = [val for val in whitespace_indices if val > operator_wise_index][0]
    else:
        print("No Delegate internal or Operator-wise* in file, data cannot be extracted!")
        return

    print("next_whitespace:", next_whitespace)
    # change all "[" and "]" to "" in LayerName, remove \t
    for i, elem in enumerate(data["name"]):
        elem = elem.replace("[", "") if "[" in str(elem) else elem
        elem = elem.replace("]", "") if "]" in str(elem) else elem
        elem = elem.replace("\t", " ") if "\t" in str(elem) else elem
        data["name"][i] = elem

    data.drop(["mem KB", "times called"], axis=1, inplace=True)
    data_subset = data.iloc[start_index+3 : next_whitespace]
    #print(data_subset)
    
    # construct the json file name from the report name
    outfile = os.path.join(outfolder, os.path.splitext(os.path.split(report)[1])[0] + ".json")
    
    logging.debug(outfile)
    
    _ = data_subset.to_json(outfile, orient='index')
    
    return data_subset

def r2a(report="report.csv", outfolder="./tmp/"):
    data = read_report(report, outfolder)

    if data is False:
        return False

    data["avg [ms]"] = data["avg [ms]"].astype(float)
    result = pandas.DataFrame(data[["name","avg [ms]"]].to_numpy(),columns=['name','time(ms)'])
    print(result)

    return result

def extract_data_from_folder(infold, outfold):
    """Extracts layer name and real time data from a folder of Rpi4 benchmark reports

    Args:
        infold: folder containing the reports generated by hw_modules/rpi4/inference.py with a bench_file
        outfold: folder where the extracted results will be saved

    Returns: boolean status of merging
    """

    # go over all profile csv in the infold, extract results and crush them into a single file?
    report_paths = sorted([os.path.join(infold, rp) for rp in os.listdir(infold) if os.path.isfile(os.path.join(infold, rp)) and ".csv" in rp])
    for rp in report_paths:
        print("Parsing:", rp)
        data = r2a(report=rp, outfolder=args.outfold)

    return True

if __name__ == "__main__":
    print("**********RPi4 PARSER**********")
    import argparse
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 network profile parser')
    parser.add_argument("-if", '--infold', default='./report',
                        help='Folder containing reports', required=False)
    parser.add_argument("-ir", '--inreport', default='./reports/report.csv',
                        help='Path to a report', required=False)
    parser.add_argument("-of", '--outfold', default='reports_extracted',
                        help='folder which will contain the output files', required=False)
    args = parser.parse_args()

    # make sure the outfolder and its parent directories exist
    os.makedirs(args.outfold, exist_ok = True)

    #data = r2a(report=args.inreport, outfolder=args.outfold) # extract data from one report
    ret_status = extract_data_from_folder(args.infold, args.outfold)