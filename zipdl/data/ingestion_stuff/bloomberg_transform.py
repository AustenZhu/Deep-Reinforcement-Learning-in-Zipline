'''
Methodology for scraping from a bloomberg terminal with excel
 
1. get a universe.csv with the names of all tickers that you want to scrape for
2. run csv_to_blbrgxl('universe.csv', metric1, metric2, ...) 
    *REMEMBER - the metrics have to be metrics from bloomberg (ex. EQY_SH_OUT)
3. Email/send the bloomberg_extract.csv to the computer with the bloomberg terminal add ons in excel
4. copy and past a formula in the first cell below the data to grab the metrics
    ex. put =BDH(B$1, B$3, "2000-01-01", "2018-03-07") into A3
5. Copy A3-to the end of the metrics for the first ticker
6. Then drag the square in the right bottom corner all the way to the end of the list of tickers.

Use of module:
python bloomberg_transform.py universe.csv "
'''


#Really bad code - someone should refactor eventually
import csv
import argparse

def csv_to_blbrgxl(csvname, *metrics):
    '''
    Note the metrics must be the labels of metrics in bloomberg
    '''
    with open(csvname, 'r') as csvfile:
        with open('bloomberg_extract.csv', 'w') as write_to:
            for row in csvfile:
                row = row.replace(' ','').replace("'",'').split(',')
                newrow = [('', str(item) + ' US EQUITY') for item in row]
            writer = csv.writer(write_to, delimiter = ',')
            write1 = []
            for i in newrow:
                write1.extend(i)
            x = [tuple([""] + list(metrics)) for i in range(len(newrow))]
            write = []
            for i in x:
                write.extend(i)
            #print(write)
            total = [write1, write]
            writer.writerows(total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help='csv of tickers', required=True)
    parser.add_argument('metrics', action='append', help='list of metrics')
    args = parser.parse_args()
    csv_to_blbrgxl(args.csv_file, args.metrics)
