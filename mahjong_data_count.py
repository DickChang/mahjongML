import os, os.path
from collections import OrderedDict
import statistics
from shutil import copyfile

data_dir = './images/mahjong/data/'
data_format_dir = './images/mahjong/data_format/'
all_files = os.listdir(data_dir)
tile_accumulator = {}

for file in all_files:
    file_split = file.split('_')
    tile_type = file_split[0] + "_" + file_split[1]
    if tile_type not in tile_accumulator:
        tile_accumulator[tile_type] = 1
    else:
        tile_accumulator[tile_type] += 1

tile_accumulator_sorted = OrderedDict(sorted(tile_accumulator.items()))
for tile_type, tile_count in tile_accumulator_sorted.items():
    print(tile_type + ": " + str(tile_count))

print("---------------------------------")
tile_count_max = max(tile_accumulator, key = lambda x: tile_accumulator[x] )
tile_count_min = min(tile_accumulator, key = lambda x: tile_accumulator[x] )
tile_count_mean = statistics.mean(tile_accumulator.values())
tile_count_median = statistics.median(tile_accumulator.values())
print( "Max: " + tile_count_max + ": " + str( tile_accumulator[tile_count_max] ) )
print( "Min: " + tile_count_min + ": " + str( tile_accumulator[tile_count_min] ) )
print( "Mean: " + str( tile_count_mean ) )
print( "Median: " + str( tile_count_median ) )
print( "Total: " + str( sum(tile_accumulator.values()) ) )

def rename_files(all_files, tile_accumulator):
    for tile_type in tile_accumulator.keys():
        tile_type_files = [x for x in all_files if tile_type in x]
        i = 0
        for file in tile_type_files:
            file_split = file.split('_')
            ext = file.split('.')[1]
            new_name = file_split[0] + "_" + file_split[1] + "_" + '%07d' % i + '.' + ext
            #print("Renaming: " + file + " to " + new_name)
            #os.rename(data_dir + file, data_dir + new_name)
            copyfile(data_dir + file, data_format_dir + new_name)
            i = i + 1
        #print(tile_type_files)

#rename_files(all_files, tile_accumulator)
