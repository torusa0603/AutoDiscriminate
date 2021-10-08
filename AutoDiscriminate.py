import mWatershed
import sys 
import re

result_folder_path = str(sys.argv[1])
mWatershed.doWatershed(False, result_folder_path, False)
print("end")