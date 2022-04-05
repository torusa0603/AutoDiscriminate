import mWatershed
import sys 
import re

result_folder_path = str(sys.argv[1])
mWatershed.main(False, result_folder_path, False, True)
print("end")