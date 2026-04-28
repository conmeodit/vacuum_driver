import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/linh-pham/Documents/vacuum_driver_1/vacuum_driver/install/vacuum_driver'
