import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/linh-pham/Downloads/vacuum_driver/install/vacuum_driver'
