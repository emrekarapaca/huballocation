import os
ui_path = "C:/Users/krpc1/OneDrive/Masaüstü/HubAllAPP/HubAllAPP/ui/huballocation.ui"
py_path = "C:/Users/krpc1/OneDrive/Masaüstü/HubAllAPP/HubAllAPP/ui/hubAppUI.py"
os.system(f"pyuic5 {ui_path} -o {py_path}")
