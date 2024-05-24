import sounddevice as sd
import numpy as np

def device_info():
    input_info = {}
    output_info = {}
    driver_info = {'name':[], 'in_latency':[], 'out_latency':[]}
    for hostapi in sd.query_hostapis():
        hostapi_name = hostapi["name"]

        # Device Info
        in_ltc = []
        out_ltc = []
        input_info[hostapi_name]    = {"id":[], "name": [], "ch":[]}
        output_info[hostapi_name]   = {"id":[], "name": [], "ch":[]}
        for device_numbar in hostapi["devices"]:
            device = sd.query_devices(device=device_numbar)
            max_in_ch  = device["max_input_channels"]
            max_out_ch = device["max_output_channels"]

            if max_in_ch > 0:
                input_info[hostapi_name]["name"].append(device['name'])
                input_info[hostapi_name]["ch"].append(max_in_ch)
                input_info[hostapi_name]["id"].append(device_numbar)
                in_ltc.append(device["default_low_input_latency"])

            if max_out_ch > 0:
                output_info[hostapi_name]["name"].append(device['name'])
                output_info[hostapi_name]["ch"].append(max_out_ch)
                output_info[hostapi_name]["id"].append(device_numbar)
                out_ltc.append(device["default_low_output_latency"])

        # Driver Info
        driver_info["name"].append(hostapi_name)
        driver_info["in_latency"].append(np.mean(np.array(in_ltc)))
        driver_info["out_latency"].append(np.mean(np.array(out_ltc)))

    return driver_info, input_info, output_info