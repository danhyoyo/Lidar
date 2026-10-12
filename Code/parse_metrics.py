import json
import glob
import os

runs = glob.glob('/home/duyennh/Downloads/Runs_Lidar/Heads/*')
for run in runs:
    if not os.path.isdir(run): continue
    eval_file = os.path.join(run, 'evaluation_ap_local_bev.json')
    if os.path.exists(eval_file):
        with open(eval_file) as f:
            data = json.load(f)
        mean_ap_9 = data.get('mean_ap_9_percent', 0)
        map_mod = 0
        classes = ['Car', 'Pedestrian', 'Cyclist']
        res = {}
        for c in classes:
            c_data = data.get('per_class', {}).get(c, {}).get('difficulties', {})
            easy = c_data.get('Easy', {}).get('ap_r40_percent', 0)
            mod = c_data.get('Moderate', {}).get('ap_r40_percent', 0)
            hard = c_data.get('Hard', {}).get('ap_r40_percent', 0)
            res[c] = (easy, mod, hard)
            map_mod += mod
        map_mod /= 3
        print(f"Run: {os.path.basename(run)}")
        print(f"Mean AP-9: {mean_ap_9:.2f}")
        print(f"mAP Mod: {map_mod:.2f}")
        for c in classes:
            print(f"{c}: Easy={res[c][0]:.2f}, Mod={res[c][1]:.2f}, Hard={res[c][2]:.2f}")
