"""Télécharge metr-la.h5 (Google Drive DCRNN) et adj_mx.pkl (dépôt liyaguang/DCRNN) dans data/.
Usage : python -m src.download_data"""
import glob
import os
import shutil
import urllib.request

DRIVE_FOLDER = "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
RAW = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/"
ADJ_URL = RAW + "adj_mx.pkl"
LOC_URL = RAW + "graph_sensor_locations.csv"  # lat/lon des 207 capteurs (pour la carte)


def main(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    h5 = os.path.join(data_dir, "metr-la.h5")
    if not os.path.exists(h5):
        import gdown
        tmp = os.path.join(data_dir, "dcrnn_drive")
        gdown.download_folder(DRIVE_FOLDER, output=tmp, quiet=False)
        found = glob.glob(os.path.join(tmp, "**", "metr-la.h5"), recursive=True)
        if not found:
            raise SystemExit(f"metr-la.h5 introuvable : télécharge-le à la main depuis {DRIVE_FOLDER} vers {h5}")
        shutil.copy(found[0], h5)
    adj = os.path.join(data_dir, "adj_mx.pkl")
    if not os.path.exists(adj):
        urllib.request.urlretrieve(ADJ_URL, adj)
    loc = os.path.join(data_dir, "graph_sensor_locations.csv")
    if not os.path.exists(loc):
        urllib.request.urlretrieve(LOC_URL, loc)
    for f in (h5, adj, loc):
        print(f, os.path.getsize(f), "octets")


if __name__ == "__main__":
    main()
