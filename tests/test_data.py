from src.data import load_metr_la


def test_load_metr_la_reads_downloaded_dataset():
    df, adj, sensor_ids = load_metr_la('data/metr-la.h5', 'data/adj_mx.pkl')

    assert df.shape[0] > 0
    assert df.shape[1] == len(sensor_ids)
    assert adj.shape == (len(sensor_ids), len(sensor_ids))
    assert list(df.columns[:3]) == ['773869', '767541', '767542']
    assert df.index.is_monotonic_increasing
