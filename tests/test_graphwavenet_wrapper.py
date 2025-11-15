import numpy as np
import pandas as pd

from traffic_forecast.evaluation.graphwavenet_wrapper import GraphWaveNetWrapper


class _DummyPredictor:
    """Simple stand-in that returns deterministic sequences for testing."""

    def predict(self, X):
        num_seq = X.shape[0]
        num_edges = X.shape[2]
        values = np.arange(num_seq * num_edges, dtype=float)
        return values.reshape(num_seq, num_edges, 1)


def test_predict_alignment_preserves_row_order():
    timestamps = pd.date_range('2025-01-01', periods=5, freq='15min')
    edges = [(1, 2), (1, 3)]

    rows = []
    speed_value = 10.0
    for ts in timestamps:
        for (node_a, node_b) in edges:
            rows.append(
                {
                    'timestamp': ts,
                    'node_a_id': node_a,
                    'node_b_id': node_b,
                    'speed': speed_value,
                }
            )
            speed_value += 1.0

    # Shuffle to ensure ordering in the raw dataframe does not match pivot order
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)

    wrapper = GraphWaveNetWrapper(sequence_length=2, max_interp_gap=1, imputation_noise=0.0)
    wrapper.model = _DummyPredictor()
    wrapper._trained = True

    predictions, _ = wrapper.predict(df, device='cpu')

    df['_edge_id'] = df.apply(wrapper._create_edge_id, axis=1)
    sorted_timestamps = sorted(df['timestamp'].unique())
    target_timestamps = sorted_timestamps[wrapper.sequence_length :]
    edge_order = wrapper.edge_order

    expected_map = {}
    current_value = 0.0
    for ts in target_timestamps:
        for edge_id in edge_order:
            expected_map[(ts, edge_id)] = current_value
            current_value += 1.0

    expected = df.apply(lambda row: expected_map.get((row['timestamp'], row['_edge_id'])), axis=1).to_numpy()

    assert np.allclose(predictions, expected, equal_nan=True)
