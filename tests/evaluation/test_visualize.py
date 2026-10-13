import numpy as np
import pandas as pd

from fluxguard.evaluation.visualize import (
    plot_class_distributions,
    plot_cwru_multi_domain,
)


def test_plot_cwru_multi_domain(tmp_path) -> None:
    signal = np.sin(2 * np.pi * 100 * np.arange(1024) / 1000)
    out_file = tmp_path / "test_multidomain.png"
    stats = plot_cwru_multi_domain(signal, 1000.0, out_file)
    assert out_file.exists()
    assert "rms" in stats
    assert "kurtosis" in stats


def test_plot_class_distributions(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "fault_type": ["healthy", "ball", "inner_race", "outer_race"],
            "rms": [0.05, 0.12, 0.15, 0.18],
            "kurtosis": [0.0, 3.2, 4.5, 2.8],
        }
    )
    out_file = tmp_path / "test_distributions.png"
    plot_class_distributions(df, out_file)
    assert out_file.exists()
