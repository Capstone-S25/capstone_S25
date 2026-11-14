from fractions import Fraction

import pytest

from data_pipeline.video_ingestion.streaming import worker


@pytest.mark.parametrize(("now", "delay"), [(100.5, 1.5), (103.0, 0.0)])
def test_waits_for_playback_time_but_does_not_delay_when_already_late(
    stream, now, delay
):
    stream.clock.monotonic.return_value = now

    stopped = worker._pace_file_frame(stream.source, Fraction(2), 100.0, stream.stop)

    stream.stop.wait.assert_called_once_with(delay)
    assert not stopped
