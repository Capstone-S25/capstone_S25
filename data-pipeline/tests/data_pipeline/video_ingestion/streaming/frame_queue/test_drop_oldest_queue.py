from queue import Empty
from threading import Event, Thread

from data_pipeline.video_ingestion.streaming.frame_queue import DropOldestQueue


def test_preserves_fifo_order_without_dropping_when_capacity_is_not_exceeded():
    queue = DropOldestQueue[int](maxsize=8)
    for value in range(8):
        queue.put_nowait(value)

    assert queue.qsize() == 8
    assert queue.dropped_count == 0
    for expected in range(8):
        assert queue.get_nowait() == expected
        queue.task_done()
    assert queue.empty()
    assert queue.unfinished_tasks == 0


def test_overflow_keeps_newest_frames_and_removes_dropped_items_from_task_count():
    queue = DropOldestQueue[int](maxsize=8)
    for value in range(12):
        queue.put_nowait(value)

    assert queue.qsize() == 8
    assert queue.dropped_count == 4
    assert queue.unfinished_tasks == 8
    for expected in range(4, 12):
        assert queue.get_nowait() == expected
        queue.task_done()

    # Dropped items must not leave join() waiting for nonexistent consumer work.
    assert queue.unfinished_tasks == 0
    queue.join()


def test_overflow_does_not_drop_or_complete_the_frame_already_being_processed():
    queue = DropOldestQueue[int](maxsize=2)
    queue.put_nowait(0)
    in_flight = queue.get_nowait()
    for value in (1, 2, 3):
        queue.put_nowait(value)

    assert in_flight == 0
    assert queue.dropped_count == 1
    assert queue.unfinished_tasks == 3  # One in flight and two waiting.
    queue.task_done()
    assert queue.get_nowait() == 2
    queue.task_done()
    assert queue.get_nowait() == 3
    queue.task_done()
    assert queue.unfinished_tasks == 0
    queue.join()


def test_concurrent_producer_and_consumer_preserve_order_and_account_for_every_frame():
    queue = DropOldestQueue[int](maxsize=8)
    ready = Event()
    consumer_started = Event()
    producer_done = Event()
    received = []
    errors = []
    total_frames = 1000

    def produce():
        try:
            # Guarantee overflow before allowing consumption, without sleeps.
            for value in range(16):
                queue.put_nowait(value)
            ready.set()
            if not consumer_started.wait(5):
                raise RuntimeError("Consumer did not start")
            for value in range(16, total_frames):
                queue.put_nowait(value)
                assert queue.qsize() <= 8
        except BaseException as error:
            errors.append(error)
        finally:
            producer_done.set()

    def consume():
        try:
            if not ready.wait(5):
                raise RuntimeError("Producer did not enqueue frames")
            while True:
                try:
                    frame = queue.get(timeout=0.01)
                except Empty:
                    if producer_done.is_set():
                        return
                    continue
                received.append(frame)
                consumer_started.set()
                queue.task_done()
        except BaseException as error:
            errors.append(error)

    producer = Thread(target=produce, daemon=True)
    consumer = Thread(target=consume, daemon=True)
    consumer.start()
    producer.start()
    producer.join(timeout=5)
    consumer.join(timeout=5)

    assert not producer.is_alive()
    assert not consumer.is_alive()
    assert not errors
    assert received == sorted(set(received))
    assert received[-1] == total_frames - 1
    assert queue.dropped_count >= 8
    assert len(received) + queue.dropped_count == total_frames
    assert queue.empty()
    assert queue.unfinished_tasks == 0
    queue.join()
