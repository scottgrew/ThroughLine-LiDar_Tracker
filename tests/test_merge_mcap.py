"""Tests for eval/merge_mcap.py. The expected topics, counts and times are
written here, not computed from the merge."""

import tempfile
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer
from merge_mcap import merge

SCHEMA = b'{"type": "object"}'


def write_log(path, topic, log_times):
    with path.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start()
        schema_id = writer.register_schema(
            name="Ping", encoding="jsonschema", data=SCHEMA)
        channel_id = writer.register_channel(
            topic=topic, message_encoding="json", schema_id=schema_id)
        for log_time in log_times:
            writer.add_message(channel_id=channel_id, log_time=log_time,
                               data=b"{}", publish_time=log_time)
        writer.finish()


def read_messages(path):
    with path.open("rb") as stream:
        return list(make_reader(stream).iter_messages())


def test_merge_interleaves_disjoint_topics_in_log_time_order():
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "first.mcap"
        second = Path(directory) / "second.mcap"
        output = Path(directory) / "merged.mcap"
        write_log(first, "/a", [1, 3])
        write_log(second, "/b", [2, 4])
        count = merge([first, second], output)
        messages = read_messages(output)
    topics = {channel.topic for _, channel, _ in messages}
    log_times = [message.log_time for _, _, message in messages]
    assert topics == {"/a", "/b"}
    assert len(messages) == 4
    assert log_times == [1, 2, 3, 4]
    assert count == 4


def test_topic_in_two_inputs_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "first.mcap"
        second = Path(directory) / "second.mcap"
        output = Path(directory) / "merged.mcap"
        write_log(first, "/a", [1])
        write_log(second, "/a", [2])
        try:
            merge([first, second], output)
        except ValueError as error:
            assert "/a" in str(error)
        else:
            raise AssertionError("merge must reject a topic in two inputs")
