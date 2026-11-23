"""Merges MCAP logs with disjoint topics into one log, in log-time order.

The tracking view records each arm as its own log, and this joins them so one
recording replays every arm on one timeline. A channel's schema is written
once however many inputs share it, and a topic that appears in two inputs is
an error, since the merged log could not tell the two apart.
"""

import argparse
import heapq
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path

from mcap.reader import make_reader
from mcap.summary import Summary
from mcap.writer import CompressionType, Writer


def _summary(path: Path, reader) -> Summary:
    summary = reader.get_summary()
    if summary is None:
        raise ValueError(f"{path} has no summary section")
    return summary


def _check_topics(paths: Sequence[Path], summaries: Sequence[Summary]) -> None:
    owner: dict[str, int] = {}
    for index, summary in enumerate(summaries):
        for channel in summary.channels.values():
            first = owner.setdefault(channel.topic, index)
            if first != index:
                raise ValueError(
                    f"topic {channel.topic} is in both {paths[first]} and "
                    f"{paths[index]}")


def _output_schema(writer: Writer, schema_ids: dict, schema) -> int:
    if schema is None:
        return 0
    key = (schema.name, schema.encoding, schema.data)
    if key not in schema_ids:
        schema_ids[key] = writer.register_schema(
            name=schema.name, encoding=schema.encoding, data=schema.data)
    return schema_ids[key]


def _register_channels(writer, schema_ids, summary) -> dict[int, int]:
    """Registers one input's channels and returns its channel ids in the
    output, keyed by the input's channel id."""
    channel_ids = {}
    for channel in summary.channels.values():
        schema_id = _output_schema(
            writer, schema_ids, summary.schemas.get(channel.schema_id))
        channel_ids[channel.id] = writer.register_channel(
            topic=channel.topic, message_encoding=channel.message_encoding,
            schema_id=schema_id, metadata=channel.metadata)
    return channel_ids


def _tagged_messages(index: int, reader) -> Iterator[tuple]:
    for _, channel, message in reader.iter_messages(log_time_order=True):
        yield index, channel, message


def merge(inputs: Sequence[Path], output: Path) -> int:
    """Writes the messages of inputs to output in log-time order and returns
    the number of messages written."""
    with ExitStack() as stack:
        readers = [
            make_reader(stack.enter_context(path.open("rb")))
            for path in inputs
        ]
        summaries = [
            _summary(path, reader) for path, reader in zip(inputs, readers,
                                                          strict=True)
        ]
        _check_topics(inputs, summaries)
        with output.open("wb") as stream:
            writer = Writer(stream, compression=CompressionType.NONE)
            writer.start()
            schema_ids: dict = {}
            channel_ids = [
                _register_channels(writer, schema_ids, summary)
                for summary in summaries
            ]
            merged = heapq.merge(
                *(_tagged_messages(index, reader)
                  for index, reader in enumerate(readers)),
                key=lambda item: item[2].log_time)
            count = 0
            for index, channel, message in merged:
                writer.add_message(
                    channel_id=channel_ids[index][channel.id],
                    log_time=message.log_time, data=message.data,
                    publish_time=message.publish_time,
                    sequence=message.sequence)
                count += 1
            writer.finish()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    arguments = parser.parse_args()
    count = merge(arguments.inputs, arguments.out)
    print(f"merged {count} messages into {arguments.out}")


if __name__ == "__main__":
    main()
