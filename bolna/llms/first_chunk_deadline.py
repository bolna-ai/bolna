import asyncio


class FirstChunkTimeout(Exception):
    """The provider sent nothing within the first-chunk deadline."""


class PrimedStream:
    """A provider stream whose first chunk was read ahead; iterating replays it, then the rest."""

    def __init__(self, stream, iterator, first):
        self.response = getattr(stream, "response", None)
        self._stream, self._iterator, self._first = stream, iterator, first

    async def __aiter__(self):
        yield self._first
        async for chunk in self._iterator:
            yield chunk

    async def close(self):
        await self._stream.close()


async def read_first_chunk(open_stream, timeout_s):
    """Open a stream and read its first chunk, closing it and raising FirstChunkTimeout past timeout_s."""

    async def opened():
        stream = await open_stream()
        iterator = aiter(stream)
        try:
            return PrimedStream(stream, iterator, await anext(iterator))
        except StopAsyncIteration:
            return stream
        except asyncio.CancelledError:
            await stream.close()
            raise

    try:
        return await asyncio.wait_for(opened(), timeout_s)
    except asyncio.TimeoutError as e:
        raise FirstChunkTimeout(f"no response within {timeout_s}s") from e
