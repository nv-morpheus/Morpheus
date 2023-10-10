import asyncio

from morpheus._lib import llm
from morpheus._lib import pycoro


async def test_pycoro():

    hit_inside = False

    async def inner():

        nonlocal hit_inside

        await asyncio.sleep(1)

        hit_inside = True

        return ['a', 'b', 'c']

    returned_val = await pycoro.wrap_coroutine(inner())

    assert returned_val == 'a'
    assert hit_inside


async def test_pycoro_many():

    expected_count = 1000
    hit_count = 0

    start_time = asyncio.get_running_loop().time()

    async def inner():

        nonlocal hit_count

        await asyncio.sleep(1)

        hit_count += 1

        return ['a', 'b', 'c']

    coros = [pycoro.wrap_coroutine(inner()) for _ in range(expected_count)]

    returned_vals = await asyncio.gather(*coros)

    end_time = asyncio.get_running_loop().time()

    assert returned_vals == ['a'] * expected_count
    assert hit_count == expected_count
    assert (end_time - start_time) < 1.5
