from morpheus._lib.tests.pycoro import call_fib_async, raise_at_depth_async, call_async
import pytest
import asyncio

@pytest.mark.asyncio
async def test_python_cpp_async_interleave():

    def fib(n):
        if n < 0:
            raise ValueError()
            
        if n < 2:
            return 1
        
        return fib(n-1) + fib(n-2)

    async def fib_async(n):
        if n < 0:
            raise ValueError()
            
        if n < 2:
            return 1
        
        task_a = call_fib_async(fib_async, n, 1)
        task_b = call_fib_async(fib_async, n, 2)
        
        [a, b] = await asyncio.gather(task_a, task_b)
        
        return a + b
    
    assert fib(15) == await fib_async(15)

@pytest.mark.asyncio
async def test_python_cpp_async_exception():

    async def py_raise_at_depth_async(n: int):
        if n <= 0:
            raise RuntimeError("depth reached zero in python")
        
        await raise_at_depth_async(py_raise_at_depth_async, n - 1)

    depth = 100

    with pytest.raises(RuntimeError) as ex:
        await raise_at_depth_async(py_raise_at_depth_async, depth + 1)
    assert "python" in str(ex.value)

    with pytest.raises(RuntimeError) as ex:
        await raise_at_depth_async(py_raise_at_depth_async, depth)
    assert "c++" in str(ex.value)

@pytest.mark.asyncio
async def test_can_cancel_coroutine_from_python():

    counter = 0

    async def increment_recursively():
        nonlocal counter
        await asyncio.sleep(0)
        counter += 1
        await call_async(increment_recursively)

    task = asyncio.ensure_future(call_async(increment_recursively))

    await asyncio.sleep(0)
    assert counter == 0
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert counter == 1
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert counter == 2

    task.cancel()

    with pytest.raises(asyncio.exceptions.CancelledError):
        await task

    assert counter == 3
