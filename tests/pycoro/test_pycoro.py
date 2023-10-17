from morpheus._lib.tests.pycoro import int_as_task, call_fib_async
import pytest
import asyncio

@pytest.mark.asyncio
async def test_hey():

    def fib(n):
        if n < 0:
            raise ValueError()
            
        elif n == 0:
            return 1
        
        elif n == 1:
            return 1
        
        return fib(n-1) + fib(n-2)

    async def fib_async(n):
        if n < 0:
            raise ValueError()
            
        if n < 2:
            return await int_as_task(1)
        
        task_a = call_fib_async(fib_async, n, 1)
        task_b = call_fib_async(fib_async, n, 2)
        
        [a, b] = await asyncio.gather(task_a, task_b)
        
        return await int_as_task(a + b)
    
    assert fib(20) == await fib_async(20)
