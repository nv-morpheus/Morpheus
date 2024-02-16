import aiohttp
import pytest

@pytest.mark.usefixtures("launch_mock_triton")
async def test_can_hit_triton_mock():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/v2/health/live") as response:
            assert response.status == 200

@pytest.mark.usefixtures("launch_mock_triton")
async def test_can_set_failure_count():
    async with aiohttp.ClientSession() as session:
        data = {"failure_count": 2}
        async with session.post("http://localhost:8000/state/failures", data=data) as response:
            assert response.status == 200
            
        async with session.get("http://localhost:8000/state/failures") as response:
            assert response.status == 500

        async with session.get("http://localhost:8000/state/failures") as response:
            assert response.status == 500

        async with session.get("http://localhost:8000/state/failures") as response:
            assert response.status == 200
