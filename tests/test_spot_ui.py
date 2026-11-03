"""The spot page is static; the backend must never receive trading authority."""
from fastapi.testclient import TestClient
from dashboard import app


def test_spot_page_and_assets_are_served_without_server_exchange_route():
    client = TestClient(app)
    response = client.get('/spot/')
    assert response.status_code == 200
    assert 'HYDRA · Spot' in response.text
    assert 'connect-src https://api.hyperliquid.xyz' in response.text
    import re
    script = re.search(r'src="(\./assets/[^"]+\.js)"', response.text).group(1)
    asset = client.get('/spot/' + script.removeprefix('./'))
    assert asset.status_code == 200
    assert client.post('/api/spot/order', json={}).status_code == 404
