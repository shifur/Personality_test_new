import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import server

def test_root_get():
    app = server.app
    with app.test_client() as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.data.decode('utf-8') == 'testing :)'
