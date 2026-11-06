"""
Interactive Web Application & REST API for Build-Failure Prediction.
Zero external web dependencies: runs using Python's standard library http.server.
"""

import http.server
import socketserver
import json
import urllib.parse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import load_artifacts, predict_single

PORT = 8050
STATIC_DIR = os.path.dirname(__file__)

class BFPRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)
        
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/' or parsed.path == '/index.html':
            self.path = '/index.html'
            return super().do_GET()
        elif parsed.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy', 'model': 'Logistic_Regression'}).encode('utf-8'))
            return
        elif parsed.path.startswith('/plots/'):
            # Serve plots from ../plots
            plot_file = os.path.join(os.path.dirname(__file__), '..', self.path.lstrip('/'))
            if os.path.exists(plot_file):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.end_headers()
                with open(plot_file, 'rb') as f:
                    self.wfile.write(f.read())
                return
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(post_body)
                model, preprocessor, metadata = load_artifacts('models')
                result = predict_single(data, model, preprocessor)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), BFPRequestHandler) as httpd:
        print(f"============================================================")
        print(f"  Build-Failure Prediction Dashboard Live at:              ")
        print(f"  👉 http://localhost:{PORT}                                ")
        print(f"============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")

if __name__ == '__main__':
    run_server()

