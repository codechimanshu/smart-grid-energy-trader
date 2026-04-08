import http.server
import socketserver
import json

class HackathonHealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "running"}')

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"status": "ok", "success": True}
        self.wfile.write(json.dumps(response).encode('utf-8'))

def main():
    PORT = 7860
    with socketserver.TCPServer(("0.0.0.0", PORT), HackathonHealthHandler) as httpd:
        print(f"Serving hackathon endpoints on port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    main()