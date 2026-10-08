#!/usr/bin/env python3
import json,pathlib,urllib.parse
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from lab import run
ROOT=pathlib.Path(__file__).resolve().parent
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  try:
   u=urllib.parse.urlparse(self.path)
   if u.path=='/':payload=(ROOT/'index.html').read_bytes();mime='text/html; charset=utf-8'
   elif u.path=='/api/run':
    q=urllib.parse.parse_qs(u.query)
    val=lambda key,default:int(q.get(key,[str(default)])[0])
    payload=json.dumps(run(val('seed',12345),val('bits',65536),val('gb',0),val('steps',1),val('mb',0))).encode();mime='application/json'
   else:self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
  except Exception as e:
   payload=json.dumps({'error':str(e)}).encode();self.send_response(400);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)
if __name__=='__main__':
 print('Open http://localhost:8765');ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
