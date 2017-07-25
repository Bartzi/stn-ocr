import argparse
import json
import threading
import tkinter
from socketserver import BaseRequestHandler, ThreadingUDPServer, DatagramRequestHandler, ThreadingMixIn, UDPServer, \
    StreamRequestHandler, TCPServer

import base64
import numpy as np
from PIL import Image, ImageTk


class ProgressWindow:

    def __init__(self, root):
        self.frame = tkinter.Frame(root)
        self.frame.pack(fill=tkinter.BOTH, expand=tkinter.YES)

        self._image = None
        self._sprite = None
        self.canvas = tkinter.Canvas(
            self.frame,
            width=850,
            height=400
        )
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        image = ImageTk.PhotoImage(value)
        self._image = image
        self._sprite = self.canvas.create_image(value.width // 2, value.height // 2, image=self._image)
        self.canvas.config(width=value.width, height=value.height)


class ImageDataHandler(StreamRequestHandler):

    def __init__(self, *args, **kwargs):
        self.window = kwargs.pop('window')
        super(ImageDataHandler, self).__init__(*args, **kwargs)

    def handle(self):
        data = self.rfile.read()
        data = json.loads(data.decode('utf-8'))
        width = data['width']
        height = data['height']
        data = np.fromstring(base64.b64decode(data['image']), dtype=np.uint8)
        data = np.resize(data, (height, width, 3))
        image = Image.fromarray(data, mode='RGB')
        self.window.image = image


class ImageServer(ThreadingMixIn, TCPServer):

    def __init__(self, *args, **kwargs):
        self.window = kwargs.pop('window')
        super(ImageServer, self).__init__(*args, **kwargs)

    def finish_request(self, request, client_address):
        self.RequestHandlerClass(request, client_address, self, window=self.window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that shows current pictures of a training')
    parser.add_argument('--host', default='0.0.0.0', help='address to listen on')
    parser.add_argument('--port', type=int, default=1337, help='port to listen on')

    args = parser.parse_args()

    root = tkinter.Tk()
    window = ProgressWindow(root)

    print("starting server")
    server = ImageServer((args.host, args.port), ImageDataHandler, window=window)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.deamon = True
    server_thread.start()

    print("starting window")
    root.mainloop()
    server.shutdown()
    server.server_close()




