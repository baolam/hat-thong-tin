const socketio = require("socket.io");
const DeviceConnection = require("./DeviceConnection");

class Socket {
    constructor(server)
    {
        /// Cài đặt thêm options
        let io = new socketio.Server(server, { });
        this.device = io.of(process.env.NAMESPACE_DEVICE);
        
        new DeviceConnection(this.device);
    }
}

module.exports = Socket;