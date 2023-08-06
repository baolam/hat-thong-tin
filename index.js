const http = require("http");
const express = require("express");
const ip = require("ip");
const cookie = require("cookie-parser");

const { config } = require("dotenv");
config("./env");

const app = express();
const server = http.createServer(app);
new (require("./src/socket.io/Socket"))(server);

const PORT = process.env.PORT || 3000;
const ADDRESS = ip.address();

// Một số middleware
app.use(express.json());
app.use(express.urlencoded({ extended : false }));
app.use(cookie(
    process.env.COOKIE_SECRET
));

// Đường dẫn của server
require("./src/routes/route")(app);
const { info, error } = require("./src/config/logger");

server.listen(PORT, () => {
    info(`Server is listening on port ${PORT}`);
    info(`Server's address is ${ADDRESS}`);
    require("./src/config/database").authenticate()
    .then(() => {
        info(`Connect to ${process.env.DATABASE} successfully!`);
        require("./src/models/initalize")()
        .then(res => info(res))
        .catch(err => {
            error(`Failed to synchronize table which error's content is '${err}'`);
            process.exit(1);
        })
    })
    .catch((err) => {
        error(`Failed to connect to database. ${err}`);
        process.exit(1);
    });
});