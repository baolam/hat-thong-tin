const morgan = require("morgan");
const { logger } = require("../config/logger");

const express = require("express");
const Router = express.Router();

Router.use((req, res, next) => require("../middleware/LoginMiddleware")(req, res, next));
Router.use("/schedule", require("./schedule"));
Router.use("/user", require("./user"));

function route(app)
{
    app.use(morgan("common", { stream : {
        write : (msg) => logger.info(msg)
    } }));
    app.use("/api", Router);
}

module.exports = route;