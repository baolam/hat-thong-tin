const { transports } = require("winston");
require("winston-daily-rotate-file");

const path = require("path");

const DailyLogger = new transports.DailyRotateFile({
    filename : `app.%DATE%.log`,
    datePattern : 'DD-MM-YYYY',
    dirname : path.join(__dirname, "../../", process.env.LOG_FOLDER),
    maxSize : 5242880
});

const HttpLogger = new transports.Http({
    host : process.env.HOST,
    port : parseInt(process.env.LOG_PORT)
});

module.exports = {
    DailyLogger, HttpLogger
};